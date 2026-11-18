import numpy as np
import pytest

from qkpt import ISING_TC
from qkpt.criticality import fit_crossing, fss_extrapolate
from qkpt.feature_maps import FEATURE_MAPS, make_feature_map
from qkpt.features import ising_features, observables
from qkpt.ising import sample_ising
from qkpt.kernels import fidelity_kernel, project_psd, shot_noise
from qkpt.metrics import geometric_difference, kernel_target_alignment
from qkpt.models import QuantumKernelSVC
from qkpt.simulator import simulate, simulate_qiskit, to_qiskit
from qkpt.tfim import ground_state, hamiltonian

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ simulator vs Qiskit
@pytest.mark.parametrize("name", FEATURE_MAPS + ["ZZ-circular"])
def test_batched_simulator_matches_qiskit(name):
    X = rng.normal(size=(6, 4))
    fm = make_feature_map(name, 4)
    assert np.allclose(simulate(fm, X), simulate_qiskit(fm, X), atol=1e-10)


@pytest.mark.parametrize("ent", ["linear", "full", "circular"])
def test_zz_map_identical_to_qiskit_library(ent):
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector
    X = rng.normal(size=(5, 3))
    ref = zz_feature_map(3, reps=2, entanglement=ent)
    R = np.array([Statevector(ref.assign_parameters(x)).data for x in X])
    assert np.allclose(R, simulate(make_feature_map(f"ZZ-{ent}", 3), X), atol=1e-10)


def test_kernel_matches_qiskit_machine_learning():
    qml = pytest.importorskip("qiskit_machine_learning.kernels")
    X = rng.normal(size=(8, 3))
    fm = make_feature_map("ZZ-linear", 3)
    K_ref = qml.FidelityStatevectorKernel(feature_map=to_qiskit(fm)).evaluate(X)
    assert np.allclose(fidelity_kernel(simulate(fm, X)), K_ref, atol=1e-10)


# ------------------------------------------------------------------ kernels
def test_kernel_is_valid_gram_matrix():
    K = fidelity_kernel(simulate(make_feature_map("HEA", 4), rng.normal(size=(20, 4))))
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10


def test_shot_noise_unbiased_and_psd():
    K = fidelity_kernel(simulate(make_feature_map("ZZ-linear", 3), rng.normal(size=(15, 3))))
    est = np.mean([shot_noise(K, 256, np.random.default_rng(s), True) for s in range(200)], axis=0)
    assert np.abs(est - K).max() < 0.02
    Kp = project_psd(shot_noise(K, 16, rng, True))
    assert np.linalg.eigvalsh(Kp).min() > -1e-10


def test_geometric_difference_of_kernel_with_itself_is_about_one():
    K = fidelity_kernel(simulate(make_feature_map("ZZ-linear", 3), 0.3 * rng.normal(size=(30, 3))))
    assert geometric_difference(K, K, reg=1e-6) == pytest.approx(1.0, abs=1e-2)


# ------------------------------------------------------------------ physics
def test_ising_low_and_high_temperature_limits():
    c, T = sample_ising(8, [1.0, 5.0], n_samples_per_T=10, n_chains=2, n_therm=50, n_between=2)
    o = observables(c)
    assert o["|m|"][T == 1.0].mean() > 0.95 and o["|m|"][T == 5.0].mean() < 0.35
    assert o["e"][T == 1.0].mean() == pytest.approx(-2.0, abs=0.06)


def test_tfim_energy_matches_exact_free_fermion_solution():
    N, h = 8, 0.7
    E0 = np.linalg.eigvalsh(hamiltonian(N, h).toarray())[0]
    # periodic chain, even-parity sector: k = (2n+1) pi / N
    k = (2 * np.arange(N) + 1) * np.pi / N
    exact = -np.sum(np.sqrt(1 + h**2 - 2 * h * np.cos(k)))
    assert E0 == pytest.approx(exact, abs=1e-9)
    assert np.linalg.norm(ground_state(N, h)) == pytest.approx(1.0)


# ------------------------------------------------------------------ end-to-end
def test_quantum_kernel_svc_separates_phases():
    c, T = sample_ising(8, [1.2, 1.6, 3.2, 3.8], n_samples_per_T=30, n_chains=3, n_therm=50, n_between=2)
    X, _ = ising_features(c, "physics")
    y = (T < ISING_TC).astype(int)
    clf = QuantumKernelSVC("ZZ-linear", bandwidth=0.3, C=10).fit(X[::2], y[::2])
    assert clf.score(X[1::2], y[1::2]) > 0.95
    assert kernel_target_alignment(clf.K_train_, y[::2]) > 0.2


def test_crossing_and_fss():
    t = np.linspace(1, 3.5, 30)
    t0, _, _ = fit_crossing(t, 1 / (1 + np.exp((t - 2.3) / 0.1)), 2.0)
    assert t0 == pytest.approx(2.3, abs=1e-3)
    L = np.array([8, 16, 32])
    tc, _, _ = fss_extrapolate(L, 2.269 + 0.5 / L, np.full(3, 0.01))
    assert tc == pytest.approx(2.269, abs=1e-6)
