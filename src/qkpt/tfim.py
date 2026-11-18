"""1D transverse-field Ising model (periodic chain), the canonical quantum phase transition.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i ,   critical point h_c / J = 1

Ground states are obtained by sparse exact diagonalisation. To mimic data coming off a
real quantum simulator, each dataset sample is built from a *finite number of projective
measurements* ("snapshots") of the ground state in the Z and X bases, so the features
carry genuine shot noise rather than being exact expectation values.

Basis convention follows Qiskit: computational-basis index b = sum_k bit_k 2^k, bit_k = 1
means Z_k = -1.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _bits(N):
    idx = np.arange(2**N)
    return ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)  # (2^N, N)


def hamiltonian(N: int, h: float, J: float = 1.0) -> sp.csr_matrix:
    dim = 2**N
    z = 1 - 2 * _bits(N).astype(float)            # Z eigenvalues, (dim, N)
    diag = -J * np.sum(z * np.roll(z, -1, axis=1), axis=1)
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [diag]
    idx = np.arange(dim)
    for k in range(N):                             # X_k flips bit k
        rows.append(idx)
        cols.append(idx ^ (1 << k))
        vals.append(np.full(dim, -h))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim))


def ground_state(N: int, h: float, J: float = 1.0) -> np.ndarray:
    H = hamiltonian(N, h, J)
    if 2**N <= 256:
        w, v = np.linalg.eigh(H.toarray())
        psi = v[:, 0]
    else:
        w, v = eigsh(H, k=1, which="SA", tol=1e-12)
        psi = v[:, 0]
    # The Z2-symmetric ground state; fix global phase so the largest amplitude is real positive
    psi = psi * np.exp(-1j * np.angle(psi[np.argmax(np.abs(psi))]))
    return psi / np.linalg.norm(psi)


def hadamard_all(psi: np.ndarray, N: int) -> np.ndarray:
    """Apply H^{(x)N} (rotate X basis -> Z basis) via a fast Walsh-Hadamard transform."""
    a = psi.astype(complex).reshape([2] * N)
    Hm = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for ax in range(N):
        a = np.moveaxis(np.tensordot(Hm, a, axes=([1], [ax])), 0, ax)
    return a.reshape(-1)


def snapshot_features(psi: np.ndarray, N: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Estimate observables from `shots` Z-basis and `shots` X-basis projective measurements.

    Features (all estimated, hence noisy):
        |m_z|        mean |sum_i Z_i| / N        (order parameter; Z2-symmetric GS has <m_z> = 0)
        C_zz(1)      <Z_i Z_{i+1}>
        C_zz(N/2)    <Z_i Z_{i+N/2}>
        m_x          <X_i>                       (transverse magnetisation)
    """
    bits = _bits(N)
    pz = np.abs(psi) ** 2
    zs = 1 - 2 * bits[rng.choice(len(pz), size=shots, p=pz / pz.sum())]
    px = np.abs(hadamard_all(psi, N)) ** 2
    xs = 1 - 2 * bits[rng.choice(len(px), size=shots, p=px / px.sum())]
    mz = np.abs(zs.mean(axis=1)).mean()
    c1 = (zs * np.roll(zs, -1, axis=1)).mean()
    cN = (zs * np.roll(zs, -(N // 2), axis=1)).mean()
    mx = xs.mean()
    return np.array([mz, c1, cN, mx])


TFIM_FEATURE_NAMES = ["|m_z|", "C_zz(1)", "C_zz(N/2)", "m_x"]


def sample_tfim(N: int, fields, n_samples_per_h: int = 30, shots: int = 100, seed: int = 0):
    """Build a measurement-based dataset.

    Returns
    -------
    X      : (n, 4) noisy features
    hs     : (n,) transverse field of each sample
    states : (len(fields), 2^N) exact ground states (for the quantum-data fidelity kernel)
    """
    rng = np.random.default_rng(seed)
    fields = np.asarray(fields, float)
    X, hs, states = [], [], []
    for h in fields:
        psi = ground_state(N, h)
        states.append(psi)
        for _ in range(n_samples_per_h):
            X.append(snapshot_features(psi, N, shots, rng))
            hs.append(h)
    return np.array(X), np.array(hs), np.array(states)
