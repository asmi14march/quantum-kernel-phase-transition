"""Estimate quantum-kernel entries with real compute-uncompute circuits via Qiskit primitives.

By default runs on the local ``StatevectorSampler`` (shot-based) and checks the estimates
against the exact fidelity. With ``--backend <name>`` it transpiles for, and submits to, an
IBM Quantum device through ``qiskit-ibm-runtime`` (requires a saved account).

    python scripts/hardware_kernel_demo.py --shots 4096
    python scripts/hardware_kernel_demo.py --backend ibm_brisbane --shots 4096 --pairs 10
"""
import argparse

import numpy as np

from qkpt import ISING_TC
from qkpt.experiment import DATA
from qkpt.feature_maps import make_feature_map
from qkpt.features import ising_features
from qkpt.kernels import fidelity_kernel
from qkpt.simulator import simulate, to_qiskit

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default=None)
ap.add_argument("--shots", type=int, default=4096)
ap.add_argument("--pairs", type=int, default=8)
ap.add_argument("--bandwidth", type=float, default=0.2)
args = ap.parse_args()

d = np.load(DATA / "ising_L16.npz")
X, _ = ising_features(d["configs"], "local")
Z = (X - X.mean(0)) / X.std(0) * args.bandwidth
rng = np.random.default_rng(0)
pairs = [tuple(rng.choice(len(Z), 2, replace=False)) for _ in range(args.pairs)]

fmap = make_feature_map("ZZ-linear", Z.shape[1])
U = to_qiskit(fmap)


def compute_uncompute(xi, xj):
    qc = U.assign_parameters(xi).compose(U.assign_parameters(xj).inverse())
    qc.measure_all()
    return qc


circuits = [compute_uncompute(Z[i], Z[j]) for i, j in pairs]

if args.backend is None:
    from qiskit.primitives import StatevectorSampler
    sampler, isa = StatevectorSampler(), circuits
    where = "local StatevectorSampler"
else:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    backend = QiskitRuntimeService().backend(args.backend)
    isa = generate_preset_pass_manager(optimization_level=3, backend=backend).run(circuits)
    sampler = SamplerV2(mode=backend)
    where = args.backend
    print("two-qubit depth after transpilation:",
          [c.depth(lambda g: g.operation.num_qubits == 2) for c in isa])

result = sampler.run(isa, shots=args.shots).result()
n = fmap.n_qubits
print(f"Kernel entries on {where}, {args.shots} shots, T = temperature of each sample")
print(f"{'T_i':>6} {'T_j':>6} {'exact':>8} {'estimate':>9} {'|err|':>7}")
for (i, j), res in zip(pairs, result):
    counts = res.data.meas.get_counts()
    est = counts.get("0" * n, 0) / args.shots
    exact = fidelity_kernel(simulate(fmap, Z[[i]]), simulate(fmap, Z[[j]]))[0, 0]
    print(f"{d['temps'][i]:6.2f} {d['temps'][j]:6.2f} {exact:8.4f} {est:9.4f} {abs(est-exact):7.4f}")
