"""qkpt — Quantum Kernel learning for Phase Transitions.

Modules
-------
ising         Monte Carlo sampler (Wolff cluster algorithm) for the 2D classical Ising model.
tfim          Exact diagonalisation + projective-measurement sampling for the 1D transverse-field Ising model.
features      Physical observables (order parameter, energy, correlation functions) and PCA features.
feature_maps  Quantum feature maps defined once and exported to both Qiskit and the batched simulator.
simulator     Batched NumPy statevector engine (validated against qiskit.quantum_info.Statevector).
kernels       Fidelity quantum kernels, finite-shot noise, PSD projection, classical kernels.
models        scikit-learn compatible QuantumKernelSVC and the classical baseline zoo.
metrics       Kernel-target alignment, spectra, concentration, geometric difference (Huang et al. 2021).
criticality   Critical-point estimation from classifier output + finite-size scaling.
"""

__version__ = "0.1.0"

ISING_TC = 2.0 / __import__("math").log(1.0 + 2.0 ** 0.5)  # Onsager: 2.269185...
TFIM_HC = 1.0  # critical transverse field h/J of the 1D TFIM
