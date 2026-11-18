"""Fidelity quantum kernels and finite-shot estimation.

    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2

Given statevectors for all samples, the full Gram matrix is a single matrix product,
so the cost is O(N) circuit simulations + O(N^2 2^n) arithmetic instead of O(N^2) circuit
simulations as in the naive compute-uncompute construction.

On hardware each entry is estimated with the compute-uncompute circuit
U(x_j)^dagger U(x_i)|0> by counting the all-zeros outcome over S shots, i.e.
K_hat ~ Binomial(S, K) / S. :func:`shot_noise` reproduces exactly that statistic.
"""
from __future__ import annotations

import numpy as np

from .simulator import simulate


def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray | None = None) -> np.ndarray:
    psi_b = psi_a if psi_b is None else psi_b
    return np.abs(psi_a.conj() @ psi_b.T) ** 2


def quantum_kernel(fmap, XA, XB=None):
    A = simulate(fmap, XA)
    return fidelity_kernel(A, None if XB is None else simulate(fmap, XB))


def shot_noise(K: np.ndarray, shots: int | None, rng: np.random.Generator, symmetric: bool):
    """Replace exact fidelities by finite-shot estimates (diagonal of a Gram matrix stays 1)."""
    if shots is None:
        return K
    Kc = np.clip(K, 0.0, 1.0)
    Kh = rng.binomial(shots, Kc) / shots
    if symmetric:
        Kh = np.triu(Kh, 1)
        Kh = Kh + Kh.T + np.eye(len(K))
    return Kh


def project_psd(K: np.ndarray) -> np.ndarray:
    """Nearest positive-semidefinite matrix (clip negative eigenvalues), needed after shot noise."""
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return (V * np.clip(w, 0, None)) @ V.T


def rbf_kernel(XA, XB=None, gamma=1.0):
    XB = XA if XB is None else XB
    d2 = (XA**2).sum(1)[:, None] + (XB**2).sum(1)[None, :] - 2 * XA @ XB.T
    return np.exp(-gamma * np.clip(d2, 0, None))


def state_fidelity_kernel(states_a, states_b=None):
    """Kernel built directly from *physical* quantum states, e.g. TFIM ground states.

    For neighbouring couplings this is the ground-state fidelity whose collapse signals a
    quantum critical point (fidelity susceptibility).
    """
    return fidelity_kernel(states_a, states_b)
