"""Diagnostics that explain *why* a kernel generalises (or fails to).

* Kernel-target alignment (Cristianini et al.; centred version of Cortes et al. 2012).
* Spectrum and participation ratio — an effective dimension of the feature space seen by the data.
* Exponential concentration — off-diagonal kernel values collapse towards 2^-n as the
  number of qubits grows, so finite shots can no longer resolve them (Thanasilp et al. 2022).
* Geometric difference g(K_C || K_Q) (Huang et al., Nat. Commun. 12, 2631, 2021). If g is
  close to 1, a classical kernel can match the quantum kernel on *any* labelling of these
  inputs, so no quantum advantage is possible for this data.
"""
from __future__ import annotations

import numpy as np


def _center(K):
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kernel_target_alignment(K, y):
    y = np.where(np.asarray(y) > 0, 1.0, -1.0)
    Kc = _center(K)
    Y = np.outer(y, y)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def spectrum(K):
    w = np.linalg.eigvalsh(0.5 * (K + K.T))[::-1]
    w = np.clip(w, 0, None)
    return w / w.sum()


def participation_ratio(K):
    p = spectrum(K)
    return float(1.0 / np.sum(p**2))


def concentration(K):
    off = K[~np.eye(len(K), dtype=bool)]
    return float(off.mean()), float(off.var())


def _sqrtm_psd(K):
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def geometric_difference(K_classical, K_quantum, reg=1e-2):
    """Regularised geometric difference of Huang et al. (2021), both traces scaled to N:

        g(K_C || K_Q) = sqrt(|| sqrt(K_Q) sqrt(K_C) (K_C + reg I)^-2 sqrt(K_C) sqrt(K_Q) ||_inf)

    g ~ 1 means the classical kernel can reproduce anything the quantum kernel learns on
    these inputs; large g only says *some* labelling exists that favours K_Q.
    """
    n = len(K_classical)
    KC = K_classical * n / np.trace(K_classical)
    KQ = K_quantum * n / np.trace(K_quantum)
    SQ, SC = _sqrtm_psd(KQ), _sqrtm_psd(KC)
    inv = np.linalg.inv(KC + reg * np.eye(n))
    M = SQ @ SC @ inv @ inv @ SC @ SQ
    return float(np.sqrt(np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))))


def best_classical_geometric_difference(Z, K_quantum, gammas=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
                                        reg=1e-2):
    """min over RBF bandwidths of g(K_RBF || K_Q): the most favourable classical comparison."""
    from .kernels import rbf_kernel
    vals = {g: geometric_difference(rbf_kernel(Z, gamma=g), K_quantum, reg) for g in gammas}
    g_best = min(vals, key=vals.get)
    return vals[g_best], g_best
