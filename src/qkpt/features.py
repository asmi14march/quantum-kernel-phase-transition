"""Physical feature extraction from Ising spin configurations.

Two hand-built feature sets are provided, deliberately differing in how much physics
they hand the classifier:

``physics``  includes the order parameter |m| -> the problem is almost linearly separable.
``local``    only short-range correlations (energy, C(2), C(3), diagonal NN) -> no direct
             access to the order parameter; the classifier must infer long-range order from
             local structure, which is genuinely harder near T_c.

``pca``      unsupervised baseline: the top-d principal components of the raw configurations,
             used to dial the number of qubits for scaling studies.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _corr(s, r):
    """Translation-averaged (not connected) correlation <s_x s_{x+r}> along both lattice axes."""
    return 0.5 * (np.mean(s * np.roll(s, r, axis=1), axis=(1, 2))
                  + np.mean(s * np.roll(s, r, axis=2), axis=(1, 2)))


def observables(configs: np.ndarray) -> dict[str, np.ndarray]:
    s = configs.astype(np.float32)
    L = s.shape[1]
    diag = np.mean(s * np.roll(np.roll(s, 1, axis=1), 1, axis=2), axis=(1, 2))
    return {
        "|m|": np.abs(s.mean(axis=(1, 2))),
        "e": -2.0 * _corr(s, 1),                 # energy per spin, J = 1
        "C(2)": _corr(s, 2),
        "C(3)": _corr(s, 3),
        "C_diag(1)": diag,
        "C(L/4)": _corr(s, max(1, L // 4)),
        "C(L/2)": _corr(s, L // 2),
    }


FEATURE_SETS = {
    "physics": ["|m|", "e", "C(2)", "C(L/4)", "C(L/2)"],
    "local": ["e", "C(2)", "C(3)", "C_diag(1)"],
}


def ising_features(configs: np.ndarray, which: str = "physics") -> tuple[np.ndarray, list[str]]:
    obs = observables(configs)
    names = FEATURE_SETS[which]
    return np.column_stack([obs[n] for n in names]).astype(float), names


def pca_features(configs: np.ndarray, d: int, fit_on: np.ndarray | None = None, seed: int = 0):
    """Top-d principal components of flattened configurations.

    The Ising Z2 symmetry (s -> -s) makes signed PCs ambiguous; we therefore return |PC|
    for the leading (magnetisation-like) component and raw values for the others.
    """
    flat = configs.reshape(len(configs), -1).astype(np.float32)
    pca = PCA(n_components=d, random_state=seed).fit(flat if fit_on is None else
                                                     fit_on.reshape(len(fit_on), -1))
    Z = pca.transform(flat)
    Z[:, 0] = np.abs(Z[:, 0])
    return Z, [f"PC{i+1}" for i in range(d)]
