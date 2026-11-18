"""2D classical Ising model on an L x L periodic square lattice (J = 1, k_B = 1).

Sampling uses the Wolff single-cluster algorithm, which removes most of the critical
slowing-down that plagues single-spin Metropolis near T_c. Numba is used when it is
installed; otherwise the same code runs as plain Python (correct, but slow).

    H = -J * sum_<ij> s_i s_j,   T_c = 2 / ln(1 + sqrt 2) ~ 2.2692
"""
from __future__ import annotations

import numpy as np

try:  # optional JIT
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _wolff_update(spins, p_add, stack):
    """Grow and flip one Wolff cluster in place. Returns the cluster size."""
    L = spins.shape[0]
    i0 = np.random.randint(L)
    j0 = np.random.randint(L)
    s0 = spins[i0, j0]
    spins[i0, j0] = -s0
    stack[0, 0] = i0
    stack[0, 1] = j0
    top = 1
    size = 1
    while top > 0:
        top -= 1
        i = stack[top, 0]
        j = stack[top, 1]
        for k in range(4):
            if k == 0:
                ni, nj = (i + 1) % L, j
            elif k == 1:
                ni, nj = (i - 1) % L, j
            elif k == 2:
                ni, nj = i, (j + 1) % L
            else:
                ni, nj = i, (j - 1) % L
            if spins[ni, nj] == s0 and np.random.random() < p_add:
                spins[ni, nj] = -s0
                stack[top, 0] = ni
                stack[top, 1] = nj
                top += 1
                size += 1
    return size


@njit(cache=True)
def _sweep_equivalent(spins, p_add, stack):
    """Apply Wolff updates until ~L^2 spins have been flipped (one 'sweep')."""
    n = spins.shape[0] * spins.shape[0]
    flipped = 0
    while flipped < n:
        flipped += _wolff_update(spins, p_add, stack)


@njit(cache=True)
def _run_chain(L, T, n_therm, n_samples, n_between, seed):
    _seed(seed)
    spins = np.ones((L, L), dtype=np.int8)
    for a in range(L):
        for b in range(L):
            if np.random.random() < 0.5:
                spins[a, b] = -1
    p_add = 1.0 - np.exp(-2.0 / T)
    stack = np.empty((L * L, 2), dtype=np.int64)
    for _ in range(n_therm):
        _sweep_equivalent(spins, p_add, stack)
    out = np.empty((n_samples, L, L), dtype=np.int8)
    for s in range(n_samples):
        for _ in range(n_between):
            _sweep_equivalent(spins, p_add, stack)
        out[s] = spins
    return out


def sample_ising(
    L: int,
    temperatures,
    n_samples_per_T: int = 50,
    n_chains: int = 5,
    n_therm: int = 200,
    n_between: int = 5,
    seed: int = 0,
):
    """Draw equilibrium spin configurations at each temperature.

    Samples are split across ``n_chains`` independent chains per temperature so
    residual autocorrelation near T_c does not dominate a single trajectory.

    Returns
    -------
    configs : (N, L, L) int8 array of +-1 spins
    temps   : (N,) float array, temperature of each configuration
    """
    temperatures = np.asarray(temperatures, dtype=float)
    per_chain = int(np.ceil(n_samples_per_T / n_chains))
    configs, temps = [], []
    rng = np.random.default_rng(seed)
    for T in temperatures:
        block = []
        for _ in range(n_chains):
            block.append(_run_chain(L, float(T), n_therm, per_chain, n_between,
                                    int(rng.integers(2**31 - 1))))
        block = np.concatenate(block)[:n_samples_per_T]
        configs.append(block)
        temps.append(np.full(len(block), T))
    return np.concatenate(configs), np.concatenate(temps)


def exact_magnetization(T):
    """Onsager/Yang spontaneous magnetisation of the infinite lattice (for reference curves)."""
    T = np.asarray(T, dtype=float)
    m = np.zeros_like(T)
    below = T < 2.0 / np.log(1 + np.sqrt(2))
    m[below] = (1.0 - np.sinh(2.0 / T[below]) ** -4) ** 0.125
    return m
