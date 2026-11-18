"""Locating the critical point from a classifier trained *away* from it.

Protocol (Carrasquilla & Melko, Nat. Phys. 13, 431, 2017): train only on configurations
deep in each phase, then let the model label every control-parameter value. The fraction
classified as 'ordered' falls from 1 to 0; its 0.5 crossing T*(L) is a pseudo-critical
point. Finite-size scaling, T*(L) = T_c + a L^(-1/nu) with nu = 1 for 2D Ising and the 1D
TFIM, extrapolates to the thermodynamic limit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def phase_curve(param, pred_ordered):
    """Mean predicted-ordered fraction and its standard error at each unique parameter value."""
    u = np.unique(param)
    mean = np.array([pred_ordered[param == t].mean() for t in u])
    sem = np.array([pred_ordered[param == t].std(ddof=1) / np.sqrt((param == t).sum()) for t in u])
    return u, mean, sem


def _logistic(t, t0, w):
    return 1.0 / (1.0 + np.exp((t - t0) / w))


def fit_crossing(t, p, t_guess):
    """Fit p(t) to a decreasing logistic; return (t*, sigma_t*, width).

    Falls back to linear interpolation of the 0.5 crossing if the curve is a sharp step.
    """
    t, p = np.asarray(t, float), np.asarray(p, float)
    try:
        popt, pcov = curve_fit(_logistic, t, p, p0=[t_guess, 0.1], maxfev=20000,
                               bounds=([t.min(), 1e-3], [t.max(), 10.0]))
        return float(popt[0]), float(np.sqrt(pcov[0, 0])), float(abs(popt[1]))
    except (RuntimeError, ValueError):
        k = int(np.argmax(p < 0.5))
        t0 = t[k - 1] + (0.5 - p[k - 1]) * (t[k] - t[k - 1]) / (p[k] - p[k - 1])
        return float(t0), float("nan"), 0.0


def fss_extrapolate(sizes, t_star, t_err=None, nu=1.0):
    """Weighted linear fit of t*(L) against L^(-1/nu). Returns (t_c, sigma_t_c, slope)."""
    x = np.asarray(sizes, float) ** (-1.0 / nu)
    y = np.asarray(t_star, float)
    w = None if t_err is None else 1.0 / np.maximum(np.asarray(t_err, float), 1e-6)
    if len(x) < 3:  # two sizes: exact line, no error estimate
        coef = np.polyfit(x, y, 1, w=w)
        return float(coef[1]), float("nan"), float(coef[0])
    coef, cov = np.polyfit(x, y, 1, w=w, cov="unscaled" if w is not None else True)
    return float(coef[1]), float(np.sqrt(cov[1, 1])), float(coef[0])
