"""Experiment 2 — locating the critical point with kernel classifiers.

(a) Supervised phase curves (Carrasquilla & Melko, Nat. Phys. 13, 431, 2017): per system
    size, train on half the data with labels from the exact transition and classify the
    other half. The fraction labelled 'ordered' sharpens with size; its 0.5 crossing T*(L)
    is extrapolated with T*(L) = T_c + a/L (nu = 1).

(b, c) Label-free protocol: train ONE classifier on samples deep inside each phase only
    (Ising T <= 1.6 or T >= 3.0; TFIM h <= 0.5 or h >= 1.6), pooled over all sizes, and apply
    it to every size. Nothing about T_c is used. We sweep the kernel's input scale
    (quantum: bandwidth; RBF: sqrt(gamma)) because, with no data near the transition, the
    position of the max-margin boundary — and hence the estimate — is set by kernel geometry.

Exact values: T_c = 2 / ln(1 + sqrt 2) = 2.2692 (Onsager), h_c / J = 1 (TFIM).
"""
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from qkpt import ISING_TC, TFIM_HC
from qkpt.criticality import fit_crossing, fss_extrapolate, phase_curve
from qkpt.experiment import DATA, FIG, TAB, split
from qkpt.feature_maps import make_feature_map
from qkpt.features import ising_features
from qkpt.kernels import fidelity_kernel, rbf_kernel
from qkpt.plotting import INK2, MODEL_COLORS, MUTED, savefig, setup
from qkpt.simulator import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
bench = pd.read_csv(TAB / "benchmark_raw.csv")
SCALES = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.7]   # input scale s: QK bandwidth = s, RBF gamma = s^2


def best(dataset, model, key):
    vals = [ast.literal_eval(b)[key] for b in bench.query("dataset==@dataset and model==@model").best_params]
    return float(pd.Series(vals).mode().iloc[0])


def load(sysname, size):
    if sysname == "ising":
        d = np.load(DATA / f"ising_L{size}.npz")
        return ising_features(d["configs"], "physics")[0], d["temps"]
    d = np.load(DATA / f"tfim_N{size}.npz")
    return d["X"], d["h"]


class KernelSVM:
    """SVM on a quantum (fidelity) or RBF kernel with a given input scale."""

    def __init__(self, kind, scale, C, mu, sd):
        self.kind, self.s, self.C, self.mu, self.sd = kind, scale, C, mu, sd

    def _enc(self, X):
        Z = (X - self.mu) / self.sd * self.s
        return simulate(make_feature_map("ZZ-linear", X.shape[1]), Z) if self.kind == "QK" else Z

    def _K(self, A, B):
        return fidelity_kernel(A, B) if self.kind == "QK" else rbf_kernel(A, B, gamma=1.0)

    def fit(self, X, y):
        self.P = self._enc(X)
        self.svc = SVC(kernel="precomputed", C=self.C).fit(self._K(self.P, self.P), y)
        return self

    def predict(self, X):
        return self.svc.predict(self._K(self._enc(X), self.P))


systems = {
    "ising": dict(sizes=[8, 12, 16] if args.quick else [8, 16, 24, 32], crit=ISING_TC, ds="ising-physics",
                  label="T / J", sym="T", size_sym="L", deep=(1.6, 3.0), rng=(1.0, 3.5)),
    "tfim": dict(sizes=[6, 8, 10] if args.quick else [6, 8, 10, 12], crit=TFIM_HC, ds="tfim",
                 label="h / J", sym="h", size_sym="N", deep=(0.5, 1.6), rng=(0.1, 2.0)),
}

setup()
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
sup_rows, free_rows = [], []
for r, (sysname, cfg) in enumerate(systems.items()):
    sizes = np.array(cfg["sizes"], float)
    data = {s: load(sysname, s) for s in cfg["sizes"]}
    blues = plt.get_cmap("Blues")(np.linspace(0.4, 0.95, len(sizes)))
    crit, sym = cfg["crit"], cfg["sym"]

    # ---------------- (a) supervised, per size
    for model, kind, scale_key, C_key in [("QK-ZZ-linear", "QK", "bandwidth", "C"),
                                          ("RBF SVM", "RBF", "svc__gamma", "svc__C")]:
        sc = best(cfg["ds"], model, scale_key)
        sc = sc if kind == "QK" else np.sqrt(sc)
        C = best(cfg["ds"], model, C_key)
        cross = []
        for size, col in zip(cfg["sizes"], blues):
            X, par = data[size]
            y = (par < crit).astype(int)
            tr, te = split(par, len(par) // 2, 0)
            clf = KernelSVM(kind, sc, C, X[tr].mean(0), X[tr].std(0)).fit(X[tr], y[tr])
            u, p, e = phase_curve(par[te], clf.predict(X[te]).astype(float))
            t0 = fit_crossing(u, p, crit)[0]
            cross.append(t0)
            sup_rows.append({"system": sysname, "model": model, "size": size, "crossing": t0})
            if kind == "QK":
                axes[r, 0].errorbar(u, p, yerr=e, color=col, marker="o", ms=3.2, lw=1.4, capsize=0,
                                    label=f"{cfg['size_sym']}={size}: {sym}* = {t0:.3f}")
        tc, tce, _ = fss_extrapolate(sizes, cross)
        sup_rows.append({"system": sysname, "model": model, "size": "FSS L→∞", "crossing": tc,
                         "stderr": tce, "rel_error_%": 100 * abs(tc - crit) / crit})
        print(f"{sysname} supervised {model}: crossings {np.round(cross, 4)} -> {tc:.4f} ± {tce:.4f}",
              flush=True)

    ax = axes[r, 0]
    ax.axvline(crit, color=MUTED, ls=":", lw=1)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xlabel=cfg["label"], ylabel="fraction classified 'ordered'",
           title=f"({'a' if r == 0 else 'd'}) {sysname.upper()}: supervised, QK-ZZ-linear")
    ax.legend(fontsize=8)

    # ---------------- (b, c) label-free, pooled deep-phase training, sweep input scale
    lo, hi = cfg["deep"]
    Xd = np.concatenate([X[(p <= lo) | (p >= hi)] for X, p in data.values()])
    pdp = np.concatenate([p[(p <= lo) | (p >= hi)] for X, p in data.values()])
    yd = (pdp < crit).astype(int)     # identical to 'which side of the gap' — no T_c needed
    sub = np.random.default_rng(0).choice(len(Xd), min(1200, len(Xd)), replace=False)
    mu, sd = Xd.mean(0), Xd.std(0)
    show = {0.05: None, 0.2: None, 1.0: None}
    for kind in ["QK", "RBF"]:
        for s in SCALES:
            clf = KernelSVM(kind, s, 10.0, mu, sd).fit(Xd[sub], yd[sub])
            cross = []
            for size in cfg["sizes"]:
                X, par = data[size]
                u, p, e = phase_curve(par, clf.predict(X).astype(float))
                cross.append(fit_crossing(u, p, crit)[0])
                if kind == "QK" and s in show and size == cfg["sizes"][-1]:
                    show[s] = (u, p, e)
            tc, tce, _ = fss_extrapolate(sizes, cross)
            free_rows.append({"system": sysname, "kernel": kind, "input_scale": s,
                              **{f"crossing_{cfg['size_sym']}{z}": c for z, c in zip(cfg["sizes"], cross)},
                              "fss_estimate": tc, "fss_stderr": tce, "exact": crit,
                              "rel_error_%": 100 * abs(tc - crit) / crit})
        f = pd.DataFrame(free_rows).query("system==@sysname and kernel==@kind")
        print(f"{sysname} label-free {kind}: estimates "
              + " ".join(f"{a}:{b:.3f}" for a, b in zip(f.input_scale, f.fss_estimate)), flush=True)

    ax = axes[r, 1]
    xmin, xmax = cfg["rng"]
    for a, b in [(xmin, lo), (hi, xmax)]:
        ax.axvspan(a, b, color="#e1e0d9", alpha=0.6, lw=0)
    ax.text((xmin + lo) / 2, 0.45, "train:\nordered", ha="center", color=INK2, fontsize=8.5)
    ax.text((hi + xmax) / 2, 0.45, "train:\ndisordered", ha="center", color=INK2, fontsize=8.5)
    for (s, (u, p, e)), c in zip(show.items(), plt.get_cmap("Blues")([0.45, 0.7, 0.95])):
        ax.errorbar(u, p, yerr=e, color=c, marker="o", ms=3.2, lw=1.4, capsize=0, label=f"bandwidth {s}")
    ax.axvline(crit, color=MUTED, ls=":", lw=1)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xlabel=cfg["label"], ylabel="fraction classified 'ordered'",
           title=f"({'b' if r == 0 else 'e'}) {sysname.upper()}: label-free QK, {cfg['size_sym']}={cfg['sizes'][-1]}")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[r, 2]
    fr = pd.DataFrame(free_rows).query("system==@sysname")
    for kind, model, mk in [("QK", "QK-ZZ-linear", "o"), ("RBF", "RBF SVM", "s")]:
        f = fr[fr.kernel == kind]
        ax.errorbar(f.input_scale, f.fss_estimate, yerr=f.fss_stderr, fmt="-" + mk,
                    color=MODEL_COLORS[model], ms=6, capsize=3, label=model)
    ax.axhline(crit, color=INK2, lw=1, ls=":")
    ax.text(SCALES[0], crit, f" exact {sym}$_c$ = {crit:.4f}", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set(xlabel="kernel input scale  (QK: bandwidth,  RBF: √γ)",
           ylabel=f"FSS-extrapolated {sym}$_c$",
           title=f"({'c' if r == 0 else 'f'}) {sysname.upper()}: label-free estimate vs kernel scale")
    ax.legend(fontsize=8)

fig.tight_layout()
savefig(fig, FIG / "fig4_critical_point.png")
sup = pd.DataFrame(sup_rows)
free = pd.DataFrame(free_rows)
sup.to_csv(TAB / "critical_point_supervised.csv", index=False)
free.to_csv(TAB / "critical_point_label_free.csv", index=False)
print(sup[sup["size"] == "FSS L→∞"].to_string(index=False))
