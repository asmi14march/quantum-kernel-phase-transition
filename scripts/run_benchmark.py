"""Experiment 1 — quantum vs classical kernels on three phase-classification tasks.

Datasets
  ising-physics : 2D Ising L=16, features [|m|, e, C(2), C(L/4), C(L/2)]  (5 qubits)
  ising-local   : 2D Ising L=16, features [e, C(2), C(3), C_diag(1)]      (4 qubits, no order parameter)
  tfim          : 1D TFIM N=10, 100-shot estimates [|m_z|, C_zz(1), C_zz(N/2), m_x]  (4 qubits)

Every model is tuned by 3-fold CV on the training split; results are averaged over seeds.
Outputs: results/tables/benchmark_{raw,summary}.csv, fig2_benchmark.png, fig3_kernels.png
"""
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qkpt import ISING_TC, TFIM_HC
from qkpt.experiment import DATA, FIG, TAB, benchmark, split, summarize, tune_and_fit
from qkpt.features import ising_features
from qkpt.kernels import fidelity_kernel, rbf_kernel
from qkpt.metrics import (best_classical_geometric_difference, concentration, kernel_target_alignment,
                          participation_ratio, spectrum)
from qkpt.models import (CLASSICAL_MODELS, QUANTUM_MODELS, PrecomputedStateKernelSVC,
                         QuantumKernelSVC)
from qkpt.plotting import INK2, MODEL_COLORS, MUTED, savefig, setup

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--plot-only", action="store_true", help="re-draw figures from saved tables")
args = ap.parse_args()
SEEDS = [0, 1] if args.quick else [0, 1, 2, 3, 4]
N_TRAIN = 200 if args.quick else 600
L, N = (16, 10)
TAB.mkdir(parents=True, exist_ok=True)

d = np.load(DATA / f"ising_L{L}.npz")
configs, temps = d["configs"], d["temps"]
y_ising = (temps < ISING_TC).astype(int)          # 1 = ordered (ferromagnet)
if not args.plot_only:
    frames = []
    raw = configs.reshape(len(configs), -1).astype(float)
    for fs in ["physics", "local"]:
        X, names = ising_features(configs, fs)
        models = QUANTUM_MODELS + CLASSICAL_MODELS + ["MLP (raw spins)"]
        inputs = {m: X for m in models}
        inputs["MLP (raw spins)"] = raw
        print(f"== ising-{fs}: features {names}")
        frames.append(benchmark(f"ising-{fs}", inputs, y_ising, temps, models, SEEDS, N_TRAIN,
                                ISING_TC, 0.25))

    t = np.load(DATA / f"tfim_N{N}.npz")
    Xt, hs, states, fields = t["X"], t["h"], t["states"], t["fields"]
    y_tfim = (hs < TFIM_HC).astype(int)
    idx = np.searchsorted(fields, hs).reshape(-1, 1)    # sample -> ground-state index
    models = QUANTUM_MODELS + CLASSICAL_MODELS + ["GS-fidelity"]
    inputs = {m: Xt for m in models}
    inputs["GS-fidelity"] = idx
    extra = {"GS-fidelity": (PrecomputedStateKernelSVC(states=states), {"C": [0.1, 1, 10, 100]})}
    print("== tfim")
    frames.append(benchmark("tfim", inputs, y_tfim, hs, models, SEEDS, N_TRAIN, TFIM_HC, 0.15, extra))

if args.plot_only:
    raw_df = pd.read_csv(TAB / "benchmark_raw.csv")
    SEEDS, N_TRAIN = sorted(raw_df.seed.unique()), int(raw_df.n_train.iloc[0])
else:
    raw_df = pd.concat(frames, ignore_index=True)
    raw_df.to_csv(TAB / "benchmark_raw.csv", index=False)
summ = summarize(raw_df)
summ.to_csv(TAB / "benchmark_summary.csv", index=False)
print(summ.to_string(index=False))

# ---------------------------------------------------------------- Figure 2: benchmark dot plot
setup()
datasets = ["ising-physics", "ising-local", "tfim"]
titles = {"ising-physics": "2D Ising — physics features (5 qubits)",
          "ising-local": "2D Ising — local correlations only (4 qubits)",
          "tfim": "1D TFIM — 100-shot measurement features (4 qubits)"}
fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
for j, ds in enumerate(datasets):
    s = summ[summ.dataset == ds].sort_values("accuracy")
    for i, metric in enumerate(["accuracy", "near_critical"]):
        ax = axes[i, j]
        ypos = np.arange(len(s))
        for k, (_, r) in enumerate(s.iterrows()):
            c = MODEL_COLORS.get(r.model, INK2)
            ax.errorbar(r[metric], k, xerr=r[metric + "_std"], fmt="o", color=c, ms=8,
                        capsize=3, elinewidth=1.5)
            ax.text(r[metric], k + 0.28, f"{r[metric]:.3f}", ha="center", fontsize=8, color=INK2)
        ax.set_yticks(ypos, s.model)
        ax.set_ylim(-0.6, len(s) - 0.3)
        if i == 0:
            ax.set_title(titles[ds])
        ax.set_xlabel("test accuracy (all T)" if metric == "accuracy" else
                      ("accuracy for |T−T$_c$| < 0.25" if ds != "tfim" else
                       "accuracy for |h−h$_c$| < 0.15"))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelleft=True)
fig.suptitle(f"Phase classification: mean ± std over {len(SEEDS)} seeds, {N_TRAIN} training samples",
             fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, FIG / "fig2_benchmark.png")

# ---------------------------------------------------------------- Figure 3: kernel geometry
tr, _ = split(temps, N_TRAIN, 0)
X, _ = ising_features(configs, "local")
order = np.argsort(temps[tr])
Xtr, ytr, Ttr = X[tr][order], y_ising[tr][order], temps[tr][order]
kern = {}
for fm in ["ZZ-linear", "Z"]:
    qname = "QK-" + fm
    best = raw_df.query("dataset=='ising-local' and model==@qname and seed==0").best_params.iloc[0]
    bw = ast.literal_eval(best)["bandwidth"]
    q = QuantumKernelSVC(feature_map=fm, bandwidth=bw).fit(Xtr, ytr)
    kern[f"Quantum {fm} (bw={bw})"] = q.K_train_
    q1 = QuantumKernelSVC(feature_map=fm, bandwidth=1.0).fit(Xtr, ytr)
    kern[f"Quantum {fm} (bw=1.0)"] = q1.K_train_
Z = (Xtr - Xtr.mean(0)) / Xtr.std(0)
gbest = ast.literal_eval(raw_df.query("dataset=='ising-local' and model=='RBF SVM' and seed==0")
             .best_params.iloc[0])["svc__gamma"]
kern[f"Classical RBF (γ={gbest})"] = rbf_kernel(Z, gamma=gbest)

rows = []
for name, K in kern.items():
    mu, var = concentration(K)
    rows.append({"kernel": name, "KTA": kernel_target_alignment(K, ytr),
                 "participation_ratio": participation_ratio(K), "offdiag_mean": mu,
                 "offdiag_var": var,
                 "min_g(K_RBF||K)": best_classical_geometric_difference(Z, K)[0]})
geo = pd.DataFrame(rows)
geo.to_csv(TAB / "kernel_geometry_ising_local.csv", index=False)
print(geo.to_string(index=False))

show = [list(kern)[0], list(kern)[1], list(kern)[-1]]
fig, axes = plt.subplots(1, 5, figsize=(17, 4.2),
                         gridspec_kw={"width_ratios": [1, 1, 1, 0.05, 1.25], "wspace": 0.25})
for ax, name in zip(axes[:3], show):
    im = ax.imshow(kern[name], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ic = np.searchsorted(Ttr, ISING_TC)
    ax.axhline(ic, color="#eb6834", lw=1), ax.axvline(ic, color="#eb6834", lw=1)
    r = geo.set_index("kernel").loc[name]
    ax.set_title(f"{name}\nKTA={r.KTA:.2f}, PR={r.participation_ratio:.1f}", fontsize=9.5)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    ax.set_xlabel("samples sorted by T  →  (orange: T$_c$)")
fig.colorbar(im, cax=axes[3], label="K(x, x′)")
ax = axes[4]
ax.set_position(ax.get_position().translated(0.04, 0))
for name, ls in zip(kern, ["-", "--", "-", "--", "-"]):
    c = MODEL_COLORS["QK-ZZ-linear"] if "ZZ" in name else (
        MODEL_COLORS["QK-Z"] if "Z " in name else MODEL_COLORS["RBF SVM"])
    ax.semilogy(np.arange(1, 41), spectrum(kern[name])[:40], ls, color=c, label=name, lw=1.6)
ax.set(xlabel="eigenvalue index", ylabel="normalised eigenvalue",
       title=f"Kernel spectra (ising-local, {N_TRAIN} train)")
ax.set_ylim(1e-6, 1)
ax.legend(fontsize=7.5)
savefig(fig, FIG / "fig3_kernels.png")
print("saved figures 2 and 3")
