"""Experiment 4 — data efficiency (learning curves) and robustness to finite measurement shots.

Learning curves: ising-local task, training-set size swept from 25 to 800, every model
re-tuned by CV at each size.

Shot noise: each kernel entry is replaced by a Binomial(S, K)/S estimate, exactly what the
compute-uncompute circuit returns on hardware with S shots; the training Gram matrix is then
projected back onto the PSD cone. A well-tuned bandwidth keeps kernel values O(1) so they are
resolvable with few shots; a concentrated kernel (bandwidth too large) is not.
"""
import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qkpt import ISING_TC, TFIM_HC
from qkpt.experiment import DATA, FIG, TAB, tune_and_fit
from qkpt.features import ising_features
from qkpt.models import QuantumKernelSVC
from qkpt.plotting import INK2, MODEL_COLORS, MUTED, PALETTE, savefig, setup

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--plot-only", action="store_true", help="redraw Figure 6 from saved tables")
args = ap.parse_args()
SEEDS = [0, 1] if args.quick else [0, 1, 2, 3, 4]
SIZES = [25, 50, 100] if args.quick else [25, 50, 100, 200, 400, 800]
SHOTS = [16, 64, 256, 1024, 4096, None]
N_TEST, N_SHOT_TRAIN = (120, 120) if args.quick else (600, 400)
bench = pd.read_csv(TAB / "benchmark_raw.csv")

d = np.load(DATA / "ising_L16.npz")
X, _ = ising_features(d["configs"], "local")
y = (d["temps"] < ISING_TC).astype(int)

if args.plot_only:
    lc = pd.read_csv(TAB / "learning_curves.csv")
    sn = pd.read_csv(TAB / "shot_noise.csv")
    models = sorted(lc.model.unique(), key=["QK-ZZ-linear", "QK-HEA", "RBF SVM", "MLP"].index)
    SIZES = sorted(lc.n_train.unique())
    N_TEST = 600 if not args.quick else N_TEST
else:
    # ------------------------------------------------------------- learning curves
    rows = []
    models = ["QK-ZZ-linear", "QK-HEA", "RBF SVM", "MLP"]
    for seed in SEEDS:
        rest, test = train_test_split(np.arange(len(y)), test_size=N_TEST, stratify=d["temps"], random_state=seed)
        for n in SIZES:
            tr = np.random.default_rng(seed).choice(rest, size=n, replace=False)
            if len(np.unique(y[tr])) < 2:
                continue
            for m in models:
                mdl, best, t = tune_and_fit(m, X[tr], y[tr], seed)
                rows.append({"model": m, "n_train": n, "seed": seed, "test_accuracy": mdl.score(X[test], y[test])})
            print(f"seed {seed} n={n}: " + "  ".join(f"{r['model']}={r['test_accuracy']:.3f}" for r in rows[-4:]),
                  flush=True)
    lc = pd.DataFrame(rows)
    lc.to_csv(TAB / "learning_curves.csv", index=False)


    # ------------------------------------------------------------- shot noise
    def best_params(ds, model):
        return ast.literal_eval(bench.query("dataset==@ds and model==@model").best_params.mode().iloc[0])


    t = np.load(DATA / "tfim_N10.npz")
    tasks = {
        "ising-local": (X, y, d["temps"]),
        "tfim": (t["X"], (t["h"] < TFIM_HC).astype(int), t["h"]),
    }
    rows = []
    for ds, (Xd, yd, par) in tasks.items():
        bp = best_params(ds, "QK-ZZ-linear")
        configs = [("tuned bandwidth", bp["bandwidth"], bp["C"])]
        if ds == "ising-local":
            configs.append(("bandwidth = 1.0", 1.0, bp["C"]))
        for seed in SEEDS:
            tr, te = train_test_split(np.arange(len(yd)), train_size=N_SHOT_TRAIN, test_size=N_TEST, stratify=par,
                                      random_state=seed)
            for tag, bw, C in configs:
                for S in SHOTS:
                    mdl = QuantumKernelSVC("ZZ-linear", bandwidth=bw, C=C, shots=S, random_state=seed)
                    mdl.fit(Xd[tr], yd[tr])
                    rows.append({"dataset": ds, "config": tag, "shots": S if S else np.inf, "seed": seed,
                                 "test_accuracy": mdl.score(Xd[te], yd[te])})
        print(ds, "done", flush=True)
    sn = pd.DataFrame(rows)
    sn.to_csv(TAB / "shot_noise.csv", index=False)

# ------------------------------------------------------------- figure
setup()
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
g = lc.groupby(["model", "n_train"]).test_accuracy.agg(["mean", "std"]).reset_index()
for m in models:
    s = g[g.model == m]
    c = MODEL_COLORS[m]
    ax.plot(s.n_train, s["mean"], "-o", color=c, label=m)
    ax.fill_between(s.n_train, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.12, lw=0)
ax.set_xscale("log")
ax.set_xticks(SIZES, [str(n) for n in SIZES])
ax.minorticks_off()
ax.set(xlabel="training samples", ylabel=f"test accuracy ({N_TEST} held-out)",
       title="(a) Learning curves — 2D Ising, local-correlation features")
ax.legend()

ax = axes[1]
g = sn.groupby(["dataset", "config", "shots"]).test_accuracy.agg(["mean", "std"]).reset_index()
xs_map = {s: (s if np.isfinite(s) else 16384 * 2) for s in g.shots.unique()}
for (ds, cfg), c, ls in [(("ising-local", "tuned bandwidth"), PALETTE[0], "-"),
                         (("ising-local", "bandwidth = 1.0"), PALETTE[6], "--"),
                         (("tfim", "tuned bandwidth"), PALETTE[2], "-")]:
    s = g[(g.dataset == ds) & (g.config == cfg)]
    xx = [xs_map[v] for v in s.shots]
    ax.errorbar(xx, s["mean"], yerr=s["std"], fmt=ls + "o", color=c, capsize=3, label=f"{ds}, {cfg}")
ax.set_xscale("log", base=2)
ticks = sorted(xs_map.values())
ax.set_xticks(ticks, [str(int(v)) if v < 30000 else "exact" for v in ticks])
ax.set(xlabel="measurement shots per kernel entry", ylabel="test accuracy",
       title="(b) Robustness to finite-shot kernel estimation (QK-ZZ-linear)")
ax.legend()
fig.tight_layout()
savefig(fig, FIG / "fig6_learning_and_noise.png")
print("saved fig6")
