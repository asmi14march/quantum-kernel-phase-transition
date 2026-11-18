"""Experiment 3 — how quantum kernels scale with the number of qubits.

Input: 2D Ising L=16 configurations compressed to d principal components (d qubits).
Measured per d:
  * exponential concentration of the kernel (variance of off-diagonal entries),
  * test accuracy at a fixed bandwidth (1.0) and at a CV-tuned bandwidth,
  * geometric difference g(K_RBF || K_Q) (Huang et al. 2021) — can any labelling separate them?
  * simulation cost: batched engine vs one qiskit Statevector per sample.
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qkpt import ISING_TC
from qkpt.experiment import DATA, FIG, TAB, tune_and_fit
from qkpt.feature_maps import make_feature_map
from qkpt.features import pca_features
from qkpt.metrics import best_classical_geometric_difference, concentration
from qkpt.models import QuantumKernelSVC
from qkpt.plotting import INK2, MODEL_COLORS, MUTED, PALETTE, savefig, setup
from qkpt.simulator import simulate, simulate_qiskit

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--plot-only", action="store_true", help="redraw Figure 5 from saved table")
args = ap.parse_args()
DIMS = [2, 4, 6] if args.quick else [2, 4, 6, 8, 10, 12, 14]
N_TOT, N_TR = (300, 200) if args.quick else (900, 600)

d = np.load(DATA / "ising_L16.npz")
configs, temps = d["configs"], d["temps"]
y = (temps < ISING_TC).astype(int)
sub, _ = train_test_split(np.arange(len(y)), train_size=N_TOT, stratify=temps, random_state=0)
tr, te = train_test_split(sub, train_size=N_TR, stratify=temps[sub], random_state=0)

if not args.plot_only:
    rows = []
    for dim in DIMS:
        Z, _ = pca_features(configs, dim, fit_on=configs[tr])
        rbf, rbf_best, _ = tune_and_fit("RBF SVM", Z[tr], y[tr], 0)
        acc_rbf = rbf.score(Z[te], y[te])
        zs = (Z - Z[tr].mean(0)) / Z[tr].std(0)
        for fm in ["ZZ-linear", "HEA"]:
            fixed, _, _ = tune_and_fit(None, Z[tr], y[tr], 0,
                                       QuantumKernelSVC(fm, bandwidth=1.0), {"C": [0.1, 1, 10, 100]})
            tuned, best, _ = tune_and_fit(None, Z[tr], y[tr], 0, QuantumKernelSVC(fm),
                                          {"bandwidth": [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
                                           "C": [0.1, 1, 10, 100]})
            for tag, mdl in [("bw=1.0", fixed), ("tuned", tuned)]:
                mu, var = concentration(mdl.K_train_)
                rows.append({"qubits": dim, "feature_map": fm, "bandwidth": tag,
                             "bandwidth_value": mdl.bandwidth, "test_accuracy": mdl.score(Z[te], y[te]),
                             "offdiag_mean": mu, "offdiag_var": var,
                             "g_RBF_vs_Q": best_classical_geometric_difference(zs[tr], mdl.K_train_)[0],
                             "rbf_accuracy": acc_rbf})
        # simulation cost per sample
        fmap = make_feature_map("ZZ-linear", dim)
        t0 = time.perf_counter(); simulate(fmap, zs[tr]); t_batch = (time.perf_counter() - t0) / len(tr)
        m = 40
        t0 = time.perf_counter(); simulate_qiskit(fmap, zs[tr][:m]); t_qk = (time.perf_counter() - t0) / m
        for r in rows[-4:]:
            r.update(time_batched_ms=1e3 * t_batch, time_qiskit_ms=1e3 * t_qk)
        print(f"d={dim:2d}  RBF={acc_rbf:.3f}  " + "  ".join(
            f"{r['feature_map']}/{r['bandwidth']}: acc={r['test_accuracy']:.3f} var={r['offdiag_var']:.2e} "
            f"g={r['g_RBF_vs_Q']:.1f}" for r in rows[-4:]) +
            f"  sim/sample: batched {1e3*t_batch:.3f} ms vs qiskit {1e3*t_qk:.2f} ms", flush=True)

if args.plot_only:
    df = pd.read_csv(TAB / "scaling.csv")
    DIMS = sorted(df.qubits.unique())
else:
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "scaling.csv", index=False)

setup()
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
styles = {("ZZ-linear", "bw=1.0"): (MODEL_COLORS["QK-ZZ-linear"], "--", "o"),
          ("ZZ-linear", "tuned"): (MODEL_COLORS["QK-ZZ-linear"], "-", "o"),
          ("HEA", "bw=1.0"): (MODEL_COLORS["QK-HEA"], "--", "s"),
          ("HEA", "tuned"): (MODEL_COLORS["QK-HEA"], "-", "s")}
for (fm, bw), (c, ls, mk) in styles.items():
    s = df[(df.feature_map == fm) & (df.bandwidth == bw)]
    lab = f"{fm}, {bw}"
    axes[0].semilogy(s.qubits, s.offdiag_mean, ls, marker=mk, color=c, label=lab)
    axes[1].plot(s.qubits, s.test_accuracy, ls, marker=mk, color=c, label=lab)
    axes[2].plot(s.qubits, s.g_RBF_vs_Q, ls, marker=mk, color=c, label=lab)
s = df.drop_duplicates("qubits")
axes[1].plot(s.qubits, s.rbf_accuracy, "-", marker="D", color=MODEL_COLORS["RBF SVM"], label="RBF SVM")
axes[2].axhline(1.0, color=MUTED, ls=":", lw=1)
axes[2].text(DIMS[0], 1.0, " g = 1: classical kernel reproduces K$_Q$", va="bottom", color=INK2, fontsize=8)
axes[3].semilogy(s.qubits, s.time_qiskit_ms, "-o", color=PALETTE[6], label="qiskit Statevector, per circuit")
axes[3].semilogy(s.qubits, s.time_batched_ms, "-o", color=PALETTE[0], label="batched engine (this repo)")
qs = np.array(DIMS, float)
axes[0].semilogy(qs, 2.0 ** -qs, ":", color=MUTED, lw=1.2, label="2$^{-n}$ (Haar-random limit)")
axes[0].set(xlabel="qubits (PCA components)", ylabel="mean off-diagonal K(x, x′)",
            title="(a) Kernel concentration")
axes[1].set(xlabel="qubits (PCA components)", ylabel="test accuracy", title="(b) Accuracy vs qubits")
axes[2].set(xlabel="qubits (PCA components)", ylabel="min$_γ$ g(K$_{RBF,γ}$ ‖ K$_Q$)",
            title="(c) Geometric difference")
axes[3].set(xlabel="qubits", ylabel="ms per sample", title="(d) Statevector simulation cost")
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, FIG / "fig5_scaling.png")
print("saved fig5")
