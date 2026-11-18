"""Generate the classical (2D Ising) and quantum (1D TFIM) datasets + Figure 1."""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from qkpt import ISING_TC, TFIM_HC
from qkpt.experiment import DATA, FIG
from qkpt.features import observables
from qkpt.ising import exact_magnetization, sample_ising
from qkpt.plotting import INK2, MUTED, PALETTE, savefig, setup
from qkpt.tfim import sample_tfim

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="small run for CI / smoke tests")
ap.add_argument("--plot-only", action="store_true", help="only redraw Figure 1 from saved data")
args = ap.parse_args()

if args.quick:
    LS, TEMPS, NS_T = [8, 12, 16], np.linspace(1.0, 3.5, 21), 20
    NS, FIELDS, NS_H = [6, 8, 10], np.linspace(0.1, 2.0, 20), 15
else:
    LS, TEMPS, NS_T = [8, 16, 24, 32], np.linspace(1.0, 3.5, 41), 50
    NS, FIELDS, NS_H = [6, 8, 10, 12], np.round(np.arange(0.1, 2.0001, 0.05), 3), 40

DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

if not args.plot_only:
    for L in LS:
        t = time.time()
        configs, temps = sample_ising(L, TEMPS, n_samples_per_T=NS_T, n_chains=5, n_therm=500,
                                      n_between=10, seed=1000 + L)
        np.savez_compressed(DATA / f"ising_L{L}.npz", configs=configs, temps=temps)
        print(f"Ising L={L}: {len(configs)} configs in {time.time() - t:.1f}s")

    for N in NS:
        t = time.time()
        X, hs, states = sample_tfim(N, FIELDS, n_samples_per_h=NS_H, shots=100, seed=2000 + N)
        np.savez_compressed(DATA / f"tfim_N{N}.npz", X=X, h=hs, states=states, fields=FIELDS)
        print(f"TFIM  N={N}: {len(X)} measurement-based samples in {time.time() - t:.1f}s")

# ---------------------------------------------------------------- Figure 1
setup()
fig = plt.figure(figsize=(12, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38, wspace=0.28)
Lmax = LS[-1]
d = np.load(DATA / f"ising_L{Lmax}.npz")
spin_cmap = ListedColormap(["#1c5cab", "#f0efec"])
for k, Tt in enumerate([1.5, ISING_TC, 3.2]):
    ax = fig.add_subplot(gs[0, k])
    T_near = d["temps"][np.argmin(np.abs(d["temps"] - Tt))]
    ax.imshow(d["configs"][d["temps"] == T_near][0], cmap=spin_cmap, interpolation="nearest")
    ax.set_title(f"L={Lmax},  T = {T_near:.2f}" + ("  (≈ T$_c$)" if k == 1 else ""))
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)

blues = plt.get_cmap("Blues")(np.linspace(0.4, 0.95, len(LS)))
ax = fig.add_subplot(gs[1, 0])
for L, c in zip(LS, blues):
    dd = np.load(DATA / f"ising_L{L}.npz")
    o = observables(dd["configs"])
    u = np.unique(dd["temps"])
    ax.plot(u, [o["|m|"][dd["temps"] == t].mean() for t in u], color=c, label=f"L={L}")
Tf = np.linspace(1.0, 3.5, 400)
ax.plot(Tf, exact_magnetization(Tf), color=INK2, lw=1.2, ls="--", label="Onsager (L→∞)")
ax.axvline(ISING_TC, color=MUTED, lw=0.8, ls=":")
ax.set(xlabel="temperature T / J", ylabel=r"$\langle |m| \rangle$", title="2D Ising: order parameter")
ax.legend()

ax = fig.add_subplot(gs[1, 1])
for L, c in zip(LS, blues):
    dd = np.load(DATA / f"ising_L{L}.npz")
    o = observables(dd["configs"])
    u = np.unique(dd["temps"])
    ax.plot(u, [o["C(2)"][dd["temps"] == t].mean() for t in u], color=c, label=f"L={L}")
ax.axvline(ISING_TC, color=MUTED, lw=0.8, ls=":")
ax.set(xlabel="temperature T / J", ylabel=r"$\langle s_x\, s_{x+2} \rangle$",
       title="2D Ising: short-range correlation C(2)")
ax.legend()

ax = fig.add_subplot(gs[1, 2])
N0 = NS[-2] if len(NS) > 2 else NS[-1]
dd = np.load(DATA / f"tfim_N{N0}.npz")
u = np.unique(dd["h"])
for j, (name, c) in enumerate([("|m$_z$|", PALETTE[0]), ("m$_x$", PALETTE[1])]):
    col = 0 if j == 0 else 3
    mu = np.array([dd["X"][dd["h"] == t, col].mean() for t in u])
    sd = np.array([dd["X"][dd["h"] == t, col].std() for t in u])
    ax.fill_between(u, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    ax.plot(u, mu, color=c)
    ax.text(0.2, 0.88 if j == 0 else 0.2, name, color=c, fontsize=11, fontweight="bold")
ax.axvline(TFIM_HC, color=MUTED, lw=0.8, ls=":")
ax.set(xlabel="transverse field h / J", ylabel="estimated observable",
       title=f"1D TFIM (N={N0}): 100-shot estimates ± 1σ")
savefig(fig, FIG / "fig1_datasets.png")
print("saved", FIG / "fig1_datasets.png")
