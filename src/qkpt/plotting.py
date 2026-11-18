"""Shared matplotlib style: recessive axes, thin marks, a fixed categorical colour order."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, assigned in fixed order and never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_CMAP = "Blues"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Colour follows the model, not its rank in any one plot.
MODEL_COLORS = {
    "QK-ZZ-linear": PALETTE[0], "RBF SVM": PALETTE[1], "MLP": PALETTE[2], "QK-HEA": PALETTE[3],
    "QK-Z": PALETTE[4], "Linear SVM": PALETTE[5], "QK-ZZ-full": PALETTE[6],
    "MLP (raw spins)": PALETTE[7], "GS-fidelity": PALETTE[7],
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
