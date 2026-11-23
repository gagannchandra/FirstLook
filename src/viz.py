"""Shared figure style (dataviz skill reference palette, light mode)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#b9b8b1"
TIER_COLOR = {"t1": SERIES[0], "t2": SERIES[1], "t3": SERIES[2]}
SITE_COLOR = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
MODEL_COLOR = {"lr": SERIES[0], "rf": SERIES[1], "xgb": SERIES[2], "lgbm": SERIES[3], "cat": SERIES[4]}
MODEL_NAME = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}
TIER_NAME = {"t1": "T1 nurse-only", "t2": "T2 + ESI", "t3": "T3 + history"}


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
        "text.color": INK, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p
