"""Static matplotlib figures for the notebooks (GitHub renders PNG outputs, not Plotly).

Same colour meaning as the app: teal = stayed, clay = left; one categorical
order; hairline grids; legends whenever two or more series share a chart.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAL, CLAY, BLUE, AMBER, VIOLET, GREEN = "#1f8a7d", "#c4553d", "#3f74c4", "#9c7020", "#7559bf", "#55894b"
NEUTRAL, INK, MUTED, GRID = "#8a959c", "#1d2a31", "#56656d", "#e3e8eb"
CATEGORICAL = [TEAL, CLAY, BLUE, AMBER, VIOLET, GREEN]
STRATEGY_COLORS = {"none": NEUTRAL, "class_weight": BLUE, "smotenc": TEAL, "random_oversample": AMBER}


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left", "axes.labelcolor": MUTED,
        "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "text.color": INK,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL), "figure.autolayout": False,
    })
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 3)


def rate_dots(ax, rates: pd.DataFrame, overall: float, title: str, ordered: bool = False) -> None:
    """Attrition rate per group with Wilson 95% intervals (hollow = n < 30)."""
    r = rates if ordered else rates.sort_values("rate")
    r = r.iloc[::-1] if ordered else r
    y = np.arange(len(r))
    ax.hlines(y, r["ci_low"], r["ci_high"], color=CLAY, alpha=0.55, lw=2)
    small = (r["n"] < 30).to_numpy()
    ax.scatter(r["rate"][~small], y[~small], color=CLAY, s=42, zorder=3, edgecolor="white", linewidth=1.2)
    ax.scatter(r["rate"][small], y[small], facecolor="white", edgecolor=CLAY, s=42, zorder=3, linewidth=1.6)
    ax.axvline(overall, color=MUTED, lw=1)
    ax.set_yticks(y, [f"{g}  (n={n})" for g, n in zip(r["group"], r["n"])])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, min(1, max(r["ci_high"].max() * 1.1, overall * 1.4)))
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def outcome_hist(ax, df: pd.DataFrame, col: str, bins: int = 25, label: str | None = None) -> None:
    for key, name, color in (("No", "Stayed", TEAL), ("Yes", "Left", CLAY)):
        s = df.loc[df["Attrition"] == key, col]
        ax.hist(s, bins=bins, density=True, alpha=0.55, color=color, label=f"{name} (n={len(s)})")
    ax.set_title(label or col)
    ax.legend()
