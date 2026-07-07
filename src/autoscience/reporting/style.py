"""Shared figure style.

Colors follow the entity (benchmark mode / model), assigned in fixed slot
order from a CVD-validated categorical palette (worst adjacent-pair CVD
deltaE 24.2 in light mode). Yellow sits below 3:1 contrast on white, so every
figure carries a legend and marks get a thin dark edge — identity is never
color-alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering everywhere (CI, Colab, laptop)

import matplotlib.pyplot as plt

# Validated categorical slots (light mode), assigned in fixed order.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

MODE_ORDER = ["automated", "baseline_default", "baseline_expert"]
MODE_COLORS = dict(zip(MODE_ORDER, CATEGORICAL[:3], strict=True))
MODE_LABELS = {
    "automated": "Automated (HPO)",
    "baseline_default": "Manual: defaults",
    "baseline_expert": "Manual: expert",
}

TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e58"
GRID = "#e4e3dd"


def apply_style() -> None:
    """Recessive grid/axes, readable text, thin marks."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def model_colors(models: list[str]) -> dict[str, str]:
    """Fixed-order slot assignment for model series (never cycled)."""
    if len(models) > len(CATEGORICAL):
        raise ValueError("more series than palette slots; fold extras into 'other'")
    return dict(zip(sorted(models), CATEGORICAL, strict=False))
