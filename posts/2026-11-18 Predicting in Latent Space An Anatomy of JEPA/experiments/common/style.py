"""Matplotlib defaults and the shared Okabe-Ito palette."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import FIGURES

PAL = {
    "blue": "#0072B2",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "grey": "#8A8A8A",
    "dark": "#2B2B2B",
    "light": "#D6E9F5",
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blog_seq", ["#FFFFFF", "#D6E9F5", "#56B4E9", "#0072B2", "#00456e"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blog_div", ["#0072B2", "#9ecae1", "#FFFFFF", "#f4a582", "#D55E00"]
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
            "svg.fonttype": "none",
            "image.interpolation": "nearest",
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", transparent=True)
    fig.savefig(FIGURES / f"{name}.svg", transparent=True)
    plt.close(fig)
    print(f"[figure] {name}")
