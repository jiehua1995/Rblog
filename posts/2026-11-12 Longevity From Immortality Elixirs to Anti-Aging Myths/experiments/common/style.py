"""Shared figure style: Okabe-Ito palette, transparent background, PNG + SVG."""

from __future__ import annotations

from pathlib import Path

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
    "yellow": "#F0E442",
    "grey": "#8A8A8A",
    "dark": "#2B2B2B",
    "light": "#D6E9F5",
}

DIVERGING = LinearSegmentedColormap.from_list(
    "blog_div", ["#0072B2", "#F7F7F7", "#D55E00"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blog_seq", ["#F7FBFE", "#9EC9E2", "#3D8DBF", "#0072B2", "#003A5C"])
EVIDENCE = LinearSegmentedColormap.from_list(
    "blog_ev", ["#F2F2F2", "#F0E442", "#E69F00", "#D55E00", "#7A2E00"])


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.3,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "figure.dpi": 120,
    })


def save(fig, name: str) -> None:
    """Write a transparent PNG (300 dpi) and an SVG next to it."""
    for ext, dpi in (("png", 300), ("svg", None)):
        path = Path(FIGURES) / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"[figure] figures/{name}.png")
