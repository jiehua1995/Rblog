"""Matplotlib defaults and the shared palette.

The site's figures are always transparent-background PNGs; here the same rule
is applied to the Python side.  Colours follow the Okabe-Ito palette already
used by theme-blog.R so the Python and R figures sit together without clashing.
"""

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
    "grey": "#8A8A8A",
    "dark": "#2B2B2B",
    "light": "#D6E9F5",
}

# diverging map for signed quantities: blue (negative) - white - vermillion (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "blog_div", ["#0072B2", "#9ecae1", "#FFFFFF", "#f4a582", "#D55E00"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blog_seq", ["#FFFFFF", "#D6E9F5", "#56B4E9", "#0072B2", "#00456e"]
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
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
    """Write figures/<name>.png (300 dpi, transparent) and .svg."""
    png = FIGURES / f"{name}.png"
    fig.savefig(png, transparent=True)
    fig.savefig(FIGURES / f"{name}.svg", transparent=True)
    plt.close(fig)
    print(f"[figure] {png.relative_to(Path.cwd()) if png.is_relative_to(Path.cwd()) else png}")
