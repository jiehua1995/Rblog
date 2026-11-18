"""Scientific figure style: Okabe-Ito + deep blue/teal, SVG and PNG."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import fig_dir

PAL = {
    "navy": "#0B3C5D",
    "blue": "#1F6FB2",
    "teal": "#2A9D8F",
    "green": "#4C9F70",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#7B6D8D",
    "grey": "#7A7A7A",
    "dark": "#1D2733",
    "light": "#DCE9F2",
    "sky": "#8ECAE6",
}

SEQ = LinearSegmentedColormap.from_list(
    "llm_seq", ["#F7FBFC", "#BFE3E0", "#6FBFB2", "#2A9D8F", "#0B3C5D"])
DIV = LinearSegmentedColormap.from_list(
    "llm_div", ["#1F6FB2", "#F7F7F7", "#E76F51"])
EV = LinearSegmentedColormap.from_list(
    "llm_ev", ["#F2F2F2", "#E9C46A", "#E76F51", "#7A2E1F"])


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
        "legend.fontsize": 7.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.22,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "figure.dpi": 120,
    })


def save(fig, name: str, article: int = 1) -> None:
    out = fig_dir(article)
    for ext, dpi in (("png", 300), ("svg", None)):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight",
                    transparent=True)
    plt.close(fig)
    print(f"[figure] figures/{name}.png")
