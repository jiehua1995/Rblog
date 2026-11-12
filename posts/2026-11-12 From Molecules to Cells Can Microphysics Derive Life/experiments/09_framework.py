"""Study 9 — a cross-scale diagram and the three gates of predictive sufficiency.

Nothing in this file is data: it draws the bookkeeping that the previous eight
experiments justify.  The layers carry the resolution at which each one is
usually posed; the right panel states the three independent conditions a model
must meet before it can be called predictively sufficient, and none of them is
implied by the one above it.

Run:  python experiments/09_framework.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

LAYERS = [
    ("electronic / quantum", "bond breaking, proton transfer, spin",
     "fs", "0.1 nm", "quantum chemistry, spectroscopy"),
    ("atomistic", "positions, velocities, forces",
     "fs-ps", "0.1 nm", "MD, cryo-EM, NMR"),
    ("molecular / complex", "conformations, binding states",
     "ns-us", "1-10 nm", "single-molecule, FRET, ITC"),
    ("regulatory network", "activities, concentrations",
     "s-min", "um", "omics, reporters, Perturb-seq"),
    ("organelle / cell", "fluxes, volumes, fate",
     "min-h", "um", "imaging, sequencing"),
    ("population / tissue", "composition, selection",
     "h-years", "mm-cm", "ecology, lineage tracing"),
]


def fig_layers():
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    cols = [PAL["blue"], PAL["sky"], PAL["green"], PAL["orange"],
            PAL["vermillion"], PAL["purple"]]
    n = len(LAYERS)
    for i, ((name, state, dt_, dx, meas), c) in enumerate(zip(LAYERS, cols)):
        y = n - 1 - i
        ax.add_patch(Rectangle((0.02, y + 0.09), 0.96, 0.82, linewidth=0,
                               facecolor=c, alpha=0.14))
        ax.add_patch(Rectangle((0.02, y + 0.09), 0.009, 0.82, linewidth=0,
                               facecolor=c, alpha=0.95))
        ax.text(0.05, y + 0.66, name, fontsize=8.2, weight="bold",
                color=PAL["dark"], va="center")
        ax.text(0.05, y + 0.30, state, fontsize=6.8, color=PAL["dark"],
                va="center")
        ax.text(0.54, y + 0.66, f"$\\Delta t$: {dt_}", fontsize=6.8,
                color=PAL["grey"], va="center")
        ax.text(0.54, y + 0.30, f"$\\Delta x$: {dx}", fontsize=6.8,
                color=PAL["grey"], va="center")
        ax.text(0.70, y + 0.48, meas, fontsize=6.6, color=PAL["grey"],
                va="center")
    ax.text(0.02, n + 0.22,
            "state variables, resolution and the measurement that constrains each layer",
            fontsize=8.4, color=PAL["dark"], va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 0.45)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-layers")


def fig_gates():
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    gates = [
        ("identifiability", "is the state that decides\nthe future observable?", PAL["blue"]),
        ("dynamical closure", "does a closed equation exist\nat this coarse-graining?", PAL["green"]),
        ("computational\ntractability", "can it be evaluated\nwithin budget?", PAL["vermillion"]),
    ]
    for i, (name, q, c) in enumerate(gates):
        x = 0.04 + i * 0.32
        ax.add_patch(Rectangle((x, 0.36), 0.26, 0.30, linewidth=0,
                               facecolor=c, alpha=0.14))
        ax.text(x + 0.13, 0.55, name, ha="center", va="center", fontsize=8,
                weight="bold", color=c)
        ax.text(x + 0.13, 0.44, q, ha="center", va="center", fontsize=6.6,
                color=PAL["dark"])
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + 0.265, 0.51), (x + 0.305, 0.51),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=PAL["grey"], lw=1.0))
    ax.text(0.04, 0.80, "correct microphysics $\\Rightarrow$ none of the three gates",
            fontsize=8.2, color=PAL["dark"])
    ax.text(0.04, 0.10,
            "the gates are independent: a model can be identifiable but intractable,\n"
            "or tractable but not identifiable. Passing all three is what 'predictive\n"
            "sufficiency' means here, and it is a property of the question, not of the model.",
            fontsize=6.6, color=PAL["grey"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-gates")


def main() -> None:
    set_seed(0)
    set_style()
    fig_layers()
    fig_gates()
    save_metrics("framework", {
        "layers": [l[0] for l in LAYERS],
        "resolution_time": [l[2] for l in LAYERS],
        "resolution_space": [l[3] for l in LAYERS],
        "gates": ["identifiability", "dynamical closure", "tractability"],
    })


if __name__ == "__main__":
    main()
