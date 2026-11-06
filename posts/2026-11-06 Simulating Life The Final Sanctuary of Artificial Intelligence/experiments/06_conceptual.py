"""Conceptual figures: the paradigm ladder, the four simulated worlds, the
cross-domain comparison, and the predict-explain-design-validate loop.

Nothing here is data.  These are diagrams, and the article says so; the only
quantitative figure in this file is the comparison matrix, which is explicitly
labelled as the author's qualitative assessment rather than a measurement.

Run:  python experiments/06_conceptual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import DIVERGING, PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGES = [
    ("recognise", "class labels", "CNN, SVM"),
    ("generate", "samples", "VAE, diffusion"),
    ("predict", "next state", "RNN, neural ODE"),
    ("simulate", "rollouts", "world models"),
    ("intervene", "do(x)", "causal graphs"),
    ("living systems", "mechanism +\nadaptation", "virtual cells"),
]


def fig_paradigm():
    fig, ax = plt.subplots(figsize=(9.4, 2.5))
    n = len(STAGES)
    xs = np.linspace(0.06, 0.94, n)
    cols = [PAL["blue"], PAL["blue"], PAL["sky"], PAL["sky"],
            PAL["green"], PAL["green"]]
    for i, ((name, what, how), x, c) in enumerate(zip(STAGES, xs, cols)):
        ax.add_patch(FancyBboxPatch((x - 0.068, 0.36), 0.136, 0.30,
                                    boxstyle="round,pad=0.012,rounding_size=0.02",
                                    linewidth=0, facecolor=c, alpha=0.92))
        ax.text(x, 0.58, name, ha="center", va="center", color="white",
                fontsize=8.2, weight="bold")
        ax.text(x, 0.47, what, ha="center", va="center", color="white", fontsize=6.6)
        ax.text(x, 0.22, how, ha="center", va="center", color=PAL["grey"], fontsize=6.4)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + 0.07, 0.51), (xs[i + 1] - 0.07, 0.51),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color=PAL["grey"], lw=0.9))
    ax.text(0.06, 0.86, "each step widens the question the model must answer",
            fontsize=8, color=PAL["dark"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-paradigm")


def fig_four_worlds():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.9))
    rng = np.random.default_rng(3)

    # language: a chain of tokens
    ax = axes[0]
    xs = np.linspace(0.1, 0.9, 7)
    for i, x in enumerate(xs):
        ax.add_patch(plt.Circle((x, 0.5), 0.055, color=PAL["blue"],
                                alpha=0.25 + 0.1 * i))
        if i:
            ax.annotate("", xy=(x - 0.06, 0.5), xytext=(xs[i - 1] + 0.06, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=PAL["grey"], lw=0.8))
    ax.set_title("language\ndiscrete chain", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # images: a 2-D manifold with samples
    ax = axes[1]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 + 0.3 * np.cos(t), 0.5 + 0.22 * np.sin(2 * t), color=PAL["blue"], lw=1.0)
    ax.scatter(0.5 + 0.3 * np.cos(rng.uniform(0, 2 * np.pi, 60)),
               0.5 + 0.22 * np.sin(2 * rng.uniform(0, 2 * np.pi, 60)),
               s=5, color=PAL["sky"], alpha=0.7)
    ax.set_title("images\nsmooth manifold", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # physics: a trajectory in state space
    ax = axes[2]
    th = np.linspace(-np.pi, np.pi, 400)
    ax.plot(th, np.cos(th) * 1.4, color=PAL["green"], lw=1.1)
    tt = np.linspace(0, 4 * np.pi, 500)
    ax.plot(0.5 * np.cos(tt), 0.5 * np.sin(tt), color=PAL["green"], lw=0.9, alpha=0.8)
    ax.set_title("physics\nconserve quantities", fontsize=9)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-1.7, 1.7); ax.axis("off")

    # life: a landscape with several basins
    ax = axes[3]
    xg, yg = np.meshgrid(np.linspace(-1.4, 1.4, 120), np.linspace(-1.1, 1.1, 120))
    z = 0.35 * (xg ** 2 - 0.55) ** 2 + 0.5 * (yg ** 2 - 0.30) ** 2
    ax.contourf(xg, yg, z, levels=12, cmap="Blues_r", alpha=0.55)
    ax.contour(xg, yg, z, levels=6, colors="white", linewidths=0.4)
    for sx, sy in [(-0.74, -0.55), (0.74, -0.55), (-0.74, 0.55), (0.74, 0.55)]:
        ax.scatter([sx], [sy], s=18, color=PAL["vermillion"], zorder=3)
    ax.set_title("life\nseveral basins", fontsize=9)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.1, 1.1); ax.axis("off")

    fig.tight_layout()
    save(fig, "fig-four-worlds")


def fig_cross_domain():
    dims = ["observable state", "time horizon", "structural constraints",
            "feedback loops", "interventions", "causal identifiability",
            "counterfactuals", "long-horizon stability", "out-of-distribution",
            "multi-scale", "evaluation metrics", "cost of ground truth"]
    domains = ["language", "images", "physics", "life"]
    # 1 = mild, 3 = severe; author's qualitative rating, not a measurement
    score = np.array([
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        [2, 1, 3, 3],
        [1, 1, 1, 3],
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        [1, 1, 2, 3],
        [1, 1, 3, 3],
        [2, 2, 2, 3],
        [1, 1, 1, 3],
        [1, 2, 1, 3],
        [1, 2, 1, 3],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 5.9))
    im = ax.imshow(score, cmap="Blues", vmin=0.6, vmax=3.2, aspect="auto")
    ax.set_xticks(range(len(domains)), domains, fontsize=8)
    ax.set_yticks(range(len(dims)), dims, fontsize=8)
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            ax.text(j, i, int(score[i, j]), ha="center", va="center",
                    color="white" if score[i, j] > 2 else PAL["dark"], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dims), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3], shrink=0.7)
    cb.ax.set_yticklabels(["mild", "moderate", "severe"], fontsize=7)
    ax.set_title("Where the difficulty comes from (author's rating, 1-3)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig-cross-domain")
    save_metrics("cross_domain", {"scores": score.tolist(), "rows": dims,
                                  "cols": domains})


def fig_loop():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    labels = ["predict", "explain", "design", "validate"]
    details = ["held-out\nperturbations", "mechanism\nnot just fit", "choose the\nnext experiment",
               "wet lab or\nnew simulator"]
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 5)[:-1]
    cols = [PAL["blue"], PAL["sky"], PAL["green"], PAL["vermillion"]]
    for a, lab, det, c in zip(ang, labels, details, cols):
        x, y = 0.5 + 0.32 * np.cos(a), 0.5 + 0.32 * np.sin(a)
        ax.add_patch(plt.Circle((x, y), 0.115, color=c, alpha=0.9, lw=0))
        ax.text(x, y + 0.02, lab, ha="center", va="center", color="white",
                fontsize=8.5, weight="bold")
        ax.text(x, y - 0.045, det, ha="center", va="center", color="white", fontsize=5.8)
    for i in range(4):
        a1, a2 = ang[i], ang[(i + 1) % 4]
        p1 = (0.5 + 0.40 * np.cos(a1), 0.5 + 0.40 * np.sin(a1))
        p2 = (0.5 + 0.40 * np.cos(a2), 0.5 + 0.40 * np.sin(a2))
        ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle="arc3,rad=-0.22",
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=PAL["grey"], lw=1.0))
    ax.text(0.5, 0.5, "the loop is the\nscientific claim", ha="center", va="center",
            fontsize=8, color=PAL["grey"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout()
    save(fig, "fig-loop")


if __name__ == "__main__":
    set_seed(0)
    set_style()
    fig_paradigm()
    fig_four_worlds()
    fig_cross_domain()
    fig_loop()
