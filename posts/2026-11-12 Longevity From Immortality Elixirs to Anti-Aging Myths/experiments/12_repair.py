"""Study 12 — what would "80 back to 20" actually require?

Panel (a) is a repair inventory, not a damage score.  Each row is a system; the
number is a published quantity (or, where no single number exists, a stated
qualitative change).  Bars are drawn only when a percentage loss relative to
young adults is available in the literature; hollow bars mean "measured, but
not as a single percentage".  Mixing the two is deliberate: it prevents a
composite index from looking more precise than the data.

Panel (b) asks a different question: which of ten systems does each intervention
family plausibly touch?  0 = no, 1 = indirect/contested, 2 = direct evidence.
The row sums are the point.

Run:  python experiments/12_repair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import EVIDENCE, PAL, save, set_style
from common.utils import save_metrics, set_seed

# (system, percent lost or None, anchor, evidence tier)
LAYERS = [
    ("somatic genome", None, "~3,000 mutations per stem-cell genome by 80", "measured"),
    ("telomeres", 30, "leucocyte telomere −2.5 kb", "measured"),
    ("epigenome", None, "methylation age +30 yr", "measured"),
    ("proteostasis", None, "aggregate load rises several-fold", "inferred"),
    ("mitochondria", 30, "OXPHOS capacity −30%", "measured"),
    ("senescent cells", None, "~1% → ~10% of cells", "measured"),
    ("stem cell pool", 80, "naive CD8 T cells −80%", "measured"),
    ("extracellular matrix", 30, "skin elasticity −30%", "measured"),
    ("vasculature / kidney", 40, "nephrons −~40% by 80", "measured"),
    ("immune system", 90, "thymic output −90%", "measured"),
]

INTERVENTIONS = [
    ("caloric restriction", [1, 1, 2, 2, 2, 1, 1, 0, 1, 1]),
    ("rapamycin", [0, 0, 1, 2, 2, 2, 1, 0, 1, 1]),
    ("senolytics", [0, 0, 1, 1, 1, 2, 1, 0, 0, 1]),
    ("NAD$^+$ precursors", [0, 0, 1, 1, 2, 0, 1, 0, 0, 0]),
    ("telomerase therapy", [0, 2, 0, 0, 1, 0, 1, 0, 0, 1]),
    ("partial reprogramming", [0, 0, 2, 1, 1, 0, 1, 0, 0, 0]),
    ("stem cell therapy", [0, 0, 0, 0, 0, 1, 2, 0, 0, 1]),
    ("AI drug discovery", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
]


def main() -> None:
    set_seed(0)
    set_style()

    fig = plt.figure(figsize=(12.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0])

    y = np.arange(len(LAYERS))[::-1]
    for yi, (name, pct, anchor, tier) in zip(y, LAYERS):
        if pct is None:
            ax.barh(yi, 100, height=0.58, color="none", edgecolor=PAL["grey"],
                    hatch="////", linewidth=0.6)
        else:
            ax.barh(yi, pct, height=0.58, color=PAL["blue"], alpha=0.85)
            ax.text(pct + 1.5, yi, f"−{pct}%", va="center", fontsize=6.2,
                    color=PAL["blue"])
        ax.text(104, yi, anchor, va="center", fontsize=5.5, color=PAL["dark"])
    ax.set_yticks(y)
    ax.set_yticklabels([l[0] for l in LAYERS], fontsize=7)
    ax.set_xlim(0, 168)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("loss relative to a 20-year-old (%), where a single number exists")
    ax.set_title("(a)  Ten systems, ten different kinds of evidence", loc="left")
    ax.grid(axis="x")

    ax2 = fig.add_subplot(gs[1])
    M = np.array([r[1] for r in INTERVENTIONS], dtype=float)
    im = ax2.imshow(M, cmap=EVIDENCE, vmin=0, vmax=2, aspect="auto")
    ax2.set_xticks(range(len(LAYERS)))
    ax2.set_xticklabels([l[0].replace(" / ", "/\n") for l in LAYERS],
                        rotation=55, ha="right", fontsize=5.8)
    ax2.set_yticks(range(len(INTERVENTIONS)))
    ax2.set_yticklabels([r[0] for r in INTERVENTIONS], fontsize=6.6)
    sums = M.sum(axis=1)
    for i in range(M.shape[0]):
        ax2.text(len(LAYERS) - 0.35, i, f"{int(sums[i])}/10", va="center",
                 fontsize=6.4, color=PAL["dark"])
    ax2.set_title(f"(b)  Best coverage: {int(sums.max())} of 20 possible points",
                  loc="left")
    ax2.set_xlim(-0.5, len(LAYERS) + 0.3)

    fig.text(0.01, -0.16,
             "Panel (a): percentages are published values rounded; hatched rows are "
             "measured changes that have no single percentage. Panel (b): 0/1/2 is the "
             "author's reading of the cited literature and is not a meta-analysis.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-repair")

    save_metrics("repair", {
        "n_layers": len(LAYERS),
        "n_with_single_pct": int(sum(1 for l in LAYERS if l[1] is not None)),
        "median_pct_loss": float(np.median([l[1] for l in LAYERS if l[1] is not None])),
        "max_coverage": int(sums.max()),
        "mean_coverage": float(sums.mean()),
        "coverage_by_intervention": {r[0]: int(s) for r, s in zip(INTERVENTIONS, sums)},
        "best_covered_layers": [LAYERS[j][0] for j in np.argsort(M.sum(0))[::-1][:3]],
        "uncovered_layers": [LAYERS[j][0] for j in range(len(LAYERS))
                             if M[:, j].max() == 0],
    })


if __name__ == "__main__":
    main()
