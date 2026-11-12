"""Study 4 — the 12 hallmarks as a network, and how strong the arrows are.

The 12 nodes and their grouping (primary / antagonistic / integrative) follow
López-Otín et al. (2023).  The edges are the author's reading of that review
and of the primary papers cited in the article; each edge is labelled

    causal        an experiment perturbs A and observes B
    correlative   A and B co-vary across ages or interventions
    hypothetical  a plausible mechanism, not yet demonstrated in vivo

This is a curated map, not a systematic review, and it is deliberately
conservative: an edge is only "causal" if a perturbation experiment supports
the direction.  The count of each type is the quantitative part.

Run:  python experiments/04_hallmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

NODES = {
    "GI": ("genomic instability", 0, 4.0, "primary"),
    "TL": ("telomere attrition", 0, 3.0, "primary"),
    "EP": ("epigenetic change", 0, 2.0, "primary"),
    "PR": ("proteostasis loss", 0, 1.0, "primary"),
    "AU": ("autophagy loss", 0, 0.0, "primary"),
    "NU": ("nutrient sensing", 1, 3.2, "antagonistic"),
    "MI": ("mitochondria", 1, 2.1, "antagonistic"),
    "SE": ("senescent cells", 1, 1.0, "antagonistic"),
    "ST": ("stem cell loss", 2, 3.7, "integrative"),
    "IC": ("communication", 2, 2.7, "integrative"),
    "IN": ("inflammation", 2, 1.7, "integrative"),
    "DY": ("dysbiosis", 2, 0.7, "integrative"),
}

GROUP_COL = {"primary": PAL["blue"], "antagonistic": PAL["orange"],
             "integrative": PAL["green"]}

# (source, target, evidence)
EDGES = [
    ("GI", "EP", "causal"), ("GI", "TL", "causal"), ("GI", "ST", "causal"),
    ("TL", "GI", "causal"), ("TL", "SE", "causal"), ("TL", "ST", "causal"),
    ("EP", "ST", "causal"), ("EP", "SE", "causal"), ("EP", "IC", "correlative"),
    ("PR", "SE", "causal"), ("PR", "AU", "causal"), ("AU", "PR", "causal"),
    ("AU", "MI", "causal"), ("AU", "IN", "correlative"),
    ("NU", "MI", "causal"), ("NU", "PR", "causal"), ("NU", "AU", "causal"),
    ("NU", "IN", "correlative"), ("MI", "SE", "causal"), ("MI", "IN", "causal"),
    ("MI", "ST", "correlative"), ("SE", "IN", "causal"), ("SE", "ST", "causal"),
    ("SE", "IC", "causal"), ("ST", "IC", "causal"), ("ST", "IN", "correlative"),
    ("IC", "IN", "causal"), ("IC", "NU", "hypothetical"),
    ("IN", "ST", "causal"), ("IN", "SE", "causal"), ("IN", "EP", "correlative"),
    ("DY", "IN", "causal"), ("DY", "IC", "correlative"), ("DY", "SE", "hypothetical"),
]

STYLE = {"causal": ("-", 1.1, 0.75), "correlative": ((0, (3, 2)), 0.9, 0.55),
         "hypothetical": ((0, (1, 2)), 0.8, 0.45)}


def main() -> None:
    set_seed(0)
    set_style()

    fig = plt.figure(figsize=(11.6, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0])

    pos = {k: (v[1], v[2]) for k, v in NODES.items()}
    for s, t, kind in EDGES:
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        ls, lw, alpha = STYLE[kind]
        arrow = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                mutation_scale=7, linewidth=lw,
                                linestyle=ls, color=PAL["grey"],
                                alpha=alpha, shrinkA=16, shrinkB=16,
                                connectionstyle="arc3,rad=0.06")
        ax.add_patch(arrow)

    for key, (label, x, y, group) in NODES.items():
        ax.text(x, y, key, ha="center", va="center", fontsize=7.6,
                color="white", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.32", fc=GROUP_COL[group],
                          ec="none"))

    for group, x in [("primary", 0), ("antagonistic", 1), ("integrative", 2)]:
        ax.text(x, 4.72, group, ha="center", fontsize=7.4,
                color=GROUP_COL[group], fontweight="bold")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-1.55, 5.0)
    ax.axis("off")

    # abbreviation key, two columns, in the empty lower-left corner
    keys = [k for k in NODES]
    for i, k in enumerate(keys):
        col, row = divmod(i, 6)
        ax.text(-0.5 + col * 1.55, -0.35 - row * 0.19,
                f"{k}  {NODES[k][0]}", fontsize=5.9, color=PAL["dark"],
                ha="left", va="center")
    for kind, lab in [("causal", "causal (perturbation)"),
                      ("correlative", "correlative"),
                      ("hypothetical", "hypothetical")]:
        ls, lw, _ = STYLE[kind]
        ax.plot([], [], color=PAL["grey"], ls=ls, lw=lw, label=lab)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.12), ncol=1,
              frameon=False, fontsize=6.6)
    ax.set_title("(a)  Twelve hallmarks, three kinds of arrow", loc="left")

    ax2 = fig.add_subplot(gs[1])
    order = ["GI", "TL", "EP", "PR", "AU", "NU", "MI", "SE", "ST", "IC",
             "IN", "DY"]
    counts = {k: {"causal": 0, "correlative": 0, "hypothetical": 0} for k in order}
    for s, t, kind in EDGES:
        counts[s][kind] += 1
        counts[t][kind] += 1
    bottom = np.zeros(len(order))
    for kind, c in [("causal", PAL["blue"]), ("correlative", PAL["orange"]),
                    ("hypothetical", PAL["grey"])]:
        vals = np.array([counts[k][kind] for k in order])
        ax2.bar(range(len(order)), vals, bottom=bottom, color=c, width=0.68,
                label=kind)
        bottom += vals
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, fontsize=7.0)
    ax2.set_ylabel("number of edges touching the hallmark")
    ax2.set_title("(b)  Hubs and weak links", loc="left")
    ax2.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax2.grid(axis="y")

    fig.text(0.01, -0.08,
             "Edge curation is the author's reading of López-Otín et al. (2023) "
             "and the primary literature; it is not a systematic review and the "
             "counts are not effect sizes.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-hallmarks")

    total = len(EDGES)
    by_kind = {k: sum(1 for e in EDGES if e[2] == k)
               for k in ("causal", "correlative", "hypothetical")}
    degree = {k: sum(1 for e in EDGES if k in (e[0], e[1])) for k in NODES}
    top = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:4]

    save_metrics("hallmarks", {
        "n_nodes": len(NODES),
        "n_edges": total,
        "edges_by_kind": by_kind,
        "fraction_causal": by_kind["causal"] / total,
        "top_hubs": [{"node": k, "degree": v} for k, v in top],
        "hubs_are_integrative": [k for k, _ in top if NODES[k][3] == "integrative"],
    })


if __name__ == "__main__":
    main()
