"""Study 3 - invariance and robustness.

Two questions: which measures survive a monotone transform, and what happens to
each measure when a fraction of the data is replaced by gross outliers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

N = 400
REPS = 40
FRACS = [0.0, 0.02, 0.05, 0.10, 0.20]
METHODS = [
    ("pearson", C.pearson), ("spearman", C.spearman),
    ("kendall", C.kendall_tau_b), ("dcor", C.dcor),
    ("mic", C.mic), ("xi", C.xi_chatterjee),
]


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)

    # ---- monotone transform
    base = {m: [] for m, _ in METHODS}
    trans = {m: [] for m, _ in METHODS}
    for _ in range(REPS):
        x = rng.normal(size=N)
        y = 0.6 * x + 0.8 * rng.normal(size=N)
        xt = np.exp(x)
        for m, fn in METHODS:
            base[m].append(fn(x, y))
            trans[m].append(fn(xt, y))
    mono = {m: {"base": float(np.mean(base[m])), "transformed": float(np.mean(trans[m]))}
            for m, _ in METHODS}

    # ---- contamination
    cont = {m: [] for m, _ in METHODS}
    for f in FRACS:
        vals = {m: [] for m, _ in METHODS}
        for _ in range(REPS):
            x = rng.normal(size=N)
            y = 0.8 * x + 0.6 * rng.normal(size=N)
            k = int(round(f * N))
            if k:
                idx = rng.choice(N, size=k, replace=False)
                x[idx] = 40 + rng.normal(size=k) * 0.5
            for m, fn in METHODS:
                vals[m].append(fn(x, y))
        for m, _ in METHODS:
            cont[m].append(float(np.mean(vals[m])))
    print("[invariance] done")

    # ---- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax = axes[0]
    for m in ["pearson", "spearman", "kendall", "dcor", "mic", "xi"]:
        ax.plot(FRACS, cont[m], marker="o", ms=3.2,
                color=PAL[{"pearson": "orange", "spearman": "blue",
                           "kendall": "teal", "dcor": "navy",
                           "mic": "gold", "xi": "purple"}[m]], label=m)
    ax.axhline(0, color=PAL["dark"], lw=0.7, ls=":")
    ax.set_xlabel("fraction of gross outliers in x")
    ax.set_ylabel("measure")
    ax.set_title("(a)  outliers", loc="left")
    ax.legend(frameon=False, fontsize=6.8, ncol=2)

    ax = axes[1]
    names = [m for m, _ in METHODS]
    ypos = np.arange(len(names))
    ax.barh(ypos - 0.18, [mono[m]["base"] for m in names], height=0.34,
            color=PAL["blue"], label="x")
    ax.barh(ypos + 0.18, [mono[m]["transformed"] for m in names], height=0.34,
            color=PAL["orange"], label="exp(x)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("measure")
    ax.set_title("(b)  monotone transform of x", loc="left")
    ax.legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig05_robust")

    save_metrics("invariance", {
        "n": N, "reps": REPS, "contamination_grid": FRACS,
        "contamination": cont, "monotone": mono,
    })


if __name__ == "__main__":
    main()
