"""Study 2 - power against six kinds of dependence.

The null distribution is built by permuting y within each draw, so marginals are
preserved and only the dependence is destroyed.  Every test is the same
two-sided absolute-value test used in the calibration study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, SEQ, save, set_style
from common.utils import save_metrics, set_seed

N_GRID = [30, 100, 300]
N_NULL = 200
N_REP = 60
ALPHA = 0.05

METHODS = [
    ("pearson", C.pearson), ("spearman", C.spearman),
    ("kendall", C.kendall_tau_b), ("gamma", C.gamma_gk),
    ("hoeffding", C.hoeffding_d), ("dcor", C.dcor), ("hsic", C.hsic),
    ("mic", C.mic), ("rdc", C.rdc), ("xi", C.xi_chatterjee),
    ("mi", C.mutual_info),
]

STRUCTURES = ["linear", "cubic", "sine", "circle", "heteroskedastic", "clusters"]


def sample(structure: str, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if structure == "linear":
        x = rng.normal(size=n)
        return x, x + 0.7 * rng.normal(size=n)
    if structure == "cubic":
        x = rng.uniform(-1.5, 1.5, n)
        return x, x ** 3 + 1.2 * rng.normal(size=n)
    if structure == "sine":
        x = rng.uniform(-1, 1, n)
        return x, np.sin(2 * np.pi * x) + 0.6 * rng.normal(size=n)
    if structure == "circle":
        th = rng.uniform(0, 2 * np.pi, n)
        return np.cos(th) + 0.05 * rng.normal(size=n), \
            np.sin(th) + 0.05 * rng.normal(size=n)
    if structure == "heteroskedastic":
        x = rng.normal(size=n)
        return x, rng.normal(size=n) * (0.2 + np.abs(x))
    if structure == "clusters":
        k = rng.integers(0, 3, n)
        centers = np.array([[0, 0], [2.5, 2.5], [0, 2.5]])
        return (centers[k, 0] + 0.5 * rng.normal(size=n),
                centers[k, 1] + 0.5 * rng.normal(size=n))
    raise ValueError(structure)


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    power = {}
    for structure in STRUCTURES:
        power[structure] = {m: [] for m, _ in METHODS}
        for n in N_GRID:
            # null quantiles by permuting y within draws
            null = {m: np.empty(N_NULL) for m, _ in METHODS}
            for i in range(N_NULL):
                x, y = sample(structure, n, rng)
                yp = rng.permutation(y)
                for m, fn in METHODS:
                    null[m][i] = abs(fn(x, yp))
            q95 = {m: np.quantile(null[m], 1 - ALPHA) for m, _ in METHODS}
            hits = {m: 0 for m, _ in METHODS}
            for _ in range(N_REP):
                x, y = sample(structure, n, rng)
                for m, fn in METHODS:
                    if abs(fn(x, y)) > q95[m]:
                        hits[m] += 1
            for m, _ in METHODS:
                power[structure][m].append(hits[m] / N_REP)
        print(f"[power] {structure} done")

    show = ["pearson", "spearman", "dcor", "mic", "xi"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True, sharey=True)
    for ax, structure in zip(axes.ravel(), STRUCTURES):
        for m in show:
            ax.plot(N_GRID, power[structure][m], marker="o", ms=3.2,
                    color=PAL[{"pearson": "grey", "spearman": "blue",
                               "dcor": "teal", "mic": "orange",
                               "xi": "navy"}[m]], label=m)
        ax.axhline(ALPHA, color=PAL["dark"], lw=0.7, ls=":")
        ax.set_title(structure, loc="left", fontsize=8.4)
        ax.set_xscale("log")
        ax.set_xticks(N_GRID)
        ax.set_xticklabels([str(n) for n in N_GRID])
    axes[0, 0].set_ylabel("power")
    axes[1, 0].set_ylabel("power")
    for ax in axes[1]:
        ax.set_xlabel("n")
    axes[1, 2].legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig04_power")

    save_metrics("power", {
        "alpha": ALPHA, "n_grid": N_GRID, "n_rep": N_REP, "n_null": N_NULL,
        "structures": STRUCTURES, "power": power,
        "note": "null by permuting y within each draw; 60 independent draws per cell",
    })


if __name__ == "__main__":
    main()
