"""Study 1 - is a nominal 5% test actually 5%?

Two ways to get a p-value: an analytic approximation (t-test for Pearson,
AS 89 for Spearman, normal approximation for Kendall) or a Monte-Carlo null
built from the data.  For each measure we build the null from 400 independent
draws, take its 95th percentile, then count how often 400 fresh independent
draws exceed it.  The analytic p-values are counted on the same draws.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

N_GRID = [20, 50, 200]
N_NULL = 400
N_TEST = 400
ALPHA = 0.05

METHODS = [
    ("pearson", C.pearson), ("spearman", C.spearman),
    ("kendall", C.kendall_tau_b), ("gamma", C.gamma_gk),
    ("somers_d", C.somers_d), ("hoeffding", C.hoeffding_d),
    ("dcor", C.dcor), ("hsic", C.hsic), ("mic", C.mic),
    ("rdc", C.rdc), ("xi", C.xi_chatterjee), ("mi", C.mutual_info),
]


def p_pearson(x, y) -> float:
    """Two-sided t-test for the linear correlation."""
    r = C.pearson(x, y)
    n = len(x)
    t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
    return float(2 * stats.t.sf(abs(t), n - 2))


def p_spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).pvalue)


def p_kendall(x, y) -> float:
    return float(stats.kendalltau(x, y).pvalue)


ANALYTIC = [("pearson_t", p_pearson), ("spearman_as89", p_spearman),
            ("kendall_asym", p_kendall)]


def draw(kind: str, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if kind == "gaussian":
        return rng.normal(size=n), rng.normal(size=n)
    if kind == "heavy":
        return rng.standard_t(3, size=n), rng.standard_t(3, size=n)
    if kind == "ties":
        return np.round(rng.normal(size=n)), np.round(rng.normal(size=n))
    if kind == "scaled":
        return 3 * rng.normal(size=n), rng.normal(size=n)
    raise ValueError(kind)


SETTINGS = ["gaussian", "heavy", "ties", "scaled"]
SET_COL = {"gaussian": PAL["blue"], "heavy": PAL["orange"],
           "ties": PAL["teal"], "scaled": PAL["gold"]}


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    out = {}
    ana = {}
    for setting in SETTINGS:
        out[setting] = {m: [] for m, _ in METHODS}
        ana[setting] = {m: [] for m, _ in ANALYTIC}
        for n in N_GRID:
            null = {m: np.empty(N_NULL) for m, _ in METHODS}
            for i in range(N_NULL):
                x, y = draw(setting, n, rng)
                for m, fn in METHODS:
                    null[m][i] = abs(fn(x, y))
            q95 = {m: np.quantile(null[m], 1 - ALPHA) for m, _ in METHODS}
            hits = {m: 0 for m, _ in METHODS}
            ahits = {m: 0 for m, _ in ANALYTIC}
            for _ in range(N_TEST):
                x, y = draw(setting, n, rng)
                for m, fn in METHODS:
                    if abs(fn(x, y)) > q95[m]:
                        hits[m] += 1
                for m, fn in ANALYTIC:
                    if fn(x, y) < ALPHA:
                        ahits[m] += 1
            for m, _ in METHODS:
                out[setting][m].append(hits[m] / N_TEST)
            for m, _ in ANALYTIC:
                ana[setting][m].append(ahits[m] / N_TEST)
        print(f"[calibration] {setting} done")

    names = [m for m, _ in METHODS]
    j = 2  # n = 50 column
    band = 2 * np.sqrt(ALPHA * (1 - ALPHA) / N_TEST)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.axvspan(ALPHA - band, ALPHA + band, color=PAL["light"], alpha=0.7, lw=0)
    ax.axvline(ALPHA, color=PAL["dark"], lw=0.8, ls=":")
    for k, setting in enumerate(SETTINGS):
        ax.scatter([out[setting][m][j] for m in names],
                   np.arange(len(names)) + (k - 1.5) * 0.13,
                   s=16, color=SET_COL[setting], label=setting, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("rejection rate at nominal 5% (n = 50)")
    ax.set_title("every measure is calibrated once the null is built from data", loc="left",
                 fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig03_type1")

    save_metrics("calibration", {
        "alpha": ALPHA, "n_grid": N_GRID, "n_null": N_NULL, "n_test": N_TEST,
        "settings": SETTINGS, "type1": out, "analytic_type1": ana,
        "n_for_plot": N_GRID[j],
        "note": "Monte-Carlo null from 400 draws; analytic p-values on the same draws",
    })


if __name__ == "__main__":
    main()
