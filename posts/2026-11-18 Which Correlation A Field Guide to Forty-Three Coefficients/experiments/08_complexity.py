"""Study 8 - how the cost grows with n.

Wall-clock time per call (median of three) for eight measures.  The O(n^2)
methods are capped at n = 2000 so that the distance matrices stay in memory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

GRID = [100, 300, 1000, 2000, 5000, 10000]
CAP = {"dcor": 2000, "hsic": 2000, "mic": 5000}

METHODS = [
    ("pearson", C.pearson), ("spearman", C.spearman),
    ("kendall", C.kendall_tau_b), ("mic", C.mic), ("xi", C.xi_chatterjee),
    ("rdc", C.rdc), ("dcor", C.dcor), ("hsic", C.hsic),
]
COL = {"pearson": "grey", "spearman": "blue", "kendall": "teal", "mic": "gold",
       "xi": "navy", "rdc": "purple", "dcor": "orange", "hsic": "green"}


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    times = {}
    for name, fn in METHODS:
        times[name] = []
        for n in GRID:
            if n > CAP.get(name, max(GRID)):
                times[name].append(float("nan"))
                continue
            x = rng.normal(size=n)
            y = 0.5 * x + rng.normal(size=n)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn(x, y)
                ts.append(time.perf_counter() - t0)
            times[name].append(float(np.median(ts)))
        print(f"[complexity] {name} done")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, _ in METHODS:
        t = np.array(times[name])
        ok = ~np.isnan(t)
        ax.plot(np.array(GRID)[ok], t[ok], marker="o", ms=3.2,
                color=PAL[COL[name]], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("seconds per call (median of 3)")
    ax.set_title("cost of one call", loc="left")
    ax.legend(frameon=False, fontsize=6.6, ncol=2)
    fig.tight_layout()
    save(fig, "fig09_complexity")

    save_metrics("complexity", {
        "n_grid": GRID, "cap": CAP, "seconds": times,
        "note": "median of 3 calls, single CPU core, NumPy/SciPy only",
    })


if __name__ == "__main__":
    main()
