"""Study 6 - confounding and Simpson's paradox.

Panel (a): a strong common cause flips the sign of the marginal association;
raw r is positive, the partial correlation is negative, the causal effect is
negative.  Panel (b): the same flip can happen with a categorical grouping
variable, which is Simpson's paradox in its usual form.  Both are simulations
with fixed coefficients; they are not fits to data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, DIV, save, set_style
from common.utils import save_metrics, set_seed

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)

    # ---- (a) continuous confounder
    n = 600
    z = rng.normal(size=n)
    x = 0.9 * z + 0.435 * rng.normal(size=n)
    y = -0.6 * x + 0.8 * z + 0.5 * rng.normal(size=n)
    raw = C.pearson(x, y)
    par = C.partial_corr(x, y, z)

    # ---- (b) Simpson's paradox with two groups
    m = 250
    xa = rng.normal(0, 1, m)
    ya = -0.8 * xa + 0.5 * rng.normal(size=m)
    xb = rng.normal(4, 1, m)
    yb = -0.8 * (xb - 4) + 12 + 0.5 * rng.normal(size=m)
    xs = np.concatenate([xa, xb])
    ys = np.concatenate([ya, yb])
    overall = C.pearson(xs, ys)
    within_a = C.pearson(xa, ya)
    within_b = C.pearson(xb, yb)
    print(f"[confound] raw={raw:.3f} partial={par:.3f} | "
          f"simpson overall={overall:.3f} within=({within_a:.3f},{within_b:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    sc = ax.scatter(x, y, c=z, cmap=DIV, s=8, alpha=0.6, edgecolor="none")
    b1, b0 = np.polyfit(x, y, 1)
    xs_ = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs_, b0 + b1 * xs_, color=PAL["orange"], lw=1.2)
    ax.set_title(f"(a)  raw r = {raw:+.3f}", loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("confounder z", fontsize=7)

    ax = axes[1]
    ax.scatter(xa, ya, s=8, color=PAL["blue"], alpha=0.45, edgecolor="none", label="group A")
    ax.scatter(xb, yb, s=8, color=PAL["orange"], alpha=0.45, edgecolor="none", label="group B")
    for xx, yy, col in ((xa, ya, PAL["blue"]), (xb, yb, PAL["orange"])):
        b1, b0 = np.polyfit(xx, yy, 1)
        grid = np.linspace(xx.min(), xx.max(), 40)
        ax.plot(grid, b0 + b1 * grid, color=col, lw=1.1)
    b1, b0 = np.polyfit(xs, ys, 1)
    grid = np.linspace(xs.min(), xs.max(), 40)
    ax.plot(grid, b0 + b1 * grid, color=PAL["dark"], lw=1.4, ls="--")
    ax.set_title(f"(b)  overall r = {overall:+.2f}; within groups "
                 f"{within_a:+.2f}, {within_b:+.2f}", loc="left", fontsize=8.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig07_confound")

    save_metrics("confound", {
        "raw": float(raw), "partial": float(par), "causal_coef": -0.6, "n": n,
        "simpson_overall": float(overall),
        "simpson_within": [float(within_a), float(within_b)],
        "simpson_slope_true": -0.8, "simpson_n_per_group": m,
    })


if __name__ == "__main__":
    main()
