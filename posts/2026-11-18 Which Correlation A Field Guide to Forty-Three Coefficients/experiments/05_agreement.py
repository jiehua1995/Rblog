"""Study 5 - correlation is not agreement.

Two "methods" measure the same quantity: one is unbiased, one has a proportional
bias.  Pearson can be nearly identical in both cases; ICC, CCC and the
Bland-Altman limits are not.

The true values are the flipper lengths of the Palmer penguins (a real marginal
distribution); the two measurement methods are simulated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

HERE = Path(__file__).resolve().parents[1]


def compare(a, b) -> dict:
    bias, lo, hi = C.bland_altman(a, b)
    return {
        "pearson": C.pearson(a, b),
        "icc11": C.icc_1_1(a, b),
        "icc21": C.icc_2_1(a, b),
        "ccc": C.ccc(a, b),
        "bias": bias, "loa_low": lo, "loa_high": hi,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
    }


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    pen = pd.read_csv(HERE / "data" / "penguins.csv")
    truth = pen["flipper_length_mm"].dropna().to_numpy()

    a = truth + rng.normal(0, 3.5, size=len(truth))
    good = truth + rng.normal(0, 3.5, size=len(truth))
    bad = 2.0 + 1.05 * truth + rng.normal(0, 4.0, size=len(truth))

    res = {"good": compare(a, good), "bad": compare(a, bad),
           "n": int(len(truth))}
    print("[agreement]", {k: round(v, 3) for k, v in res["bad"].items()})

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    for row, (label, b) in enumerate([("unbiased", good), ("proportional bias", bad)]):
        ax = axes[row, 0]
        ax.scatter(a, b, s=6, color=PAL["navy"], alpha=0.35, edgecolor="none")
        lim = [min(a.min(), b.min()) - 3, max(a.max(), b.max()) + 3]
        ax.plot(lim, lim, color=PAL["orange"], lw=0.9, ls="--")
        r = res["good" if row == 0 else "bad"]
        ax.set_title(f"({'a' if row == 0 else 'c'})  {label}: r={r['pearson']:.3f}, "
                     f"ICC={r['icc21']:.3f}", loc="left", fontsize=8.2)
        ax.set_xlabel("method A (mm)")
        ax.set_ylabel("method B (mm)")

        ax = axes[row, 1]
        d = a - b
        m = (a + b) / 2
        ax.scatter(m, d, s=6, color=PAL["teal"], alpha=0.35, edgecolor="none")
        ax.axhline(r["bias"], color=PAL["orange"], lw=0.9)
        ax.axhline(r["loa_low"], color=PAL["grey"], lw=0.7, ls=":")
        ax.axhline(r["loa_high"], color=PAL["grey"], lw=0.7, ls=":")
        ax.set_title(f"({'b' if row == 0 else 'd'})  Bland-Altman: bias={r['bias']:.2f} mm",
                     loc="left", fontsize=8.2)
        ax.set_xlabel("mean of the two methods (mm)")
        ax.set_ylabel("A - B (mm)")
    fig.tight_layout()
    save(fig, "fig06_agreement")

    save_metrics("agreement", res)


if __name__ == "__main__":
    main()
