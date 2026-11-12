"""Study 5 — what an age clock can and cannot see.

A ten-dimensional "damage state" x(a) grows with age.  Two clocks are trained
on a simulated cohort:

  * M-clock: ridge regression on 20 methylation-like features z = W x + noise.
  * P-clock: ridge regression on 6 phenotypic biomarkers y = C x + noise.

Both predict chronological age well in the training distribution.  We then
apply two kinds of intervention at age 70:

  * a true component intervention: halve one damage component (x_i *= 0.5);
  * a methylation-only intervention: move z along the direction the M-clock
    reads, leaving x untouched.

The panel asks a narrow question: when the clock says -30 years, how much of
the *state* actually moved?  This is a demonstration of a logical possibility,
not evidence that any real intervention works this way.

Run:  python experiments/05_clocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import DIVERGING, PAL, save, set_style
from common.utils import save_metrics, set_seed

NAMES = ["DNA damage", "telomere", "epigenetic", "proteostasis", "mitochondria",
         "senescence", "stem cells", "inflammation", "ECM", "immune"]
K = np.array([0.020, 0.030, 0.015, 0.025, 0.022, 0.018, 0.030, 0.024, 0.017, 0.020])
AGE = np.arange(20, 91, 1.0)


def state(age):
    """Damage state in [0, 1] for a single age."""
    return 1 - np.exp(-K * age)


def cohort(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 90, n)
    x = np.array([state(a) for a in age])
    # person-specific offsets
    x = np.clip(x + rng.normal(0, 0.03, x.shape), 0, 1)
    return age, x


def build(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age, x = cohort(n, seed)

    W = np.abs(rng.normal(0, 1, (20, 10)))
    W /= np.linalg.norm(W, axis=0)          # equal weight per axis...
    W *= 2.0
    W[:, 2] *= 1.5                          # ...except the epigenetic axis
    W[:, 7] *= 1.2
    Z = x @ W.T + rng.normal(0, 0.18, (n, 20))

    C = np.abs(rng.normal(0, 1, (6, 10)))
    C[:, 4] *= 2.2            # a phenotypic panel weighs mitochondria heavily
    C[:, 8] *= 1.4
    Y = x @ C.T + rng.normal(0, 0.10, (n, 6))

    def ridge(F, t, lam=1.0):
        Fm, tm = F.mean(0), t.mean()
        Fc, tc = F - Fm, t - tm
        beta = np.linalg.solve(Fc.T @ Fc + lam * np.eye(F.shape[1]), Fc.T @ tc)
        return beta, Fm, tm

    bM, mM, tM = ridge(Z, age)
    bP, mP, tP = ridge(Y, age)

    predM = (Z - mM) @ bM + tM
    predP = (Y - mP) @ bP + tP
    return dict(age=age, x=x, W=W, C=C, bM=bM, mM=mM, tM=tM,
                bP=bP, mP=mP, tP=tP, predM=predM, predP=predP)


def clock_m(d, x):
    z = x @ d["W"].T
    return float((z - d["mM"]) @ d["bM"] + d["tM"])


def clock_p(d, x):
    y = x @ d["C"].T
    return float((y - d["mP"]) @ d["bP"] + d["tP"])


def function_score(x):
    return float(1 - np.mean(x))


def main() -> None:
    set_seed(0)
    set_style()
    d = build()

    age0 = 70.0
    x0 = state(age0)
    base = {"M": clock_m(d, x0), "P": clock_p(d, x0), "S": function_score(x0)}

    rows = []
    for i, name in enumerate(NAMES):
        x1 = x0.copy()
        x1[i] *= 0.5
        rows.append(("true:" + name, i,
                     base["M"] - clock_m(d, x1),
                     base["P"] - clock_p(d, x1),
                     100 * (function_score(x1) - base["S"])))

    # methylation-only shift: the M-clock reads -30 years, x is untouched
    rows.append(("methylation-only", -1, 30.0, 0.0, 0.0))

    # a "rejuvenating" methylation shift accompanied by worse mitochondria
    x_worse = x0.copy()
    x_worse[4] = min(1.0, x_worse[4] * 1.25)
    rows.append(("methylation + mito harm", -2,
                 30.0, base["P"] - clock_p(d, x_worse),
                 100 * (function_score(x_worse) - base["S"])))

    names = [r[0] for r in rows]
    dM = np.array([r[2] for r in rows])
    dP = np.array([r[3] for r in rows])
    dS = np.array([r[4] for r in rows])

    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax.scatter(dS, dM, s=36, color=PAL["blue"], label="M-clock (methylation)",
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(dS, dP, s=36, color=PAL["orange"], marker="s",
               label="P-clock (phenotype)", edgecolor="white", linewidth=0.6,
               zorder=3)
    lim = 33
    ax.plot([0, 30], [0, 30], color=PAL["grey"], ls=":", lw=1)
    ax.axhline(0, color=PAL["grey"], lw=0.6)
    ax.axvline(0, color=PAL["grey"], lw=0.6)
    for i, nm in enumerate(names):
        if nm in ("methylation-only", "methylation + mito harm"):
            xy = (2.2, 27.5) if nm == "methylation-only" else (-3.6, 23.5)
            ax.annotate(nm.replace(" + ", "\n+ "), (dS[i], dM[i]),
                        xytext=xy, textcoords="data",
                        fontsize=6.2, ha="left", color=PAL["blue"],
                        arrowprops=dict(arrowstyle="-", color=PAL["blue"],
                                        lw=0.5))
    ax.set_xlabel("true functional gain (%)")
    ax.set_ylabel("reported clock reduction (years)")
    ax.set_xlim(-4, 14)
    ax.set_ylim(-2, lim)
    ax.set_title("(a)  A clock can move without the state moving", loc="left")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.grid(True)

    # panel b: matrix of standardised changes
    ax2 = fig.add_subplot(gs[1])
    mat = np.column_stack([dM, dP, dS])
    mat = mat / np.maximum(np.abs(mat).max(axis=0), 1e-9)
    im = ax2.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=6.0)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["M-clock", "P-clock", "function"], fontsize=7)
    for i in range(len(names)):
        for j in range(3):
            v = mat[i, j]
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     fontsize=5.4, color="white" if abs(v) > 0.6 else PAL["dark"])
    ax2.set_title("(b)  Discordance, row by row", loc="left")
    cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    fig.text(0.01, -0.07,
             "Simulated state vector (10 components), ridge clocks trained on 4,000 "
             "synthetic subjects, intervention at age 70. The example is constructed to "
             "show that clock readings and state changes are logically separable.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-clocks")

    idx_epi = NAMES.index("epigenetic")
    ratio_M = dM[:10] / np.maximum(dS[:10], 1e-9)
    ratio_P = dP[:10] / np.maximum(dS[:10], 1e-9)
    save_metrics("clocks", {
        "clock_mae_M": float(np.mean(np.abs(d["predM"] - d["age"]))),
        "clock_mae_P": float(np.mean(np.abs(d["predP"] - d["age"]))),
        "epigenetic_intervention": {
            "clock_M": float(dM[idx_epi]),
            "clock_P": float(dP[idx_epi]),
            "function_pct": float(dS[idx_epi]),
        },
        "methylation_only_intervention": {
            "clock_M": 30.0, "function_pct": 0.0,
        },
        "clockM_years_per_pct_median": float(np.median(ratio_M)),
        "clockM_years_per_pct_range": [float(ratio_M.min()), float(ratio_M.max())],
        "clockP_years_per_pct_median": float(np.median(ratio_P)),
        "clockP_years_per_pct_range": [float(ratio_P.min()), float(ratio_P.max())],
        "max_clock_reduction": float(np.max(dM)),
        "min_function_gain": float(np.min(dS)),
    })


if __name__ == "__main__":
    main()
