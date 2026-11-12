"""Supplementary study B — five clocks, one person, five answers.

Five simulated clocks are trained on the same ten-dimensional damage state but
with different probe weights, panel sizes and noise.  All predict chronological
age well on average; the question is whether they agree about *individuals*.
The CpG counts in the labels are the published panel sizes (Horvath 353,
Hannum 71, PhenoAge 513, GrimAge 1030, DunedinPACE 173); the weights here are
simulated analogues, not the real clocks.

Run:  python experiments/14_clock_zoo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import DIVERGING, PAL, save, set_style
from common.utils import save_metrics, set_seed

CLOCKS = [
    ("Horvath-like (353)", 353, 1.0, 0.18),
    ("Hannum-like (71)", 71, 0.8, 0.28),
    ("PhenoAge-like (513)", 513, 1.2, 0.16),
    ("GrimAge-like (1030)", 1030, 1.4, 0.14),
    ("DunedinPACE-like (173)", 173, 0.9, 0.22),
]
K = np.array([0.020, 0.030, 0.015, 0.025, 0.022, 0.018, 0.030, 0.024, 0.017, 0.020])


def main() -> None:
    set_seed(0)
    set_style()
    rng = np.random.default_rng(7)

    n = 3000
    age = rng.uniform(20, 90, n)
    x = np.array([1 - np.exp(-K * a) for a in age])
    x = np.clip(x + rng.normal(0, 0.03, x.shape), 0, 1)

    preds = []
    for name, n_probe, scale, noise in CLOCKS:
        W = np.abs(rng.normal(0, 1, (min(n_probe, 40), 10)))
        W /= np.linalg.norm(W, axis=0)
        W *= scale
        Z = x @ W.T + rng.normal(0, noise, (n, W.shape[0]))
        Fm, tm = Z.mean(0), age.mean()
        Fc, tc = Z - Fm, age - tm
        beta = np.linalg.solve(Fc.T @ Fc + np.eye(Z.shape[1]), Fc.T @ tc)
        preds.append((Z - Fm) @ beta + tm)
    preds = np.array(preds)
    resid = preds - age

    C = np.corrcoef(resid)
    iu = np.triu_indices(len(CLOCKS), 1)
    mean_r = float(C[iu].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax = axes[0]
    im = ax.imshow(C, cmap=DIVERGING, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLOCKS)))
    ax.set_yticks(range(len(CLOCKS)))
    short = ["Horvath", "Hannum", "PhenoAge", "GrimAge", "Dunedin"]
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=6.6)
    ax.set_yticklabels(short, fontsize=6.6)
    for i in range(len(CLOCKS)):
        for j in range(len(CLOCKS)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.0, color="white" if C[i, j] > 0.65 else PAL["dark"])
    ax.set_title("(a)  Clocks agree on average, not on people", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6)

    ax2 = axes[1]
    r1, r2 = resid[0], resid[3]
    ax2.scatter(r1, r2, s=5, color=PAL["blue"], alpha=0.25, linewidths=0)
    ax2.axhline(0, color=PAL["grey"], lw=0.6)
    ax2.axvline(0, color=PAL["grey"], lw=0.6)
    discord = np.mean(np.sign(r1) != np.sign(r2))
    ax2.set_xlabel("Horvath-like age acceleration (years)")
    ax2.set_ylabel("GrimAge-like age acceleration (years)")
    ax2.set_title(f"(b)  Opposite signs in {100 * discord:.0f}% of people", loc="left")
    ax2.grid(True, alpha=0.15)
    lim = 18
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

    fig.text(0.01, -0.07,
             "Simulated cohort (n = 3,000) from a ten-dimensional damage state; the five "
             "clocks differ in panel size, weights and noise. CpG counts are the published "
             "panel sizes, but the weights are analogues, not the real algorithms.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-clock-zoo")

    save_metrics("clock_zoo", {
        "n_clocks": len(CLOCKS),
        "mean_pairwise_residual_correlation": mean_r,
        "min_pairwise": float(C[iu].min()),
        "max_pairwise": float(C[iu].max()),
        "discordant_sign_fraction": float(discord),
        "cpg_counts": [c[1] for c in CLOCKS],
    })


if __name__ == "__main__":
    main()
