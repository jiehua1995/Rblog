"""Supplementary study C — when an intervention acts matters as much as how much.

A Gompertz life table (A = 5e-5 /yr, B = 0.085 /yr) with a permanent
proportional reduction delta in the hazard, starting at age a0.  The figure is
a grid of life-expectancy gains.  It is a standard demographic calculation, not
a model of any therapy: it shows how quickly the return on a fixed
mortality reduction falls when the intervention starts late.

Run:  python experiments/15_timing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, save, set_style
from common.utils import save_metrics, set_seed

A_HUMAN = 5e-5
B_HUMAN = 0.085
T = np.arange(0, 130, 0.1)


def survival(delta=0.0, a0=0.0):
    H0 = A_HUMAN / B_HUMAN * (np.exp(B_HUMAN * T) - 1)
    Ha0 = A_HUMAN / B_HUMAN * (np.exp(B_HUMAN * min(a0, 130)) - 1)
    late = Ha0 + (1 - delta) * A_HUMAN / B_HUMAN * (
        np.exp(B_HUMAN * np.maximum(T, a0)) - np.exp(B_HUMAN * min(a0, 130)))
    return np.exp(-np.where(T < a0, H0, late))


def e0(delta=0.0, a0=0.0):
    s = survival(delta, a0)
    dt = T[1] - T[0]
    return float((s.sum() - 0.5 * (s[0] + s[-1])) * dt)


def main() -> None:
    set_seed(0)
    set_style()
    base = e0()

    deltas = np.linspace(0.05, 0.5, 10)
    ages = np.arange(20, 81, 5)
    G = np.zeros((len(ages), len(deltas)))
    for i, a0 in enumerate(ages):
        for j, d in enumerate(deltas):
            G[i, j] = e0(d, a0) - base

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    im = ax.imshow(G, origin="lower", aspect="auto", cmap=SEQUENTIAL,
                   extent=[deltas[0], deltas[-1], ages[0], ages[-1]])
    ax.set_xlabel("mortality reduction $\\delta$")
    ax.set_ylabel("age at which it starts")
    ax.set_title("(a)  The same drug, different timing", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("gain in life expectancy (years)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    ax2 = axes[1]
    d = 0.2
    tt = np.linspace(20, 80, 200)
    gains = np.array([e0(d, a) - base for a in tt])
    ax2.plot(tt, gains, color=PAL["blue"])
    for a in (30, 50, 70):
        g = e0(d, a) - base
        ax2.plot(a, g, "o", color=PAL["orange"], ms=5, mec="white")
        ax2.annotate(f"{a} yr: +{g:.1f}", (a, g), xytext=(6, -10),
                     textcoords="offset points", fontsize=6.4)
    ax2.set_xlabel("age at intervention start")
    ax2.set_ylabel("life-expectancy gain (years)")
    ax2.set_title("(b)  Late starts buy little", loc="left")
    ax2.grid(True)

    fig.text(0.01, -0.06,
             "Gompertz life table with A = 5e-5 /yr and B = 0.085 /yr; a permanent 20% "
             "mortality reduction in panel (b). This is arithmetic, not a therapy model.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-timing")

    save_metrics("timing", {
        "baseline_e0": base,
        "delta": 0.2,
        "gain_start_30": e0(0.2, 30) - base,
        "gain_start_50": e0(0.2, 50) - base,
        "gain_start_70": e0(0.2, 70) - base,
        "gain_start_20_delta_0_5": e0(0.5, 20) - base,
        "gain_start_80_delta_0_5": e0(0.5, 80) - base,
    })


if __name__ == "__main__":
    main()
