"""Study 9 — survival, competing risks, and why a hazard floor forbids infinity.

Death is modelled as the first failure among eight organ systems, each with a
Gompertz hazard mu_i(t) = A_i exp(B t).  Because the failures are independent,
the population survives with S(t) = exp(-sum_i H_i(t)), where H_i is the
integrated organ hazard.  Several strategies are compared:

  baseline                 all eight organs as they are
  slower aging             B halved for every organ
  aging halted at 60       hazard frozen at its age-60 value
  four organs repaired     A and B reduced in four of eight
  all repaired, floor      each hazard scaled by 0.3 plus a floor h0

The last strategy is the interesting one: while the hazard has a positive
floor, S(t) decays exponentially and the mean lifetime is finite.  The tail
panel shows the floor needed for a 1 % chance of reaching 200.

Run:  python experiments/09_survival.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

N_ORGANS = 8
A_TOTAL = 5e-5
B_HUMAN = 0.085
A_ORG = A_TOTAL / N_ORGANS
AGE = np.linspace(0, 200, 4001)


def hazard(t, A, B):
    return A * np.exp(B * t)


def cum_hazard(t, A, B):
    return A / B * (np.exp(B * t) - 1)


def survival(t, A, B):
    return np.exp(-cum_hazard(t, A, B))


def strategy_curves():
    out = {}
    out["baseline"] = survival(AGE, A_TOTAL, B_HUMAN)

    out["slower aging"] = survival(AGE, A_TOTAL, B_HUMAN / 2)

    h60 = hazard(60, A_TOTAL, B_HUMAN)
    H = np.where(AGE <= 60, cum_hazard(AGE, A_TOTAL, B_HUMAN),
                 cum_hazard(np.minimum(AGE, 60), A_TOTAL, B_HUMAN)
                 + h60 * np.clip(AGE - 60, 0, None))
    out["aging halted at 60"] = np.exp(-H)

    # four organs repaired: 4 at 0.1x, 4 at 1x
    H4 = 4 * cum_hazard(AGE, 0.1 * A_ORG, B_HUMAN / 2) + 4 * cum_hazard(AGE, A_ORG, B_HUMAN)
    out["four organs repaired"] = np.exp(-H4)

    floor = 2e-4
    Hall = N_ORGANS * (0.1 * cum_hazard(AGE, A_ORG, B_HUMAN) + floor * AGE)
    out["all repaired, floor"] = np.exp(-Hall)
    return out


def median_time(S, level=0.5):
    idx = np.where(S <= level)[0]
    return float(AGE[idx[0]]) if len(idx) else float("nan")


def main() -> None:
    set_seed(0)
    set_style()

    curves = strategy_curves()
    colors = [PAL["grey"], PAL["green"], PAL["blue"], PAL["orange"],
              PAL["vermillion"]]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axes[0]
    for (name, S), c in zip(curves.items(), colors):
        ax.plot(AGE, S, color=c, label=name)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.05)
    ax.set_xlim(0, 160)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("survival S(t)")
    ax.set_title("(a)  Five ways to die", loc="left")
    ax.legend(frameon=False, fontsize=6.0, loc="lower left")
    ax.grid(True, which="major")

    ax2 = axes[1]
    causes = [("cancer", 0.35, PAL["vermillion"]), ("cardiovascular", 0.30, PAL["blue"]),
              ("infection", 0.20, PAL["orange"]), ("neurodegeneration", 0.15, PAL["purple"])]
    bottom = np.zeros_like(AGE)
    for name, frac, c in causes:
        h = hazard(AGE, A_TOTAL * frac, B_HUMAN)
        ax2.plot(AGE, h, color=c, label=name)
        bottom = bottom + h
    ax2.plot(AGE, bottom, color=PAL["dark"], ls="--", lw=1.0, label="total")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 2)
    ax2.set_xlim(20, 110)
    ax2.set_xlabel("age (years)")
    ax2.set_ylabel("cause-specific hazard (1/yr)")
    ax2.set_title("(b)  Competing risks share one curve", loc="left")
    ax2.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax2.grid(True, which="major")

    ax3 = axes[2]
    floors = np.logspace(-4, -2, 120)
    for t_target, c in [(150, PAL["blue"]), (200, PAL["vermillion"])]:
        p = np.exp(-floors * t_target)
        ax3.plot(floors, p, color=c, label=f"survive to {t_target}")
    ax3.axhline(0.01, color=PAL["grey"], ls=":", lw=0.8)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("residual hazard floor $h_0$ (1/yr)")
    ax3.set_ylabel("probability")
    ax3.set_title("(c)  A floor buys a finite tail", loc="left")
    ax3.legend(frameon=False, fontsize=6.4)
    ax3.grid(True, which="major")

    fig.text(0.01, -0.07,
             "Eight independent Gompertz organ hazards; S(t) = exp(-sum H_i). "
             "Panel (c) isolates the floor: P(T > t) = exp(-h0 t) once all other "
             "hazards are negligible.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-survival")

    x150 = float(-np.log(0.01) / 200)
    save_metrics("survival", {
        "median_baseline": median_time(curves["baseline"]),
        "median_slower_aging": median_time(curves["slower aging"]),
        "median_halted": median_time(curves["aging halted at 60"]),
        "median_four_organs": median_time(curves["four organs repaired"]),
        "median_all_repaired_floor": median_time(curves["all repaired, floor"]),
        "p150_baseline": float(np.interp(150, AGE, curves["baseline"])),
        "p150_four_organs": float(np.interp(150, AGE, curves["four organs repaired"])),
        "p200_all_repaired_floor": float(np.interp(200, AGE, curves["all repaired, floor"])),
        "healthspan_baseline": median_time(curves["baseline"], 0.7),
        "healthspan_four_organs": median_time(curves["four organs repaired"], 0.7),
        "healthspan_slower_aging": median_time(curves["slower aging"], 0.7),
        "floor_for_1pct_at_200": float(-np.log(0.01) / 200),
        "floor_grid_1pct": float(x150),
        "hazard_floor_policy": 2e-4,
    })


if __name__ == "__main__":
    main()
