"""Study 3 — why evolution does not build a body that lasts forever.

A one-sex Leslie matrix with age-dependent survival.  An individual carries a
heritable "maintenance" trait m in [0, 1] that slows the Gompertz rise of
intrinsic mortality,

    mu_int(a) = mu0 * exp(B(m) * a),        B(m) = B0 * (1 - m)

and costs fecundity,

    F(m) = F0 * (1 - c * m).

For each level of extrinsic mortality mu_e we find the m that maximises the
intrinsic growth rate r (the clonal ESS in this simple model) and then read off
the mortality curve and the life expectancy.  The model is a toy: the trade-off
is imposed by hand, and real maintenance has many dimensions.  What it shows is
only that the *direction* of the trade-off follows from demography, not from a
death programme.

Run:  python experiments/03_evolution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

OMEGA = 80          # maximum age class
ALPHA = 5           # age at first reproduction
MU0 = 2e-3          # baseline intrinsic mortality
B0 = 0.50           # senescence slope without maintenance
C_COST = 0.95       # fecundity cost at full maintenance (quadratic)
F0 = 0.075


def intrinsic_mu(age, m: float):
    """Gompertz hazard; maintenance slows the slope, starting at maturity."""
    return MU0 * np.exp(B0 * (1 - m) * np.clip(age - ALPHA, 0, None))


def survival(m: float, mu_e: float) -> np.ndarray:
    """Survival l(a) from birth to age a, for a = 0..OMEGA."""
    age = np.arange(OMEGA + 1)
    mu = mu_e + intrinsic_mu(age, m)
    l = np.ones(OMEGA + 1)
    l[1:] = np.cumprod(np.exp(-mu[:-1]))
    return l


def growth_rate(m: float, mu_e: float) -> float:
    age = np.arange(OMEGA)
    s = np.exp(-(mu_e + intrinsic_mu(age, m)))
    F = F0 * (1 - C_COST * m ** 2)
    A = np.zeros((OMEGA, OMEGA))
    for a in range(OMEGA - 1):
        A[a + 1, a] = s[a]
    A[0, ALPHA:] = F
    eig = np.linalg.eigvals(A)
    lam = float(np.max(eig.real))
    return np.log(lam) if lam > 0 else -np.inf


def life_expectancy(m: float, mu_e: float) -> float:
    l = survival(m, mu_e)
    return float(np.sum(l) - 0.5)


def main() -> None:
    set_seed(0)
    set_style()

    m_grid = np.linspace(0, 1, 81)
    mu_grid = np.logspace(-2.3, -0.1, 26)      # 0.005 .. 0.79 per year

    best_m = []
    for mu_e in mu_grid:
        r = np.array([growth_rate(m, mu_e) for m in m_grid])
        best_m.append(m_grid[int(np.argmax(r))])
    best_m = np.array(best_m)

    e_low = life_expectancy(best_m[0], mu_grid[0])
    e_high = life_expectancy(best_m[-1], mu_grid[-1])

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    ax.plot(mu_grid, best_m, color=PAL["blue"], marker="o", ms=2.6)
    ax.set_xscale("log")
    ax.set_xlabel("extrinsic mortality $\\mu_e$ (1/yr)")
    ax.set_ylabel("evolutionarily favoured maintenance $m^*$")
    ax.set_title("(a)  Safe worlds buy repair", loc="left")
    ax.grid(True, which="major")
    ax.annotate("mouse-like\n(high $\\mu_e$)", (mu_grid[-1], best_m[-1]),
                xytext=(-8, 18), textcoords="offset points", fontsize=6.6,
                ha="right", color=PAL["dark"])
    ax.annotate("human-like\n(low $\\mu_e$)", (mu_grid[0], best_m[0]),
                xytext=(10, -22), textcoords="offset points", fontsize=6.6,
                ha="left", color=PAL["dark"])

    ax2 = axes[1]
    show = [0.03, 0.15, 0.6]
    for mu_e, c in zip(show, [PAL["blue"], PAL["green"], PAL["vermillion"]]):
        r = np.array([growth_rate(m, mu_e) for m in m_grid])
        rel = r - r.max()
        ax2.plot(m_grid, rel, color=c, label=f"$\\mu_e$ = {mu_e:g} /yr")
        k = int(np.argmax(r))
        ax2.plot(m_grid[k], 0.0, "o", color=c, ms=5, mec="white")
    ax2.axhline(0, color=PAL["grey"], lw=0.6, ls=":")
    ax2.set_ylim(-0.06, 0.002)
    ax2.set_xlabel("maintenance investment $m$")
    ax2.set_ylabel("fitness relative to best strategy")
    ax2.set_title("(b)  The optimum moves with the environment", loc="left")
    ax2.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax2.grid(True)

    ax3 = axes[2]
    age = np.arange(OMEGA + 1)
    for mu_e, c, lab in [(0.03, PAL["blue"], "low $\\mu_e$"),
                         (0.6, PAL["vermillion"], "high $\\mu_e$")]:
        m = best_m[int(np.argmin(np.abs(mu_grid - mu_e)))]
        ax3.plot(age, intrinsic_mu(age, m), color=c,
                 label=f"{lab}, $m^*$ = {m:.2f}")
    ax3.plot(age, intrinsic_mu(age, 0.0), color=PAL["grey"], ls="--",
             label="no maintenance")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-3, 5)
    ax3.set_xlim(0, 70)
    ax3.set_xlabel("age (years)")
    ax3.set_ylabel("intrinsic hazard (1/yr)")
    ax3.set_title("(c)  Faster senescence is a choice, not a sentence", loc="left")
    ax3.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax3.grid(True, which="major")

    fig.text(0.01, -0.05,
             "Leslie matrix, $\\omega$ = 80, maturity at 5 yr, quadratic fecundity "
             "cost ($c$ = 0.95 at $m$ = 1). The trade-off is imposed by hand; "
             "the point is demographic, not biochemical.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-evolution")

    save_metrics("evolution", {
        "m_star_low_mu": float(best_m[0]),
        "m_star_high_mu": float(best_m[-1]),
        "mu_range": [float(mu_grid[0]), float(mu_grid[-1])],
        "senescence_slope_ratio": float((1 - best_m[-1]) / (1 - best_m[0])),
        "life_expectancy_low_mu": e_low,
        "life_expectancy_high_mu": e_high,
        "fecundity_cost": C_COST,
        "omega": OMEGA,
    })


if __name__ == "__main__":
    main()
