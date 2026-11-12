"""Study 8 — regeneration, mutations and the cancer trade-off.

A multistage (Armitage-Doll) model: a tissue becomes malignant after m rate-
limiting events, each occurring at rate proportional to the stem-cell division
rate.  If regeneration multiplies the division rate by r, the cancer hazard
scales as r^m — the central, uncomfortable arithmetic of regenerative medicine.

The model is calibrated so that lifetime cancer risk at r = 1 is about 40 %,
a round number for a mixed set of epithelial cancers, and the other-cause
mortality is a Gompertz curve.  Monte Carlo checks the analytic incidence.
The numbers are illustrative: this is a mechanism, not an epidemiological
estimate for any therapy.

Run:  python experiments/08_cancer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

M_STAGES = 6
T_MAX = 100.0
TARGET_RISK = 0.40
GOM_A = 1e-4          # baseline mortality 1/yr
GOM_B = 0.09          # Gompertz slope 1/yr


def k0_from_risk(m=M_STAGES, target=TARGET_RISK, horizon=85.0):
    """Choose k0 so that 1 - exp(-H(horizon)) equals the target risk."""
    return -np.log(1 - target) * m / horizon ** m


K0 = k0_from_risk()


def cancer_time(r, m=M_STAGES, u=None):
    """Inverse-transform a Weibull-like cumulative hazard H(t)=k0 r^m t^m / m."""
    if u is None:
        u = np.random.default_rng(0).random()
    e = -np.log(1 - u)          # H(T) is unit exponential
    return (e * m / (K0 * r ** m)) ** (1 / m)


def other_time(u):
    """Inverse Gompertz survival, S(t)=exp(-A/B (e^{Bt}-1))."""
    return np.log(1 - GOM_B * np.log(u) / GOM_A) / GOM_B


def simulate(r, n=20000, seed=0, m=M_STAGES):
    rng = np.random.default_rng(seed)
    tc = cancer_time(r, m, rng.random(n))
    to = other_time(rng.random(n))
    return {
        "incidence_to_85": float(np.mean(tc < 85.0)),
        "cancer_death_by_100": float(np.mean(tc < to)),
        "mean_age_at_death": float(np.mean(np.minimum(tc, to))),
    }


def incidence_curve(r, m=M_STAGES, horizon=85.0):
    return 1 - np.exp(-K0 * r ** m * horizon ** m / m)


def benefit(r):
    """Tissue function gain from regeneration; saturating, heuristic."""
    return 1 - np.exp(-(r - 1) / 1.2)


def main() -> None:
    set_seed(0)
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ax = axes[0]
    rr = np.linspace(1.0, 2.0, 100)
    for m, c in zip([3, 4, 5, 6, 7],
                    [PAL["sky"], PAL["green"], PAL["orange"],
                     PAL["blue"], PAL["vermillion"]]):
        ax.plot(rr, incidence_curve(rr, m), color=c, label=f"m = {m}")
    ax.set_yscale("log")
    ax.set_xlabel("regeneration intensity $r$ (division-rate multiplier)")
    ax.set_ylabel("lifetime cancer incidence (to age 85)")
    ax.set_title("(a)  Each extra stage multiplies the penalty", loc="left")
    ax.legend(frameon=False, fontsize=6.6, ncol=2)
    ax.grid(True, which="major")

    ax2 = axes[1]
    rs = np.linspace(1.0, 2.0, 41)
    risk = np.array([incidence_curve(r, 6) for r in rs])
    ben = np.array([benefit(r) for r in rs])
    base = incidence_curve(1.0, 6)
    ratio = risk / base
    ax2.plot(ratio, ben, color=PAL["blue"], marker="o", ms=3)
    ok = ratio <= 2.0
    ax2.axvline(2.0, color=PAL["grey"], ls=":", lw=0.9)
    ax2.scatter(ratio[ok], ben[ok], color=PAL["green"], s=22, zorder=3,
                label="risk ≤ 2× baseline")
    ax2.scatter(ratio[~ok], ben[~ok], color=PAL["vermillion"], s=22, zorder=3,
                label="risk > 2× baseline")
    best = np.argmax(np.where(ok, ben, -1))
    ax2.plot(ratio[best], ben[best], "*", ms=13, color=PAL["orange"],
             mec="white", mew=0.6, zorder=4)
    ax2.set_xscale("log")
    ax2.set_xlim(0.95, 200)
    ax2.set_xlabel("cancer risk relative to baseline (log scale)")
    ax2.set_ylabel("tissue function gain (heuristic)")
    ax2.set_title("(b)  The feasible frontier is short", loc="left")
    ax2.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax2.grid(True)

    ax3 = axes[2]
    tgrid = np.linspace(0, 95, 400)
    for r, c in [(1.0, PAL["grey"]), (1.3, PAL["blue"]), (1.6, PAL["vermillion"])]:
        H = K0 * r ** 6 * tgrid ** 6 / 6
        ax3.plot(tgrid, 1 - np.exp(-H), color=c, label=f"$r$ = {r}")
    ax3.set_xlabel("age (years)")
    ax3.set_ylabel("cumulative cancer incidence")
    ax3.set_title("(c)  Timing, not only totals", loc="left")
    ax3.legend(frameon=False, fontsize=6.6)
    ax3.grid(True)

    fig.text(0.01, -0.07,
             "Multistage model with m = 6 rate-limiting events, calibrated to a 40 % "
             "lifetime risk at r = 1; other-cause mortality is Gompertz. The function "
             "gain is a heuristic curve, not a clinical endpoint.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-cancer")

    mc = {str(r): simulate(r, seed=int(100 * r)) for r in (1.0, 1.3, 1.6)}

    save_metrics("cancer", {
        "m_stages": M_STAGES,
        "k0": K0,
        "baseline_lifetime_risk": float(incidence_curve(1.0)),
        "risk_ratio_r1_3": float(incidence_curve(1.3) / incidence_curve(1.0)),
        "risk_ratio_r1_6": float(incidence_curve(1.6) / incidence_curve(1.0)),
        "risk_multiplier_per_doubling": 2.0 ** M_STAGES,
        "max_safe_r": float(rs[best]),
        "benefit_at_max_safe_r": float(ben[best]),
        "mc_lifetime_risk": mc,
        "gompertz_A": GOM_A,
        "gompertz_B": GOM_B,
    })


if __name__ == "__main__":
    main()
