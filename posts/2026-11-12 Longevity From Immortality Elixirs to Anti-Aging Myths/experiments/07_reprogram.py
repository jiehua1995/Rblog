"""Study 7 — partial reprogramming in a two-dimensional state space.

The state is (a, d): a is an age coordinate (0 young, 1 old) and d is a
differentiation coordinate (0 pluripotent, 1 differentiated).  A smooth
potential has three wells: young-differentiated, old-differentiated, and
pluripotent.  Reprogramming is a transient force that pushes both coordinates
down; if it lasts too long, cells fall into the pluripotent well.

The model is a cartoon of a real tension (rejuvenation versus identity loss),
not a fit to any experiment.  What it contributes is a *safety window*: a
region of dose and duration where cells move toward the young well without
crossing the identity boundary.

Run:  python experiments/07_reprogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, save, set_style
from common.utils import save_metrics, set_seed

WELLS = [(0.15, 0.88, 1.00), (0.90, 0.88, 0.95), (0.22, 0.12, 1.15)]
SIG_A, SIG_D = 0.24, 0.20
BACKGROUND = 0.10
SAFE_D = 0.45
DT = 0.02
TREAT_END_MAX = 9.0
TOTAL = 18.0


def potential(a, d):
    V = BACKGROUND * ((a - 0.5) ** 2 + (d - 0.5) ** 2)
    for ca, cd, w in WELLS:
        V = V - w * np.exp(-((a - ca) ** 2 / (2 * SIG_A ** 2)
                             + (d - cd) ** 2 / (2 * SIG_D ** 2)))
    # a ridge that separates the young and pluripotent wells: crossing it is
    # what "losing identity" means in this model
    V = V + 0.55 * np.exp(-((d - 0.50) ** 2 / (2 * 0.07 ** 2)
                            + (a - 0.18) ** 2 / (2 * 0.22 ** 2)))
    return V


def grad(a, d, eps=1e-4):
    Va = (potential(a + eps, d) - potential(a - eps, d)) / (2 * eps)
    Vd = (potential(a, d + eps) - potential(a, d - eps)) / (2 * eps)
    return Va, Vd


def simulate(k: float, tau: float, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = 0.90 + rng.normal(0, 0.06, n)
    d = 0.88 + rng.normal(0, 0.06, n)
    t = 0.0
    while t < TOTAL:
        treat = 1.0 if t < tau else 0.0
        Va, Vd = grad(a, d)
        a = a - DT * Va - DT * k * treat + np.sqrt(DT) * 0.035 * rng.normal(size=n)
        d = d - DT * Vd - DT * 0.14 * k * treat + np.sqrt(DT) * 0.035 * rng.normal(size=n)
        a = np.clip(a, -0.3, 1.4)
        d = np.clip(d, -0.3, 1.4)
        t += DT
    return a, d


def main() -> None:
    set_seed(0)
    set_style()

    fig = plt.figure(figsize=(12.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.0], wspace=0.34)
    ax = fig.add_subplot(gs[0])

    aa, dd = np.meshgrid(np.linspace(-0.25, 1.35, 220),
                         np.linspace(-0.25, 1.35, 220))
    V = potential(aa, dd)
    ax.contourf(aa, dd, V, levels=26, cmap=SEQUENTIAL, alpha=0.85)
    ax.contour(aa, dd, V, levels=10, colors="white", linewidths=0.35, alpha=0.6)
    ga, gd = grad(aa, dd)
    ax.quiver(aa[::14, ::14], dd[::14, ::14], -ga[::14, ::14], -gd[::14, ::14],
              color=PAL["dark"], alpha=0.35, scale=260, width=0.002)
    for label, (ca, cd, _), c in zip(["young", "old", "pluripotent"],
                                     WELLS,
                                     [PAL["green"], PAL["vermillion"], PAL["purple"]]):
        ax.plot(ca, cd, "o", color=c, ms=6, mec="white", mew=0.8)
        ax.text(ca, cd + 0.07, label, ha="center", fontsize=6.8, color=c)
    ax.axhline(SAFE_D, color=PAL["dark"], ls="--", lw=0.7)
    ax.text(1.22, SAFE_D + 0.02, "identity boundary", fontsize=6.0,
            ha="right", color=PAL["dark"])
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.25, 1.35)
    ax.set_xlabel("age coordinate $a$ (0 young, 1 old)")
    ax.set_ylabel("differentiation $d$")
    ax.set_title("(a)  A landscape with three exits", loc="left")

    # dose-duration scan
    doses = np.linspace(3.0, 6.5, 15)
    taus = np.linspace(0.5, 9.0, 10)
    unsafe = np.zeros((len(taus), len(doses)))
    gain = np.zeros_like(unsafe)
    for i, tau in enumerate(taus):
        for j, k in enumerate(doses):
            a, d = simulate(k, tau, n=300, seed=j + 17 * i)
            unsafe[i, j] = np.mean(d < SAFE_D)
            gain[i, j] = 0.90 - np.mean(a[d >= SAFE_D]) if np.any(d >= SAFE_D) else 0.0

    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(unsafe, origin="lower", aspect="auto", cmap=SEQUENTIAL,
                    extent=[doses[0], doses[-1], taus[0], taus[-1]], vmin=0, vmax=1)
    cs = ax2.contour(doses, taus, gain, levels=[0.5, 0.7], colors=[PAL["orange"],
                                                               PAL["vermillion"]],
                     linewidths=1.0, linestyles="--")
    ax2.clabel(cs, fmt={0.5: "Δa 0.5", 0.7: "Δa 0.7"}, fontsize=6)
    ax2.set_xlabel("reprogramming dose $k$")
    ax2.set_ylabel("duration $\\tau$")
    ax2.set_title("(b)  Unsafe fraction and efficacy contours", loc="left")
    cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    cb.set_label("P(identity lost)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    ax3 = fig.add_subplot(gs[2])
    safe = 1 - unsafe
    ax3.scatter(gain.ravel(), safe.ravel(), s=22,
                c=np.repeat(taus, len(doses)), cmap=SEQUENTIAL,
                edgecolor="white", linewidth=0.4)
    ax3.axhline(0.95, color=PAL["grey"], ls=":", lw=0.8)
    ax3.axvline(0.4, color=PAL["grey"], ls=":", lw=0.8)
    ax3.set_xlabel("mean age reduction in safe cells")
    ax3.set_ylabel("fraction of cells keeping identity")
    ax3.set_title("(c)  Efficacy and safety trade off", loc="left")
    ax3.set_xlim(-0.02, 0.85)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL,
                               norm=plt.Normalize(taus.min(), taus.max()))
    cb2 = fig.colorbar(sm, ax=ax3, fraction=0.045, pad=0.02)
    cb2.set_label("duration $\\tau$", fontsize=6.5)
    cb2.ax.tick_params(labelsize=6)

    fig.text(0.01, -0.07,
             "Two-dimensional toy potential (three wells) with Langevin noise, 300 cells "
             "per condition, 18 time units. The safety window is a property of this "
             "model, not an estimate for any real protocol.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-reprogram")

    score = np.where(unsafe < 0.05, gain, -1)
    kbest, taubest = None, None
    if np.max(score) > 0:
        i, j = np.unravel_index(np.argmax(score), score.shape)
        kbest, taubest = float(doses[j]), float(taus[i])

    save_metrics("reprogram", {
        "unsafe_at_max_dose_long": float(unsafe[-1, -1]),
        "unsafe_at_low_dose_short": float(unsafe[0, 0]),
        "max_safe_age_reduction": float(np.max(score)),
        "safe_dose": kbest,
        "safe_duration": taubest,
        "fraction_conditions_safe": float(np.mean(unsafe < 0.05)),
        "safe_identity_boundary": SAFE_D,
    })


if __name__ == "__main__":
    main()
