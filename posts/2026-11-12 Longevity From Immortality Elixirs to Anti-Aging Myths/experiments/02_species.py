"""Study 2 — how much of lifespan diversity does body size explain?

The species table is a small, hand-curated subset of the AnAge / HAGR
longevity records (Tacutu et al., 2018) plus a few classic outliers.  Values
are maximum reported lifespans in captivity or in the wild, rounded; they are
not a systematic sample.  The point of the panel is the residual spread: at a
fixed body mass, lifespan varies by two orders of magnitude, so allometry is a
weak explanation and the interesting variation is in maintenance strategy.

Run:  python experiments/02_species.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

# name, adult mass (kg), maximum lifespan (years), taxon
SPECIES = [
    ("C. elegans", 2e-8, 0.05, "invertebrate"),
    ("Drosophila", 1e-6, 0.25, "invertebrate"),
    ("Hydra", 1e-6, 4.0, "invertebrate"),
    ("killifish", 2e-3, 0.75, "fish"),
    ("mouse", 0.03, 4.0, "mammal"),
    ("naked mole rat", 0.05, 37.0, "mammal"),
    ("rat", 0.4, 4.0, "mammal"),
    ("pigeon", 0.35, 30.0, "bird"),
    ("dog", 20.0, 20.0, "mammal"),
    ("macaque", 8.0, 40.0, "mammal"),
    ("chimpanzee", 45.0, 60.0, "mammal"),
    ("human", 65.0, 122.0, "mammal"),
    ("elephant", 4000.0, 65.0, "mammal"),
    ("bowhead whale", 75000.0, 211.0, "mammal"),
    ("Greenland shark", 900.0, 392.0, "fish"),
    ("ocean quahog", 0.1, 507.0, "invertebrate"),
    ("rockfish", 2.0, 205.0, "fish"),
]

COL = {"mammal": PAL["blue"], "bird": PAL["green"], "fish": PAL["sky"],
       "invertebrate": PAL["orange"]}

LABEL = {"naked mole rat", "human", "mouse", "bowhead whale",
         "ocean quahog", "Greenland shark", "Hydra", "killifish"}


def gompertz(x, A, B):
    return A * np.exp(B * x)


def main() -> None:
    set_seed(0)
    set_style()

    mass = np.array([s[1] for s in SPECIES])
    life = np.array([s[2] for s in SPECIES])
    tax = [s[3] for s in SPECIES]

    x = np.log10(mass)
    y = np.log10(life)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    resid_sd = np.std(y - yhat, ddof=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    for t in COL:
        sel = np.array([s == t for s in tax])
        ax.scatter(mass[sel], life[sel], s=34, color=COL[t], label=t,
                   edgecolor="white", linewidth=0.6, zorder=3)
    xx = np.logspace(-8, 5.2, 100)
    ax.plot(xx, 10 ** (a + b * np.log10(xx)), color=PAL["grey"], lw=1.2,
            ls="--", label=f"allometric fit (R² = {r2:.2f})")
    for (name, m, l, t) in SPECIES:
        if name in LABEL:
            ax.annotate(name, (m, l), xytext=(4, 4), textcoords="offset points",
                        fontsize=6.2, color=PAL["dark"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("adult body mass (kg)")
    ax.set_ylabel("maximum reported lifespan (years)")
    ax.set_title("(a)  Allometry explains two thirds; the residual is still 20-fold",
                 loc="left")
    ax.text(0.03, 0.06, f"R² = {r2:.2f}\nresidual ≈ {10 ** (2 * resid_sd):.0f}× "
            "at fixed mass", transform=ax.transAxes, fontsize=7,
            color=PAL["dark"], va="bottom")
    ax.legend(loc="upper left", frameon=False, fontsize=6.6)
    ax.grid(True, which="major")

    ax2 = axes[1]
    age = np.linspace(0, 100, 400)
    curves = [
        ("mouse (B ≈ 1.2 /yr)", 0.02, 1.2, PAL["orange"]),
        ("human (B ≈ 0.085 /yr)", 2e-4, 0.085, PAL["blue"]),
        ("naked mole rat (flat)", 0.01, 0.004, PAL["green"]),
    ]
    for name, A, B, c in curves:
        h = gompertz(age, A, B)
        ax2.plot(age, h, color=c, label=name)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 50)
    ax2.set_xlim(0, 90)
    ax2.set_xlabel("age (years)")
    ax2.set_ylabel("mortality hazard (1/yr, illustrative)")
    ax2.set_title("(b)  Three ways to grow old", loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=6.6)
    ax2.grid(True, which="major")

    fig.text(0.01, -0.04,
             "Species values are rounded maxima from AnAge/HAGR and the primary "
             "literature; panel (b) uses illustrative Gompertz parameters, not fits. "
             "The naked-mole-rat curve is drawn flat to represent negligible senescence.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-species")

    save_metrics("species", {
        "n_species": len(SPECIES),
        "slope_log10_life_per_log10_mass": float(b),
        "r2_allometry": float(r2),
        "residual_sd_dex": float(resid_sd),
        "fold_scatter_at_fixed_mass": float(10 ** (2 * resid_sd)),
        "mass_range_kg": [float(mass.min()), float(mass.max())],
        "lifespan_range_y": [float(life.min()), float(life.max())],
    })


if __name__ == "__main__":
    main()
