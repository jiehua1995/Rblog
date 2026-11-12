"""Study 10 — where quantum mechanics enters, and where it does not.

Three scale arguments, all computed rather than asserted:

  * the thermal energy kT at 300 K compared with hydrogen bonds, ATP, photons
    and covalent bonds;
  * WKB proton versus deuterium tunnelling through a 5 kcal/mol barrier;
  * a collisional decoherence estimate for a molecular superposition in water,
    tau ~ (lambda_th / dx)^2 / nu_coll.

The conclusion drawn in the article is deliberately narrow: quantum mechanics
sets the parameters (energies, tunnelling rates, spin selection rules) of the
classical stochastic model, but coherent superpositions do not survive at the
scale of a cell.

Run:  python experiments/10_quantum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

KB = 1.380649e-23
H = 6.62607015e-34
HBAR = H / (2 * np.pi)
EV = 1.602176634e-19
KCAL = 4184.0
T_ROOM = 300.0


def main() -> None:
    set_seed(0)
    set_style()
    kT = KB * T_ROOM / EV
    kT_kcal = kT * 23.0605

    scales = [
        ("$kT$ (300 K)", kT, PAL["grey"]),
        ("hydrogen bond", 0.12, PAL["sky"]),
        ("ATP hydrolysis", 0.32, PAL["green"]),
        ("visible photon", 2.0, PAL["orange"]),
        ("C-C bond", 3.6, PAL["vermillion"]),
        ("ionisation", 10.0, PAL["purple"]),
    ]

    # ---- tunnelling --------------------------------------------------------
    barrier = 5.0 / 23.0605                 # 5 kcal/mol in eV
    me = 9.109e-31
    mp = 1.6726e-27
    widths = np.linspace(0.15, 1.4, 200) * 1e-10
    kappa_h = np.sqrt(2 * mp * barrier * EV) / HBAR
    kappa_d = np.sqrt(2 * 2 * mp * barrier * EV) / HBAR
    t_h = np.exp(-2 * kappa_h * widths)
    t_d = np.exp(-2 * kappa_d * widths)

    # ---- decoherence -------------------------------------------------------
    m_water = 18 * 1.66e-27
    dx = np.logspace(-11, -8.3, 200)
    temps = [(300.0, PAL["vermillion"]), (4.0, PAL["blue"])]
    tau = {}
    for T, _ in temps:
        lam = H / np.sqrt(2 * np.pi * m_water * KB * T)
        nu = 1e13 * np.sqrt(T / 300.0)
        tau[T] = (lam / dx) ** 2 / nu

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8))
    ax = axes[0, 0]
    for i, (name, e, c) in enumerate(scales):
        ax.plot([1e-3, e], [i, i], color=c, lw=2.4)
        ax.plot([e], [i], "o", color=c, ms=5)
        ax.text(e * 1.25, i, f"{e:.3g} eV", fontsize=6.6, color=c, va="center")
    ax.set_yticks(range(len(scales)), [s[0] for s in scales], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 30)
    ax.set_xlabel("energy (eV)")
    ax.set_title("(a) the energy budget of a cell")

    ax = axes[0, 1]
    ax.semilogy(widths * 1e10, t_h, color=PAL["blue"], label="proton")
    ax.semilogy(widths * 1e10, t_d, color=PAL["orange"], label="deuteron")
    ax.set_xlabel("barrier width (Å)")
    ax.set_ylabel("WKB transmission")
    ax.set_title("(b) tunnelling through 5 kcal/mol")
    ax.legend(fontsize=6.8, frameon=False)
    ax.set_ylim(1e-12, 1)
    ax.axhline(1e-3, color=PAL["grey"], lw=0.6, ls=":")

    ax = axes[1, 0]
    for T, c in temps:
        ax.loglog(dx * 1e9, tau[T], color=c, label=f"{T:g} K")
    ax.set_xlabel("superposition size $\\Delta x$ (nm)")
    ax.set_ylabel("decoherence time (s)")
    ax.set_title("(c) coherence dies before it travels")
    ax.legend(fontsize=6.8, frameon=False)
    for y, lab in [(1e-15, "fs"), (1e-12, "ps"), (1e-9, "ns")]:
        ax.axhline(y, color=PAL["grey"], lw=0.5, ls=":")
        ax.text(2e-2, y * 1.5, lab, fontsize=6, color=PAL["grey"])

    ax = axes[1, 1]
    gaps = np.linspace(0.02, 3.0, 200)
    ax.semilogy(gaps, np.exp(-gaps / kT), color=PAL["vermillion"])
    ax.set_xlabel("electronic gap (eV)")
    ax.set_ylabel("thermal population of the excited state")
    ax.set_title("(d) 300 K cannot excite electrons")
    ax.set_ylim(1e-40, 1)
    ax.axvline(2.0, color=PAL["grey"], lw=0.6, ls=":")
    ax.text(2.05, 1e-32, "2 eV photon", fontsize=6.4, color=PAL["grey"])
    fig.tight_layout()
    save(fig, "fig-quantum")

    save_metrics("quantum", {
        "kT_eV": round(kT, 5),
        "kT_kcal": round(kT_kcal, 4),
        "atp_over_kT": round(0.32 / kT, 2),
        "barrier_eV": round(barrier, 4),
        "tunnelling_at_0.5A": round(float(np.interp(0.5, widths * 1e10, t_h)), 6),
        "tunnelling_D_at_0.5A": round(float(np.interp(0.5, widths * 1e10, t_d)), 6),
        "tau_300K_at_0.1nm": round(float(np.interp(0.1e-9, dx, tau[300.0])), 18),
        "tau_4K_at_0.1nm": round(float(np.interp(0.1e-9, dx, tau[4.0])), 18),
        "boltzmann_2eV": float(np.exp(-2.0 / kT)),
    })


if __name__ == "__main__":
    main()
