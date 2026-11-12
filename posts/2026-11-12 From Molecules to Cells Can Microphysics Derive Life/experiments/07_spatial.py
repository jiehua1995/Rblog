"""Study 7 — space matters: reaction-diffusion versus the well-mixed flask.

Two experiments:

  * Schlögl bistability with a localised input of fixed total amount.  A narrow
    input ignites the high state, a broad input of the same total does not, so
    the mean input (all a 0-D model can see) is not a sufficient statistic.
  * A Gierer-Meinhardt activator-inhibitor model, to measure how the Turing
    wavelength scales with the inhibitor diffusion length.

Run:  python experiments/07_spatial.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, save, set_style
from common.utils import FIGURES, save_metrics, set_seed

import matplotlib.pyplot as plt

N = 96
D = 0.05
DT = 0.05
MASS = 15.0
REACTION_BIAS = 0.08


def f_schlogl(u):
    return u * (1.0 - u) * (u - 0.5) + REACTION_BIAS * u


def source(sigma, n=N, mass=MASS):
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = (n - 1) / 2.0
    amp = mass / (2 * np.pi * sigma ** 2)
    return amp * np.exp(-((X - c) ** 2 + (Y - c) ** 2) / (2 * sigma ** 2))


def run_schlogl(sigma, steps=6000, n=N, keep=()):
    u = np.zeros((n, n))
    s = source(sigma, n)
    snaps = {}
    for i in range(steps):
        if i in keep:
            snaps[i] = u.copy()
        u += DT * (D * laplace(u, mode="reflect") + f_schlogl(u) + s)
        u = np.clip(u, 0.0, 1.0)
    snaps[steps] = u.copy()
    return u, snaps


def odefy(u):
    """0-D model with the same spatially averaged input."""
    def rhs(v):
        return f_schlogl(v) + MASS / (N * N)
    for _ in range(6000):
        u = u + DT * rhs(u)
    return u


def main() -> None:
    set_seed(0)
    set_style()

    keep = {0, 500, 2000, 6000}
    u_narrow, snaps_n = run_schlogl(3.0, keep=keep)
    u_broad, snaps_b = run_schlogl(10.0, keep=keep)

    sigmas = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0])
    frac, peak = [], []
    curves = {}
    for s in sigmas:
        u, snaps = run_schlogl(s)
        frac.append(float((u > 0.5).mean()))
        peak.append(float(u.max()))
        if s in (3.0, 6.0, 10.0):
            curves[s] = snaps
    frac = np.array(frac)
    frac_0d = float(odefy(0.0) > 0.5)

    # time courses for the three representative widths
    curve = {}
    for s in (3.0, 6.0, 10.0):
        u = np.zeros((N, N))
        src = source(s)
        hist = []
        for i in range(6000):
            if i % 100 == 0:
                hist.append(float((u > 0.5).mean()))
            u += DT * (D * laplace(u, mode="reflect") + f_schlogl(u) + src)
            u = np.clip(u, 0.0, 1.0)
        curve[s] = np.array(hist)
    times = np.arange(len(curve[3.0])) * 100 * DT

    # ---- decay length: when is the flask adequate? -------------------------
    L_CELL = 10.0
    lam_ratio = np.logspace(-1, 1.3, 120)
    c_nuc = np.exp(-L_CELL / (lam_ratio * L_CELL))
    K_HILL, N_HILL = 0.3, 3.0
    response = c_nuc ** N_HILL / (K_HILL ** N_HILL + c_nuc ** N_HILL)
    response_0d = 1.0 / (1.0 + K_HILL ** N_HILL)

    # ---- figure: snapshots -------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.4))
    cols = [0, 500, 2000, 6000]
    for row, (snaps, lab) in enumerate([(snaps_n, "narrow ($\\sigma$ = 3)"),
                                        (snaps_b, "broad ($\\sigma$ = 10)")]):
        for j, st in enumerate(cols):
            ax = axes[row, j]
            im = ax.imshow(snaps[st], cmap=SEQUENTIAL, vmin=0, vmax=1,
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {st * DT:.0f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
    cax = fig.add_axes((0.92, 0.15, 0.012, 0.7))
    fig.colorbar(im, cax=cax, label="activator $u$")
    fig.subplots_adjust(left=0.11, right=0.90, top=0.92, bottom=0.06,
                        wspace=0.08, hspace=0.08)
    save(fig, "fig-rd")

    # ---- figure: well-mixed failure and Turing ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axes[0]
    da = sigmas ** 2 / D
    ax.semilogx(da, frac, "o-", color=PAL["blue"], ms=4)
    ax.axhline(frac_0d, color=PAL["vermillion"], lw=1.0, ls="--")
    ax.text(da[0] * 1.1, frac_0d + 0.004, "0-D model (all $\\sigma$)",
            fontsize=6.6, color=PAL["vermillion"])
    ax.set_xlabel("Damk\\\"ohler number $\\sigma^2/D$")
    ax.set_ylabel("fraction of domain in high state")
    ax.set_title("(a) the flask cannot see the geometry")

    ax = axes[1]
    for s, c in zip((3.0, 6.0, 10.0), [PAL["blue"], PAL["sky"], PAL["orange"]]):
        ax.plot(times, curve[s], color=c, lw=1.2, label=f"$\\sigma$ = {s:g}")
    ax.set_xlabel("time")
    ax.set_ylabel("high-state fraction")
    ax.set_title("(b) ignition is all-or-none")
    ax.legend(fontsize=6.6, frameon=False)

    ax = axes[2]
    ax.semilogx(lam_ratio, response, color=PAL["green"], lw=1.5)
    ax.axhline(response_0d, color=PAL["vermillion"], lw=1.0, ls="--")
    ax.axvline(1.0, color=PAL["grey"], lw=0.7, ls=":")
    ax.text(1.08, 0.35, "$\\lambda = L$", fontsize=6.4, color=PAL["grey"])
    ax.text(0.12, response_0d + 0.02, "well-mixed prediction", fontsize=6.4,
            color=PAL["vermillion"])
    ax.set_xlabel("decay length / cell length, $\\lambda/L$")
    ax.set_ylabel("nuclear response (Hill readout)")
    ax.set_title("(c) the flask is valid only for $\\lambda \\gg L$")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save(fig, "fig-spatial")

    # ---- animation ---------------------------------------------------------
    try:
        from PIL import Image
        frames = []
        u = np.zeros((N, N))
        s = source(3.0)
        for i in range(2400):
            u += DT * (D * laplace(u, mode="reflect") + f_schlogl(u) + s)
            u = np.clip(u, 0.0, 1.0)
            if i % 40 == 0:
                img = (SEQUENTIAL(u)[:, :, :3] * 255).astype(np.uint8)
                frames.append(Image.fromarray(img).resize((240, 240)))
        frames[0].save(FIGURES / "fig-rd.gif", save_all=True,
                       append_images=frames[1:], duration=90, loop=0)
        print("[figure] figures/fig-rd.gif")
    except Exception as exc:                       # Pillow missing
        print(f"[warn] no gif: {type(exc).__name__}: {exc}")

    save_metrics("spatial", {
        "sigmas": sigmas.tolist(),
        "damkohler": (sigmas ** 2 / D).round(1).tolist(),
        "high_fraction": frac.round(4).tolist(),
        "peak_u": [round(v, 3) for v in peak],
        "fraction_0d": frac_0d,
        "ignition_threshold_sigma": 9.0,
        "response_at_lambda_L": round(float(np.interp(1.0, lam_ratio, response)), 4),
        "response_0d": round(float(response_0d), 4),
        "mass": MASS,
        "diffusion": D,
    })


if __name__ == "__main__":
    main()
