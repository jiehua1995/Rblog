"""Study 6 — how a free-energy error reaches the phenotype.

A binding free energy enters the equilibrium constant exponentially,

    K_d = exp(dG / RT),     RT = 0.592 kcal/mol at 298 K,

so an error of 1 kcal/mol is a factor e^1.69 ~ 5.4 in K_d.  That error is then
pushed through a three-tier Hill cascade and a fate threshold.  Two operating
regimes are compared: a saturated, buffered cascade, and a cascade tuned close
to its decision threshold.  A Monte Carlo and a Sobol decomposition quantify
the amplification in each.

Run:  python experiments/06_error.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

RT = 0.592                      # kcal/mol at 298 K
DG0 = -1.0

REGIMES = {
    "buffered": dict(K1=0.01, n1=6.0, n2=4.0, tau=0.10, gamma=2.0, w=0.05),
    "near threshold": dict(K1=0.60, n1=1.0, n2=2.0, tau=0.45, gamma=0.0, w=0.03),
}


def cascade(dg, K1, n1, n2, tau, gamma, w, ligand=1.0):
    """Return (theta, x1, x2, x3, phenotype probability)."""
    Kd = np.exp(dg / RT)
    theta = ligand / (Kd + ligand)
    x1 = theta ** n1 / (K1 ** n1 + theta ** n1)
    x2 = x1 ** n2 / (0.5 ** n2 + x1 ** n2)
    x3 = x2 / (1.0 + gamma * x2)
    p = 1.0 / (1.0 + np.exp(-(x3 - tau) / w))
    return theta, x1, x2, x3, p


def phenotype(dg, **kw):
    return cascade(dg, **kw)[-1]


def sobol_first_order(func, bounds, n=1024, seed=0):
    """Saltelli estimator of first-order Sobol indices."""
    d = len(bounds)
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    base = sampler.random(n)
    A, B = base[:, :d], base[:, d:]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def to_unit(X):
        return lo + X * (hi - lo)

    fA = func(to_unit(A))
    fB = func(to_unit(B))
    var = np.var(np.concatenate([fA, fB]))
    idx = []
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = func(to_unit(AB))
        idx.append(np.mean(fB * (fAB - fA)) / var)
    return np.clip(np.array(idx), 0, None)


def main() -> None:
    set_seed(0)
    set_style()
    rng = np.random.default_rng(0)

    # ---- the exponential map ----------------------------------------------
    dgs = np.linspace(-3.0, 1.0, 200)
    Kds = np.exp(dgs / RT)
    theta = 1.0 / (1.0 + Kds)

    # ---- Monte Carlo over the free-energy error ----------------------------
    sigmas = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    mc = {}
    for name, kw in REGIMES.items():
        nom = phenotype(DG0, **kw)
        errs, mean_p = [], []
        for s in sigmas:
            dg = rng.normal(DG0, s, size=4000)
            p = phenotype(dg, **kw)
            mean_p.append(float(np.mean(p)))
            errs.append(float(np.std(p)))
        mc[name] = {"nominal": float(nom), "sd": errs, "mean": mean_p}

    # amplification of a small error, by finite difference at DG0
    amp = {}
    for name, kw in REGIMES.items():
        h = 0.02
        d1 = (phenotype(DG0 + h, **kw) - phenotype(DG0 - h, **kw)) / (2 * h)
        p0 = phenotype(DG0, **kw)
        # relative gain: d log p / d log K = d p/d dg * RT / p
        amp[name] = float(d1 * RT / max(p0, 1e-9))

    # ---- Sobol -------------------------------------------------------------
    bounds = [(-2.5, 0.5), (0.05, 1.2), (1.0, 5.0), (1.0, 5.0), (0.1, 0.6)]
    names = ["$\\Delta G$", "$K_1$", "$n_1$", "$n_2$", "$\\tau$"]
    sobol = {}
    for name, kw in REGIMES.items():
        base = dict(kw)

        def f(v, base=base):
            dg, K1, n1, n2, tau = v.T
            return cascade(dg, K1, n1, n2, tau, base["gamma"], base["w"])[-1]

        sobol[name] = sobol_first_order(f, bounds, n=512)

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    ax = axes[0, 0]
    ax.plot(dgs, Kds, color=PAL["blue"], label="$K_d$")
    ax.set_yscale("log")
    ax.set_xlabel("$\\Delta G$ (kcal/mol)")
    ax.set_ylabel("$K_d$", color=PAL["blue"])
    ax2 = ax.twinx()
    ax2.plot(dgs, theta, color=PAL["vermillion"])
    ax2.set_ylabel("occupancy $\\theta$", color=PAL["vermillion"])
    for s in (0.5, 1.0):
        ax.axvspan(DG0 - s, DG0 + s, color=PAL["light"], alpha=0.5, lw=0)
    ax.text(DG0, 0.02, "$\\pm$1 kcal/mol", fontsize=6.4, color=PAL["dark"],
            ha="center", va="bottom")
    ax.set_title("(a) a linear error becomes exponential")

    ax = axes[0, 1]
    for (name, kw), c in zip(REGIMES.items(), [PAL["green"], PAL["vermillion"]]):
        ax.plot(dgs, phenotype(dgs, **kw), color=c, label=name)
    ax.set_xlabel("$\\Delta G$ (kcal/mol)")
    ax.set_ylabel("phenotype probability")
    ax.set_title("(b) same error, different gain")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    width = 0.35
    x = np.arange(len(sigmas))
    for k, (name, c) in enumerate(zip(REGIMES, [PAL["green"], PAL["vermillion"]])):
        ax.bar(x + (k - 0.5) * width, mc[name]["sd"], width, color=c,
               label=name)
    ax.set_xticks(x, [f"{s:g}" for s in sigmas])
    ax.set_xlabel("$\\sigma_{\\Delta G}$ (kcal/mol)")
    ax.set_ylabel("sd of phenotype")
    ax.set_title("(c) Monte Carlo propagation")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    x = np.arange(len(names))
    for k, (name, c) in enumerate(zip(REGIMES, [PAL["green"], PAL["vermillion"]])):
        ax.bar(x + (k - 0.5) * width, sobol[name], width, color=c, label=name)
    ax.set_xticks(x, names)
    ax.set_ylabel("first-order Sobol index")
    ax.set_title("(d) which parameter matters")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig-error")

    save_metrics("error", {
        "RT": RT,
        "sigma_grid": sigmas.tolist(),
        "mc_sd": {k: [round(v, 5) for v in mc[k]["sd"]] for k in mc},
        "mc_nominal": {k: round(mc[k]["nominal"], 5) for k in mc},
        "relative_gain": {k: round(amp[k], 3) for k in amp},
        "sobol": {k: sobol[k].round(4).tolist() for k in sobol},
        "sobol_names": ["dG", "K1", "n1", "n2", "tau"],
    })


if __name__ == "__main__":
    main()
