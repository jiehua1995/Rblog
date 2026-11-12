"""Study 2 — can molecular structure pin down the cell state?

A bistable toggle is used as a stand-in for a cell-fate decision.  The hidden
variable delta is a chromatin-like bias that is not visible in the "structure"
readout; the readout is a noisy, compressed measurement of delta.  Everything
below is a finite-sample estimate of quantities that are defined exactly:

  * Bayes error of the fate given a readout (numerically integrated posterior),
  * mutual information I(readout; fate) and I(readout; delta),
  * conditional variance of the future state given the readout,
  * the same quantities at several prediction horizons.

Run:  python experiments/02_identifiability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

ALPHA, K, N_HILL, DEG = 2.6, 0.5, 4.0, 1.0
DT = 0.02
T_END = 40.0


def rhs(state: np.ndarray, delta: float) -> np.ndarray:
    a, b = state
    prod_a = ALPHA / (1.0 + (b / K) ** N_HILL) + 1.0 + delta
    prod_b = ALPHA / (1.0 + (a / K) ** N_HILL) + 1.0 - delta
    return np.array([prod_a - DEG * a, prod_b - DEG * b])


def simulate(delta: float, t_end: float = T_END, a0: float = 1.0,
             b0: float = 1.0, dt: float = DT) -> np.ndarray:
    """RK4 trajectory, returned as an array of shape (n_steps + 1, 2)."""
    n = int(round(t_end / dt))
    s = np.array([a0, b0], dtype=float)
    out = np.empty((n + 1, 2))
    out[0] = s
    for i in range(n):
        k1 = rhs(s, delta)
        k2 = rhs(s + 0.5 * dt * k1, delta)
        k3 = rhs(s + 0.5 * dt * k2, delta)
        k4 = rhs(s + dt * k3, delta)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = s
    return out


def fate_at(traj: np.ndarray, t: float) -> float:
    i = min(int(round(t / DT)), len(traj) - 1)
    a, b = traj[i]
    return float(a > b)


def posterior(delta_grid, p_delta, s_obs, sigma):
    """P(delta | s_obs) on the grid, Gaussian readout centred on delta."""
    lik = np.exp(-0.5 * ((s_obs - delta_grid) / sigma) ** 2) / sigma
    w = lik * p_delta
    return w / np.trapezoid(w, delta_grid)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(delta_grid, p_delta, sigma, fate_fn, n_s=241):
    """I(s; fate) for s = delta + N(0, sigma^2), fate = fate_fn(delta)."""
    fate = np.array([fate_fn(d) for d in delta_grid])
    p1 = np.trapezoid(p_delta * fate, delta_grid)
    if p1 <= 0 or p1 >= 1:
        return 0.0, 0.0
    h_fate = entropy(np.array([p1, 1 - p1]))
    s_grid = np.linspace(delta_grid[0] - 3 * sigma, delta_grid[-1] + 3 * sigma, n_s)
    post = np.exp(-0.5 * ((s_grid[:, None] - delta_grid[None, :]) / sigma) ** 2)
    post = post * p_delta[None, :]
    post = post / post.sum(axis=1, keepdims=True)
    p1_s = post @ fate
    s_marg = post.sum(axis=1)
    s_marg = s_marg / s_marg.sum()
    h_cond = 0.0
    for w, q in zip(s_marg, p1_s):
        q = min(max(q, 1e-12), 1 - 1e-12)
        h_cond += -w * (q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return float(h_fate - h_cond), h_fate


def bayes_error(delta_grid, p_delta, sigma, fate_fn, n_s=241):
    """Expected error of the optimal decoder of fate from s."""
    fate = np.array([fate_fn(d) for d in delta_grid])
    s_grid = np.linspace(delta_grid[0] - 4 * sigma, delta_grid[-1] + 4 * sigma, n_s)
    lik = np.exp(-0.5 * ((s_grid[:, None] - delta_grid[None, :]) / sigma) ** 2)
    w = lik * p_delta[None, :]
    post = w / w.sum(axis=1, keepdims=True)
    p1 = post @ fate
    err = np.sum(np.minimum(p1, 1 - p1) * w.sum(axis=1)) / np.sum(w)
    return float(err)


def conditional_sd(delta_grid, p_delta, sigma, t_end, n_s=241):
    """sqrt(E[Var(a_T | s)]) and the unconditional sd of a_T."""
    vals = np.array([simulate(d, t_end=t_end)[-1, 0] for d in delta_grid])
    s_grid = np.linspace(delta_grid[0] - 4 * sigma, delta_grid[-1] + 4 * sigma, n_s)
    lik = np.exp(-0.5 * ((s_grid[:, None] - delta_grid[None, :]) / sigma) ** 2)
    w = lik * p_delta[None, :]
    post = w / w.sum(axis=1, keepdims=True)
    mean = post @ vals
    var = post @ (vals ** 2) - mean ** 2
    marg = w.sum(axis=1) / w.sum()
    cond = float(np.sqrt(max(np.sum(np.clip(var, 0, None) * marg), 0.0)))
    prior = float(np.sqrt(np.average((vals - np.average(vals, weights=p_delta)) ** 2,
                                     weights=p_delta)))
    return cond, prior


def main() -> None:
    set_seed(0)
    set_style()

    delta_grid = np.linspace(-0.6, 0.6, 241)
    p_delta = np.full_like(delta_grid, 1.0 / (delta_grid[-1] - delta_grid[0]))

    # ---- trajectories from an identical observed structure -----------------
    deltas_show = [-0.45, -0.12, 0.0, 0.12, 0.45]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    cols = [PAL["blue"], PAL["sky"], PAL["grey"], PAL["orange"], PAL["vermillion"]]
    for d, c in zip(deltas_show, cols):
        tr = simulate(d)
        t = np.arange(len(tr)) * DT
        ax.plot(t, tr[:, 0], color=c, lw=1.3)
        ax.plot(t, tr[:, 1], color=c, lw=1.0, ls=":")
        ax.text(41.5, tr[-1, 0], f"{d:+.2f}", color=c, fontsize=6.6,
                va="center")
    ax.set_xlabel("time")
    ax.set_ylabel("protein level (solid $a$, dotted $b$)")
    ax.set_title("(a) same start, same structure, five hidden biases")
    ax.set_xlim(0, 50)

    ax = axes[1]
    final_gap = []
    for d in delta_grid:
        tr = simulate(d)
        final_gap.append(tr[-1, 0] - tr[-1, 1])
    final_gap = np.array(final_gap)
    ax.plot(delta_grid, final_gap, color=PAL["dark"], lw=1.2)
    ax.axhline(0, color=PAL["grey"], lw=0.7)
    ax.fill_between(delta_grid, -0.35, 0.35, color=PAL["light"], alpha=0.55,
                    lw=0)
    ax.set_xlabel("hidden bias $\\delta$")
    ax.set_ylabel("final $a-b$")
    ax.set_title("(b) fate is decided by the hidden variable")
    ax.text(-0.55, 0.22, "ambiguous\nfor any finite\nstructure readout",
            fontsize=6.4, color=PAL["dark"])
    fig.tight_layout()
    save(fig, "fig-identifiability")

    # ---- Bayes error vs readout noise --------------------------------------
    sigmas = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    regimes = {
        "structure only": sigmas,
        "structure + RNA": sigmas / 2.0,
        "structure + chromatin": sigmas / 5.0,
    }
    err = {k: [bayes_error(delta_grid, p_delta, s, lambda d: fate_at(simulate(d), 40.0))
               for s in v] for k, v in regimes.items()}

    # ---- how much of the future is pinned down, vs horizon ------------------
    horizons = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    cond = {}
    prior_sd = []
    for label, s in [("structure only", 0.2), ("+ RNA", 0.1),
                     ("+ chromatin", 0.04)]:
        vals, pri = [], []
        for T in horizons:
            c, p = conditional_sd(delta_grid, p_delta, s, T)
            vals.append(c)
            pri.append(p)
        cond[label] = vals
        prior_sd = pri

    mi_fate, h_fate = mutual_information(
        delta_grid, p_delta, 0.2, lambda d: fate_at(simulate(d), 40.0))
    mi_fate_rich, _ = mutual_information(
        delta_grid, p_delta, 0.04, lambda d: fate_at(simulate(d), 40.0))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.15))
    ax = axes[0]
    ax.semilogx(sigmas, err["structure only"], "o-", color=PAL["vermillion"],
                ms=4, label="structure only")
    ax.semilogx(sigmas, err["structure + RNA"], "o-", color=PAL["orange"],
                ms=4, label="structure + RNA")
    ax.semilogx(sigmas, err["structure + chromatin"], "o-", color=PAL["green"],
                ms=4, label="structure + chromatin")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.minorticks_off()
    ax.set_xlabel("readout noise $\\sigma_s$")
    ax.set_ylabel("Bayes error of fate")
    ax.set_title("(a) more measurements, lower floor")
    ax.legend(fontsize=6.4, frameon=False)

    ax = axes[1]
    for k, c in zip(cond, [PAL["vermillion"], PAL["orange"], PAL["green"]]):
        ax.plot(horizons, cond[k], "o-", color=c, ms=4, label=k)
    ax.plot(horizons, prior_sd, color=PAL["grey"], lw=0.9, ls="--",
            label="no readout")
    ax.set_xscale("log")
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("conditional sd of $a_T$")
    ax.set_title("(b) uncertainty grows with the horizon")
    ax.legend(fontsize=6.2, frameon=False)

    ax = axes[2]
    labels = ["$H$(fate)", "$I$($s$; fate)\nstructure", "$I$($s$; fate)\n+ chromatin"]
    vals = [h_fate, mi_fate, mi_fate_rich]
    ax.bar(range(3), vals, color=[PAL["grey"], PAL["vermillion"], PAL["green"]],
           width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(3), labels, fontsize=6.8)
    ax.set_ylabel("bits")
    ax.set_title("(c) the readout is not sufficient")
    ax.set_ylim(0, max(vals) * 1.28)
    fig.tight_layout()
    save(fig, "fig-information")

    save_metrics("identifiability", {
        "sigma_grid": sigmas.tolist(),
        "bayes_error": {k: [round(v, 4) for v in err[k]] for k in err},
        "horizons": horizons.tolist(),
        "conditional_sd": {k: [round(v, 4) for v in cond[k]] for k in cond},
        "prior_sd_by_horizon": [round(v, 4) for v in prior_sd],
        "H_fate": round(h_fate, 4),
        "I_structure_fate": round(mi_fate, 4),
        "I_rich_fate": round(mi_fate_rich, 4),
        "final_gap_range": [round(float(final_gap.min()), 3),
                            round(float(final_gap.max()), 3)],
    })


if __name__ == "__main__":
    main()
