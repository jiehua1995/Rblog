"""Study 4 — open, driven reaction networks: what a concentration cannot tell you.

A three-state cycle is driven out of equilibrium by a chemical-potential
difference mu.  Local detailed balance fixes the ratio of forward to backward
rates on every edge, and the affinity around the cycle is

    A/kT = ln( w12 w23 w31 / (w21 w32 w13) ) = 3 * 2 * theta

The steady-state distribution of the symmetric cycle is uniform for every
drive, so the concentrations carry no information about the drive; the net
flux and the entropy production rate do:

    J = (w+ - w-)/3,      sigma = J * A / T   (kB = 1)

Run:  python experiments/04_thermo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

K0 = 1.0
N_STATE = 3


def rates(affinity: float, scale: float = 1.0):
    """Forward / backward rates for a 3-cycle with total affinity `affinity`."""
    theta = affinity / 6.0                 # per-edge half-affinity in kT units
    wf = scale * K0 * np.exp(theta)
    wb = scale * K0 * np.exp(-theta)
    return wf, wb


def generator(affinity: float, scale: float = 1.0):
    """Rate matrix Q with Q[i, j] the rate i -> j and rows summing to zero."""
    wf, wb = rates(affinity, scale)
    Q = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        j = (i + 1) % N_STATE
        Q[i, j] = wf                      # i -> i+1
        Q[j, i] = wb                      # i+1 -> i
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def steady_state(W: np.ndarray) -> np.ndarray:
    """Right null vector of the generator, normalised to sum to one."""
    evals, evecs = np.linalg.eig(W.T)
    idx = np.argmin(np.abs(evals))
    p = np.real(evecs[:, idx])
    p = np.abs(p)
    return p / p.sum()


def flux_and_sigma(Q: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    wf = Q[0, 1]
    wb = Q[1, 0]
    J = p[0] * wf - p[1] * wb
    sigma = 0.0
    for i in range(N_STATE):
        for j in range(N_STATE):
            if i != j and Q[i, j] > 0 and p[i] > 0 and p[j] > 0:
                sigma += p[i] * Q[i, j] * np.log(p[i] * Q[i, j] /
                                                 (p[j] * Q[j, i]))
    return float(J), float(sigma)


def gillespie(affinity: float, t_end: float, rng: np.random.Generator):
    """Exact stochastic simulation of the 3-state cycle; returns (times, states)."""
    wf, wb = rates(affinity)
    state = 0
    t = 0.0
    ts = [t]
    ss = [state]
    while t < t_end:
        props = np.array([wf, wb]) if state == 0 else np.array([wf, wb])
        total = props.sum()
        t += rng.exponential(1.0 / total)
        if t > t_end:
            break
        if rng.random() < props[0] / total:
            state = (state + 1) % N_STATE
        else:
            state = (state - 1) % N_STATE
        ts.append(t)
        ss.append(state)
    return np.array(ts), np.array(ss)


def main() -> None:
    set_seed(0)
    set_style()
    rng = np.random.default_rng(1)

    affinities = np.linspace(0.0, 12.0, 61)
    J, sig, pvec = [], [], []
    for a in affinities:
        W = generator(a)
        p = steady_state(W)
        j, s = flux_and_sigma(W, p)
        J.append(j)
        sig.append(s)
        pvec.append(p)
    J = np.array(J)
    sig = np.array(sig)
    pvec = np.array(pvec)

    scales = [1.0, 5.0, 25.0]
    same_p = []
    for sc in scales:
        W = generator(4.0, sc)
        p = steady_state(W)
        j, s = flux_and_sigma(W, p)
        same_p.append((p.copy(), j, s))

    # ---- stochastic runs ---------------------------------------------------
    t_end = 300.0
    n_runs = 40
    traj = {0.0: [], 4.0: []}
    cycles = {0.0: [], 4.0: []}
    for a in (0.0, 4.0):
        for _ in range(n_runs):
            ts, ss = gillespie(a, t_end, rng)
            traj[a].append((ts, ss))
            # net clockwise windings: count 1->2 minus 2->1
            fwd = np.sum((ss[:-1] == 0) & (ss[1:] == 1))
            bwd = np.sum((ss[:-1] == 1) & (ss[1:] == 0))
            cycles[a].append(int(fwd - bwd))

    # ---- figure 1: deterministic steady state ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    ax.semilogy(affinities, np.clip(J, 1e-12, None), color=PAL["blue"],
                label="net flux $J$")
    ax.semilogy(affinities, np.clip(sig, 1e-12, None), color=PAL["vermillion"],
                label="entropy production $\\sigma$")
    ax.set_xlabel("cycle affinity $A/k_BT$")
    ax.set_ylabel("rate (arbitrary units)")
    ax.set_title("(a) drive buys flux and dissipation")
    ax.legend(fontsize=6.8, frameon=False)
    ax.axvline(0, color=PAL["grey"], lw=0.7)

    ax = axes[1]
    xpos = np.arange(len(scales))
    ax.bar(xpos, [s[1] for s in same_p], width=0.5, color=PAL["vermillion"],
           label="net flux $J$")
    ax.axhline(same_p[0][0][0], color=PAL["blue"], lw=1.1, ls="--")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PAL["vermillion"], label="net flux $J$"),
                       Line2D([], [], color=PAL["blue"], ls="--",
                              label="$p_1=1/3$ (identical)")],
              fontsize=6.6, frameon=False, loc="upper left")
    for i, s in enumerate(same_p):
        ax.text(i, s[1] * 1.02, f"$\\sigma$={s[2]:.1f}", ha="center",
                fontsize=6.4, color=PAL["vermillion"])
    ax.set_xticks(xpos, [f"$\\times${s:g}" for s in scales])
    ax.set_xlabel("all rates scaled by")
    ax.set_ylabel("value at $A/k_BT=4$")
    ax.set_title("(b) same concentrations, different turnover")
    ax.set_ylim(0, max(s[1] for s in same_p) * 1.18)
    fig.tight_layout()
    save(fig, "fig-thermo")

    # ---- figure 2: stochastic trajectories ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    for k, a in enumerate((0.0, 4.0)):
        ts, ss = traj[a][0]
        ax.step(ts, ss + 3 * k, where="post",
                color=[PAL["grey"], PAL["blue"]][k], lw=0.8,
                label=f"$A/k_BT$ = {a:g}")
    ax.set_yticks([0, 1, 2, 4, 5, 6], ["1", "2", "3", "1", "2", "3"])
    ax.set_xlabel("time")
    ax.set_ylabel("state")
    ax.set_title("(a) one trajectory each")
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    ax.set_xlim(0, 120)

    ax = axes[1]
    bins = np.arange(-80, 240, 10)
    ax.hist(cycles[4.0], bins=bins, density=True, color=PAL["blue"], alpha=0.65,
            label="$A/k_BT$ = 4")
    ax.hist(cycles[0.0], bins=bins, density=True, color=PAL["grey"], alpha=0.55,
            label="$A/k_BT$ = 0")
    ax.axvline(np.mean(cycles[4.0]), color=PAL["vermillion"], lw=1.0, ls="--")
    ax.set_xlabel("net clockwise cycles in $T=300$")
    ax.set_ylabel("probability density")
    ax.set_title("(b) windings distinguish the two worlds")
    ax.legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    save(fig, "fig-thermo-stochastic")

    save_metrics("thermo", {
        "affinities": affinities.tolist(),
        "flux": J.round(6).tolist(),
        "sigma": sig.round(6).tolist(),
        "p_at_A4": same_p[0][0].round(5).tolist(),
        "J_scaled": [round(s[1], 5) for s in same_p],
        "sigma_scaled": [round(s[2], 5) for s in same_p],
        "mean_cycles_A4": float(np.mean(cycles[4.0])),
        "sd_cycles_A4": float(np.std(cycles[4.0])),
        "mean_cycles_A0": float(np.mean(cycles[0.0])),
        "expected_cycles_A4": float(J[np.argmin(np.abs(affinities - 4.0))] * t_end),
        "A4_flux": float(J[np.argmin(np.abs(affinities - 4.0))]),
        "A4_sigma": float(sig[np.argmin(np.abs(affinities - 4.0))]),
    })


if __name__ == "__main__":
    main()
