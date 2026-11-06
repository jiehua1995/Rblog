"""Experiment 4 — a minimal virtual cell.

What the model is: ten genes, two mutual-inhibition toggles, one hub and five
readouts, all described by an explicit ODE (see common/grn.py).  What the
experiment asks: does a system this small already show the properties that make
living systems hard to model — multiple stable states, environment-dependent
state selection, hysteresis, and a response that depends on where the cell was?

Figures: the network, the expression dynamics under an environmental pulse, and
the state space with its attractors.

Run:  python experiments/04_virtual_cell.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grn import C_TRUE, GENES, N, W_TRUE, find_attractors, simulate, step
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed, timer

DT = 0.05


def main():
    set_seed(0)
    set_style()

    with timer("find attractors"):
        atts = find_attractors(n_starts=400, steps=1500, u=0.0, seed=1)
    print(f"{len(atts)} attractors under u = 0")
    for i, a in enumerate(atts):
        print(f"  A{i}: " + " ".join(f"{v:.2f}" for v in a))

    # ---- environmental pulse ---------------------------------------------
    # start in the first attractor, switch the signal on, then off again
    x0 = atts[0] * 1.0
    steps_on, steps_total = 400, 2400          # 20 time units on, 120 in total
    u = np.zeros(steps_total)
    u[steps_on:steps_on + 600] = 1.0           # 30 time units of signal
    xs = simulate(x0, 0.0, np.zeros(N, dtype=bool), 1)  # warm-up no-op
    x = x0.copy()
    traj = np.empty((steps_total + 1, N))
    traj[0] = x
    for i in range(steps_total):
        x = step(x, u[i], np.zeros(N, dtype=bool))
        traj[i + 1] = x

    # second cell, started in a different attractor, same pulse
    y0 = atts[-1] * 1.0
    y = y0.copy()
    traj2 = np.empty((steps_total + 1, N))
    traj2[0] = y
    for i in range(steps_total):
        y = step(y, u[i], np.zeros(N, dtype=bool))
        traj2[i + 1] = y

    # ---- hysteresis check: sweep u up and down ----------------------------
    us = np.linspace(0, 2.0, 41)
    up = []
    x = atts[0].copy()
    for uu in us:
        for _ in range(600):
            x = step(x, uu, np.zeros(N, dtype=bool))
        up.append(x.copy())
    down = []
    x = np.array(up[-1])
    for uu in us[::-1]:
        for _ in range(600):
            x = step(x, uu, np.zeros(N, dtype=bool))
        down.append(x.copy())
    up = np.array(up)
    down = np.array(down)[::-1]

    # ---- state-space cloud for PCA ----------------------------------------
    rng = np.random.default_rng(3)
    cloud = []
    for _ in range(260):
        xi = rng.uniform(0, 2.0, N)
        t = simulate(xi, rng.uniform(0, 1.2), np.zeros(N, dtype=bool), 400)
        cloud.append(t[::8])
    cloud = np.vstack(cloud)
    mu = cloud.mean(0)
    U, S, Vt = np.linalg.svd(cloud - mu, full_matrices=False)
    pcs = Vt[:2]

    def proj(a):
        return (np.asarray(a) - mu) @ pcs.T

    save_metrics(
        "cell",
        {
            "n_genes": N,
            "n_edges": int(np.count_nonzero(W_TRUE)),
            "n_attractors": int(len(atts)),
            "attractors": [[round(float(v), 3) for v in a] for a in atts],
            "pc_var_explained": [float(S[0] ** 2 / np.sum(S ** 2)),
                                 float(S[1] ** 2 / np.sum(S ** 2))],
            "pulse_start": 0,
            "signal_on_steps": 600,
            "total_steps": steps_total,
            "final_state_A": [round(float(v), 3) for v in traj[-1]],
            "start_state_A": [round(float(v), 3) for v in atts[0]],
            "final_state_B": [round(float(v), 3) for v in traj2[-1]],
            "start_state_B": [round(float(v), 3) for v in atts[-1]],
            "hysteresis_max_gap": float(np.max(np.abs(up - down))),
        },
    )

    # ---- figures ----------------------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    # (a) network
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    pos = {0: (-0.62, 1.00), 1: (0.12, 1.00),      # toggle A
           2: (-1.02, 0.18), 3: (-0.28, 0.18),     # toggle B
           4: (-0.10, -0.52),                     # hub
           5: (0.86, 0.92), 6: (1.02, 0.54), 7: (1.06, 0.16),
           8: (0.98, -0.22), 9: (0.78, -0.58)}     # readouts
    for i in range(N):
        for j in range(N):
            if abs(W_TRUE[i, j]) < 1e-9:
                continue
            colour = PAL["blue"] if W_TRUE[i, j] > 0 else PAL["vermillion"]
            ax.add_patch(FancyArrowPatch(
                pos[j], pos[i], connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>", mutation_scale=8, lw=0.4 + 0.55 * abs(W_TRUE[i, j]),
                color=colour, alpha=0.85, shrinkA=11, shrinkB=11))
    for i in range(N):
        x, y = pos[i]
        node_col = PAL["green"] if i < 4 else (PAL["blue"] if i == 4 else PAL["grey"])
        ax.scatter([x], [y], s=320, color=node_col, zorder=3, edgecolor="white", lw=1.2)
        ax.text(x, y, GENES[i], ha="center", va="center", fontsize=8,
                color="white", zorder=4)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.95, 1.35)
    ax.axis("off")
    ax.set_title("Ten genes, two toggles, one hub")
    ax.plot([], [], color=PAL["blue"], lw=1.4, label="activation")
    ax.plot([], [], color=PAL["vermillion"], lw=1.4, label="inhibition")
    ax.legend(loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, "fig-cell-network")

    # (b) dynamics under an environmental pulse
    tt = np.arange(steps_total + 1) * DT
    u_full = np.r_[u, u[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for g, col, lab in ((0, PAL["blue"], "g0 (toggle A)"),
                        (1, PAL["sky"], "g1 (toggle A)"),
                        (2, PAL["green"], "g2 (toggle B)"),
                        (4, PAL["vermillion"], "g4 (hub)")):
        ax.plot(tt, traj[:, g], color=col, lw=1.1, label=lab)
    ax.fill_between(tt, 0, u_full * 3.2, color=PAL["light"], lw=0, alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("expression")
    ax.set_title("(a) cell A: the pulse flips a toggle")
    ax.legend(loc="upper right", fontsize=6.5)

    ax = axes[1]
    ax.plot(tt, traj[:, 0], color=PAL["blue"], lw=1.2, label="cell A, g0")
    ax.plot(tt, traj2[:, 0], color=PAL["green"], lw=1.2, label="cell B, g0")
    ax.fill_between(tt, 0, u_full * 3.2, color=PAL["light"], lw=0, alpha=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("g0 expression")
    ax.set_title("(b) same pulse, different starting state")
    ax.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig-cell-dynamics")

    # (c) state space
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sub = rng.choice(len(cloud), 4000, replace=False)
    ax.scatter(*proj(cloud)[sub].T, s=3, color=PAL["sky"], alpha=0.28, lw=0,
               label="state cloud")
    for i, a in enumerate(atts):
        p = proj(a)
        ax.scatter(*p, s=90, color=PAL["vermillion"], zorder=4)
        ax.annotate(f"attractor {i + 1}", p, textcoords="offset points",
                    xytext=(7, 6), fontsize=7, color=PAL["vermillion"])
    for tr, col, lab in ((traj, PAL["blue"], "cell A"),
                         (traj2, PAL["green"], "cell B")):
        p = proj(tr)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.1, alpha=0.9, label=lab)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("The state space has a few discrete destinations")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig-cell-attractors")


if __name__ == "__main__":
    main()
