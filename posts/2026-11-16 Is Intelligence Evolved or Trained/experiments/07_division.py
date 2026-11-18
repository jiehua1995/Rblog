"""Article 2, study 4 — does division of labour beat one bigger agent?

A deliberately simple coordination task.  Two patches produce a concave
reward, q_k * sqrt(n_k), but only if at least one agent is present.  A round
contains 8 agents with 1 action each; a single agent is given 8 actions but
can visit only one patch per round (embodiment constraint), and a second
single-agent control is allowed to split its 8 actions across both patches.

Treatments for the 8-agent group:
  homogeneous   every agent picks the better patch
  fixed roles   four agents are permanently assigned to each patch
  adaptive      each agent best-responds to crowding
  heritable     a role bias is inherited and mutated across 10 generations
                of group selection

Run:  python experiments/evolutionary_learning/07_division.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import bootstrap_ci, save_metrics, set_seed

ROUNDS = 40
FLIP = 20
N_AGENTS = 8
SEEDS = range(20)


def qualities(rng):
    q = np.array([1.0, 1.0])
    traj = []
    for t in range(ROUNDS):
        if t == FLIP:
            q = q[::-1].copy()
        q = np.clip(q + rng.normal(0, 0.05, 2), 0.3, 1.7)
        traj.append(q.copy())
    return np.array(traj)


def reward(q, counts):
    return float(np.sum(q * np.sqrt(np.maximum(counts, 1e-9)) * (counts > 0)))


def run_group(kind, qtraj, rng, role_bias=None):
    totals = []
    for t, q in enumerate(qtraj):
        if kind == "homogeneous":
            k = int(np.argmax(q + rng.normal(0, 0.08, 2)))
            counts = np.array([N_AGENTS, 0]) if k == 0 else np.array([0, N_AGENTS])
        elif kind == "fixed":
            counts = np.array([N_AGENTS // 2, N_AGENTS // 2])
        elif kind == "adaptive":
            # myopic best response with a crowding correction, sequentially
            counts = np.zeros(2)
            for _ in range(N_AGENTS):
                score = q / np.sqrt(counts + 1)
                k = int(np.argmax(score + rng.normal(0, 0.05, 2)))
                counts[k] += 1
        elif kind == "heritable":
            bias = role_bias if role_bias is not None else np.zeros(2)
            score = q * np.exp(bias) / np.sqrt(np.arange(2) * 0 + 1)
            counts = np.zeros(2)
            for _ in range(N_AGENTS):
                p = np.exp(bias) / np.exp(bias).sum()
                k = int(rng.choice(2, p=p))
                counts[k] += 1
            if counts.sum() == 0:
                counts[0] = N_AGENTS
        totals.append(reward(q, counts))
    return np.array(totals)


def run_single(kind, qtraj, rng):
    totals = []
    for q in qtraj:
        if kind == "single-one-patch":
            counts = np.array([N_AGENTS, 0])
        else:  # can split actions
            counts = np.array([N_AGENTS // 2, N_AGENTS // 2])
        totals.append(reward(q, counts))
    return np.array(totals)


def evolve_roles(seed):
    rng = np.random.default_rng(9000 + seed)
    pop = rng.normal(0, 0.4, (12, 2))
    best = pop[0]
    for gen in range(10):
        scores = []
        for bias in pop:
            q = qualities(rng)
            scores.append(run_group("heritable", q, rng, bias).mean())
        scores = np.array(scores)
        idx = np.argsort(-scores)[:4]
        elite = pop[idx]
        kids = elite[rng.integers(0, 4, 8)] + rng.normal(0, 0.15, (8, 2))
        pop = np.vstack([elite, kids])
        best = pop[0]
    return best


def main() -> None:
    set_seed(0)
    set_style()

    treatments = ["homogeneous", "fixed", "adaptive", "heritable",
                  "single-one-patch", "single-split"]
    out = {k: [] for k in treatments}

    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        qtraj = qualities(rng)
        for k in ["homogeneous", "fixed", "adaptive"]:
            out[k].append(run_group(k, qtraj, rng).mean())
        out["heritable"].append(
            run_group("heritable", qtraj, rng, evolve_roles(seed)).mean())
        out["single-one-patch"].append(run_single("single-one-patch", qtraj, rng).mean())
        out["single-split"].append(run_single("single-split", qtraj, rng).mean())

    order = ["homogeneous", "fixed", "adaptive", "heritable",
             "single-one-patch", "single-split"]
    labels = ["homogeneous\ngroup", "fixed roles", "adaptive\nroles",
              "heritable\nroles", "one agent\n(one patch)", "one agent\n(split actions)"]
    cols = [PAL["orange"], PAL["gold"], PAL["teal"], PAL["navy"],
            PAL["grey"], PAL["sky"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    means, los, his = [], [], []
    for k in order:
        arr = np.array(out[k])
        lo, hi = bootstrap_ci(arr)
        means.append(float(arr.mean()))
        los.append(lo)
        his.append(hi)
    ax.bar(range(len(order)), means, color=cols, width=0.62)
    ax.errorbar(range(len(order)), means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                fmt="none", ecolor=PAL["dark"], capsize=3, lw=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_ylabel("mean reward per round")
    ax.set_title("(a)  The gain comes from parallelism, not from numbers",
                 loc="left", fontsize=8.4)
    ax.grid(axis="y")

    ax = axes[1]
    rng = np.random.default_rng(7)
    qtraj = qualities(rng)
    for k, c in [("homogeneous", PAL["orange"]), ("fixed", PAL["gold"]),
                 ("adaptive", PAL["teal"]), ("heritable", PAL["navy"])]:
        ax.plot(run_group(k, qtraj, np.random.default_rng(11), np.array([0.0, 0.0])),
                color=c, lw=1.1, label=k)
    ax.axvline(FLIP, color=PAL["dark"], lw=0.8, ls="--")
    ax.annotate("patch qualities swap", (FLIP + 0.6, ax.get_ylim()[1] * 0.9),
                fontsize=6.2, color=PAL["dark"])
    ax.set_xlabel("round")
    ax.set_ylabel("reward")
    ax.set_title("(b)  One trajectory, four rules", loc="left", fontsize=8.4)
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig22_division", 2)

    save_metrics("division", {
        "reward": {k: float(np.mean(v)) for k, v in out.items()},
        "reward_sd": {k: float(np.std(v)) for k, v in out.items()},
        "n_seeds": len(list(SEEDS)),
        "rounds": ROUNDS,
        "note": "reward is q_k * sqrt(n_k); one agent is limited to one patch per round",
    })


if __name__ == "__main__":
    main()
