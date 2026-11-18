"""Article 2, study 3 — a minimal artificial ecosystem.

Agents live on a toroidal grid, eat a regenerating resource, pay a metabolic
cost, reproduce with mutation and die.  A mid-run environmental shift halves
regrowth and makes the resource patchy.  We record population size, trait
diversity, spatial organisation and the genealogy of a single run.

This is a toy model: it shows that selection can maintain diversity and
recover from a shock, not that it produces intelligence.
Run:  python experiments/artificial_life/06_ecosystem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import FIGURES as ROOT_FIG, POST as PROJECT, save_metrics, set_seed

G = 48
STEPS = 600
SHIFT = 300
POP_CAP = 250
K_CROWD = 130.0
SEEDS = range(20)
NEIGH = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


class World:
    def __init__(self, seed, record=False):
        self.rng = np.random.default_rng(seed)
        self.R = self.rng.uniform(0.3, 1.0, (G, G))
        self.patch = self.rng.normal(0, 1, (G, G))
        self.patch = (self.patch - self.patch.min()) / np.ptp(self.patch)
        self.agents = []
        self.next_id = 0
        self.record = record
        self.tree = {}
        for _ in range(30):
            self.spawn()

    def spawn(self, parent=None):
        if len(self.agents) >= POP_CAP:
            return None
        rng = self.rng
        if parent is None:
            pos = (int(rng.integers(G)), int(rng.integers(G)))
            genes = np.array([rng.uniform(0.3, 0.9), rng.uniform(0.4, 0.9)])
            pid = -1
        else:
            pos = parent["pos"]
            genes = parent["genes"] + rng.normal(0, 0.06, 2)
            genes[0] = np.clip(genes[0], 0.05, 1.0)
            genes[1] = np.clip(genes[1], 0.05, 1.0)
            pid = parent["id"]
        a = {"id": self.next_id, "pos": pos, "genes": genes, "energy": 0.6,
             "age": 0, "parent": pid}
        self.next_id += 1
        self.agents.append(a)
        if self.record:
            self.tree[a["id"]] = (pid, pos[0], pos[1], a["genes"].copy())
        return a

    def step(self, t):
        rng = self.rng
        if t == SHIFT:
            keep = rng.random(len(self.agents)) > 0.7
            self.agents = [a for a, k in zip(self.agents, keep) if k]
        growth = 0.08 if t < SHIFT else 0.05
        cap = 1.0 if t < SHIFT else 0.5 + 0.5 * self.patch
        self.R += growth * self.R * (1 - self.R / np.maximum(cap, 1e-6))
        self.R = np.clip(self.R, 0, 1)

        alive = []
        n = len(self.agents)
        crowd = max(0.1, 1 - n / K_CROWD)
        for a in list(self.agents):
            a["age"] += 1
            speed, eff = a["genes"]
            i, j = a["pos"]
            if rng.random() < speed:
                best, bestv = (i, j), self.R[i, j]
                for di, dj in NEIGH:
                    v = self.R[(i + di) % G, (j + dj) % G]
                    if v > bestv:
                        bestv, best = v, ((i + di) % G, (j + dj) % G)
                a["pos"] = best
                i, j = best
            intake = min(self.R[i, j], 0.28 * eff * crowd)
            self.R[i, j] -= intake
            a["energy"] += intake - (0.02 + 0.03 * speed)
            if a["energy"] > 0.85:
                a["energy"] -= 0.4
                kid = self.spawn(a)
                if kid is not None:
                    alive.append(kid)
            if a["energy"] > 0 and a["age"] < 200:
                alive.append(a)
        self.agents = alive

    def stats(self):
        if not self.agents:
            return 0, np.nan
        g = np.array([a["genes"] for a in self.agents])
        return len(self.agents), float(g[:, 0].std() + g[:, 1].std())


def run(seed, record=False):
    w = World(seed, record=record)
    pops, divs, snaps = [], [], {}
    tree_snaps = []
    for t in range(STEPS):
        w.step(t)
        if t % 10 == 0:
            n, d = w.stats()
            pops.append(n)
            divs.append(d)
            if t in (250, 560) and record:
                snaps[t] = (w.R.copy(), np.array([a["pos"] for a in w.agents]),
                            np.array([a["genes"][0] for a in w.agents]))
            if record and t % 60 == 0 and t > 0:
                tree_snaps.append((dict(w.tree), [a["id"] for a in w.agents]))
        if not w.agents:
            for _ in range(20):
                w.spawn()
    if record:
        tree_snaps.append((dict(w.tree), [a["id"] for a in w.agents]))
    return np.array(pops), np.array(divs), snaps, w, tree_snaps


def draw_tree(ax, tree, leaf_ids=None, max_leaves=90):
    alive = (leaf_ids or [k for k in tree if tree[k][0] != -1])[-max_leaves:]
    keep = set()
    for k in alive:
        while k != -1 and k not in keep:
            keep.add(k)
            k = tree[k][0]
    children = {}
    for k in keep:
        p = tree[k][0]
        children.setdefault(p, []).append(k)
    for p in list(children):
        children[p].sort()

    def leaves(node):
        if node not in children:
            return 1
        return sum(leaves(c) for c in children[node])

    total = leaves(-1) if -1 in children else 1
    counter = {"y": 0}

    def place(node, depth, xspan):
        if node not in children:
            y = counter["y"]
            counter["y"] += 1
            return (y, y), [(node, depth, y)]
        sub = [place(c, depth + 1, xspan) for c in children[node]]
        ys = []
        pts = []
        for (lo, hi), p in sub:
            ys += [lo, hi]
            pts += p
        y = (min(ys) + max(ys)) / 2
        ax.plot([depth, depth + 0.8], [min(ys), min(ys)], color=PAL["grey"], lw=0.3)
        ax.plot([depth, depth + 0.8], [max(ys), max(ys)], color=PAL["grey"], lw=0.3)
        ax.plot([depth + 0.8, depth + 0.8], [min(ys), max(ys)], color=PAL["grey"], lw=0.3)
        ax.plot([depth + 0.8, depth + 1.6], [y, y], color=PAL["grey"], lw=0.3)
        return (min(ys), max(ys)), pts

    place(-1, 0, total)
    ax.set_yticks([])
    ax.set_xlabel("generations of descent")
    ax.set_title("(b)  Genealogy of one run", loc="left", fontsize=8.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main() -> None:
    set_seed(0)
    set_style()

    pops, divs, snaps, world = [], [], None, None
    tree_snaps = []
    for seed in SEEDS:
        p, d, sn, w, ts = run(seed, record=(seed == 0))
        pops.append(p)
        divs.append(d)
        if seed == 0:
            snaps, world, tree_snaps = sn, w, ts

    P = np.array(pops)
    D = np.array(divs)
    tt = np.arange(P.shape[1]) * 10

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))
    ax = axes[0]
    ax.plot(tt, P.mean(0), color=PAL["navy"], lw=1.2)
    ax.fill_between(tt, np.quantile(P, 0.05, 0), np.quantile(P, 0.95, 0),
                    color=PAL["navy"], alpha=0.16, lw=0)
    ax.axvline(SHIFT, color=PAL["orange"], lw=1.0, ls="--")
    ax.annotate("shock: 70% mortality,\nregrowth falls", (SHIFT + 8, P.mean(0).max() * 0.85),
                fontsize=6.2, color=PAL["orange"])
    ax.set_xlabel("time step")
    ax.set_ylabel("population")
    ax.set_title("(a)  A shock, then rapid recovery", loc="left", fontsize=8.6)
    ax.grid(True)

    ax = axes[1]
    ax.plot(tt, D.mean(0), color=PAL["teal"], lw=1.2)
    ax.fill_between(tt, np.quantile(D, 0.05, 0), np.quantile(D, 0.95, 0),
                    color=PAL["teal"], alpha=0.16, lw=0)
    ax.axvline(SHIFT, color=PAL["orange"], lw=1.0, ls="--")
    ax.set_xlabel("time step")
    ax.set_ylabel("trait diversity (sd of speed + sd of efficiency)")
    ax.set_title("(b)  Selection erodes trait diversity", loc="left", fontsize=8.6)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig19_ecosystem", 2)

    # ---------- spatial snapshots ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, t in zip(axes, [150, 560]):
        if t not in snaps:
            continue
        R, pos, speed = snaps[t]
        im = ax.imshow(R, cmap="YlGn", vmin=0, vmax=1, origin="lower")
        if len(pos):
            ax.scatter(pos[:, 1], pos[:, 0], c=speed, cmap="coolwarm", s=7,
                       vmin=0.2, vmax=1.0, edgecolors="none")
        ax.set_title(f"({'ab'[t > SHIFT]})  t = {t}", loc="left", fontsize=8.6)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("resource", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig20_snapshots", 2)

    # ---------- genealogy ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    draw_tree(ax, world.tree, [a["id"] for a in world.agents])
    save(fig, "fig21_tree", 2)

    # ---------- animated genealogy ----------
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig2, ax2 = plt.subplots(figsize=(7.6, 4.6))

    def frame(i):
        ax2.clear()
        tree_i, leaves_i = tree_snaps[i]
        draw_tree(ax2, tree_i, leaves_i)
        ax2.set_title(f"(b)  Genealogy, step {60 * (i + 1)}", loc="left", fontsize=8.6)

    ani = FuncAnimation(fig2, frame, frames=len(tree_snaps), interval=800)
    out = ROOT_FIG / "fig21_tree.gif"
    ani.save(out, writer=PillowWriter(fps=1.5), dpi=110)
    print(f"[figure] {out.relative_to(PROJECT)}")

    pre = P[:, 20:SHIFT // 10].mean(1)
    post = P[:, -20:].mean(1)
    trough = P[:, SHIFT // 10:SHIFT // 10 + 6].min(1)
    save_metrics("ecosystem", {
        "pop_pre_shift": float(pre.mean()),
        "pop_trough": float(trough.mean()),
        "pop_post_shift": float(post.mean()),
        "recovery_ratio": float(post.mean() / max(pre.mean(), 1e-6)),
        "diversity_pre": float(np.nanmean(D[:, 20:SHIFT // 10])),
        "diversity_post": float(np.nanmean(D[:, -20:])),
        "n_seeds": len(list(SEEDS)),
        "n_steps": STEPS,
    })


if __name__ == "__main__":
    main()
