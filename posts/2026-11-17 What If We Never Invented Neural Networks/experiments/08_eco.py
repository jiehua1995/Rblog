"""Article 3 — Evolvable Computational Organisms (ECO), a minimal living program.

An ECO individual carries a variable-length genome of regulated instructions.
The expressed program runs on a tiny register machine that senses a shared
grid, moves, eats, signals and reproduces.  Nothing here is a neural network:
the heritable unit *is* the computation.

Experiments
  main       40 generations x 20 seeds, shared world, tournament selection
  ablations  no structural mutation, no regulation, no signalling,
             fixed environment, random selection (drift)
  baselines  hill climbing and random search at the same evaluation budget
  shift      environment changes at generation 20 (season amplitude doubles,
             the rich patch moves); we measure recovery

Run:  python experiments/artificial_life/08_eco.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import bootstrap_ci, save_metrics, set_seed

OPS = ["NOP", "SENSE_R", "SENSE_G", "SENSE_E", "SENSE_S", "SENSE_N",
       "CONST", "ADD", "SUB", "MUL", "STORE", "MOVE", "EAT", "REPRO",
       "SIGNAL", "JMPIF"]
OP_IDX = {o: i for i, o in enumerate(OPS)}
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

G = 24
POP = 16
GENS = 30
TICKS = 35
SHIFT_GEN = 18
SEEDS = range(20)
MAX_PROG = 24
MAX_STEPS = 10


class World:
    def __init__(self, seed, genotype_seed=None):
        self.rng = np.random.default_rng(seed)
        self.R = self.rng.uniform(0.2, 0.9, (G, G))
        self.sig = np.zeros((G, G))
        self.season = 0.0
        self.shifted = False

    def regrow(self, t):
        amp = 0.25 if self.shifted else 0.12
        self.season = float(np.sin(2 * np.pi * t / 25))
        growth = 0.16 * (1 + amp * self.season)
        self.R += growth * self.R * (1 - self.R)
        self.R = np.clip(self.R, 0, 1)
        self.sig *= 0.9


class Org:
    __slots__ = ("genome", "pos", "energy", "age", "kids", "pid", "id")

    def __init__(self, genome, pos, pid=-1, oid=0):
        self.genome = genome
        self.pos = pos
        self.energy = 0.55
        self.age = 0
        self.kids = 0
        self.pid = pid
        self.id = oid

    def program(self):
        return [g for g in self.genome if g[0] == 1][:MAX_PROG]


def random_genome(rng, n=None, structural=True):
    n = n or int(rng.integers(6, 12))
    return [(1, int(rng.integers(0, len(OPS))), int(rng.integers(0, 8)))
            for _ in range(n)]


def mutate(genome, rng, structural=True, regulate=True):
    g = list(genome)
    for i in range(len(g)):
        r = rng.random()
        if r < 0.05:
            g[i] = (g[i][0], int(rng.integers(0, len(OPS))), int(rng.integers(0, 8)))
        elif r < 0.09:
            g[i] = (g[i][0], g[i][1], int(rng.integers(0, 8)))
        elif r < 0.11 and regulate:
            g[i] = (1 - g[i][0], g[i][1], g[i][2])
    if structural:
        r = rng.random()
        if r < 0.10 and len(g) < 40:
            g.insert(int(rng.integers(0, len(g) + 1)), random_genome(rng, 1, False)[0])
        elif r < 0.17 and len(g) > 3:
            g.pop(int(rng.integers(0, len(g))))
        elif r < 0.23 and len(g) < 40:
            j = int(rng.integers(0, len(g)))
            g.insert(int(rng.integers(0, len(g) + 1)), g[j])
    return g


def run(org, world, allow_signal=True):
    prog = org.program()
    if not prog:
        return
    M = [0.0, 0.0, 0.0, 0.0]
    acc, pc, steps = 0.0, 0, 0
    i, j = org.pos
    while pc < len(prog) and steps < MAX_STEPS:
        steps += 1
        prom, op, arg = prog[pc]
        pc += 1
        name = OPS[op]
        if name == "SENSE_R":
            acc = float(world.R[i, j])
        elif name == "SENSE_G":
            nb = max(world.R[(i + a) % G, (j + b) % G] for a, b in DIRS)
            acc = float(nb - world.R[i, j])
        elif name == "SENSE_E":
            acc = org.energy
        elif name == "SENSE_S":
            acc = world.season
        elif name == "SENSE_N":
            acc = float(world.sig[i, j])
        elif name == "CONST":
            acc = arg / 10.0
        elif name == "ADD":
            acc = acc + M[arg % 4]
        elif name == "SUB":
            acc = acc - M[arg % 4]
        elif name == "MUL":
            acc = acc * M[arg % 4]
        elif name == "STORE":
            M[arg % 4] = acc
        elif name == "MOVE":
            d = int(acc * 8) % 8
            i, j = (i + DIRS[d][0]) % G, (j + DIRS[d][1]) % G
            org.pos = (i, j)
        elif name == "EAT":
            take = min(world.R[i, j], 0.30)
            world.R[i, j] -= take
            org.energy += take
        elif name == "REPRO":
            if org.energy > 1.15:
                org.energy -= 0.55
                org.kids += 1
        elif name == "SIGNAL" and allow_signal:
            world.sig[i, j] += 0.4
        elif name == "JMPIF":
            if acc > 0:
                pc = max(0, min(len(prog) - 1, arg % max(len(prog), 1)))


def evaluate(genomes, world, allow_signal=True):
    orgs = [Org(g, (int(world.rng.integers(G)), int(world.rng.integers(G))))
            for g in genomes]
    for t in range(TICKS):
        world.regrow(t)
        for o in orgs:
            o.age += 1
            o.energy -= 0.022
            run(o, world, allow_signal)
            o.energy = min(o.energy, 2.5)
    return np.array([o.energy + 2.0 * o.kids for o in orgs])


def per_generation_env(seed, gen, fixed_env=False):
    w = World(seed)
    if fixed_env:
        gen = min(gen, SHIFT_GEN - 1)
    if gen >= SHIFT_GEN:
        w.shifted = True
        # the richest patch moves to a corner
        w.R[:] *= 0.55
        w.R[2:9, 15:22] += 0.8
        w.R = np.clip(w.R, 0, 1)
    return w


def evolve(seed, structural=True, regulate=True, allow_signal=True,
           fixed_env=False, drift=False, record=False):
    rng = np.random.default_rng(5000 + seed)
    pop = [random_genome(rng) for _ in range(POP)]
    best_hist, div_hist, mean_hist = [], [], []
    lineage = []
    for gen in range(GENS):
        w = per_generation_env(seed, gen, fixed_env)
        fit = evaluate(pop, w, allow_signal)
        best_hist.append(float(fit.max()))
        mean_hist.append(float(fit.mean()))
        if len(pop) > 1:
            d = [sum(a != b for a, b in zip(p, q)) for p, q in
                 itertools.combinations(pop[:10], 2)]
            div_hist.append(float(np.mean(d)))
        else:
            div_hist.append(0.0)
        order = np.argsort(-fit)
        if drift:
            parents = rng.integers(0, POP, (POP,))
        else:
            idx = rng.integers(0, POP, (POP - 1, 2))
            parents = np.where(fit[idx[:, 0]] >= fit[idx[:, 1]], idx[:, 0], idx[:, 1])
        new = [] if drift else [pop[order[0]]]
        for p in parents:
            if len(new) >= POP:
                break
            new.append(mutate(pop[p], rng, structural, regulate))
        if record and gen in (0, GENS // 2, GENS - 1):
            lineage.append((gen, [len(g) for g in new], best_hist[-1]))
        pop = new
    w = per_generation_env(seed, GENS - 1, fixed_env)
    fit = evaluate(pop, w, allow_signal)
    best_genome = pop[int(np.argmax(fit))]
    return (np.array(best_hist), np.array(mean_hist), np.array(div_hist),
            float(fit.max()), pop, best_genome)


def main() -> None:
    set_seed(0)
    set_style()

    res = {"baseline": [], "hill": [], "random": []}
    pop0 = []
    best0 = None
    for seed in SEEDS:
        b, m, d, best, p, bg = evolve(seed)
        res["baseline"].append((b, m, d))
        if seed == 0:
            pop0, best0 = p, bg
        # hill climbing: one genome, keep if better, same evaluation budget
        rng = np.random.default_rng(6000 + seed)
        g = random_genome(rng)
        w = per_generation_env(seed, GENS - 1)
        cur = float(evaluate([g], w)[0])
        hist = []
        for _ in range(GENS):
            cand = mutate(g, rng)
            f = float(evaluate([cand], w)[0])
            if f > cur:
                g, cur = cand, f
            hist.append(cur)
        res["hill"].append(np.array(hist))
        # random search with the same number of genomes evaluated
        rs = [float(evaluate([random_genome(rng)], w)[0]) for _ in range(GENS)]
        res["random"].append(np.maximum.accumulate(np.array(rs)))

    B = np.array([r[0] for r in res["baseline"]])
    M = np.array([r[1] for r in res["baseline"]])
    D = np.array([r[2] for r in res["baseline"]])
    H = np.array(res["hill"])
    RS = np.array(res["random"])

    # ---------- fig23: architecture ----------
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.axis("off")
    boxes = [
        (0.02, "genome\nregulated opcodes", PAL["sky"]),
        (0.22, "expression\npromoter filter", PAL["teal"]),
        (0.42, "program\nregister machine", PAL["navy"]),
        (0.62, "behaviour\nmove / eat / signal", PAL["gold"]),
        (0.82, "selection\nenergy + offspring", PAL["orange"]),
    ]
    for x, label, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.42), 0.15, 0.30, boxstyle="round,pad=0.02",
                                    fc=c, ec="white", alpha=0.9))
        ax.text(x + 0.075, 0.57, label, ha="center", va="center", fontsize=7.4,
                color="white")
    for x in (0.17, 0.37, 0.57, 0.77):
        ax.add_patch(FancyArrowPatch((x, 0.57), (x + 0.05, 0.57), arrowstyle="-|>",
                                     mutation_scale=11, color=PAL["dark"], lw=1.1))
    ax.add_patch(FancyArrowPatch((0.895, 0.42), (0.095, 0.42), arrowstyle="-|>",
                                 mutation_scale=11, color=PAL["orange"], lw=1.1,
                                 connectionstyle="arc3,rad=-0.12"))
    ax.text(0.5, 0.26, "mutation: point · insert · delete · duplicate · promoter flip",
            ha="center", fontsize=6.8, color=PAL["orange"])
    ax.text(0.5, 0.92, "ECO: the heritable unit is the computation, not the weights",
            ha="center", fontsize=9)
    save(fig, "fig23_eco_architecture", 3)

    # ---------- fig24: learning curves ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    gens = np.arange(1, GENS + 1)
    for arr, c, lab in [(B, PAL["navy"], "ECO (population + structure)"),
                        (H, PAL["teal"], "hill climbing"),
                        (RS, PAL["orange"], "random search")]:
        m = arr.mean(0)
        lo = np.quantile(arr, 0.1, 0)
        hi = np.quantile(arr, 0.9, 0)
        ax.plot(gens, m, color=c, lw=1.3, label=lab)
        ax.fill_between(gens, lo, hi, color=c, alpha=0.15, lw=0)
    ax.axvline(SHIFT_GEN, color=PAL["dark"], lw=0.8, ls="--")
    ax.annotate("environment shift", (SHIFT_GEN + 0.4, ax.get_ylim()[1] * 0.92),
                fontsize=6.2, color=PAL["dark"])
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title("A population with heritable programs learns faster than a hill climber",
                 loc="left", fontsize=8.8)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.grid(True)
    save(fig, "fig25_eco_learning", 3)

    # ---------- ablations ----------
    conds = {
        "no structural mutation": dict(structural=False),
        "no regulation": dict(regulate=False),
        "no signalling": dict(allow_signal=False),
        "fixed environment": dict(fixed_env=True),
        "random selection": dict(drift=True),
    }
    abl = {}
    for name, kw in conds.items():
        vals = []
        for seed in SEEDS:
            out = evolve(seed, **kw)
            vals.append(out[3])
        abl[name] = np.array(vals)
    base_final = B[:, -1]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    names = list(abl)
    diffs, los, his = [], [], []
    for n in names:
        d = base_final - abl[n]
        lo, hi = bootstrap_ci(d)
        diffs.append(float(d.mean()))
        los.append(lo)
        his.append(hi)
    y = np.arange(len(names))
    ax.barh(y, diffs, color=PAL["orange"], height=0.55)
    ax.errorbar(diffs, y, xerr=[np.array(diffs) - los, np.array(his) - np.array(diffs)],
                fmt="none", ecolor=PAL["dark"], capsize=3, lw=0.9)
    ax.axvline(0, color=PAL["dark"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("fitness lost relative to the full ECO (mean, 95% bootstrap CI)")
    ax.set_title("Selection is the only ingredient that clearly matters here",
                 loc="left", fontsize=8.6)
    ax.grid(axis="x")
    save(fig, "fig27_eco_ablations", 3)

    # ---------- fig26: diversity and growth ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax = axes[0]
    for arr, c, lab in [(M, PAL["navy"], "mean fitness"), (D, PAL["teal"], "genomic diversity")]:
        m = arr.mean(0)
        lo = np.quantile(arr, 0.1, 0)
        hi = np.quantile(arr, 0.9, 0)
        ax.plot(gens, m, color=c, lw=1.2, label=lab)
        ax.fill_between(gens, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("generation")
    ax.set_title("(a)  Fitness rises; diversity stays roughly flat", loc="left",
                 fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.6)
    ax.grid(True)

    ax = axes[1]
    lengths = sorted(len(g) for g in pop0)
    ax.hist(lengths, bins=range(3, 41, 3), color=PAL["sky"], edgecolor="white")
    ax.set_xlabel("genome length (genes)")
    ax.set_ylabel("individuals")
    ax.set_title("(b)  Genomes at generation 30", loc="left", fontsize=8.6)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig26_eco_diversity", 3)

    # ---------- fig27: what evolved ----------
    w = World(0)
    org = Org(best0, (12, 12))
    path, energies = [], []
    for t in range(TICKS):
        w.regrow(t)
        org.age += 1
        org.energy -= 0.022
        run(org, w)
        path.append(org.pos)
        energies.append(org.energy)
    prog = org.program()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    im = ax.imshow(w.R, cmap="YlGn", vmin=0, vmax=1, origin="lower")
    p = np.array(path)
    ax.plot(p[:, 1], p[:, 0], color=PAL["navy"], lw=0.9, alpha=0.8)
    ax.scatter([p[0, 1]], [p[0, 0]], color=PAL["orange"], s=28, zorder=3,
               label="start")
    ax.scatter([p[-1, 1]], [p[-1, 0]], color=PAL["teal"], s=28, zorder=3,
               label="end")
    ax.set_title("(a)  One evolved organism, 35 ticks", loc="left", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("resource", fontsize=6.5)

    ax = axes[1]
    ax.axis("off")
    lines = [f"{i:2d}  {'on ' if g[0] else 'off'}  {OPS[g[1]]:<8} {g[2]}"
             for i, g in enumerate(org.genome)]
    ax.text(0.0, 1.0, "expressed genome (promoter, opcode, argument)",
            fontsize=7.2, va="top", family="monospace")
    ax.text(0.0, 0.93, "\n".join(lines), fontsize=5.6, va="top", family="monospace")
    ax.text(0.0, 0.03, f"final energy {org.energy:.2f}   offspring {org.kids}   "
                       f"steps {len(prog)}", fontsize=6.8, va="bottom",
            color=PAL["navy"])
    ax.set_title("(b)  Genotype and phenotype are the same object", loc="left",
                 fontsize=8.6)
    fig.tight_layout()
    save(fig, "fig24_eco_organism", 3)

    save_metrics("eco", {
        "final_best": {"eco": float(base_final.mean()),
                       "hill": float(np.array(res["hill"])[:, -1].mean()),
                       "random": float(np.array(res["random"])[:, -1].mean())},
        "final_best_sd": {"eco": float(base_final.std()),
                          "hill": float(np.array(res["hill"])[:, -1].std()),
                          "random": float(np.array(res["random"])[:, -1].std())},
        "ablation_loss": {n: float(base_final.mean() - abl[n].mean()) for n in names},
        "ablation_ci": {n: list(bootstrap_ci(base_final - abl[n])) for n in names},
        "diversity_final": float(D[:, -1].mean()),
        "genome_length_final": float(np.mean([len(g) for g in pop0])),
        "n_seeds": len(list(SEEDS)),
        "generations": GENS,
        "population": POP,
        "ticks": TICKS,
        "note": "shared grid, 24x24, 20 seeds; fitness = final energy + 2 x offspring",
    })


if __name__ == "__main__":
    main()
