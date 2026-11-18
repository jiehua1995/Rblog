"""Article 2, study 2 — evolution changes structure, not only weights.

Two experiments on *structure*:

  (a) a minimal NEAT-style neuroevolution: genomes grow by adding nodes and
      connections, and we ask whether topological innovation is necessary for
      XOR and 3-bit parity (it is, for parity);
  (b) MAP-Elites over one-dimensional cellular automata with a radius-2
      neighbourhood: the genome is a 32-bit rule table, i.e. a program, not a
      weight vector.  We map the archive of behaviour, not just the best score.

Run:  python experiments/evolutionary_learning/05_structure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQ, save, set_style
from common.utils import save_metrics, set_seed

SEEDS = range(20)
POP = 120
GENS = 80


# ---------------- NEAT-lite ----------------
class Net:
    """Feed-forward network with a growable topology.

    Node 0 is a bias unit, inputs are 1..n_in, the output is n_in+1 and new
    hidden nodes get increasing ids.  Evaluation is recursive, so the ordering
    of ids does not matter and accidental cycles simply contribute zero.
    """

    def __init__(self, n_in, rng):
        self.n_in = n_in
        self.n_out = n_in + 1
        self.bias = 0
        self.hidden0 = n_in + 2
        self.n_nodes = n_in + 3
        self.conns = {(i, self.hidden0): rng.normal(0, 0.7)
                      for i in range(1, n_in + 1)}
        self.conns[(0, self.hidden0)] = rng.normal(0, 0.3)
        self.conns[(self.hidden0, self.n_out)] = rng.normal(0, 0.7)
        for i in range(1, n_in + 1):
            self.conns[(i, self.n_out)] = rng.normal(0, 0.3)

    def copy(self):
        c = Net.__new__(Net)
        c.n_in, c.n_out, c.bias = self.n_in, self.n_out, self.bias
        c.n_nodes = self.n_nodes
        c.conns = dict(self.conns)
        return c

    def forward(self, x):
        vals = {i: float(x[i - 1]) for i in range(1, self.n_in + 1)}
        vals[0] = 1.0

        def get(node, seen):
            if node in vals:
                return vals[node]
            if node in seen:
                return 0.0
            seen.add(node)
            s = 0.0
            for (a, b), w in self.conns.items():
                if b == node:
                    s += w * get(a, seen)
            v = np.tanh(s)
            vals[node] = v
            return v

        return 0.5 * (1.0 + np.tanh(get(self.n_out, set())))

    def mutate(self, rng):
        r = rng.random()
        if r < 0.70 and self.conns:
            k = list(self.conns)[int(rng.integers(len(self.conns)))]
            self.conns[k] += rng.normal(0, 0.3)
        elif r < 0.80:
            b = int(rng.integers(self.n_in + 1, self.n_nodes))
            a = int(rng.integers(0, self.n_nodes))
            if a != b:
                self.conns[(a, b)] = rng.normal(0, 0.5)
        elif r < 0.90:
            k = list(self.conns)[int(rng.integers(len(self.conns)))]
            self.conns[k] = rng.normal(0, 0.8)
        else:
            # split an existing connection through a new node
            k = list(self.conns)[int(rng.integers(len(self.conns)))]
            w = self.conns.pop(k)
            new = self.n_nodes
            self.n_nodes += 1
            self.conns[(k[0], new)] = 1.0
            self.conns[(new, k[1])] = w


def fitness(net, data):
    err = 0.0
    for x, y in data:
        err += (net.forward(np.array(x, float)) - y) ** 2
    return 1 - err / len(data)


def run_neat(task, seed):
    rng = np.random.default_rng(seed)
    if task == "xor":
        data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
        n_in = 2
    else:
        data = [([int(b) for b in f"{i:03b}"], sum(int(b) for b in f"{i:03b}") % 2)
                for i in range(8)]
        n_in = 3
    pop = [Net(n_in, rng) for _ in range(POP)]
    fit = np.array([fitness(n, data) for n in pop])
    hist_success, hist_conn, hist_fit = [], [], []
    for _ in range(GENS):
        idx = rng.integers(0, POP, (POP // 2, 2))
        w = np.where(fit[idx[:, 0]] >= fit[idx[:, 1]], idx[:, 0], idx[:, 1])
        kids = []
        for i in w:
            c = pop[i].copy()
            c.mutate(rng)
            kids.append(c)
        pop = [pop[int(np.argmax(fit))]] + kids + kids
        pop = pop[:POP]
        fit = np.array([fitness(n, data) for n in pop])
        hist_success.append(float(fit.max() > 0.95))
        hist_fit.append(float(fit.max()))
        best = pop[int(np.argmax(fit))]
        hist_conn.append(len(best.conns))
    return np.array(hist_success), np.array(hist_conn), np.array(hist_fit)


# ---------------- cellular automata ----------------
N_CELLS, STEPS, N_IC = 32, 20, 10
NEIGH_IDX = ((np.arange(N_CELLS) - 2) % N_CELLS,
             (np.arange(N_CELLS) - 1) % N_CELLS,
             np.arange(N_CELLS),
             (np.arange(N_CELLS) + 1) % N_CELLS,
             (np.arange(N_CELLS) + 2) % N_CELLS)


def ca_step(state, rule_int):
    neigh = (state[NEIGH_IDX[0]] << 4 | state[NEIGH_IDX[1]] << 3
             | state[NEIGH_IDX[2]] << 2 | state[NEIGH_IDX[3]] << 1
             | state[NEIGH_IDX[4]])
    return (rule_int >> neigh) & 1


def rule_int(rule_bits):
    return int("".join(str(int(b)) for b in rule_bits), 2)


def run_ca(rule_bits, ic, steps=STEPS):
    r = rule_int(rule_bits)
    state = ic.copy()
    activity = 0
    for _ in range(steps):
        new = ca_step(state, r)
        activity += np.mean(new != state)
        state = new
    return state, activity / steps


def evaluate_rule(rule_bits, ics):
    score, dens, act = 0.0, [], []
    for ic in ics:
        final, a = run_ca(rule_bits, ic)
        majority = 0.0 if ic.mean() < 0.5 else 1.0
        score += float(np.mean(final) == majority)
        dens.append(final.mean())
        act.append(a)
    return score / len(ics), float(np.mean(dens)), float(np.mean(act))


def main() -> None:
    set_seed(0)
    set_style()

    # ---------- NEAT-lite ----------
    res = {}
    for task in ("xor", "parity"):
        succ, conn, fits = [], [], []
        for seed in SEEDS:
            s, c, f = run_neat(task, 100 + seed)
            succ.append(s)
            conn.append(c)
            fits.append(f)
        res[task] = (np.array(succ), np.array(conn), np.array(fits))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    for task, c, lab in [("xor", PAL["teal"], "XOR (2 inputs)"),
                         ("parity", PAL["orange"], "parity-3 (3 inputs)")]:
        s = res[task][2]
        ax.plot(s.mean(0), color=c, lw=1.3, label=lab)
        ax.fill_between(range(GENS), s.mean(0) - s.std(0), s.mean(0) + s.std(0),
                        color=c, alpha=0.15, lw=0)
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness in the population")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("(a)  Both improve; parity stays harder", loc="left",
                 fontsize=8.4)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.grid(True)

    ax = axes[1]
    for task, c in [("xor", PAL["teal"]), ("parity", PAL["orange"])]:
        cn = res[task][1]
        ax.plot(cn.mean(0), color=c, lw=1.3, label=task)
        ax.fill_between(range(GENS), cn.mean(0) * 0.9, cn.mean(0) * 1.1,
                        color=c, alpha=0.12, lw=0)
    ax.set_xlabel("generation")
    ax.set_ylabel("connections in the best genome")
    ax.set_title("(b)  Genomes grow structure on their own", loc="left", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.4)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig17_neat", 2)

    # ---------- MAP-Elites over CA rules ----------
    rng = np.random.default_rng(11)
    ics = [rng.integers(0, 2, N_CELLS) for _ in range(N_IC)]
    # make densities far from 0.5 so the task is well posed
    ics = [ic if abs(ic.mean() - 0.5) > 0.12 else (ic ^ 1) for ic in ics]

    grid = 28
    archive = {}
    best_rule, best_fit = None, -1
    history = []

    def cell_of(d, a):
        return (min(grid - 1, int(d * grid)), min(grid - 1, int(a * grid)))

    for it in range(8000):
        if archive and rng.random() < 0.8:
            k = list(archive)[int(rng.integers(len(archive)))]
            rule = list(archive[k][0])
            for _ in range(rng.integers(1, 4)):
                rule[int(rng.integers(0, 32))] ^= 1
        else:
            rule = list(rng.integers(0, 2, 32))
        f, d, a = evaluate_rule(rule, ics)
        key = cell_of(d, a)
        if key not in archive:
            archive[key] = (tuple(rule), f)
        elif f > archive[key][1]:
            archive[key] = (tuple(rule), f)
        if f > best_fit:
            best_fit, best_rule = f, rule

    Z = np.full((grid, grid), np.nan)
    for (i, j), (_, f) in archive.items():
        Z[i, j] = f

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("final density of ones")
    ax.set_ylabel("mean activity (cells changed per step)")
    ax.set_title(f"(a)  MAP-Elites archive: {len(archive)} niches", loc="left",
                 fontsize=8.6)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(
        "density-classification score", fontsize=6.5)

    ax = axes[1]
    examples = [("best rule", best_rule), ("rule 30", list(int(b) for b in f"{30:032b}")),
                ("rule 110", list(int(b) for b in f"{110:032b}"))]
    strip = np.zeros((3 * STEPS + 10, N_CELLS))
    row = 0
    labels = []
    for name, rule in examples:
        state = ics[0].copy()
        r = rule_int(rule)
        for _ in range(STEPS):
            strip[row] = state
            state = ca_step(state, r)
            row += 1
        labels.append((name, row - STEPS / 2))
        row += 5
    ax.imshow(strip, cmap="binary", aspect="auto", interpolation="nearest")
    for name, y in labels:
        ax.text(N_CELLS + 0.8, y, name, fontsize=6.5, va="center")
    ax.set_xlim(-0.5, N_CELLS + 9)
    ax.set_xlabel("cell")
    ax.set_ylabel("time step")
    ax.set_title("(b)  Three programs, same substrate", loc="left", fontsize=8.6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save(fig, "fig18_mapelites", 2)

    save_metrics("structure", {
        "xor_solved_by_gen80": float(res["xor"][0].mean(0)[-1]),
        "parity_solved_by_gen80": float(res["parity"][0].mean(0)[-1]),
        "mean_connections_xor": float(res["xor"][1].mean(0)[-1]),
        "mean_connections_parity": float(res["parity"][1].mean(0)[-1]),
        "archive_size": len(archive),
        "archive_best": float(best_fit),
        "archive_coverage": len(archive) / (grid * grid),
        "ca_evaluations": 8000,
        "n_seeds": len(list(SEEDS)),
    })


if __name__ == "__main__":
    main()
