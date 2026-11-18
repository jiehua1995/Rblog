"""Article 2, study 1 — gradient descent and natural selection are not the same
optimiser, and neither dominates.

Same landscape, two information channels:

  gradient flow   follows dL/dx, one point, exact local slope
  selection       samples fitness of a population, no slope, keeps variants

Experiments
  (1) static multimodal landscape (Rastrigin, 2-D): who finds the global basin?
  (2) moving optimum: how well does each track a target that drifts?
  (3) quantised fitness ("staircase"): gradients are zero almost everywhere.

20 seeds, equal evaluation budget (4 000 fitness evaluations).
Run:  python experiments/evolutionary_learning/04_optimizers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

BUDGET = 4000
POP = 40
SEEDS = range(20)


def rastrigin(x, y, cx=0.0, cy=0.0):
    u, v = x - cx, y - cy
    return 20 + u ** 2 - 10 * np.cos(2 * np.pi * u) + v ** 2 - 10 * np.cos(2 * np.pi * v)


def grad(x, y, cx=0.0, cy=0.0):
    u, v = x - cx, y - cy
    return (2 * u + 20 * np.pi * np.sin(2 * np.pi * u),
            2 * v + 20 * np.pi * np.sin(2 * np.pi * v))


def run_gradient(cx=0.0, cy=0.0, lr=0.01, steps=4000, start=(-3.5, 3.2)):
    x, y = start
    best, hist = rastrigin(x, y, cx, cy), []
    for _ in range(steps):
        gx, gy = grad(x, y, cx, cy)
        x -= lr * gx
        y -= lr * gy
        best = min(best, rastrigin(x, y, cx, cy))
        hist.append(best)
    return np.array(hist), (x, y)


def run_evolution(cx=0.0, cy=0.0, rng=None, pop=POP, budget=BUDGET, sigma=0.35):
    rng = rng or np.random.default_rng(0)
    P = rng.uniform(-4, 4, (pop, 2))
    fit = rastrigin(P[:, 0], P[:, 1], cx, cy)
    hist, used = [], 0
    best = float(fit.min())
    while used < budget:
        idx = rng.integers(0, pop, (pop // 2, 2))
        winners = np.where(fit[idx[:, 0]] < fit[idx[:, 1]], idx[:, 0], idx[:, 1])
        kids = P[winners] + rng.normal(0, sigma, (pop // 2, 2))
        P = np.clip(np.vstack([kids, kids]), -5.5, 5.5)
        fit = rastrigin(P[:, 0], P[:, 1], cx, cy)
        used += len(P)
        best = min(best, float(fit.min()))
        hist.append(best)
    return np.array(hist), P, fit


def main() -> None:
    set_seed(0)
    set_style()

    # ---------- fig15: one landscape, two trajectories ----------
    gh, g_end = run_gradient()
    eh, P, fit = run_evolution(rng=np.random.default_rng(3))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    xs = np.linspace(-5.5, 5.5, 400)
    X, Y = np.meshgrid(xs, xs)
    cs = ax.contourf(X, Y, rastrigin(X, Y), levels=28, cmap="Blues_r", alpha=0.85)
    ax.contour(X, Y, rastrigin(X, Y), levels=10, colors="white", linewidths=0.3)
    ax.plot([-3.5], [3.2], "o", color=PAL["orange"], ms=5, label="gradient start")
    ax.plot(P[:, 0], P[:, 1], "o", color=PAL["teal"], ms=3, alpha=0.55,
            label="final population")
    ax.set_title("(a)  Rugged landscape, two search styles", loc="left", fontsize=8.6)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.02).set_label("loss", fontsize=6.5)

    ax = axes[1]
    ev = np.arange(1, len(eh) + 1) * POP
    ax.plot(np.arange(1, len(gh) + 1), gh, color=PAL["orange"], lw=1.2,
            label="gradient flow (slope information)")
    ax.plot(ev, eh, color=PAL["teal"], lw=1.2, label="selection + mutation (samples)")
    ax.set_xscale("log")
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best loss so far")
    ax.set_title("(b)  Local slope versus global sampling", loc="left", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.4)
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "fig15_landscape", 2)

    # ---------- experiment 2 and 3 ----------
    shift_rates = [0, 200, 500, 1000, 2000]  # evaluations between moves
    track = {"gradient": [], "selection": []}
    stair = {"gradient": [], "selection": []}
    reach = {"gradient": [], "selection": []}

    for seed in SEEDS:
        rng = np.random.default_rng(500 + seed)
        gh, _ = run_gradient()
        eh, _, _ = run_evolution(rng=rng)
        reach["gradient"].append(gh[-1])
        reach["selection"].append(eh[-1])

        for rate in shift_rates:
            rng = np.random.default_rng(900 + seed)
            cx = cy = 0.0
            x, y = -3.5, 3.2
            errs = []
            for step in range(4000):
                if rate and step % rate == 0:
                    cx = rng.uniform(-2.5, 2.5)
                    cy = rng.uniform(-2.5, 2.5)
                gx, gy = grad(x, y, cx, cy)
                x -= 0.01 * gx
                y -= 0.01 * gy
                errs.append(rastrigin(x, y, cx, cy))
            track["gradient"].append(np.mean(errs[500:]))

            rng = np.random.default_rng(900 + seed)
            cx = cy = 0.0
            pop = rng.uniform(-4, 4, (POP, 2))
            best_err = []
            for step in range(0, 4000, POP):
                if rate and step % rate == 0:
                    cx = rng.uniform(-2.5, 2.5)
                    cy = rng.uniform(-2.5, 2.5)
                f = rastrigin(pop[:, 0], pop[:, 1], cx, cy)
                best_err.append(float(f.min()))
                idx = rng.integers(0, POP, (POP // 2, 2))
                w = np.where(f[idx[:, 0]] < f[idx[:, 1]], idx[:, 0], idx[:, 1])
                kids = pop[w] + rng.normal(0, 0.3, (POP // 2, 2))
                pop = np.clip(np.vstack([kids, kids]), -5.5, 5.5)
            track["selection"].append(np.mean(best_err))

        # staircase fitness: gradient is zero almost everywhere
        def stair_loss(x, y):
            return np.floor(rastrigin(x, y) / 4) * 4

        x, y = -3.5, 3.2
        rng_st = np.random.default_rng(3000 + seed)
        for _ in range(4000):
            gx, gy = grad(x, y)
            if abs(stair_loss(x + 1e-3 * gx, y + 1e-3 * gy) - stair_loss(x, y)) < 1e-9:
                # slope gives no information: the gradient is zero almost everywhere
                gx, gy = rng_st.normal(size=2)
            x -= 0.01 * gx
            y -= 0.01 * gy
        stair["gradient"].append(stair_loss(x, y))

        rng = np.random.default_rng(700 + seed)
        pop = rng.uniform(-4, 4, (POP, 2))
        f = stair_loss(pop[:, 0], pop[:, 1])
        for _ in range(4000 // POP):
            idx = rng.integers(0, POP, (POP // 2, 2))
            w = np.where(f[idx[:, 0]] < f[idx[:, 1]], idx[:, 0], idx[:, 1])
            kids = pop[w] + rng.normal(0, 0.3, (POP // 2, 2))
            pop = np.clip(np.vstack([kids, kids]), -5.5, 5.5)
            f = stair_loss(pop[:, 0], pop[:, 1])
        stair["selection"].append(f.min())

    # ---------- fig16: tracking and quantised fitness ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    sr = np.array(shift_rates, float)
    g = np.array(track["gradient"]).reshape(len(list(SEEDS)), -1)
    s = np.array(track["selection"]).reshape(len(list(SEEDS)), -1)
    ax.plot(sr, g.mean(0), "o-", color=PAL["orange"], ms=3.5, label="gradient flow")
    ax.plot(sr, s.mean(0), "o-", color=PAL["teal"], ms=3.5, label="selection")
    for arr, c in ((g, PAL["orange"]), (s, PAL["teal"])):
        ax.fill_between(sr, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                        color=c, alpha=0.15, lw=0)
    ax.set_xlabel("evaluations between environmental shifts (0 = static)")
    ax.set_ylabel("loss at the tracked point")
    ax.set_title("(a)  Tracking a moving optimum", loc="left", fontsize=8.6)
    ax.legend(frameon=False, fontsize=6.4)
    ax.grid(True)

    ax = axes[1]
    bp = ax.boxplot([stair["gradient"], stair["selection"]], widths=0.5,
                    patch_artist=True, tick_labels=["gradient\n(zero slope)", "selection"])
    for patch, c in zip(bp["boxes"], [PAL["orange"], PAL["teal"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color(PAL["dark"])
    ax.set_ylabel("final loss (lower is better)")
    ax.set_title("(b)  When the slope is uninformative", loc="left", fontsize=8.6)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig16_tracking", 2)

    # ---------- control: a convex bowl, where the gradient should win ----------
    def bowl(x, y):
        return (x - 1.0) ** 2 + (y + 1.0) ** 2

    steps_to_tol, evals_to_tol = [], []
    for seed in SEEDS:
        x, y = -3.5, 3.2
        for i in range(5000):
            x -= 0.01 * 2 * (x - 1.0)
            y -= 0.01 * 2 * (y + 1.0)
            if bowl(x, y) < 1e-3:
                break
        steps_to_tol.append(i + 1)
        rng = np.random.default_rng(4000 + seed)
        pop = rng.uniform(-4, 4, (POP, 2))
        f = bowl(pop[:, 0], pop[:, 1])
        used = POP
        while f.min() > 1e-3 and used < 200_000:
            idx = rng.integers(0, POP, (POP // 2, 2))
            w = np.where(f[idx[:, 0]] < f[idx[:, 1]], idx[:, 0], idx[:, 1])
            kids = pop[w] + rng.normal(0, 0.3, (POP // 2, 2))
            pop = np.clip(np.vstack([kids, kids]), -5.5, 5.5)
            f = bowl(pop[:, 0], pop[:, 1])
            used += POP
        evals_to_tol.append(used)

    save_metrics("optimizers", {
        "final_loss_static": {"gradient": float(np.mean(reach["gradient"])),
                              "selection": float(np.mean(reach["selection"]))},
        "shift_rates": shift_rates,
        "tracking_error": {"gradient": [float(v) for v in g.mean(0)],
                           "selection": [float(v) for v in s.mean(0)]},
        "staircase_loss": {"gradient": float(np.mean(stair["gradient"])),
                           "selection": float(np.mean(stair["selection"]))},
        "convex_bowl": {"gradient_steps": float(np.mean(steps_to_tol)),
                        "selection_evals": float(np.mean(evals_to_tol))},
        "n_seeds": len(list(SEEDS)),
        "budget": BUDGET,
    })


if __name__ == "__main__":
    main()
