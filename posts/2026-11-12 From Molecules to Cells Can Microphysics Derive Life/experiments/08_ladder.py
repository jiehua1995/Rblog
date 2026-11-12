"""Study 8 — six models of the same cell, on one error-cost plane.

A 12-variable ground truth is used as the "cell": six slow regulatory
variables with a tanh nonlinearity, and six fast variables that relax on a
ten-times shorter timescale.

The six models differ in resolution, in what they are told, and in how much
they cost:

  fine         12 variables, dt = 0.002, exact parameters
  fine + error same, but every parameter is off by 15 % (a realistic force-field
               or rate-constant error)
  coarse       6 slow variables with the fast subsystem at quasi-steady state
  VAR          linear one-step map fitted on simulated trajectories
  MLP          feed-forward network, initial state -> phenotype
  hybrid       coarse simulation + a learned residual
  structure    knows the network A but not the state; predicts the mean fate

Cost is measured as wall-clock seconds for the same 400 held-out predictions,
including the one-off fitting cost of the learned models.

Run:  python experiments/08_ladder.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

N_SLOW = 6
N_FAST = 6
TAU_FAST = 0.1
T_END = 20.0
DT_DATA = 0.01


def make_network(seed=0):
    rng = np.random.default_rng(seed)
    A = 0.8 * rng.normal(size=(N_SLOW, N_SLOW)) / np.sqrt(N_SLOW)
    np.fill_diagonal(A, 1.4 + 0.2 * rng.normal(size=N_SLOW))
    A -= 0.3 * np.ones((N_SLOW, N_SLOW))          # global inhibition
    b = -0.35 * np.ones(N_SLOW) + 0.1 * rng.normal(size=N_SLOW)
    B = 0.9 * rng.normal(size=(N_FAST, N_SLOW)) / np.sqrt(N_SLOW)
    return A, b, B


A_TRUE, B0_TRUE, B_TRUE = make_network(0)


def slow_rhs(x, A, b, f):
    return -x + np.tanh(x @ A.T + b + 0.4 * f.sum(axis=1, keepdims=True))


def simulate_fast(x0, A, b, B, n_sub=10, dt=DT_DATA * 1.0, t_end=T_END, tau=TAU_FAST):
    """Integrate the 12-variable system with a small internal step."""
    n = x0.shape[0]
    x = x0.copy()
    f = x @ B.T
    steps = int(round(t_end / dt))
    for _ in range(steps):
        for _ in range(n_sub):
            h = dt / n_sub
            f = f + h * ((x @ B.T) - f) / tau
            x = x + h * slow_rhs(x, A, b, f)
    return x


def quasi_steady(x0, A, b, B, dt=DT_DATA, t_end=T_END):
    x = x0.copy()
    for _ in range(int(round(t_end / dt))):
        f = x @ B.T
        x = x + dt * slow_rhs(x, A, b, f)
    return x


def phenotype(x):
    z = np.clip((x[:, 0] - 0.5) / 0.05, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sample_states(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, N_SLOW))


def main() -> None:
    set_seed(0)
    set_style()
    rng = np.random.default_rng(2)

    n_train, n_test = 600, 400
    x_train = sample_states(n_train, 11)
    x_test = sample_states(n_test, 12)

    t0 = time.perf_counter()
    y_train = phenotype(simulate_fast(x_train, A_TRUE, B0_TRUE, B_TRUE))
    y_test = phenotype(simulate_fast(x_test, A_TRUE, B0_TRUE, B_TRUE))
    t_truth = time.perf_counter() - t0
    print(f"ground truth: {n_train} train + {n_test} test in {t_truth:.1f} s")

    preds, costs, notes = {}, {}, {}

    def measure(name, predict, fit=None):
        t0 = time.perf_counter()
        if fit is not None:
            fit()
        p = predict()
        costs[name] = time.perf_counter() - t0
        preds[name] = np.asarray(p).ravel()
        print(f"  {name:20s} mse {np.mean((preds[name] - y_test) ** 2):.4f}"
              f"   cost {costs[name]:.2f} s")

    # fine, exact parameters
    measure("fine", lambda: phenotype(simulate_fast(x_test, A_TRUE, B0_TRUE, B_TRUE)))

    # fine, parameters perturbed by 15 %
    A_p = A_TRUE * (1 + 0.15 * rng.normal(size=A_TRUE.shape))
    B_p = B_TRUE * (1 + 0.15 * rng.normal(size=B_TRUE.shape))
    measure("fine + 15% error",
            lambda: phenotype(simulate_fast(x_test, A_p, B0_TRUE, B_p)))

    # coarse: fast subsystem at quasi-steady state
    measure("coarse", lambda: phenotype(quasi_steady(x_test, A_TRUE, B0_TRUE, B_TRUE)))

    # VAR: linear one-step map on the slow variables, rolled out
    def one_step(X):
        return X + DT_DATA * slow_rhs(X, A_TRUE, B0_TRUE, X @ B_TRUE.T)

    X1 = one_step(x_train)
    M, *_ = np.linalg.lstsq(np.hstack([x_train, np.ones((n_train, 1))]), X1, rcond=None)
    Mx, mb = M[:N_SLOW].T, M[N_SLOW]
    def var_predict(X):
        x = X.copy()
        for _ in range(int(T_END / DT_DATA)):
            x = x @ Mx.T + mb
        return phenotype(x)
    measure("VAR", lambda: var_predict(x_test))

    # MLP on the initial state
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0)
    measure("MLP", lambda: mlp.predict(x_test), fit=lambda: mlp.fit(x_train, y_train))

    # hybrid: coarse simulation + MLP residual
    base_train = phenotype(quasi_steady(x_train, A_TRUE, B0_TRUE, B_TRUE))
    residual = y_train - base_train
    hyb = MLPRegressor(hidden_layer_sizes=(32,), max_iter=3000, random_state=0)
    def hybrid_predict(X):
        return phenotype(quasi_steady(X, A_TRUE, B0_TRUE, B_TRUE)) + hyb.predict(X)
    measure("hybrid", lambda: hybrid_predict(x_test),
            fit=lambda: hyb.fit(x_train, residual))

    # structure only: knows A, assumes the mean initial state
    mean_state = x_train.mean(axis=0, keepdims=True)
    pred_mean = float(phenotype(quasi_steady(mean_state, A_TRUE, B0_TRUE, B_TRUE))[0])
    measure("structure only", lambda: np.full(n_test, pred_mean))

    # ---- precision versus observation -------------------------------------
    param_noise = np.array([0.0, 0.1, 0.2, 0.3])
    n_obs = np.array([1, 2, 3, 6])
    grid = np.zeros((len(param_noise), len(n_obs)))
    for i, pn in enumerate(param_noise):
        Ap = A_TRUE * (1 + pn * rng.normal(size=A_TRUE.shape))
        Bp = B_TRUE * (1 + pn * rng.normal(size=B_TRUE.shape))
        for j, k in enumerate(n_obs):
            X = x_test.copy()
            X[:, k:] = x_train[:, k:].mean(axis=0)   # unobserved states -> mean
            pred = phenotype(simulate_fast(X, Ap, B0_TRUE, Bp, n_sub=4))
            grid[i, j] = np.mean((pred - y_test) ** 2)

    # ---- error versus training size ---------------------------------------
    sizes = np.array([50, 100, 200, 600])
    err_size = {"MLP": [], "hybrid": []}
    for s in sizes:
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)
        m.fit(x_train[:s], y_train[:s])
        err_size["MLP"].append(float(np.mean((m.predict(x_test) - y_test) ** 2)))
        h = MLPRegressor(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)
        h.fit(x_train[:s], y_train[:s] - base_train[:s])
        pred = phenotype(quasi_steady(x_test, A_TRUE, B0_TRUE, B_TRUE)) + h.predict(x_test)
        err_size["hybrid"].append(float(np.mean((pred - y_test) ** 2)))

    # ---- figure: Pareto ----------------------------------------------------
    names = list(preds)
    mse = {k: float(np.mean((preds[k] - y_test) ** 2)) for k in names}
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.3),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    style = {"fine": (PAL["vermillion"], "o"), "fine + 15% error": (PAL["orange"], "s"),
             "coarse": (PAL["blue"], "o"), "VAR": (PAL["sky"], "s"),
             "MLP": (PAL["green"], "d"), "hybrid": (PAL["purple"], "P"),
             "structure only": (PAL["grey"], "X")}
    offsets = {"fine": (-32, 10), "fine + 15% error": (-120, 6), "coarse": (-58, 10),
               "VAR": (6, -12), "MLP": (10, 4), "hybrid": (6, -18),
               "structure only": (8, 4)}
    for k in names:
        c, m = style[k]
        ax.scatter(costs[k], max(mse[k], 1e-8), color=c, marker=m, s=42, zorder=3)
        ax.annotate(k, (costs[k], max(mse[k], 1e-8)), textcoords="offset points",
                    xytext=offsets[k], fontsize=6.3, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("seconds for 400 predictions (fitting included)")
    ax.set_ylabel("phenotype MSE")
    ax.set_title("(a) more resolution is not cheaper or better")

    ax = axes[1]
    for i, (pn, c) in enumerate(zip(param_noise, [PAL["sky"], PAL["blue"],
                                                  PAL["green"], PAL["vermillion"]])):
        ax.semilogy(n_obs, np.clip(grid[i], 1e-4, None), "o-", color=c, ms=4,
                    label=f"param. error {pn:.0%}")
    ax.set_xticks(n_obs)
    ax.minorticks_off()
    ax.set_xlabel("slow states actually observed")
    ax.set_ylabel("phenotype MSE")
    ax.set_title("(b) missing state caps precision")
    ax.legend(fontsize=6.2, frameon=False)

    ax = axes[2]
    for k, c in zip(err_size, [PAL["green"], PAL["purple"]]):
        ax.plot(sizes, err_size[k], "o-", color=c, ms=4, label=k)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("training trajectories")
    ax.set_ylabel("phenotype MSE")
    ax.set_title("(c) learned corrections need data")
    ax.legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    save(fig, "fig-ladder")

    save_metrics("ladder", {
        "mse": {k: round(v, 6) for k, v in mse.items()},
        "cost_s": {k: round(v, 4) for k, v in costs.items()},
        "precision_grid_mse": grid.round(6).tolist(),
        "param_noise": param_noise.tolist(),
        "n_observed": n_obs.tolist(),
        "size_curve": {k: [round(v, 6) for v in err_size[k]] for k in err_size},
        "sizes": sizes.tolist(),
        "structure_pred": round(pred_mean, 3),
        "truth_time_s": round(t_truth, 2),
        "n_train": n_train,
        "n_test": n_test,
    })


if __name__ == "__main__":
    main()
