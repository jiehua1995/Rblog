"""Study 3 — coarse-graining a linear system: what closure costs.

A 10-dimensional linear system is split into 2 slow and 8 fast variables.
Because the system is linear, the Mori-Zwanzig memory kernel can be written
down exactly:

    dX/dt = A_ss X + A_sf Y
    dY/dt = A_fs X + A_ff Y
    =>  dX/dt = A_ss X + int_0^t A_sf e^{A_ff (t-s)} A_fs X(s) ds + (initial slip)

so K(tau) = A_sf exp(A_ff tau) A_fs.  The Markov ("no memory") closure replaces
the kernel by its time integral, which is exact only in the limit tau_f -> 0.
This script measures the error of four reduced models as the timescale
separation eps = tau_f / tau_s is varied.

Run:  python experiments/03_projection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

N_SLOW, N_FAST = 2, 8
TAU_S = 1.0
DT = 0.01
T_END = 12.0
N_STEPS = int(round(T_END / DT))


def build_system(eps: float, seed: int = 0):
    """Slow/fast blocks with a prescribed timescale separation.

    The fast block is built from damped rotations, so its memory kernel
    oscillates while it decays.  A purely relaxational fast block would make
    the Markov closure look better than it deserves to look.
    """
    rng = np.random.default_rng(seed)
    tau_f = TAU_S * eps
    A_ss = -np.eye(N_SLOW) / TAU_S + 0.15 * np.array([[0, 1], [-1, 0]])
    blocks = []
    for k in range(N_FAST // 2):
        gamma = 1.0 / tau_f
        omega = 1.0 + 0.7 * k
        blocks.append(np.array([[-gamma, -omega], [omega, -gamma]]))
    A_ff = np.zeros((N_FAST, N_FAST))
    for k, blk in enumerate(blocks):
        A_ff[2 * k:2 * k + 2, 2 * k:2 * k + 2] = blk
    A_sf = 1.6 / np.sqrt(N_FAST) * rng.normal(size=(N_SLOW, N_FAST))
    A_fs = 1.6 / np.sqrt(N_SLOW) * rng.normal(size=(N_FAST, N_SLOW))
    A = np.block([[A_ss, A_sf], [A_fs, A_ff]])
    return A, A_ss, A_sf, A_fs, A_ff


def kernel(A_sf, A_fs, A_ff, lag):
    return A_sf @ expm(A_ff * lag) @ A_fs


def simulate_truth(A, x0, n_steps=N_STEPS, dt=DT):
    M = expm(A * dt)
    x = np.empty((n_steps + 1, A.shape[0]))
    x[0] = x0
    for i in range(n_steps):
        x[i + 1] = M @ x[i]
    return x


def markov_closure(A_ss, A_sf, A_fs, A_ff, x0, n_steps=N_STEPS, dt=DT):
    A_eff = A_ss - A_sf @ np.linalg.solve(A_ff, A_fs)
    M = expm(A_eff * dt)
    out = np.empty((n_steps + 1, N_SLOW))
    out[0] = x0
    for i in range(n_steps):
        out[i + 1] = M @ out[i]
    return out


def memory_model(A_ss, A_sf, A_fs, A_ff, x0, n_steps=N_STEPS, dt=DT):
    """dX/dt = A_ss X + int K(t-s) X(s) ds, with the exact kernel."""
    lags = np.arange(n_steps + 1) * dt
    K = np.array([kernel(A_sf, A_fs, A_ff, tau) for tau in lags])  # (n, 2, 2)
    out = np.empty((n_steps + 1, N_SLOW))
    out[0] = x0
    for i in range(n_steps):
        hist = out[: i + 1]
        w = np.ones(i + 1)
        w[0] = w[-1] = 0.5
        Ki = K[i::-1]                       # lag t_i - t_j
        conv = np.einsum("j,jab,jb->a", w, Ki, hist) * dt
        out[i + 1] = out[i] + dt * (A_ss @ out[i] + conv)
    return out


def ar2_fit(X, dt=DT):
    """Least-squares AR(2) on the coarse trajectory: x_{k+1} = M1 x_k + M2 x_{k-1}."""
    Xk = X[2:]
    F = np.hstack([X[1:-1], X[:-2]])
    coef, *_ = np.linalg.lstsq(F, Xk, rcond=None)
    M1, M2 = coef[:N_SLOW].T, coef[N_SLOW:].T
    return M1, M2


def ar2_rollout(M1, M2, x0, n_steps=N_STEPS):
    out = np.empty((n_steps + 1, N_SLOW))
    out[0] = x0
    out[1] = x0
    for i in range(1, n_steps):
        out[i + 1] = M1 @ out[i] + M2 @ out[i - 1]
    return out


def rel_error(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) /
                 np.sqrt(np.mean(truth ** 2)))


def main() -> None:
    set_seed(0)
    set_style()

    eps_list = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    rows = {k: [] for k in ["Markov", "AR(2)", "memory"]}
    x0 = np.array([1.0, -0.5])
    traj_store = {}

    for eps in eps_list:
        A, A_ss, A_sf, A_fs, A_ff = build_system(eps)
        full = simulate_truth(A, np.concatenate([x0, np.zeros(N_FAST)]))
        truth = full[:, :N_SLOW]
        mk = markov_closure(A_ss, A_sf, A_fs, A_ff, x0)
        me = memory_model(A_ss, A_sf, A_fs, A_ff, x0)
        M1, M2 = ar2_fit(truth)
        ar = ar2_rollout(M1, M2, x0)
        rows["Markov"].append(rel_error(mk, truth))
        rows["AR(2)"].append(rel_error(ar, truth))
        rows["memory"].append(rel_error(me, truth))
        traj_store[eps] = (truth, mk, me, ar)
        print(f"eps={eps:5.2f}  Markov {rows['Markov'][-1]:.4f}  "
              f"AR(2) {rows['AR(2)'][-1]:.4f}  memory {rows['memory'][-1]:.4f}")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.15),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    lags = np.linspace(0, 3.0, 300)
    for eps, c in zip([0.02, 0.1, 0.5], [PAL["blue"], PAL["sky"], PAL["vermillion"]]):
        _, A_ss, A_sf, A_fs, A_ff = build_system(eps)
        nrm = [np.linalg.norm(kernel(A_sf, A_fs, A_ff, tau)) for tau in lags]
        ax.plot(lags, nrm, color=c, lw=1.2, label=f"$\\varepsilon$ = {eps}")
    ax.set_xlabel("lag $\\tau$ (time)")
    ax.set_ylabel("$\\|K(\\tau)\\|_F$")
    ax.set_title("(a) the memory kernel decays in $\\tau_f$")
    ax.legend(fontsize=6.6, frameon=False)

    ax = axes[1]
    truth, mk, me, ar = traj_store[0.5]
    t = np.arange(len(truth)) * DT
    denom = np.linalg.norm(truth, axis=1) + 1e-3
    for pred, c, ls, lab in [(mk, PAL["vermillion"], "--", "Markov"),
                             (ar, PAL["orange"], ":", "AR(2)"),
                             (me, PAL["green"], "-", "memory")]:
        rel = np.linalg.norm(pred - truth, axis=1) / denom
        ax.semilogy(t, np.clip(rel, 1e-6, None), color=c, lw=1.0, ls=ls, label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("relative error of $X$")
    ax.set_title("(b) $\\varepsilon$ = 0.5: error, not trajectory")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=6.6, frameon=False)

    ax = axes[2]
    for k, c, m in [("Markov", PAL["vermillion"], "o"),
                    ("AR(2)", PAL["orange"], "s"),
                    ("memory", PAL["green"], "d")]:
        ax.loglog(eps_list, rows[k], m + "-", color=c, ms=3.6, label=k)
    ax.set_xlabel("timescale separation $\\varepsilon=\\tau_f/\\tau_s$")
    ax.set_ylabel("relative rollout error")
    ax.set_title("(c) closure error scales with $\\varepsilon$")
    ax.legend(fontsize=6.6, frameon=False)
    fig.tight_layout()
    save(fig, "fig-projection")

    save_metrics("projection", {
        "eps": eps_list.tolist(),
        "rel_error": {k: [round(v, 5) for v in rows[k]] for k in rows},
        "markov_slope": round(float(np.polyfit(np.log(eps_list[:5]),
                                               np.log(rows["Markov"][:5]), 1)[0]), 3),
        "n_slow": N_SLOW,
        "n_fast": N_FAST,
        "dt": DT,
    })


if __name__ == "__main__":
    main()
