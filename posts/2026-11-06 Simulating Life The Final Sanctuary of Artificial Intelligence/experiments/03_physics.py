"""Experiment 3 — learning a physical law from trajectories.

System: a damped, periodically driven pendulum

    theta'' = -gamma * theta' - sin(theta) + A cos(omega t)

Two learners see exactly the same data and the same inputs (gamma and the drive
amplitude A are both given to them):

  * black box  — an MLP that maps a state to the next state, iterated forward;
  * structured — the same MLP, but it only predicts the angular acceleration,
    and the step is taken with a fourth-order Runge-Kutta integrator that
    already knows theta' = omega.

A numerical solver with the true parameters is the reference.  The questions
are (i) one-step accuracy, (ii) error growth over hundreds of steps,
(iii) energy conservation in the undriven, undamped case, (iv) extrapolation to
damping values never seen in training, and (v) sensitivity to initial
conditions.

Run:  python experiments/03_physics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed, timer

OMEGA, DT = 0.7, 0.05
# (drive amplitude, damping) pairs used for training
TRAIN_PAIRS = [(1.2, 0.0), (1.2, 0.05), (1.2, 0.10), (0.0, 0.0), (0.0, 0.05)]
GAMMA_TEST = [0.0, 0.05, 0.15, 0.20]
A_TEST = 1.2


# ---------------------------------------------------------------------------
# 1. Reference solver
# ---------------------------------------------------------------------------
def accel(theta, w, gamma, t, A, omega=OMEGA):
    return -gamma * w - np.sin(theta) + A * np.cos(omega * t)


def rk4_true(theta, w, gamma, A, t, dt=DT):
    k1t, k1w = w, accel(theta, w, gamma, t, A)
    k2t = w + 0.5 * dt * k1w
    k2w = accel(theta + 0.5 * dt * k1t, k2t, gamma, t + 0.5 * dt, A)
    k3t = w + 0.5 * dt * k2w
    k3w = accel(theta + 0.5 * dt * k2t, k3t, gamma, t + 0.5 * dt, A)
    k4t = w + dt * k3w
    k4w = accel(theta + dt * k3t, k4t, gamma, t + dt, A)
    return (theta + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t),
            w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w))


def simulate(theta0, w0, gamma, A, steps=400):
    th = np.empty(steps + 1)
    om = np.empty(steps + 1)
    th[0], om[0] = theta0, w0
    for i in range(steps):
        th[i + 1], om[i + 1] = rk4_true(th[i], om[i], gamma, A, i * DT)
    return th, om


def energy(th, om):
    return 0.5 * om ** 2 + (1 - np.cos(th))


# ---------------------------------------------------------------------------
# 2. Learners
# ---------------------------------------------------------------------------
def features(theta, w, t, gamma, A):
    return torch.stack([torch.sin(theta), torch.cos(theta), w,
                        torch.sin(OMEGA * t), torch.cos(OMEGA * t),
                        gamma, A], dim=-1)


class Net(nn.Module):
    def __init__(self, out_dim=2, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(7, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, out_dim))

    def forward(self, theta, w, t, gamma, A):
        return self.body(features(theta, w, t, gamma, A))


def blackbox_step(net, s, t, gamma, A, dt=DT):
    delta = net(s[:, 0], s[:, 1], t, gamma, A)
    return s + dt * delta


def structured_step(net, s, t, gamma, A, dt=DT):
    """RK4 where the only learned quantity is the angular acceleration."""
    def a(th, om, tt):
        return net(th, om, tt, gamma, A)[:, 0]

    th, om = s[:, 0], s[:, 1]
    k1t, k1w = om, a(th, om, t)
    k2t = om + 0.5 * dt * k1w
    k2w = a(th + 0.5 * dt * k1t, k2t, t + 0.5 * dt)
    k3t = om + 0.5 * dt * k2w
    k3w = a(th + 0.5 * dt * k2t, k3t, t + 0.5 * dt)
    k4t = om + dt * k3w
    k4w = a(th + dt * k3t, k4t, t + dt)
    th_new = th + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
    om_new = om + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return torch.stack([th_new, om_new], dim=-1)


def make_data(n_traj=260, steps=300, seed=0):
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n_traj):
        A, g = TRAIN_PAIRS[rng.integers(len(TRAIN_PAIRS))]
        th0 = rng.uniform(-2.5, 2.5)
        w0 = rng.uniform(-1.0, 1.0)
        th, om = simulate(th0, w0, g, A, steps)
        trajs.append((th, om, g, A))
    return trajs


class TrajDataset:
    def __init__(self, trajs, horizon=10):
        self.trajs = trajs
        self.horizon = horizon
        self.n = len(trajs)
        self.steps = len(trajs[0][0]) - horizon - 1

    def batch(self, bs):
        idx = np.random.randint(0, self.n, bs)
        starts = np.random.randint(0, self.steps, bs)
        th = np.stack([self.trajs[i][0][s:s + self.horizon + 1] for i, s in zip(idx, starts)])
        om = np.stack([self.trajs[i][1][s:s + self.horizon + 1] for i, s in zip(idx, starts)])
        g = torch.tensor([self.trajs[i][2] for i in idx], dtype=torch.float32)
        a = torch.tensor([self.trajs[i][3] for i in idx], dtype=torch.float32)
        return (torch.tensor(th, dtype=torch.float32),
                torch.tensor(om, dtype=torch.float32), g, a,
                torch.tensor(starts * DT, dtype=torch.float32))


def rollout_loss(step_fn, net, th, om, g, A, t0, horizon):
    s = torch.stack([th[:, 0], om[:, 0]], dim=-1)
    loss = 0.0
    for k in range(horizon):
        s = step_fn(net, s, t0 + k * DT, g, A)
        target = torch.stack([th[:, k + 1], om[:, k + 1]], dim=-1)
        loss = loss + ((torch.sin(s[:, 0]) - torch.sin(target[:, 0])) ** 2
                       + (torch.cos(s[:, 0]) - torch.cos(target[:, 0])) ** 2
                       + (s[:, 1] - target[:, 1]) ** 2).mean()
    return loss / horizon


def train_model(kind, ds, epochs=24, bs=64, horizon=10, seed=0):
    torch.manual_seed(seed)
    net = Net(out_dim=2 if kind == "blackbox" else 1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    step_fn = blackbox_step if kind == "blackbox" else structured_step
    hist = []
    for ep in range(epochs):
        run = 0.0
        for _ in range(120):
            th, om, g, A, t0 = ds.batch(bs)
            loss = rollout_loss(step_fn, net, th, om, g, A, t0, horizon)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        hist.append(run / 120)
        print(f"  {kind:10s} epoch {ep + 1:2d}  rollout loss {hist[-1]:.5f}")
    return net, hist


@torch.no_grad()
def rollout_model(net, kind, theta0, w0, gamma, A, steps=400):
    step_fn = blackbox_step if kind == "blackbox" else structured_step
    th = np.empty(steps + 1)
    om = np.empty(steps + 1)
    th[0], om[0] = theta0, w0
    s = torch.tensor([[theta0, w0]], dtype=torch.float32)
    g = torch.tensor([gamma], dtype=torch.float32)
    a = torch.tensor([A], dtype=torch.float32)
    for i in range(steps):
        s = step_fn(net, s, torch.tensor([i * DT]), g, a)
        th[i + 1], om[i + 1] = s[0, 0].item(), s[0, 1].item()
    return th, om


def main():
    set_seed(0)
    set_style()
    torch.set_num_threads(4)

    with timer("generate trajectories"):
        ds = TrajDataset(make_data(260, 300, seed=0), horizon=10)
    print(f"{ds.n} training trajectories from {len(TRAIN_PAIRS)} (A, gamma) pairs")

    nets, hists = {}, {}
    with timer("train both models"):
        for kind in ("blackbox", "structured"):
            nets[kind], hists[kind] = train_model(kind, ds)

    # ---- rollout accuracy across damping values ---------------------------
    rng = np.random.default_rng(7)
    ic = [(rng.uniform(-2.5, 2.5), rng.uniform(-1.0, 1.0)) for _ in range(24)]
    horizon_short, horizon = 100, 400
    results = {}
    for g in GAMMA_TEST:
        errs = {}
        for kind in ("blackbox", "structured"):
            e = []
            for th0, w0 in ic:
                th_t, om_t = simulate(th0, w0, g, A_TEST, horizon_short)
                th_p, om_p = rollout_model(nets[kind], kind, th0, w0, g, A_TEST, horizon_short)
                e.append(np.mean((np.sin(th_p) - np.sin(th_t)) ** 2
                                 + (np.cos(th_p) - np.cos(th_t)) ** 2
                                 + (om_p - om_t) ** 2))
            errs[kind] = float(np.mean(e))
        results[g] = errs
        print(f"gamma {g:.2f}: blackbox {errs['blackbox']:.4f}  "
              f"structured {errs['structured']:.4f}")

    # ---- error growth at a training gamma ---------------------------------
    g = 0.05
    curves = {"blackbox": [], "structured": []}
    for kind in curves:
        acc = []
        for th0, w0 in ic[:8]:
            th_t, om_t = simulate(th0, w0, g, A_TEST, horizon)
            th_p, om_p = rollout_model(nets[kind], kind, th0, w0, g, A_TEST, horizon)
            acc.append((np.sin(th_p) - np.sin(th_t)) ** 2
                       + (np.cos(th_p) - np.cos(th_t)) ** 2
                       + (om_p - om_t) ** 2)
        curves[kind] = np.mean(acc, axis=0)

    # ---- energy in the undriven, undamped case (in the training set) -------
    energy_curves = {"truth": [], "blackbox": [], "structured": []}
    for th0, w0 in ic[:8]:
        th_t, om_t = simulate(th0, w0, 0.0, 0.0, horizon)
        energy_curves["truth"].append(np.abs(energy(th_t, om_t) - energy(th0, w0)))
        for kind in ("blackbox", "structured"):
            th_p, om_p = rollout_model(nets[kind], kind, th0, w0, 0.0, 0.0, horizon)
            energy_curves[kind].append(np.abs(energy(th_p, om_p) - energy(th0, w0)))
    energy_mean = {k: np.mean(v, axis=0) for k, v in energy_curves.items()}

    # ---- sensitivity to initial conditions (true system only) -------------
    eps = 1e-3
    chaos_A, chaos_steps = 1.6, 800
    th_a, _ = simulate(0.3, 0.2, 0.05, chaos_A, chaos_steps)
    th_b, _ = simulate(0.3 + eps, 0.2, 0.05, chaos_A, chaos_steps)
    divergence = np.abs(th_a - th_b)
    log_div = np.log(divergence[50:] + 1e-16)
    slope = float(np.polyfit(np.arange(len(log_div)) * DT, log_div, 1)[0])

    save_metrics(
        "physics",
        {
            "train_pairs": TRAIN_PAIRS,
            "gamma_test": [str(g) for g in GAMMA_TEST],
            "rollout_mse_100": {str(g): {k: round(v, 5) for k, v in r.items()}
                                for g, r in results.items()},
            "train_loss": {k: [round(x, 5) for x in v] for k, v in hists.items()},
            "final_error_400": {k: float(v[-1]) for k, v in curves.items()},
            "mean_error_400": {k: float(np.mean(v)) for k, v in curves.items()},
            "max_energy_error": {k: float(np.max(v)) for k, v in energy_mean.items()},
            "energy_error_400": {k: float(v[-1]) for k, v in energy_mean.items()},
            "divergence_final": float(divergence[-1]),
            "lyapunov_per_time": round(slope, 3),
            "chaos_A": chaos_A,
            "eps": eps,
            "n_train_traj": ds.n,
            "dt": DT,
        },
    )

    # ---- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    th0, w0 = ic[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    th_t, _ = simulate(th0, w0, 0.05, A_TEST, 300)
    tt = np.arange(301) * DT
    ax.plot(tt, th_t, color=PAL["dark"], lw=1.3, label="numerical solver")
    for kind, col in (("blackbox", PAL["vermillion"]), ("structured", PAL["blue"])):
        th_p, _ = rollout_model(nets[kind], kind, th0, w0, 0.05, A_TEST, 300)
        ax.plot(tt, th_p, color=col, lw=1.1, label=kind)
    ax.set_xlabel("time")
    ax.set_ylabel(r"angle $\theta$ (rad)")
    ax.set_title("(a) one trajectory, gamma = 0.05")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    ax.semilogy(np.arange(horizon + 1) * DT, curves["blackbox"] + 1e-8,
                color=PAL["vermillion"], label="black box")
    ax.semilogy(np.arange(horizon + 1) * DT, curves["structured"] + 1e-8,
                color=PAL["blue"], label="structured (RK4)")
    ax.set_xlabel("rollout time")
    ax.set_ylabel("mean squared state error")
    ax.set_title("(b) error growth over 400 steps")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig-physics-rollout")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    x = np.arange(len(GAMMA_TEST))
    w = 0.36
    for j, (kind, col) in enumerate((("blackbox", PAL["vermillion"]),
                                     ("structured", PAL["blue"]))):
        vals = [results[g][kind] for g in GAMMA_TEST]
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=col, label=kind)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{g:.2f}" for g in GAMMA_TEST])
    ax.axvspan(1.5, 3.5, color=PAL["light"], alpha=0.6, lw=0)
    ax.text(2.5, ax.get_ylim()[1] * 0.35, "outside training range", ha="center",
            fontsize=7, color=PAL["grey"])
    ax.set_xlabel("damping $\\gamma$")
    ax.set_ylabel("100-step rollout MSE")
    ax.set_title("(a) extrapolation across a physical parameter")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    tt = np.arange(horizon + 1) * DT
    for key, col, lab in (("truth", PAL["dark"], "numerical solver"),
                          ("structured", PAL["blue"], "structured (RK4)"),
                          ("blackbox", PAL["vermillion"], "black box")):
        ax.semilogy(tt, energy_mean[key] + 1e-12, color=col, lw=1.2, label=lab)
    ax.set_xlabel("time")
    ax.set_ylabel("absolute energy error")
    ax.set_title("(b) energy at gamma = 0, no drive")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig-physics-generalize")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    tt = np.arange(chaos_steps + 1) * DT
    ax = axes[0]
    ax.plot(tt, th_a, color=PAL["dark"], lw=1.0, label="initial condition A")
    ax.plot(tt, th_b, color=PAL["vermillion"], lw=1.0, label="A + 0.001 rad")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.set_title("(a) the same law, two nearly identical starts")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    ax.semilogy(tt[1:], divergence[1:] + 1e-16, color=PAL["vermillion"], lw=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\Delta\theta|$ (rad)")
    ax.set_title(f"(b) separation grows, slope {slope:.2f} per unit time")
    fig.tight_layout()
    save(fig, "fig-physics-chaos")


if __name__ == "__main__":
    main()
