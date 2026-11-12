"""Study 5 — bifurcations, noise and cell fate in a regulatory network.

Two canonical motifs are used:

  * a symmetric toggle switch,  da/dt = beta/(1+b^n) - a,  with a pitchfork
    bifurcation at beta_c = x + x^(n+1), x = n^(-1/(n+1))  (n = 4 -> 1.013);
  * a three-variable Goodwin oscillator with a Hopf bifurcation in the Hill
    coefficient p.

The stochastic version is an exact Gillespie simulation of the toggle with
system size Omega.  Switching rates are measured on a (beta, Omega) grid.

Run:  python experiments/05_bifurcation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, save, set_style
from common.utils import save_metrics, set_seed

import matplotlib.pyplot as plt

N_HILL = 4.0
K = 1.0
BETA_C = (N_HILL ** (-1.0 / (N_HILL + 1))) + (N_HILL ** (-1.0 / (N_HILL + 1))) ** (N_HILL + 1)


def toggle_rhs(state, beta):
    a, b = state
    return np.array([beta / (1.0 + (b / K) ** N_HILL) - a,
                     beta / (1.0 + (a / K) ** N_HILL) - b])


def toggle_fixed_points(beta):
    pts = []
    for a0 in np.linspace(0.05, 3.0, 12):
        for b0 in np.linspace(0.05, 3.0, 12):
            sol = root(toggle_rhs, [a0, b0], args=(beta,), tol=1e-10)
            if sol.success and np.all(sol.x > 0):
                p = sol.x
                if not any(np.linalg.norm(p - q) < 1e-6 for q in pts):
                    pts.append(p)
    return np.array(pts)


def toggle_stability(p, beta):
    a, b = p
    g = beta * N_HILL * a ** (N_HILL - 1) / (1 + a ** N_HILL) ** 2
    h = beta * N_HILL * b ** (N_HILL - 1) / (1 + b ** N_HILL) ** 2
    J = np.array([[-1.0, -g], [-h, -1.0]])
    return np.max(np.real(np.linalg.eigvals(J)))


def goodwin(z, k=(1.0, 0.4, 1.0, 0.4, 1.0, 0.4), p=8.0):
    x, y, zz = z
    k1, k2, k3, k4, k5, k6 = k
    return np.array([k1 / (1.0 + zz ** p) - k2 * x,
                     k3 * x - k4 * y,
                     k5 * y - k6 * zz])


def goodwin_amplitude(p, t_end=400.0, dt=0.01):
    z = np.array([0.5, 0.5, 0.5])
    n = int(t_end / dt)
    zs = np.empty(n + 1)
    zs[0] = z[2]
    for i in range(n):
        k1 = goodwin(z, p=p)
        k2 = goodwin(z + 0.5 * dt * k1, p=p)
        k3 = goodwin(z + 0.5 * dt * k2, p=p)
        k4 = goodwin(z + dt * k3, p=p)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        zs[i + 1] = z[2]
    tail = zs[n // 2:]
    return float(tail.max() - tail.min())


def gillespie_toggle(beta, omega, t_end, rng, max_events=2_000_000):
    """Exact simulation; returns (times, x, y) at event times."""
    x = y = int(round(omega * 0.8))
    t = 0.0
    ts = [0.0]
    xs = [x]
    ys = [y]
    scale = K * omega
    for _ in range(max_events):
        bx = omega * beta / (1.0 + (y / scale) ** N_HILL)
        by = omega * beta / (1.0 + (x / scale) ** N_HILL)
        props = np.array([bx, x, by, y])
        total = props.sum()
        if total <= 0:
            break
        t += rng.exponential(1.0 / total)
        if t > t_end:
            break
        r = rng.random() * total
        if r < props[0]:
            x += 1
        elif r < props[0] + props[1]:
            x -= 1
        elif r < props[0] + props[1] + props[2]:
            y += 1
        else:
            y -= 1
        ts.append(t)
        xs.append(x)
        ys.append(y)
    return np.array(ts), np.array(xs), np.array(ys)


def switch_rate(beta, omega, t_end, n_runs, seed=0):
    """Committed transitions between wells, after smoothing.

    A transition is counted only when the smoothed separation crosses the
    threshold and stays on the other side for at least 5 time units, which
    removes the rapid sign flicker of a monostable well.
    """
    rng = np.random.default_rng(seed)
    switches = 0
    total_t = 0.0
    thr = 0.6 * omega
    for _ in range(n_runs):
        ts, xs, ys = gillespie_toggle(beta, omega, t_end, rng)
        if len(ts) < 10:
            continue
        grid = np.arange(0.0, ts[-1], 0.5)
        d = np.interp(grid, ts, (xs - ys).astype(float))
        k = 9
        ds = np.convolve(d, np.ones(k) / k, mode="same")
        lab = np.where(ds > thr, 1, np.where(ds < -thr, -1, 0))
        idx = np.where(lab != 0, np.arange(len(lab)), 0)
        idx = np.maximum.accumulate(idx)
        lab = lab[idx]
        # keep only runs that last at least 10 samples (5 time units)
        change = np.flatnonzero(np.diff(lab) != 0) + 1
        bounds = np.concatenate([[0], change, [len(lab)]])
        committed = [lab[bounds[i]] for i in range(len(bounds) - 1)
                     if bounds[i + 1] - bounds[i] >= 10]
        switches += max(len(committed) - 1, 0)
        total_t += ts[-1]
    return switches / max(total_t, 1e-9)


def main() -> None:
    set_seed(0)
    set_style()

    print(f"beta_c = {BETA_C:.4f}")
    betas = np.linspace(0.15, 3.0, 60)
    stable_hi, stable_lo, unstable = [], [], []
    for beta in betas:
        pts = toggle_fixed_points(beta)
        for p in pts:
            ev = toggle_stability(p, beta)
            if ev < -1e-6:
                if p[0] > p[1]:
                    stable_hi.append((beta, p[0]))
                else:
                    stable_lo.append((beta, p[0]))
            else:
                unstable.append((beta, p[0]))

    ps = np.arange(2, 13, 1)
    amps = [goodwin_amplitude(p) for p in ps]

    # ---- stochastic switching ---------------------------------------------
    rng = np.random.default_rng(3)
    traj = {}
    for beta in (0.9 * BETA_C, 1.08 * BETA_C):
        traj[beta] = gillespie_toggle(beta, 160, 1200.0, rng)

    grid_beta = np.array([0.8, 1.0, 1.2, 1.5]) * BETA_C
    grid_omega = np.array([40.0, 80.0, 160.0, 320.0])
    rates = np.zeros((len(grid_beta), len(grid_omega)))
    for i, b in enumerate(grid_beta):
        for j, om in enumerate(grid_omega):
            rates[i, j] = switch_rate(b, om, 200.0, 4, seed=100 + 10 * i + j)
        print(f"  beta/beta_c = {b / BETA_C:.1f}: "
              + "  ".join(f"{r:.2e}" for r in rates[i]))

    # ---- figure: bifurcations ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
    ax = axes[0]
    hi = np.array(stable_hi)
    lo = np.array(stable_lo)
    un = np.array(unstable)
    ax.plot(hi[:, 0], hi[:, 1], ".", ms=3.5, color=PAL["blue"],
            label="stable")
    ax.plot(lo[:, 0], lo[:, 1], ".", ms=3.5, color=PAL["blue"])
    ax.plot(un[:, 0], un[:, 1], ".", ms=2.6, color=PAL["vermillion"], alpha=0.7,
            label="unstable")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.axvline(BETA_C, color=PAL["grey"], lw=0.7, ls="--")
    ax.text(BETA_C * 1.03, 0.12, f"$\\beta_c$ = {BETA_C:.3f}", fontsize=6.8,
            color=PAL["grey"])
    ax.set_xlabel("production rate $\\beta$")
    ax.set_ylabel("steady-state $a$")
    ax.set_title("(a) toggle switch: pitchfork")
    ax.set_xlim(0, 3)

    ax = axes[1]
    ax.plot(ps, amps, "o-", color=PAL["green"], ms=4)
    ax.set_xlabel("Hill coefficient $p$ (Goodwin loop)")
    ax.set_ylabel("peak-to-peak amplitude of $z$")
    ax.set_title("(b) negative feedback: Hopf onset")
    fig.tight_layout()
    save(fig, "fig-bifurcation")

    # ---- figure: noise ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    for k, beta in enumerate((0.9 * BETA_C, 1.08 * BETA_C)):
        ts, xs, ys = traj[beta]
        ax.plot(ts, xs, color=[PAL["sky"], PAL["blue"]][k], lw=0.6,
                label=f"$\\beta/\\beta_c$={beta / BETA_C:.2f}, $x$")
        ax.plot(ts, ys, color=[PAL["orange"], PAL["vermillion"]][k], lw=0.6,
                ls=":", label=f"$\\beta/\\beta_c$={beta / BETA_C:.2f}, $y$")
    ax.set_xlabel("time")
    ax.set_ylabel("molecule count ($\\Omega=160$)")
    ax.set_title("(a) noise flips the switch")
    ax.legend(fontsize=5.6, frameon=False, ncol=2)
    ax.set_xlim(0, 1200)

    ax = axes[1]
    for beta, c in zip([0.85, 1.0, 1.2], [PAL["sky"], PAL["green"], PAL["vermillion"]]):
        ts, xs, ys = gillespie_toggle(beta * BETA_C, 40, 800.0, rng)
        ax.hist(xs - ys, bins=np.arange(-70, 72, 4), density=True, alpha=0.55,
                color=c, label=f"$\\beta/\\beta_c$={beta:.2f}")
    ax.set_xlabel("$x-y$")
    ax.set_ylabel("density")
    ax.set_title("(b) one well or two")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    im = ax.imshow(np.log10(np.clip(rates, 1e-6, None)), cmap=SEQUENTIAL,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(grid_omega)), [f"{int(v)}" for v in grid_omega])
    ax.set_yticks(range(len(grid_beta)),
                  [f"{b / BETA_C:.1f}" for b in grid_beta])
    ax.set_xlabel("system size $\\Omega$")
    ax.set_ylabel("$\\beta/\\beta_c$")
    ax.set_title("(c) switching rate, log$_{10}$")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("per unit time")
    fig.tight_layout()
    save(fig, "fig-noise")

    save_metrics("bifurcation", {
        "beta_c": round(float(BETA_C), 4),
        "bifurcation_beta": betas.tolist(),
        "stable_branch_high": [[round(float(b), 4), round(float(a), 4)]
                               for b, a in stable_hi],
        "goodwin_p": ps.tolist(),
        "goodwin_amplitude": [round(a, 4) for a in amps],
        "switch_grid_beta_over_bc": (grid_beta / BETA_C).round(3).tolist(),
        "switch_grid_omega": grid_omega.tolist(),
        "switch_rate": rates.round(6).tolist(),
    })


if __name__ == "__main__":
    main()
