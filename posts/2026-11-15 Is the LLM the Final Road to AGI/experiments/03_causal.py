"""Article 1, study 3 — does next-token prediction identify causal structure?

A matched-pair simulation.  Two environments produce *observationally very
similar* time series of two visible variables (x, y), but they differ in the
hidden mechanism that links them:

  environment A (direct cause)   x -> y, no confounder
  environment B (confounder)     z -> x and z -> y, no x -> y edge

The confounder z is never observed.  The parameters of B are fitted so that
the autocorrelation of x, the autocorrelation of y and the cross-correlation of
x and y agree with A up to lag 2.  The residual mismatch is reported.

Learners of the *same* model class (linear regression of y_t on
[y_{t-1}, x_{t-1}]) see the same number of samples:

  passive    20 000 observational samples
  active     20 000 samples, half from randomised interventions on x

Metrics: in-distribution R^2, R^2 after an unseen intervention, recovery of
the x -> y coefficient, adaptation with k fresh interventional samples, and
control regret in a third environment C where the true effect is negative but
confounding makes the observational association positive.

Run:  python experiments/causal_world_models/03_causal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

# environment A
A_XX, A_XY, A_YY = 0.80, 0.60, 0.50
S_X, S_Y = 0.45, 0.45

# environment B (re-fitted at run time, these are the starting values)
B_RHO, B_P, B_Q, B_SX, B_SY = 0.88, 1.00, 0.55, 0.55, 0.45

# environment C: x really lowers y, but a confounder reverses the sign
C_C, C_P, C_Q, C_RHO = -0.60, 1.0, 1.8, 0.90

SEEDS = range(20)
ADAPT_K = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]


def step_A(x_prev, y_prev, rng):
    x = A_XX * x_prev + S_X * rng.normal()
    y = A_XY * x + A_YY * y_prev + S_Y * rng.normal()
    return x, y


def simulate_B(n, rng, params, intervene_every=0):
    rho, p, q, sx, sy = params
    xs, ys = np.empty(n), np.empty(n)
    z = y = 0.0
    burn = 200
    for i in range(n + burn):
        z = rho * z + 0.35 * rng.normal()
        do = intervene_every and (i % intervene_every == 0) and i >= burn
        x = rng.uniform(-3, 3) if do else p * z + sx * rng.normal()
        y = q * z + A_YY * y + sy * rng.normal()
        if i >= burn:
            xs[i - burn], ys[i - burn] = x, y
    return xs, ys


def simulate_A(n, rng):
    xs, ys = np.empty(n), np.empty(n)
    x = y = 0.0
    burn = 200
    for i in range(n + burn):
        x, y = step_A(x, y, rng)
        if i >= burn:
            xs[i - burn], ys[i - burn] = x, y
    return xs, ys


def simulate_C(n, rng, intervene_every=0):
    xs, ys = np.empty(n), np.empty(n)
    x = y = z = 0.0
    burn = 200
    for i in range(n + burn):
        z = C_RHO * z + 0.30 * rng.normal()
        do = intervene_every and (i % intervene_every == 0) and i >= burn
        x = rng.uniform(-3, 3) if do else C_P * z + 0.5 * rng.normal()
        y = C_C * x + C_Q * z + A_YY * y + 0.45 * rng.normal()
        if i >= burn:
            xs[i - burn], ys[i - burn] = x, y
    return xs, ys


def stats(xs, ys, lags=2):
    x = (xs - xs.mean()) / xs.std()
    y = (ys - ys.mean()) / ys.std()
    out = [np.mean(x[:-k] * x[k:]) if k else 1.0 for k in range(lags + 1)]
    out += [np.mean(y[:-k] * y[k:]) if k else 1.0 for k in range(lags + 1)]
    out += [np.mean(x[:-k] * y[k:]) if k else np.mean(x * y) for k in range(lags + 1)]
    return np.array(out)


def design(xs, ys):
    """Predict y_t from y_{t-1} and x_t (the action available at time t)."""
    X = np.column_stack([np.ones(len(xs) - 1), ys[:-1], xs[1:]])
    return X, ys[1:]


def fit(xs, ys):
    X, target = design(xs, ys)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return beta


def r2_of(beta, xs, ys):
    X, target = design(xs, ys)
    pred = X @ beta
    return 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def main() -> None:
    set_seed(1)
    set_style()

    # ---- fit B so that its observed statistics match A ----
    rng = np.random.default_rng(7)
    xa_ref, ya_ref = simulate_A(120_000, rng)
    target = stats(xa_ref, ya_ref)

    def residual(theta):
        params = (theta[0], theta[1], theta[2], abs(theta[3]), abs(theta[4]))
        r = np.random.default_rng(11)
        xb, yb = simulate_B(60_000, r, params)
        return stats(xb, yb) - target

    sol = least_squares(residual, x0=np.array([B_RHO, B_P, B_Q, B_SX, B_SY]),
                        bounds=([0.5, 0.3, 0.1, 0.1, 0.1],
                                [0.99, 2.0, 1.5, 2.0, 1.5]), max_nfev=400)
    B = (sol.x[0], sol.x[1], sol.x[2], abs(sol.x[3]), abs(sol.x[4]))

    rng = np.random.default_rng(0)
    xa, ya = simulate_A(200_000, rng)
    xb, yb = simulate_B(200_000, rng, B)
    sa, sb = stats(xa, ya), stats(xb, yb)
    mismatch = float(np.max(np.abs(sa - sb)))

    # ---------- fig12: design and matched statistics ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for ax, title, edges in [
        (axes[0], "(a)  Environment A: direct cause", [("x", "y", True)]),
        (axes[1], "(b)  Environment B: hidden confounder",
         [("z", "x", True), ("z", "y", True), ("x", "y", False)]),
    ]:
        pos = {"x": (0.15, 0.25), "y": (0.85, 0.25), "z": (0.5, 0.8)}
        for name in (["x", "y"] if len(edges) == 1 else ["x", "y", "z"]):
            px, py = pos[name]
            ax.add_patch(plt.Circle((px, py), 0.10, color=PAL["sky"], ec=PAL["navy"],
                                    lw=1.0, zorder=2))
            ax.text(px, py, name, ha="center", va="center", fontsize=10)
        for u, v, real in edges:
            ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>",
                                         mutation_scale=12, lw=1.4 if real else 1.1,
                                         color=PAL["navy"] if real else PAL["orange"],
                                         ls="-" if real else "--",
                                         shrinkA=14, shrinkB=14, zorder=1))
        if len(edges) > 1:
            ax.text(0.5, 0.02, "solid: causal edge   dashed: induced association",
                    ha="center", fontsize=6.2, color=PAL["grey"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.5)

    ax = axes[2]
    labels = ["ACF x(1)", "ACF x(2)", "ACF y(1)", "ACF y(2)",
              "CCF(0)", "CCF(1)", "CCF(2)"]
    idx = [1, 2, 4, 5, 6, 7, 8]
    xx = np.arange(len(idx))
    ax.bar(xx - 0.19, sa[idx], width=0.36, color=PAL["navy"], label="A (direct)")
    ax.bar(xx + 0.19, sb[idx], width=0.36, color=PAL["orange"], label="B (confounded)")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=5.8, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.axhline(0, color=PAL["dark"], lw=0.6)
    ax.set_ylabel("correlation")
    ax.set_title(f"(c)  Observed statistics agree to {mismatch:.3f}", loc="left",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower center")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig12_matched_envs", 1)

    # ---------- main experiment ----------
    n_obs, n_active = 20_000, 20_000
    res = {"A_passive": [], "B_passive": [], "B_active": [], "B_passive10x": []}
    coefB = {"passive": [], "active": [], "passive10x": []}
    coefC = {"passive": [], "active": []}
    adapt = {"passive": [], "active": []}

    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        xa_tr, ya_tr = simulate_A(n_obs, rng)
        xb_tr, yb_tr = simulate_B(n_obs, rng, B)
        xb_i, yb_i = simulate_B(n_active, rng, B, intervene_every=1)
        xb_big, yb_big = simulate_B(10 * n_obs, rng, B)

        betas = {
            "A_passive": fit(xa_tr, ya_tr),
            "B_passive": fit(xb_tr, yb_tr),
            "B_active": fit(np.concatenate([xb_i, xb_tr[:2000]]),
                            np.concatenate([yb_i, yb_tr[:2000]])),
            "B_passive10x": fit(xb_big, yb_big),
        }
        coefB["passive"].append(betas["B_passive"][2])
        coefB["active"].append(betas["B_active"][2])
        coefB["passive10x"].append(betas["B_passive10x"][2])

        xv, yv = simulate_B(4000, rng, B, intervene_every=2)
        xb_obs, yb_obs = simulate_B(4000, rng, B)
        xa_te, ya_te = simulate_A(4000, rng)
        Xs, ts = design(xv, yv)
        ss_tot = np.sum((ts - ts.mean()) ** 2)
        for k, b in betas.items():
            if k == "A_passive":
                r2_in, r2_shift = r2_of(b, xa_te, ya_te), r2_of(b, xa_te, ya_te)
            else:
                r2_in = r2_of(b, xb_obs, yb_obs)
                r2_shift = 1 - np.sum((Xs @ b - ts) ** 2) / ss_tot
            res[k].append((r2_in, r2_shift))

        # adaptation with k fresh interventional samples
        for name in ("passive", "active"):
            curve = []
            for k in ADAPT_K:
                xi, yi = simulate_B(max(k, 200), rng, B, intervene_every=1)
                if name == "passive":
                    xs2, ys2 = np.concatenate([xb_tr[:2000], xi]), np.concatenate([yb_tr[:2000], yi])
                else:
                    xs2, ys2 = np.concatenate([xb_i[:1000], xi]), np.concatenate([yb_i[:1000], yi])
                bb = fit(xs2, ys2)
                Xa, ta = design(xv, yv)
                curve.append(float(1 - np.sum((Xa @ bb - ta) ** 2)
                                   / np.sum((ta - ta.mean()) ** 2)))
            adapt[name].append(curve)

        # environment C: coefficient recovery under a sign reversal
        xc_tr, yc_tr = simulate_C(8000, rng)
        xc_i, yc_i = simulate_C(8000, rng, intervene_every=1)
        coefC["passive"].append(fit(xc_tr, yc_tr)[2])
        coefC["active"].append(fit(np.concatenate([xc_i, xc_tr[:1000]]),
                                   np.concatenate([yc_i, yc_tr[:1000]]))[2])

    # ---------- fig13: fit, shift, and coefficient recovery ----------
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9))
    order = ["A_passive", "B_passive", "B_active", "B_passive10x"]
    labels = ["A\npassive", "B\npassive", "B\nactive", "B\npassive\n10× data"]
    cols = [PAL["navy"], PAL["orange"], PAL["teal"], PAL["gold"]]

    ax = axes[0]
    vals = [np.mean([r[0] for r in res[k]]) for k in order]
    errs = [np.std([r[0] for r in res[k]]) for k in order]
    ax.bar(range(4), vals, yerr=errs, color=cols, capsize=3, width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel(r"in-distribution $R^2$")
    ax.set_ylim(0, 1)
    ax.set_title("(a)  All learners fit the data", loc="left", fontsize=8.5)
    ax.grid(axis="y")

    ax = axes[1]
    vals = [np.mean([r[1] for r in res[k]]) for k in order]
    errs = [np.std([r[1] for r in res[k]]) for k in order]
    ax.bar(range(4), vals, yerr=errs, color=cols, capsize=3, width=0.6)
    ax.axhline(0, color=PAL["dark"], lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel(r"$R^2$ after an unseen intervention")
    ax.set_title("(b)  Only the confounded passive model breaks", loc="left",
                 fontsize=8.5)
    ax.grid(axis="y")

    ax = axes[2]
    rngj = np.random.default_rng(5)
    for k, (name, c) in enumerate([("passive", PAL["orange"]), ("active", PAL["teal"]),
                                   ("passive10x", PAL["gold"])]):
        ax.scatter(np.full(len(coefB[name]), k) + rngj.normal(0, 0.04, len(coefB[name])),
                   coefB[name], s=14, color=c, alpha=0.85)
    ax.axhline(0, color=PAL["dark"], lw=1.0, ls="--", label="true effect in B = 0")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["passive", "active", "passive\n10× data"], fontsize=6.6)
    ax.set_ylabel(r"estimated coefficient on $x_{t-1}$")
    ax.set_title("(c)  Confounding is a bias, not noise", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig13_learning", 1)

    # ---------- fig14: adaptation and control ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    kk = np.array(ADAPT_K)
    for name, c in [("passive", PAL["orange"]), ("active", PAL["teal"])]:
        arr = np.array(adapt[name])
        m = arr.mean(0)
        lo, hi = np.quantile(arr, 0.025, 0), np.quantile(arr, 0.975, 0)
        ax.plot(kk, m, color=c, lw=1.3, label=f"{name} initial fit")
        ax.fill_between(kk, lo, hi, color=c, alpha=0.18, lw=0)
    ax.set_xscale("symlog", linthresh=25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("fresh interventional samples")
    ax.set_ylabel(r"$R^2$ under intervention")
    ax.set_title("(a)  The right data, not more data", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")
    ax.grid(True, which="major")

    ax = axes[1]
    vals = [np.mean(coefB["passive"]), np.mean(coefB["active"]),
            np.mean(coefC["passive"]), np.mean(coefC["active"])]
    errs = [np.std(coefB["passive"]), np.std(coefB["active"]),
            np.std(coefC["passive"]), np.std(coefC["active"])]
    ax.bar(range(4), vals, yerr=errs, capsize=3, width=0.55,
           color=[PAL["orange"], PAL["teal"], PAL["orange"], PAL["teal"]])
    ax.axhline(0, color=PAL["dark"], lw=0.9, ls="--")
    ax.axhline(C_C, color=PAL["navy"], lw=0.9, ls=":")
    ax.annotate("truth in B = 0", (3.35, 0), xytext=(0, 4), textcoords="offset points",
                fontsize=5.8, ha="right", color=PAL["dark"])
    ax.annotate("truth in C = −0.6", (3.35, C_C), xytext=(0, -9),
                textcoords="offset points", fontsize=5.8, ha="right", color=PAL["navy"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["B\npassive", "B\nactive", "C\npassive", "C\nactive"],
                       fontsize=6.4)
    ax.set_ylabel(r"predicted effect of $x \uparrow$ on $y$")
    ax.set_title("(b)  Counterfactual effect, not correlation", loc="left",
                 fontsize=8.5)
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "fig14_recovery", 1)

    save_metrics("causal_world", {
        "B_params": {"rho": float(B[0]), "p": float(B[1]), "q": float(B[2]),
                     "sx": float(B[3]), "sy": float(B[4])},
        "max_stat_mismatch": mismatch,
        "stats_A": [float(v) for v in sa],
        "stats_B": [float(v) for v in sb],
        "r2_indist": {k: float(np.mean([r[0] for r in res[k]])) for k in order},
        "r2_shift": {k: float(np.mean([r[1] for r in res[k]])) for k in order},
        "coef_x": {k: float(np.mean(v)) for k, v in coefB.items()},
        "coef_x_sd": {k: float(np.std(v)) for k, v in coefB.items()},
        "coef_x_envC_true": C_C,
        "coef_x_envC_passive": float(np.mean(coefC["passive"])),
        "coef_x_envC_active": float(np.mean(coefC["active"])),
        "adapt_k": ADAPT_K,
        "adapt_r2": {k: [float(x) for x in np.array(v).mean(0)]
                     for k, v in adapt.items()},
        "n_seeds": len(list(SEEDS)),
    })


if __name__ == "__main__":
    main()
