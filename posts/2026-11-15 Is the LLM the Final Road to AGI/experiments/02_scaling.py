"""Article 1, study 2 — what scaling laws do and do not imply.

Uses the Chinchilla parametric fit  L(N, D) = E + A/N^alpha + B/D^beta
(Hoffmann et al. 2022, Table A3: E=1.69, A=406.4, B=410.7, alpha=0.34,
beta=0.28) and the accounting identity C = 6ND.  Four figures:

  fig08  isoFLOP slices: loss versus parameters at fixed compute
  fig09  compute-optimal N*, D* and the Kaplan vs Chinchilla exponents
  fig10  marginal loss per additional 10x compute, with a data wall
  fig11  total-cost Pareto front when inference is counted

Everything here is a *model*, not a measurement of any particular run.
Run:  python experiments/scaling_laws/02_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
FLOPS_PER_PARAM_TOKEN = 6.0

# Kaplan et al. 2020 fit, expressed as power laws in compute alone.
KAPLAN_N_EXP, KAPLAN_D_EXP, KAPLAN_L_EXP = 0.73, 0.27, -0.050


def loss(N, D, E=E, A=A, B=B, alpha=ALPHA, beta=BETA):
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def optimal(N_or_C):
    """Chinchilla allocation: minimise loss subject to 6ND = C."""
    exp_n = BETA / (ALPHA + BETA)
    G = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    C = np.asarray(N_or_C, float)
    n_star = G * (C / FLOPS_PER_PARAM_TOKEN) ** exp_n
    return n_star, C / (FLOPS_PER_PARAM_TOKEN * n_star)


def main() -> None:
    set_seed(0)
    set_style()

    # ---------- fig08: isoFLOP slices ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    Cs = [1e21, 1e23, 1e25]
    for C, c in zip(Cs, [PAL["sky"], PAL["teal"], PAL["navy"]]):
        N = np.logspace(7, 12.5, 300)
        L = loss(N, C / (FLOPS_PER_PARAM_TOKEN * N))
        ax.plot(N, L, color=c, lw=1.2, label=f"C = 10^{int(np.log10(C))} FLOP")
        n_opt = optimal(C)[0]
        ax.plot([n_opt], [loss(n_opt, C / (FLOPS_PER_PARAM_TOKEN * n_opt))], "o",
                ms=4, color=PAL["orange"], zorder=4,
                label="Chinchilla optimum" if C == Cs[0] else None)
    ax.set_xscale("log")
    ax.set_xlabel("parameters N")
    ax.set_ylabel("predicted loss (nats/token)")
    ax.set_ylim(1.6, 3.4)
    ax.set_title("(a)  At fixed compute, loss is U-shaped in model size", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    ax.grid(True, which="major")

    ax = axes[1]
    N = np.logspace(7, 12.5, 300)
    for ratio, c, lab in [(1, PAL["navy"], "20 tokens/param (heuristic)"),
                          (0.2, PAL["orange"], "4 tokens/param"),
                          (20, PAL["teal"], "400 tokens/param")]:
        D = ratio * N
        C = FLOPS_PER_PARAM_TOKEN * N * D
        ax.plot(C, loss(N, D), color=c, lw=1.2, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("training compute C (FLOP)")
    ax.set_ylabel("predicted loss")
    ax.set_ylim(1.6, 4.0)
    ax.set_xlim(1e19, 1e28)
    ax.set_title("(b)  Loss falls as a power law until the floor dominates", loc="left")
    ax.legend(frameon=False, fontsize=6.4)
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "fig08_scaling", 1)

    # ---------- fig09: allocation exponents ----------
    C = np.logspace(18, 28, 200)
    n_star, d_star = optimal(C)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax = axes[0]
    ax.plot(C, n_star, color=PAL["navy"], lw=1.3, label="Chinchilla N*")
    ax.plot(C, d_star, color=PAL["teal"], lw=1.3, label="Chinchilla D*")
    # Kaplan allocation: N ~ C^0.73, D ~ C^0.27, anchored at 1e21 FLOP
    anchor_N = n_star[np.argmin(np.abs(C - 1e21))]
    ax.plot(C, anchor_N * (C / 1e21) ** KAPLAN_N_EXP, color=PAL["orange"],
            ls="--", lw=1.0, label="Kaplan-style N ~ C^0.73")
    ax.plot(C, anchor_N * 20 * (C / 1e21) ** KAPLAN_D_EXP, color=PAL["gold"],
            ls=":", lw=1.0, label="Kaplan-style D ~ C^0.27")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training compute (FLOP)")
    ax.set_ylabel("count (log)")
    ax.set_title("(a)  Allocation exponents differ between recipes", loc="left")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.grid(True, which="major")

    ax = axes[1]
    L_star = loss(n_star, d_star)
    excess = L_star - E
    ax.loglog(C, excess, color=PAL["navy"], lw=1.3)
    slope = np.polyfit(np.log10(C), np.log10(excess), 1)[0]
    ref = excess[0] * (C / C[0]) ** (-ALPHA * BETA / (ALPHA + BETA))
    ax.loglog(C, ref, color=PAL["orange"], lw=1.0, ls="--",
              label=f"analytic slope −αβ/(α+β) = {-ALPHA * BETA / (ALPHA + BETA):.3f}\n"
                    f"fitted slope {slope:.3f}")
    ax.set_xlabel("training compute (FLOP)")
    ax.set_ylabel("loss above the irreducible floor (log)")
    ax.set_title("(b)  The reducible part is a clean power law", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "fig09_allocation", 1)

    # ---------- fig10: marginal returns and a data wall ----------
    C = np.logspace(19, 27, 80)
    L_unc = loss(*optimal(C))
    # data wall: total unique high-quality tokens saturates at 3e13 (~ Chinchilla
    # scale for a 1.5e12 model, with 20 tokens/param).
    D_MAX = 3e13
    n_opt, d_opt = optimal(C)
    D_eff = np.minimum(d_opt, D_MAX)
    L_wall = loss(n_opt, D_eff)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax = axes[0]
    ax.plot(C, L_unc, color=PAL["navy"], lw=1.3, label="unconstrained data")
    ax.plot(C, L_wall, color=PAL["orange"], lw=1.3, ls="--",
            label=f"data capped at {D_MAX:.0e} tokens")
    ax.set_xscale("log")
    ax.set_xlabel("training compute (FLOP)")
    ax.set_ylabel("loss")
    ax.set_title("(a)  A data wall bends the frontier", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    ax.grid(True, which="major")

    ax = axes[1]
    gain_unc = -(np.diff(L_unc))
    gain_wall = -(np.diff(L_wall))
    mid = np.sqrt(C[1:] * C[:-1])
    ax.plot(mid, gain_unc, color=PAL["navy"], lw=1.3, label="unconstrained")
    ax.plot(mid, gain_wall, color=PAL["orange"], lw=1.3, ls="--", label="data-capped")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training compute (FLOP)")
    ax.set_ylabel("loss reduction per 10× compute")
    ax.set_title("(b)  Diminishing returns are arithmetic, not a surprise", loc="left")
    ax.legend(frameon=False, fontsize=6.4)
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "fig10_marginal", 1)

    # ---------- fig11: training + inference total cost ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    target = 2.0  # nats/token
    Ns = np.logspace(8, 13, 220)
    for queries, c in [(1e9, PAL["sky"]), (1e11, PAL["teal"]), (1e13, PAL["navy"])]:
        need = (B / (target - E - A / Ns ** ALPHA)) ** (1 / BETA)
        total = FLOPS_PER_PARAM_TOKEN * Ns * need + 2 * Ns * queries
        total = np.where(np.isfinite(total), total, np.nan)
        ax.plot(Ns, total, color=c, lw=1.4, label=f"{queries:.0e} inference queries")
        k = np.nanargmin(total)
        ax.plot([Ns[k]], [total[k]], "o", ms=4, color=PAL["orange"], zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters N")
    ax.set_ylabel("total compute over model lifetime (FLOP)")
    ax.set_title("Cost-optimal size shrinks when inference is counted", loc="left")
    ax.legend(frameon=False, fontsize=6.6)
    ax.grid(True, which="major")
    save(fig, "fig11_pareto", 1)

    # ---------- metrics ----------
    n21, d21 = optimal(1e21)
    n25, d25 = optimal(1e25)
    save_metrics("scaling", {
        "constants": {"E": E, "A": A, "B": B, "alpha": ALPHA, "beta": BETA},
        "tokens_per_param_optimal": 20.0,
        "tokens_per_param_at_1e23": float(optimal(1e23)[1] / optimal(1e23)[0]),
        "allocation_note": "the parametric fit implies 50-120 tokens/param, "
                           "more data-heavy than the 20x rule of thumb",
        "n_star_1e21": float(n21), "d_star_1e21": float(d21),
        "n_star_1e25": float(n25), "d_star_1e25": float(d25),
        "loss_at_1e21": float(loss(n21, d21)),
        "loss_at_1e25": float(loss(n25, d25)),
        "loss_reduction_1e21_to_1e25": float(loss(n21, d21) - loss(n25, d25)),
        "data_wall_tokens": D_MAX,
        "loss_gap_at_1e27_from_wall": float(loss(*optimal(1e27)) - loss(optimal(1e27)[0], D_MAX)),
    })


if __name__ == "__main__":
    main()
