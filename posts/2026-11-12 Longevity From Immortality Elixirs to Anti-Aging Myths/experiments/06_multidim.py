"""Study 6 — rejuvenation as a multi-objective control problem.

Eight coupled damage variables (telomeres, epigenome, proteostasis,
mitochondria, senescence, stem-cell pool, inflammation, extracellular matrix)
follow

    dx_i/dt = a_i (1 - x_i) - b_i u_i x_i
              + c * sum_j A_ij x_j (1 - x_i),

with u_i in [0, 1] the intensity of a repair intervention.  The composite
score S = 1 - mean(x) is an arbitrary index, used only to compare strategies
inside this model; it is not a biological age.

Strategies: A telomere-only, B epigenetic-only, C mitochondrial-only,
D four-way, E four-way plus a senolytic with a proliferative cost.  The run
starts at age 40, treats for 20 years, then stops, so the panel also shows
relapse.  A synergy scan over (u_tel, u_epi) asks whether the two repairs
interact.

Run:  python experiments/06_multidim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, DIVERGING, save, set_style
from common.utils import save_metrics, set_seed

NAMES = ["telomere", "epigenetic", "proteostasis", "mitochondria",
         "senescence", "stem cells", "inflammation", "ECM"]
A_RATE = np.array([0.012, 0.008, 0.015, 0.014, 0.010, 0.009, 0.011, 0.006])
B_RATE = np.array([0.150, 0.120, 0.140, 0.160, 0.130, 0.100, 0.130, 0.080])
COUPLING = 0.050

# coupling: row i is affected by column j
A_MAT = np.zeros((8, 8))
A_MAT[0, 3] = 1.0          # mito damage -> telomere attrition
A_MAT[1, 4] = 1.0          # senescence -> epigenetic drift
A_MAT[4, 3] = 1.0          # mito -> senescence
A_MAT[4, 0] = 0.6          # telomere -> senescence
A_MAT[6, 4] = 1.2          # senescence -> inflammation
A_MAT[6, 7] = 0.6          # ECM -> inflammation
A_MAT[5, 6] = 1.0          # inflammation -> stem cells
A_MAT[5, 4] = 0.8          # senescence -> stem cells
A_MAT[7, 6] = 0.5          # inflammation -> ECM

STRATEGIES = {
    "control": np.zeros(8),
    "A telomere": np.array([0.8, 0, 0, 0, 0, 0, 0, 0]),
    "B epigenetic": np.array([0, 0.8, 0, 0, 0, 0, 0, 0]),
    "C mitochondrial": np.array([0, 0, 0, 0.8, 0, 0, 0, 0]),
    "D four-way": np.array([0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0]),
    "E four-way + senolytic": np.array([0.5, 0.5, 0.5, 0.5, 0.6, 0, -0.15, -0.2]),
}

TREAT_START, TREAT_END, T_END = 40.0, 60.0, 100.0


def rhs(t, x, u):
    x = np.clip(x, 0, 1)
    repair = B_RATE * u * x
    spread = COUPLING * (A_MAT @ x) * (1 - x)
    return A_RATE * (1 - x) - repair + spread


def simulate(u, t_end=T_END):
    def f(t, x):
        uu = u if TREAT_START <= t <= TREAT_END else np.zeros(8)
        return rhs(t, x, uu)
    sol = solve_ivp(f, (0, t_end), np.full(8, 1e-3),
                    t_eval=np.arange(0, t_end + 1e-9, 0.5),
                    method="LSODA", rtol=1e-7, atol=1e-9)
    return sol.t, sol.y


def score(y):
    return 1 - y.mean(axis=0)


def score_at(u, t=60.0):
    tt, yy = simulate(u, t_end=t)
    return float(score(yy)[-1])


def jacobian(x, u):
    """Numerical Jacobian of the drift at state x with intervention u."""
    x = np.clip(x, 0, 1)
    J = np.zeros((8, 8))
    for j in range(8):
        h = 1e-6
        xp = x.copy()
        xp[j] += h
        J[:, j] = (rhs(0, xp, u) - rhs(0, x, u)) / h
    return J


def main() -> None:
    set_seed(0)
    set_style()

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    ax = axes[0]
    colors = [PAL["grey"], PAL["sky"], PAL["green"], PAL["orange"],
              PAL["blue"], PAL["vermillion"]]
    finals = {}
    for (name, u), c in zip(STRATEGIES.items(), colors):
        tt, yy = simulate(u)
        s = score(yy)
        finals[name] = float(s[-1])
        ax.plot(tt, s, color=c, label=name)
    ax.axvspan(TREAT_START, TREAT_END, color=PAL["light"], alpha=0.35, lw=0)
    ax.text((TREAT_START + TREAT_END) / 2, 0.06, "treatment", ha="center",
            fontsize=6.4, color=PAL["grey"])
    ax.set_xlabel("age (years)")
    ax.set_ylabel("composite score (arbitrary units)")
    ax.set_title("(a)  Partial repair, partial relapse", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.grid(True)
    ax.set_ylim(-0.05, 1.02)

    ax2 = axes[1]
    grid = np.linspace(0, 0.8, 9)
    S = np.zeros((len(grid), len(grid)))
    for i, ut in enumerate(grid):
        for j, ue in enumerate(grid):
            u = np.zeros(8)
            u[0], u[1] = ut, ue
            S[i, j] = score_at(u)
    base = score_at(np.zeros(8))
    tel = np.array([score_at(np.r_[ut, [0, 0, 0, 0, 0, 0, 0]]) for ut in grid])
    epi = np.array([score_at(np.r_[0, ue, [0, 0, 0, 0, 0, 0]]) for ue in grid])
    syn = S - tel[:, None] - epi[None, :] + base
    im = ax2.imshow(syn * 1e5, origin="lower", cmap=DIVERGING,
                    extent=[grid[0], grid[-1], grid[0], grid[-1]], aspect="auto")
    ax2.set_xlabel("epigenetic repair $u_{epi}$")
    ax2.set_ylabel("telomere repair $u_{tel}$")
    ax2.set_title("(b)  Synergy is positive but tiny", loc="left")
    cb = fig.colorbar(im, ax=ax2, orientation="horizontal", fraction=0.055,
                      pad=0.22)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("synergy (×10$^{-5}$ score)", fontsize=6.5)

    ax3 = axes[2]
    bottom = np.zeros(len(STRATEGIES))
    comp_colors = [PAL["blue"], PAL["sky"], PAL["green"], PAL["orange"],
                   PAL["vermillion"], PAL["purple"], PAL["yellow"], PAL["grey"]]
    mats = []
    labels = []
    for name, u in STRATEGIES.items():
        tt, yy = simulate(u, t_end=60.0)
        mats.append(yy[:, -1])
        labels.append(name.split()[0])
    mats = np.array(mats).T       # components x strategies
    for i in range(8):
        ax3.bar(range(len(labels)), mats[i], bottom=bottom,
                color=comp_colors[i], width=0.7, label=NAMES[i])
        bottom += mats[i]
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylabel("remaining damage at age 60 (0-8)")
    ax3.set_title("(c)  What each strategy leaves behind", loc="left")
    ax3.legend(frameon=False, fontsize=5.4, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.13))
    ax3.grid(axis="y")

    fig.text(0.01, -0.06,
             "Toy 8-variable ODE; the score is an arbitrary index, not a biological "
             "age. Interventions act from 40 to 60 yr and then stop. Synergy = S(tel,epi) "
             "- S(tel) - S(epi) + S(none) at age 60.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-multidim")

    multi_u = STRATEGIES["D four-way"]
    tt, yy = simulate(multi_u, t_end=100.0)
    s = score(yy)
    peak = float(s[(tt >= TREAT_START) & (tt <= 60)].max())
    relapse = float(peak - s[-1])

    # stability of the treated steady state at age 60
    tt60, yy60 = simulate(multi_u, t_end=60.0)
    ev = np.linalg.eigvals(jacobian(yy60[:, -1], multi_u))
    max_real = float(np.max(ev.real))

    save_metrics("multidim", {
        "final_score": finals,
        "treatment_peak_score_D": peak,
        "relapse_after_stop_D": relapse,
        "max_real_eigenvalue_treated": max_real,
        "synergy_max": float(syn.max()),
        "synergy_min": float(syn.min()),
        "baseline_score_age60": float(base),
        "components": NAMES,
    })


if __name__ == "__main__":
    main()
