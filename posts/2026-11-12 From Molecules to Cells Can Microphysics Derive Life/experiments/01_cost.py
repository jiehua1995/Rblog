"""Study 1 — what would an all-atom simulation of a cell actually cost?

The model is deliberately simple and fully stated:

    wall time = N_atoms * (T_bio / dt) / throughput

where `throughput` is measured in atom-steps per second.  Throughputs are
calibrated from published benchmarks rather than invented:

  * GROMACS 2024 on one A100: 148 ns/day for DHFR (23,558 atoms, 2 fs step)
        -> 148e-9 / 2e-15 / 86400 * 23558 = 2.0e7 atom-steps/s
  * Anton 3, 512 nodes: 100 us/day for 1e6 atoms, 2 fs
        -> 100e-6 / 2e-15 / 86400 * 1e6 = 5.8e8 atom-steps/s
  * H100 is taken as 2x A100 (an assumption, stated in the article).

System sizes: the syn3A all-atom count (~6e9 atoms) is the number quoted by
Stevens et al. (2023) for the coarse-grained whole-cell model's atom count.

Run:  python experiments/01_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, SEQUENTIAL, save, set_style
from common.utils import save_metrics, set_seed, years

import matplotlib.pyplot as plt

DT = 2e-15                       # s, 2 fs integration step
BYTES_PER_ATOM = 150.0           # positions, velocities, forces, neighbour list
TRAJ_BYTES_PER_ATOM = 12.0       # float32 xyz

# atom-steps per second, calibrated or assumed (see module docstring)
SCENARIOS = {
    "one A100": 2.0e7,
    "one H100": 4.0e7,
    "Anton 3 (512 nodes)": 5.8e8,
    "1,000 H100, ideal": 4.0e10,
    "10,000 H100, ideal": 4.0e11,
    "10,000 H100, 25%": 1.0e11,
}
CLUSTER_MEMORY = {              # bytes available for the particle arrays
    "one A100": 80e9,
    "one H100": 80e9,
    "Anton 3 (512 nodes)": 512 * 8e9,
    "1,000 H100, ideal": 80e12,
    "10,000 H100, ideal": 800e12,
    "10,000 H100, 25%": 800e12,
}

SYSTEMS = {
    "protein complex": 1e5,
    "syn3A (all-atom)": 6e9,
    "E. coli": 3e10,
    "yeast cell": 1.6e12,
    "human cell": 1e14,
}
BIO_TIMES = np.array([1e-6, 1e-3, 1.0, 60.0, 6300.0, 3600.0])  # s
BIO_LABELS = ["1 us", "1 ms", "1 s", "1 min", "cell cycle", "1 h"]
# cell cycle (syn3A, 105 min) is plotted before 1 h so the axis is ordered
ORDER = [0, 1, 2, 3, 5, 4]
BIO_TIMES = BIO_TIMES[ORDER]
BIO_LABELS = [BIO_LABELS[i] for i in ORDER]

REFS_YEARS = {
    "1 day": 1 / 365.25,
    "1 year": 1.0,
    "recorded history": 5e3,
    "Earth age": 4.54e9,
    "universe age": 1.38e10,
}


def wall_seconds(n_atoms: float, t_bio: float, throughput: float) -> float:
    return n_atoms * (t_bio / DT) / throughput


def main() -> None:
    set_seed(0)
    set_style()

    # ---- table for one scenario -------------------------------------------
    scen = "10,000 H100, ideal"
    P = SCENARIOS[scen]
    rows = {}
    print(f"scenario: {scen}  ({P:.1e} atom-steps/s)")
    for name, n in SYSTEMS.items():
        for t, lab in zip([1e-3, 1.0, 3600.0], ["1 ms", "1 s", "1 h"]):
            sec = wall_seconds(n, t, P)
            rows[f"{name}|{lab}"] = years(sec)
            print(f"  {name:20s} {lab:6s} {years(sec):10.2e} yr"
                  f"   memory {n * BYTES_PER_ATOM / 1e12:8.2f} TB")

    # ---- heat map + feasibility contours ----------------------------------
    Ns = np.logspace(5, 14.5, 160)
    Ts = np.logspace(-7, 4, 160)
    NN, TT = np.meshgrid(Ns, Ts, indexing="ij")
    wall_years = NN * (TT / DT) / P / (365.25 * 24 * 3600)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    z = np.log10(np.clip(wall_years, 1e-6, None))
    im = ax.pcolormesh(Ns, Ts, z.T, cmap=SEQUENTIAL, shading="auto",
                       vmin=-4, vmax=10)
    ax.contour(Ns, Ts, wall_years.T, levels=[1 / 365, 1, 1e3, 1e6],
               colors="white", linewidths=0.6, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms in the simulated system")
    ax.set_ylabel("biological time simulated (s)")
    ax.set_title(f"(a) wall-clock years, {scen}")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("log$_{10}$ years")
    for name, n in [("syn3A", 6e9), ("E. coli", 3e10), ("human cell", 1e14)]:
        ax.axvline(n, color=PAL["vermillion"], lw=0.7, ls=":")
        ax.text(n * 1.15, 2e-5, name, rotation=90, fontsize=6.3,
                color=PAL["vermillion"], va="bottom")
    ax.set_xlim(Ns[0], Ns[-1])
    ax.set_ylim(Ts[0], Ts[-1])

    # panel b: constant-cost contours with reference durations
    ax = axes[1]
    for label, yv in REFS_YEARS.items():
        if yv < 1e-2:
            continue
        t_bio = yv * 365.25 * 24 * 3600 * P * DT / Ns
        ax.plot(Ns, t_bio, color=PAL["grey"], lw=0.7, ls="--")
        idx = int(0.75 * len(Ns))
        ax.text(Ns[idx], t_bio[idx] * 1.6, label, fontsize=6.2,
                color=PAL["grey"], rotation=-24, ha="center")
    t_cycle = 6300.0
    ax.plot(Ns, np.full_like(Ns, t_cycle), color=PAL["blue"], lw=1.2)
    ax.text(3e5, t_cycle * 1.25, "105-min cell cycle", fontsize=6.5,
            color=PAL["blue"])
    for name, n in [("syn3A", 6e9), ("E. coli", 3e10), ("human cell", 1e14)]:
        ax.axvline(n, color=PAL["vermillion"], lw=0.6, ls=":")
        ax.text(n * 1.15, 3e-6, name, rotation=90, fontsize=6.2,
                color=PAL["vermillion"], va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(Ns[0], Ns[-1])
    ax.set_ylim(1e-7, 1e4)
    ax.set_xlabel("atoms in the simulated system")
    ax.set_ylabel("biological time reachable in 1 wall-clock year (s)")
    ax.set_title("(b) feasibility frontier")
    fig.tight_layout()
    save(fig, "fig-cost-landscape")

    # ---- scenarios: how long is one second of one cell? --------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names = list(SCENARIOS)
    for j, (sysname, n) in enumerate([("syn3A", 6e9), ("E. coli", 3e10)]):
        vals = [years(wall_seconds(n, 3600.0, SCENARIOS[s])) for s in names]
        ypos = np.arange(len(names)) - 0.2 + 0.4 * j
        ax.barh(ypos, vals, height=0.34,
                color=[PAL["blue"], PAL["sky"]][j], label=f"{sysname}, 1 h")
    for yv in REFS_YEARS.values():
        ax.axvline(yv, color=PAL["grey"], lw=0.6, ls="--")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xscale("log")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(list(REFS_YEARS.values()))
    sec.set_xticklabels(["1 d", "1 yr", "5 kyr", "4.5 Gyr\n", "\n13.8 Gyr"],
                        fontsize=6.2, color=PAL["grey"])
    sec.tick_params(length=0, pad=1)
    sec.spines["top"].set_visible(False)
    ax.set_xlabel("wall-clock years for one hour of biological time")
    ax.set_title("(a) hardware does not close the gap")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.invert_yaxis()

    # panel b: memory and trajectory storage
    ax = axes[1]
    n = np.logspace(5, 14.5, 120)
    ax.plot(n, n * BYTES_PER_ATOM / 1e12, color=PAL["blue"], label="state (150 B/atom)")
    ax.plot(n, n * TRAJ_BYTES_PER_ATOM * 1e4 / 1e12, color=PAL["vermillion"],
            label="10$^4$ frames (12 B/atom)")
    for lab, cap in [("one A100, 80 GB", 80e9), ("1,000 GPUs, 80 TB", 80e12),
                     ("10,000 GPUs, 800 TB", 800e12)]:
        ax.axhline(cap / 1e12, color=PAL["grey"], lw=0.7, ls=":")
        ax.text(2e5, cap / 1e12 * 1.25, lab, fontsize=6.2, color=PAL["grey"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms")
    ax.set_ylabel("storage (TB)")
    ax.set_title("(b) memory and trajectory I/O")
    ax.legend(loc="lower right", fontsize=6.8, frameon=False)
    ax.set_ylim(1e-3, 1e10)
    fig.tight_layout()
    save(fig, "fig-cost-memory")

    # ---- coarse-graining: speedup vs bead size ----------------------------
    # m atoms per bead, and a smoother potential allows a larger timestep
    m = np.array([1, 2, 4, 10, 25, 100], dtype=float)
    dt_factor = np.minimum(m ** 0.6, 20.0)
    speedup = m * dt_factor
    wall_syn = years(wall_seconds(6e9, 6300.0, P))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(m, wall_syn / speedup, "o-", color=PAL["blue"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms represented by one coarse-grained bead")
    ax.set_ylabel("wall-clock years for one syn3A cycle")
    ax.set_title("coarse-graining: orders of magnitude, not miracles")
    ax.axhline(1 / 365, color=PAL["grey"], lw=0.7, ls="--")
    ax.text(1.2, 1 / 365 * 1.4, "one day", fontsize=6.5, color=PAL["grey"])
    ax.text(1.2, wall_syn * 2.5,
            f"all-atom: {wall_syn:.1e} yr on {scen}", fontsize=7,
            color=PAL["dark"])
    for mi, si in zip(m, speedup):
        if mi in (4, 25):
            ax.annotate(f"{si:.0f}x", (mi, wall_syn / si), textcoords="offset points",
                        xytext=(4, 6), fontsize=6.5, color=PAL["grey"])
    ax.set_xlim(0.8, 130)
    fig.tight_layout()
    save(fig, "fig-cost-coarsegrain")

    save_metrics("cost", {
        "dt_fs": 2.0,
        "bytes_per_atom": BYTES_PER_ATOM,
        "scenarios_atom_steps_per_s": SCENARIOS,
        "systems_atoms": SYSTEMS,
        "years_table": rows,
        "syn3a_cycle_years_10k_gpu": wall_syn,
        "syn3a_state_TB": 6e9 * BYTES_PER_ATOM / 1e12,
        "human_state_PB": 1e14 * BYTES_PER_ATOM / 1e15,
        "cg_speedups": dict(zip([int(v) for v in m], speedup.round(1).tolist())),
    })


if __name__ == "__main__":
    main()
