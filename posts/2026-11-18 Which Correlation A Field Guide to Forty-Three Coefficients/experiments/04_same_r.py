"""Study 4 - the same r, different data.

Anscombe's quartet and the Datasaurus Dozen all share (almost) the same Pearson
correlation.  We ask which measures separate them, and add a set of real pairs
from the Palmer penguins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, SEQ, save, set_style
from common.utils import save_metrics, set_seed

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"

ANSCOMBE = {
    "I": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
          [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
           [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": ([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}

COEFS = [("pearson", C.pearson), ("spearman", C.spearman),
         ("kendall", C.kendall_tau_b), ("dcor", C.dcor),
         ("hsic", C.hsic), ("mic", C.mic), ("xi", C.xi_chatterjee),
         ("rdc", C.rdc)]


def all_coefs(x, y) -> dict:
    return {name: float(fn(x, y)) for name, fn in COEFS}


def main() -> None:
    set_seed(0)

    ans = {k: all_coefs(np.array(v[0], float), np.array(v[1], float))
           for k, v in ANSCOMBE.items()}

    # ---- figure 1: Anscombe
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9), sharex=True, sharey=True)
    for ax, (k, (x, y)) in zip(axes, ANSCOMBE.items()):
        ax.scatter(x, y, s=16, color=PAL["navy"], alpha=0.85, edgecolor="none")
        a = ans[k]
        ax.set_title(f"({k})", loc="left", fontsize=9)
        ax.text(0.03, 0.97,
                f"r={a['pearson']:.2f}\ndCor={a['dcor']:.2f}\nMIC={a['mic']:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.6,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    save(fig, "fig01_anscombe")

    # ---- Datasaurus Dozen
    dsz = pd.read_csv(DATA / "datasauzoo.csv")
    rows, labels = [], []
    for name, g in dsz.groupby("dataset", sort=True):
        rows.append(all_coefs(g["x"].to_numpy(), g["y"].to_numpy()))
        labels.append(name)
    M = np.array([[r[c] for c, _ in COEFS] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(M, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(COEFS)))
    ax.set_xticklabels([c for c, _ in COEFS], rotation=30, ha="right", fontsize=7.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if M[i, j] > 0.6 else PAL["dark"])
    ax.set_title("Datasaurus Dozen: thirteen datasets, one Pearson r", loc="left",
                 fontsize=8.6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save(fig, "fig02_datasaurus")

    # ---- penguins
    pen = pd.read_csv(DATA / "penguins.csv")
    pairs = {
        "bill_length~bill_depth": ("bill_length_mm", "bill_depth_mm"),
        "flipper~body_mass": ("flipper_length_mm", "body_mass_g"),
        "bill_length~body_mass": ("bill_length_mm", "body_mass_g"),
        "bill_depth~flipper": ("bill_depth_mm", "flipper_length_mm"),
    }
    penres = {}
    for label, (a, b) in pairs.items():
        d = pen[[a, b]].dropna()
        penres[label] = all_coefs(d[a].to_numpy(), d[b].to_numpy())
        penres[label]["n"] = int(len(d))

    # categorical and ordinal use of the same table
    cat = pen.dropna(subset=["species", "island"])
    sp = pd.factorize(cat["species"])[0]
    isl = pd.factorize(cat["island"])[0]
    ordd = pen.dropna(subset=["bill_length_mm", "bill_depth_mm"])
    binned = pd.qcut(ordd["bill_length_mm"], 5, labels=False)
    binned2 = pd.qcut(ordd["bill_depth_mm"], 5, labels=False)
    extra = {
        "cramers_v(species,island)": C.cramers_v(sp, isl),
        "theil_u(species,island)": C.theil_u(sp, isl),
        "lambda(species,island)": C.lambda_gk(sp, isl),
        "polychoric(bill_len,bill_depth,5 bins)": C.polychoric(binned.to_numpy(), binned2.to_numpy()),
        "pearson(same 5 bins)": C.pearson(binned.to_numpy(), binned2.to_numpy()),
        "spearman(same 5 bins)": C.spearman(binned.to_numpy(), binned2.to_numpy()),
        "n_penguins": int(len(cat)),
    }
    print("[same-r] done")

    save_metrics("same_r", {
        "anscombe": ans, "datasaurus": dict(zip(labels, rows)),
        "penguins": penres, "penguins_categorical": extra,
        "datasaurus_pearson_range": [float(M[:, 0].min()), float(M[:, 0].max())],
        "datasaurus_dcor_range": [float(M[:, 3].min()), float(M[:, 3].max())],
    })


if __name__ == "__main__":
    main()
