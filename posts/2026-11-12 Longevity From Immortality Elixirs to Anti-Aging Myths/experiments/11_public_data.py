"""Study 11 — real effect sizes from the mouse ITP, and what they would mean in humans.

Panel (a) plots reported median-lifespan changes from the NIA Interventions
Testing Program for agents that reached significance, separately for males and
females.  The values are rounded from the published reports and are used here
as a real-data summary of the *mouse* evidence base.

Panel (b) is a transparent "perfect translation" calculation: if a drug reduced
human age-specific mortality by the same fraction, what would life expectancy
gain?  It is an upper bound under an assumption that has never been tested —
which is exactly the gap the article is about.  The observed human gain is not
plotted because no completed trial has demonstrated one.

Run:  python experiments/11_public_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

# (agent, male % change, female % change, source note)
ITP = [
    ("rapamycin", 23, 26, "42 ppm, both sexes"),
    ("acarbose", 22, 5, "1000 ppm"),
    ("17$\\alpha$-estradiol", 12, 0, "males only"),
    ("canagliflozin", 14, 0, "males only"),
    ("NDGA", 10, 0, "males only"),
    ("aspirin", 0, 0, "not significant"),
    ("metformin", 0, 0, "not significant"),
    ("resveratrol", 0, 0, "not significant"),
]

A_HUMAN = 5e-5
B_HUMAN = 0.085


def survival_human(t, mortality_reduction=0.0):
    return np.exp(-(1 - mortality_reduction) * A_HUMAN / B_HUMAN
                  * (np.exp(B_HUMAN * t) - 1))


def life_expectancy(reduction):
    t = np.linspace(0, 130, 13001)
    s = survival_human(t, reduction)
    dt = t[1] - t[0]
    return float((s.sum() - 0.5 * (s[0] + s[-1])) * dt)


def main() -> None:
    set_seed(0)
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    y = np.arange(len(ITP))[::-1]
    male = np.array([r[1] for r in ITP])
    female = np.array([r[2] for r in ITP])
    ax.barh(y + 0.19, male, height=0.34, color=PAL["blue"], label="male")
    ax.barh(y - 0.19, female, height=0.34, color=PAL["vermillion"], label="female")
    for yi, (name, m, f, note) in zip(y, ITP):
        if m == 0 and f == 0:
            ax.text(0.6, yi, note, va="center", fontsize=5.8, color=PAL["grey"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ITP], fontsize=7)
    ax.set_xlabel("change in median lifespan (%)")
    ax.set_xlim(0, 30)
    ax.set_title("(a)  The mouse evidence is real — and small", loc="left")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.grid(axis="x")

    ax2 = axes[1]
    red = np.linspace(0, 0.35, 80)
    gains = np.array([life_expectancy(x) - life_expectancy(0) for x in red])
    ax2.plot(100 * red, gains, color=PAL["blue"])
    for value, c, lab in [(0.23, PAL["orange"], "rapamycin-like 23%"),
                          (0.26, PAL["vermillion"], "26%")]:
        g = life_expectancy(value) - life_expectancy(0)
        ax2.plot(100 * value, g, "o", color=c, ms=6, mec="white")
        ax2.annotate(f"{lab}\n+{g:.1f} yr", (100 * value, g),
                     xytext=(6, -14), textcoords="offset points", fontsize=6.4,
                     color=c)
    ax2.set_xlabel("reduction in age-specific mortality (%)")
    ax2.set_ylabel("gain in life expectancy at birth (years)")
    ax2.set_title("(b)  If the mouse effect transferred perfectly", loc="left")
    ax2.grid(True)

    fig.text(0.01, -0.06,
             "ITP values are rounded from the published reports (42 ppm rapamycin; "
             "1000 ppm acarbose; sex-specific significant results only). Panel (b) "
             "assumes a proportional mortality reduction at every age, which no "
             "human trial has demonstrated.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-itp")

    gains = {int(100 * r): float(life_expectancy(r) - life_expectancy(0))
             for r in (0.23, 0.26)}
    save_metrics("public_data", {
        "n_agents": len(ITP),
        "n_significant_male": int(np.sum([r[1] > 0 for r in ITP])),
        "n_significant_female": int(np.sum([r[2] > 0 for r in ITP])),
        "mean_male_effect": float(np.mean([r[1] for r in ITP if r[1] > 0])),
        "mean_female_effect": float(np.mean([r[2] for r in ITP if r[2] > 0])),
        "max_effect": int(max(max(r[1] for r in ITP), max(r[2] for r in ITP))),
        "human_e0_gain_at_23pct": gains[23],
        "human_e0_gain_at_26pct": gains[26],
        "human_e0_baseline": life_expectancy(0.0),
    })


if __name__ == "__main__":
    main()
