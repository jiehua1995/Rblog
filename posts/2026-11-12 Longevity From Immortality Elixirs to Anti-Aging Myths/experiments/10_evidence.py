"""Study 10 — how strong is the evidence, and how long would a real trial take?

Part 1 is a curated evidence matrix: nine intervention families against six
levels of evidence, scored 0-3 from the literature cited in the article.  The
scores are the author's summary, not a meta-analysis.

Part 2 is arithmetic that does not depend on curation.  For a mortality
endpoint, the number of events needed is

    D = 4 (z_{1-alpha/2} + z_{1-beta})^2 / (ln HR)^2,

and the number of participants is D divided by the event proportion.  The
right panel adds the minimum calendar time of a drug-development programme.

Run:  python experiments/10_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import EVIDENCE, PAL, save, set_style
from common.utils import save_metrics, set_seed

LEVELS = ["in vitro", "mouse\nlifespan", "mammalian\ntissue",
          "human\nsafety", "human\nRCT", "healthspan\nendpoint"]

ROWS = [
    ("caloric restriction", [3, 3, 2, 2, 2, 1]),
    ("rapamycin / mTOR", [3, 3, 2, 2, 1, 0]),
    ("metformin", [2, 1, 1, 2, 2, 0]),
    ("senolytics (D+Q)", [3, 2, 1, 2, 1, 0]),
    ("NAD$^+$ precursors", [2, 1, 1, 2, 1, 0]),
    ("telomerase gene therapy", [2, 2, 1, 1, 0, 0]),
    ("partial reprogramming", [3, 2, 1, 1, 0, 0]),
    ("stem cells / regenerative", [2, 2, 2, 2, 1, 0]),
    ("AI-driven drug discovery", [1, 1, 1, 0, 0, 0]),
]

PHASES = [("target discovery", 3.0), ("phase 1", 1.5), ("phase 2", 2.0),
          ("phase 3", 3.0), ("regulatory review", 1.5)]


def required_events(hr, alpha=0.05, power=0.8):
    z_a = 1.959963984540054
    z_b = 0.8416212335729143
    return 4 * (z_a + z_b) ** 2 / np.log(hr) ** 2


def main() -> None:
    set_seed(0)
    set_style()

    fig = plt.figure(figsize=(12.8, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.95], wspace=0.42)
    ax = fig.add_subplot(gs[0])

    M = np.array([r[1] for r in ROWS], dtype=float)
    im = ax.imshow(M, cmap=EVIDENCE, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVELS, fontsize=6.6)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "●" if M[i, j] >= 2 else ("·" if M[i, j] == 1 else ""),
                    ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] >= 2 else PAL["grey"])
    ax.set_title("(a)  The evidence staircase stops early", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0, 1, 2, 3])
    cb.set_ticklabels(["none", "early", "moderate", "strong"])
    cb.ax.tick_params(labelsize=6)

    ax2 = fig.add_subplot(gs[1])
    hr = np.linspace(0.7, 0.95, 120)
    for event_rate, c in [(0.15, PAL["sky"]), (0.30, PAL["blue"]),
                          (0.50, PAL["vermillion"])]:
        n = required_events(hr) / event_rate
        ax2.plot(hr, n, color=c, label=f"{int(event_rate*100)}% 10-yr mortality")
    ax2.set_yscale("log")
    ax2.set_xlabel("hazard ratio (mortality)")
    ax2.set_ylabel("total participants")
    ax2.set_title("(b)  A mortality endpoint is expensive", loc="left")
    ax2.legend(frameon=False, fontsize=6.4)
    ax2.grid(True, which="major")
    n80 = required_events(0.8) / 0.30
    ax2.plot(0.8, n80, "o", color=PAL["dark"], ms=5, mec="white")
    ax2.annotate(f"HR 0.8\n≈ {n80:.0f} people", (0.8, n80),
                 xytext=(8, 10), textcoords="offset points", fontsize=6.4)

    ax3 = fig.add_subplot(gs[2])
    left = 0.0
    colors = [PAL["sky"], PAL["green"], PAL["orange"], PAL["vermillion"], PAL["grey"]]
    for i, (name, dur) in enumerate(PHASES):
        ax3.barh(i, dur, left=left, height=0.62, color=colors[i])
        ax3.text(left + dur / 2, i, f"{dur:g}", ha="center", va="center",
                 fontsize=6.2, color="white")
        left += dur
    ax3.set_yticks(range(len(PHASES)))
    ax3.set_yticklabels([n for n, _ in PHASES], fontsize=6.6)
    ax3.set_xlim(0, 12)
    ax3.set_ylim(-0.7, len(PHASES) - 0.3)
    ax3.set_xlabel("minimum calendar years")
    ax3.set_title(f"(c)  ≈ {left:g} years before an answer", loc="left")
    ax3.grid(axis="x")

    fig.text(0.01, -0.08,
             "Panel (a) scores are the author's summary of the cited literature "
             "(0 none, 1 early, 2 moderate, 3 strong). Panel (b) uses the standard "
             "log-rank event formula with alpha = 0.05, power = 0.8.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-evidence")

    highest = M.max(axis=1)
    save_metrics("evidence", {
        "interventions": [r[0] for r in ROWS],
        "levels": LEVELS,
        "n_with_healthspan_endpoint": int(np.sum(M[:, 5] > 0)),
        "n_with_human_rct": int(np.sum(M[:, 4] > 0)),
        "median_highest_level": float(np.median(highest)),
        "participants_per_arm_hr08": float(n80),
        "events_needed_hr08": float(required_events(0.8)),
        "minimum_trial_years": float(sum(d for _, d in PHASES)),
        "matrix": M.astype(int).tolist(),
    })


if __name__ == "__main__":
    main()
