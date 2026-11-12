"""Study 1 — a timeline of promises, and how long evidence has taken.

The historical dates are from primary texts (Shiji, Hanshu, Baopuzi) and from
standard histories of alchemy; the modern dates are the publication years of
the experiments cited in the article.  The second panel is a simple
bookkeeping exercise: how many years separate a mechanism's demonstration from
a *completed* human healthspan trial.  As of 2026 the count is zero, which is
the point of the panel.

Run:  python experiments/01_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

PRESENT = 2026

# (year, row, label, evidence class, label dx, label dy, vertical alignment)
# rows: 0 = China, 1 = Europe / Islamic world, 2 = modern laboratory
EVENTS = [
    (-219, 0, "Qin Shi Huang sends Xu Fu\nfor the elixir", "legend", 0, 9, "bottom"),
    (-133, 0, "Han Wudi's fangshi\nand the dew terrace", "legend", 0, -9, "top"),
    (320, 0, "Ge Hong: Baopuzi\nouter alchemy", "text", 0, 9, "bottom"),
    (650, 0, "Tang emperors die\nof cinnabar pills", "text", 0, -9, "top"),
    (1100, 1, "Latin alchemy:\nelixir of life", "text", 0, 9, "bottom"),
    (1530, 1, "Paracelsus:\nquintessence", "text", 0, -9, "top"),
    (1889, 1, "organ extracts\ninjected for vigour", "experiment", 0, 9, "bottom"),
    (1935, 2, "caloric restriction\nextends rat lifespan", "experiment", 0, -10, "top"),
    (1961, 2, "Hayflick limit", "experiment", 0, 9, "bottom"),
    (1990, 2, "telomerase and\ntelomere maintenance", "experiment", 0, -10, "top"),
    (2006, 2, "OSKM reprogramming\nof somatic cells", "experiment", 0, 9, "bottom"),
    (2013, 2, "pan-tissue\nmethylation clock", "experiment", 0, -10, "top"),
    (2015, 2, "senolytics\nin mice", "experiment", 0, 9, "bottom"),
    (2016, 2, "partial reprogramming\nin vivo", "experiment", 0, -10, "top"),
    (2026, 2, "first-in-human\nOSK gene therapy (phase 1)", "trial", 0, 9, "bottom"),
]

COL = {"legend": PAL["grey"], "text": PAL["orange"],
       "experiment": PAL["blue"], "trial": PAL["vermillion"]}

# (mechanism, year demonstrated, status, class)
GAPS = [
    ("caloric restriction", 1935, "primate RCTs mixed", "partial"),
    ("Hayflick limit", 1961, "no human intervention", "none"),
    ("telomerase", 1990, "no healthspan endpoint", "none"),
    ("iPSC reprogramming", 2006, "phase 1 safety only", "none"),
    ("rapamycin / mTOR", 2009, "no mortality RCT", "none"),
    ("epigenetic clock", 2013, "no causal endpoint", "none"),
    ("senolytics", 2015, "small, mixed trials", "partial"),
    ("partial reprogramming", 2016, "phase 1 safety only", "none"),
]


def main() -> None:
    set_seed(0)
    set_style()

    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.42)
    ax = fig.add_subplot(gs[0])

    # x: years before present, log scale, inverted so time runs left -> right
    ev = sorted(EVENTS, key=lambda e: e[0])
    ypos = np.arange(len(ev))[::-1]
    for y, (year, row, label, cls, dx, dy, va) in zip(ypos, ev):
        x = max(PRESENT - year, 0.6)
        ax.plot([x], [y], "o", ms=5.0, color=COL[cls], zorder=3,
                mec="white", mew=0.7)
        to_left = cls == "trial"
        ax.annotate(label.replace("\n", " "), (x, y),
                    xytext=(-5 if to_left else 5, 0),
                    textcoords="offset points",
                    ha="right" if to_left else "left", va="center",
                    fontsize=6.2, color=COL[cls])

    ax.set_xscale("log")
    ax.set_xlim(5200, 0.30)
    ax.set_ylim(-0.7, len(ev) + 0.48)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{abs(e[0])} BCE" if e[0] < 0 else f"{e[0]} CE"
                        for e in ev], fontsize=7.0)
    ax.set_xticks([5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 1])
    ax.set_xticklabels(["5000", "2000", "1000", "500", "200", "100",
                        "50", "20", "10", "5", "1"])
    ax.set_xlabel("years before 2026 (log scale)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", which="major")
    ax.set_title("(a)  Two millennia of promises, accelerating mechanisms",
                 loc="left", pad=20)
    for cls, name in [("legend", "legend / court record"), ("text", "alchemical text"),
                      ("experiment", "controlled experiment"),
                      ("trial", "human trial")]:
        ax.plot([], [], "o", color=COL[cls], ms=5, label=name)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              ncol=4, handletextpad=0.2, columnspacing=1.4, fontsize=6.8)

    ax2 = fig.add_subplot(gs[1])
    gaps = sorted(GAPS, key=lambda g: g[1])
    ypos = np.arange(len(gaps))
    lengths = [PRESENT - g[1] for g in gaps]
    colors = [PAL["orange"] if g[3] == "partial" else PAL["grey"] for g in gaps]
    ax2.barh(ypos, lengths, color=colors, height=0.58, alpha=0.85)
    for i, (name, year, status, cls) in enumerate(gaps):
        ax2.text(lengths[i] + 0.7, i, f"{status}  ({lengths[i]} yr)",
                 va="center", fontsize=6.4,
                 color=PAL["dark"] if cls == "partial" else PAL["grey"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([g[0] for g in gaps], fontsize=7.2)
    ax2.set_xlim(0, max(lengths) * 1.55)
    ax2.set_xlabel("years since the mechanism was demonstrated (to 2026)")
    ax2.set_title("(b)  Time to a completed human healthspan endpoint: "
                  "none of these has one", loc="left")
    ax2.grid(axis="x")

    fig.text(0.01, 0.005,
             "Historical dates follow the Shiji, Hanshu and Baopuzi; modern dates are "
             "publication years of the cited experiments. Panel (b) records whether a "
             "completed, positive human healthspan trial exists, not whether the biology is real.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-timeline")

    save_metrics("history", {
        "n_events": len(EVENTS),
        "n_mechanisms": len(GAPS),
        "n_with_completed_human_healthspan_trial": 0,
        "median_years_since_mechanism": float(np.median(
            [PRESENT - g[1] for g in GAPS])),
        "longest_gap": {"name": max(GAPS, key=lambda g: PRESENT - g[1])[0],
                        "years": max(PRESENT - g[1] for g in GAPS)},
        "shortest_gap": {"name": min(GAPS, key=lambda g: PRESENT - g[1])[0],
                         "years": min(PRESENT - g[1] for g in GAPS)},
    })


if __name__ == "__main__":
    main()
