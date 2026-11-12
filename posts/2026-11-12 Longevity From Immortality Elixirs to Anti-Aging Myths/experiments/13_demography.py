"""Supplementary study A — the demographic background.

Life expectancy at birth for Japan, the United States and the world, rounded
from UN World Population Prospects (2024 revision), plus the decadal gain.  The
point is not to forecast: it is to show that the twentieth-century gains are
decelerating, so any claim of systemic rejuvenation must explain how it breaks
this curve.

Run:  python experiments/13_demography.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

YEARS = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2023])

# rounded UN WPP 2024 values, life expectancy at birth (both sexes)
DATA = {
    "Japan": (np.array([61.2, 67.7, 72.0, 76.1, 78.9, 81.2, 82.8, 84.6, 84.0]),
              PAL["vermillion"]),
    "United States": (np.array([68.1, 69.9, 70.9, 73.8, 75.4, 76.8, 78.7, 77.0, 78.4]),
                      PAL["blue"]),
    "World": (np.array([45.7, 49.3, 56.0, 61.3, 64.5, 67.0, 70.6, 72.5, 73.2]),
              PAL["green"]),
}


def main() -> None:
    set_seed(0)
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    for name, (vals, c) in DATA.items():
        ax.plot(YEARS, vals, marker="o", ms=3.2, color=c, label=name)
    ax.set_xlabel("year")
    ax.set_ylabel("life expectancy at birth (years)")
    ax.set_title("(a)  Gains are real, and they are slowing", loc="left")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.grid(True)
    ax.set_ylim(40, 90)

    ax2 = axes[1]
    for i, (name, (vals, c)) in enumerate(DATA.items()):
        v, y = vals[:-1], YEARS[:-1]          # full decades only, 1950-2020
        rate = np.diff(v) / np.diff(y) * 10
        x = (y[:-1] + y[1:]) / 2 + (i - 1) * 2.6
        ax2.bar(x, rate, width=2.3, color=c, label=name)
    ax2.axhline(0, color=PAL["dark"], lw=0.6)
    ax2.set_xlabel("midpoint of decade")
    ax2.set_ylabel("annualised gain (years per decade)")
    ax2.set_title("(b)  Decadal gains are smaller and more variable", loc="left")
    ax2.legend(frameon=False, fontsize=6.6, loc="upper right")
    ax2.grid(axis="y")

    fig.text(0.01, -0.06,
             "Rounded from UN World Population Prospects 2024. Panel (b) uses the full "
             "1950–2020 decades; panel (a) includes 2023. The 2020–2023 rebound after "
             "COVID-19 is not a trend.",
             fontsize=6.2, color=PAL["grey"], ha="left")

    save(fig, "fig-demography")

    jp = DATA["Japan"][0]
    world = DATA["World"][0]
    gains_jp = np.diff(jp)
    save_metrics("demography", {
        "years": YEARS.tolist(),
        "japan_1950": float(jp[0]),
        "japan_2023": float(jp[-1]),
        "world_1950": float(world[0]),
        "world_2023": float(world[-1]),
        "japan_max_decadal_gain": float(gains_jp.max()),
        "japan_latest_gain": float(gains_jp[-1]),
        "world_latest_gain": float(np.diff(world)[-1]),
    })


if __name__ == "__main__":
    main()
