"""Single entry point: run every experiment, rebuild all figures and metrics.

    python experiments/run_all.py          # all six, in order (CPU, no GPU)
    python experiments/run_all.py 2 3      # only 02_image.py and 03_physics.py

Each script is independent: it seeds its own RNG, writes its numbers into
outputs/metrics.json and its figures into ../figures as PNG + SVG.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_language.py",
    "02_image.py",
    "03_physics.py",
    "04_virtual_cell.py",
    "05_perturbation.py",
    "06_conceptual.py",
]


def main(argv: list[str]) -> int:
    chosen = ([s for s in SCRIPTS if s[:2] in argv] if argv else SCRIPTS)
    if not chosen:
        print(f"no script matches {argv}; available: {[s[:2] for s in SCRIPTS]}")
        return 2
    for name in chosen:
        t0 = time.perf_counter()
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        code = subprocess.call([sys.executable, str(HERE / name)])
        if code != 0:
            print(f"!! {name} failed with exit code {code}; stopping.")
            return code
        print(f"--- {name} finished in {time.perf_counter() - t0:.0f} s")
    print("\nall requested experiments finished; figures in ../figures, "
          "numbers in outputs/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
