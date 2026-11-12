"""Run every experiment in order and rebuild all figures and metrics.

    python experiments/run_all.py          # all ten
    python experiments/run_all.py 3 7      # only 03_projection.py and 07_spatial.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_cost.py",
    "02_identifiability.py",
    "03_projection.py",
    "04_thermo.py",
    "05_bifurcation.py",
    "06_error.py",
    "07_spatial.py",
    "08_ladder.py",
    "09_framework.py",
    "10_quantum.py",
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
    print("\ndone: figures in ../figures, numbers in outputs/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
