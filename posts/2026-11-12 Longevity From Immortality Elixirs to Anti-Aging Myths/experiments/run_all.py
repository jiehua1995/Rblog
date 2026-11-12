"""Run every experiment in order and rebuild all figures and metrics.

    python experiments/run_all.py          # all twelve
    python experiments/run_all.py 4 9      # only 04_hallmarks.py and 09_survival.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_history.py",
    "02_species.py",
    "03_evolution.py",
    "04_hallmarks.py",
    "05_clocks.py",
    "06_multidim.py",
    "07_reprogram.py",
    "08_cancer.py",
    "09_survival.py",
    "10_evidence.py",
    "11_public_data.py",
    "12_repair.py",
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
