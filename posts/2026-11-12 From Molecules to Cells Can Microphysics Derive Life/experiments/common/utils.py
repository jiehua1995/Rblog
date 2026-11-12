"""Shared paths, seeding and JSON I/O for the molecule-to-cell experiments.

Every script is run from the post directory:

    python experiments/01_cost.py

Figures go to ../figures (PNG + SVG); numbers are merged into
./outputs/metrics.json so the article can read them instead of retyping.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # experiments/common
EXPERIMENTS = HERE.parent                       # experiments
POST = EXPERIMENTS.parent                       # post directory
FIGURES = POST / "figures"
OUTPUTS = EXPERIMENTS / "outputs"
DATA = POST / "data"
for _p in (FIGURES, OUTPUTS, DATA):
    _p.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUTS / "metrics.json"


def set_seed(seed: int = 0) -> None:
    """Seed every RNG used here (scipy and numpy only, no torch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def _clean(obj):
    """Replace NaN/inf with null so the file is valid strict JSON."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def save_metrics(key: str, values: dict) -> None:
    allm = load_metrics()
    allm[key] = values
    METRICS_PATH.write_text(
        json.dumps(_clean(allm), indent=2, ensure_ascii=False,
                   sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[metrics] wrote '{key}' -> {METRICS_PATH.relative_to(POST)}")


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    print(f"[time] {label}: {time.perf_counter() - t0:.1f} s")


def years(seconds: float) -> float:
    return seconds / (365.25 * 24 * 3600)


def fmt_years(y: float) -> str:
    """Human-readable duration for tables printed to stdout."""
    if y < 1 / 365:
        return f"{y * 365 * 24:.1f} h"
    if y < 1:
        return f"{y * 365:.1f} d"
    if y < 1e4:
        return f"{y:.1f} yr"
    return f"{y:.2e} yr"
