"""Shared paths, seeding and JSON I/O for the experiments in this post.

Every script is run from the post directory:

    python experiments/01_language.py

Artifacts go to ../figures (PNG + SVG) and ./outputs (metrics.json), so the
article can read numbers instead of having them retyped.
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
    """Seed every RNG used in the experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def _clean(obj):
    """Replace NaN/inf with null so the JSON is valid outside Python."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def save_metrics(key: str, values: dict) -> None:
    """Merge one experiment's numbers into outputs/metrics.json."""
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
