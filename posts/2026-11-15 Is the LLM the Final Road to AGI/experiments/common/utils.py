"""Shared paths, seeding and JSON I/O for the "another road" experiments.

Every script is run from the post directory:

    python experiments/02_scaling.py

Figures go to ../figures (PNG + SVG); numbers are merged into
./outputs/metrics.json so the article reads them instead of retyping.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # experiments/common
EXPERIMENTS = HERE.parent                       # experiments
POST = EXPERIMENTS.parent                       # post directory
FIGURES = POST / "figures"
OUTPUTS = EXPERIMENTS / "outputs"
DATA = POST / "data"
for _p in (FIGURES, OUTPUTS, DATA / "raw"):
    _p.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUTS / "metrics.json"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


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


def bootstrap_ci(x, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0):
    rng_ = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    stats = np.array([statistic(rng_.choice(x, size=len(x), replace=True))
                      for _ in range(n_boot)])
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def fig_dir(article: int = 1) -> Path:
    """All figures of a post live in one flat figures/ folder."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES
