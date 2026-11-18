"""Shared paths, seeding and JSON I/O for the JEPA experiments."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
POST = EXPERIMENTS.parent
FIGURES = POST / "figures"
OUTPUTS = EXPERIMENTS / "outputs"
DATA = POST / "data"
for _p in (FIGURES, OUTPUTS, DATA):
    _p.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUTPUTS / "metrics.json"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(min(8, os.cpu_count() or 4))


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def _clean(obj):
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
    print(f"[metrics] wrote '{key}'")


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    print(f"[time] {label}: {time.perf_counter() - t0:.1f} s")
