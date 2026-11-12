"""Fast checks of the formulas and of one number per experiment.

Run from the post directory:

    python experiments/tests/test_core.py

These are assertions, not a full regression suite: the expensive simulations
are not re-run.  Each test recomputes a closed-form or cheap quantity and
compares it with the value stored in outputs/metrics.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
POST = EXP.parent
sys.path.insert(0, str(EXP))

import numpy as np

METRICS = json.loads((EXP / "outputs" / "metrics.json").read_text(encoding="utf-8"))


def load(script: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXP / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def close(a, b, rel=0.05):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def test_cost():
    m = load("01_cost.py", "cost")
    p = m.SCENARIOS["10,000 H100, ideal"]
    sec = m.wall_seconds(6e9, 1.0, p)
    yrs = sec / (365.25 * 24 * 3600)
    assert close(yrs, METRICS["cost"]["years_table"]["syn3A (all-atom)|1 s"], 0.01)
    assert yrs > 1e4, "one second of syn3A should be far out of reach"
    # memory of the human cell is petabyte scale
    assert m.SYSTEMS["human cell"] * m.BYTES_PER_ATOM > 1e15


def test_identifiability():
    m = load("02_identifiability.py", "ident")
    grid = np.linspace(-0.6, 0.6, 121)
    p = np.full_like(grid, 1 / (grid[-1] - grid[0]))
    e = m.bayes_error(grid, p, 0.2, lambda d: m.fate_at(m.simulate(d), 40.0))
    assert 0.0 <= e <= 0.5
    mi, h = m.mutual_information(grid, p, 0.2,
                                 lambda d: m.fate_at(m.simulate(d), 40.0))
    assert mi <= h + 1e-9, "data processing inequality"
    assert close(e, 0.136, 0.25)   # stored value is computed on a finer grid


def test_projection():
    m = load("03_projection.py", "proj")
    err = METRICS["projection"]["rel_error"]
    assert err["Markov"][-1] > 5 * err["Markov"][0]
    assert err["memory"][-1] < 0.01
    assert err["Markov"][-1] > 0.1


def test_thermo():
    m = load("04_thermo.py", "thermo")
    Q0 = m.generator(0.0)
    p0 = m.steady_state(Q0)
    J0, s0 = m.flux_and_sigma(Q0, p0)
    assert abs(J0) < 1e-9 and abs(s0) < 1e-9, "equilibrium has no flux"
    assert np.allclose(p0, 1 / 3, atol=1e-6)
    Q4 = m.generator(4.0)
    p4 = m.steady_state(Q4)
    J4, s4 = m.flux_and_sigma(Q4, p4)
    assert J4 > 0 and s4 > J4 * 3.9, "affinity 4 means sigma = 4 J"
    assert np.allclose(p4, 1 / 3, atol=1e-6), "concentrations cannot see the drive"


def test_bifurcation():
    m = load("05_bifurcation.py", "bif")
    x = m.N_HILL ** (-1 / (m.N_HILL + 1))
    beta_c = x + x ** (m.N_HILL + 1)
    assert close(beta_c, METRICS["bifurcation"]["beta_c"], 0.02)
    amp = METRICS["bifurcation"]["goodwin_amplitude"]
    assert amp[0] < 1e-6 and amp[-1] > 0.5, "Hopf onset"


def test_error():
    m = load("06_error.py", "err")
    kd = np.exp(-1.0 / m.RT)
    assert close(kd, 0.184, 0.02)
    mc = METRICS["error"]["mc_sd"]
    assert mc["buffered"][3] < mc["near threshold"][3]


def test_spatial():
    m = load("07_spatial.py", "spa")
    assert m.f_schlogl(0.0) == 0.0
    fr = METRICS["spatial"]["high_fraction"]
    assert fr[1] > 0.01 and fr[-1] == 0.0
    assert METRICS["spatial"]["fraction_0d"] == 0.0


def test_ladder():
    mse = METRICS["ladder"]["mse"]
    assert mse["fine"] < 1e-4
    assert mse["coarse"] < mse["fine + 15% error"], "the counterexample"
    assert mse["hybrid"] < mse["MLP"]


def test_quantum():
    m = load("10_quantum.py", "q")
    kT = 1.380649e-23 * 300 / 1.602176634e-19
    assert close(kT, METRICS["quantum"]["kT_eV"], 0.001)
    assert METRICS["quantum"]["tunnelling_at_0.5A"] > \
        METRICS["quantum"]["tunnelling_D_at_0.5A"]
    assert METRICS["quantum"]["tau_300K_at_0.1nm"] < 1e-13


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
