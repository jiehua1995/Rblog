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


def test_history():
    m = METRICS["history"]
    assert m["n_with_completed_human_healthspan_trial"] == 0
    assert m["n_events"] == 15 and m["n_mechanisms"] == 8
    assert m["median_years_since_mechanism"] > 5


def test_species():
    mod = load("02_species.py", "species")
    x = np.log10([s[1] for s in mod.SPECIES])
    y = np.log10([s[2] for s in mod.SPECIES])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    assert close(r2, METRICS["species"]["r2_allometry"], 0.01)
    assert METRICS["species"]["fold_scatter_at_fixed_mass"] > 10


def test_evolution():
    mod = load("03_evolution.py", "evolution")
    m = METRICS["evolution"]
    assert m["m_star_low_mu"] > m["m_star_high_mu"], \
        "safe environments should buy more maintenance"
    r_low = mod.growth_rate(m["m_star_low_mu"], mod.MU0)
    assert np.isfinite(r_low)


def test_hallmarks():
    mod = load("04_hallmarks.py", "hallmarks")
    m = METRICS["hallmarks"]
    assert m["n_edges"] == len(mod.EDGES)
    assert 0.4 < m["fraction_causal"] < 1.0


def test_clocks():
    m = METRICS["clocks"]
    assert m["clock_mae_M"] < 10
    assert m["methylation_only_intervention"]["clock_M"] == 30.0
    assert m["methylation_only_intervention"]["function_pct"] == 0.0
    assert 0.5 < m["clockM_years_per_pct_median"] < 5


def test_multidim():
    mod = load("06_multidim.py", "multidim")
    m = METRICS["multidim"]
    assert m["final_score"]["D four-way"] > m["final_score"]["control"]
    assert m["relapse_after_stop_D"] > 0
    assert m["max_real_eigenvalue_treated"] < 0, "treated state should be stable"
    base = mod.score_at(np.zeros(8))
    assert close(base, m["baseline_score_age60"], 0.05)


def test_reprogram():
    m = METRICS["reprogram"]
    assert m["unsafe_at_max_dose_long"] > 0.8
    assert m["unsafe_at_low_dose_short"] < 0.1
    assert m["max_safe_age_reduction"] > 0.3


def test_cancer():
    mod = load("08_cancer.py", "cancer")
    m = METRICS["cancer"]
    assert close(m["k0"], mod.K0, 1e-6)
    assert close(m["baseline_lifetime_risk"], 0.40, 0.01)
    assert close(m["risk_multiplier_per_doubling"], 2 ** m["m_stages"])
    mc = mod.simulate(1.0, n=20000, seed=7)["incidence_to_85"]
    assert close(mc, 0.40, 0.05)


def test_survival():
    m = METRICS["survival"]
    assert m["median_slower_aging"] > m["median_baseline"]
    assert m["healthspan_baseline"] < m["median_baseline"]
    assert close(m["floor_for_1pct_at_200"], -np.log(0.01) / 200, 0.01)
    assert m["p150_baseline"] < 1e-20


def test_evidence():
    mod = load("10_evidence.py", "evidence")
    m = METRICS["evidence"]
    assert close(m["events_needed_hr08"], mod.required_events(0.8), 0.01)
    assert m["n_with_healthspan_endpoint"] == 1
    assert m["minimum_trial_years"] == 11.0


def test_public_data():
    mod = load("11_public_data.py", "public")
    m = METRICS["public_data"]
    assert close(m["human_e0_baseline"], mod.life_expectancy(0.0), 0.01)
    assert m["human_e0_gain_at_23pct"] > 2.0
    assert close(m["human_e0_gain_at_23pct"] / m["human_e0_gain_at_26pct"], 23 / 26, 0.1)


def test_repair():
    mod = load("12_repair.py", "repair")
    m = METRICS["repair"]
    assert m["max_coverage"] == int(max(sum(r[1]) for r in mod.INTERVENTIONS))
    assert m["n_layers"] == len(mod.LAYERS)
    assert m["n_with_single_pct"] == sum(1 for l in mod.LAYERS if l[1] is not None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
