# Experiments

Run from the post directory:

    python experiments/01_calibration.py
    python experiments/02_power.py
    python experiments/03_invariance.py
    python experiments/04_same_r.py
    python experiments/05_agreement.py
    python experiments/06_confound.py
    python experiments/07_timeseries.py
    python experiments/08_complexity.py

Each script writes its figures to `../figures` (PNG + SVG) and merges its
numbers into `outputs/metrics.json`, which `index.qmd` reads.  All measures are
implemented in `common/corr.py` on top of NumPy/SciPy; no specialist package is
used, so the definition in the article and the code are the same object.

## Data

* `data/penguins.csv` — Palmer penguins, from
  `allisonhorst/palmerpenguins` (`inst/extdata/penguins.csv`).
* `data/datasauzoo.csv` — the Datasaurus Dozen (13 synthetic datasets with the
  same summary statistics), exported from the CRAN package `datasauRus` 0.1.8,
  object `datasaurus_dozen`.

Both files are committed so that the scripts run offline.  `04_same_r.py` uses
Anscombe's quartet as literal values, as published in Anscombe (1973).

## Notes on implementation

* `mic()` is a grid approximation (equal-frequency bins, `k*l <= B(n)`), not the
  Reshef et al. ApproxMaxMI algorithm; the article labels it as an approximation.
* `hoeffding_d()` uses the empirical-copula form, which has range [0, 1]; the
  classical U-statistic formula has a different finite-sample scale.
* `cdCor` is approximated by the maximum lagged distance correlation over lags
  0..10; the article says so.
