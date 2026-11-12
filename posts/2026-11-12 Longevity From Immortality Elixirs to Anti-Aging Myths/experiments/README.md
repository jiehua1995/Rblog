# Longevity — reproduction package

Twelve CPU-only Python scripts produce every number and every figure in
`../index.qmd`. Nothing is hand-drawn or hand-typed: `outputs/metrics.json` is
written by the scripts and read by the article.

- Article: `../index.qmd`
- Scripts: this directory
- Figures: `../figures/*.png` and `../figures/*.svg`
- Numbers: `outputs/metrics.json`
- Tests: `tests/test_core.py`

## Environment

Python 3.10+; tested with Python 3.13 on Windows 11, numpy 2.x, scipy 1.x and
matplotlib 3.11. No GPU, no network at run time, no pretrained weights.

```powershell
python -m pip install -r experiments/requirements.txt
```

## Run

From the **post directory** (the parent of `experiments/`):

```powershell
python experiments/run_all.py          # all twelve, in order
python experiments/run_all.py 4 9      # only 04_hallmarks.py and 09_survival.py
```

Or one at a time:

```powershell
python experiments/01_history.py       # timeline of claims and evidence gaps
python experiments/02_species.py       # cross-species lifespan vs body mass
python experiments/03_evolution.py     # Leslie matrix, maintenance ESS
python experiments/04_hallmarks.py     # hallmark network and evidence types
python experiments/05_clocks.py        # clocks vs a simulated damage state
python experiments/06_multidim.py      # 8-variable coupled ODE, relapse, synergy
python experiments/07_reprogram.py     # 2-D potential, safety window
python experiments/08_cancer.py        # multistage model, regeneration vs cancer
python experiments/09_survival.py      # Gompertz organs, competing risks, tails
python experiments/10_evidence.py      # evidence matrix and trial sample size
python experiments/11_public_data.py   # ITP effect sizes, "perfect transfer"
python experiments/12_repair.py        # ten-system repair inventory and coverage
```

Nothing is expensive: the whole set is under two minutes on a laptop. `07` is
the slowest (a 15 × 10 dose–duration scan with 300 Langevin cells per
condition).

## What each script assumes

| script | model | key assumptions | headline result |
|---|---|---|---|
| `01_history.py` | timeline + bookkeeping | dates from the *Shiji*, *Hanshu*, *Baopuzi* and standard histories; modern dates are publication years | 15 dated events, 8 mechanisms, median 18.5 years since demonstration, 0 completed human healthspan trials |
| `02_species.py` | cross-species regression | 17 hand-picked species, maximum reported lifespans, rounded | log–log slope 0.27, R² = 0.67, residual ≈ 22× at fixed mass |
| `03_evolution.py` | Leslie matrix, 80 age classes | quadratic fecundity cost, maintenance slows the Gompertz slope from maturity | ESS maintenance falls from 0.61 (low extrinsic mortality) to 0.30 (high) |
| `04_hallmarks.py` | curated directed network | 34 edges classified by the author as causal / correlative / hypothetical | 73.5% causal; inflammation, senescence, stem cells and communication are the hubs |
| `05_clocks.py` | 10-D damage state, two ridge clocks | linear methylation and phenotype projections, Gaussian noise | clock MAE ≈ 3.4–3.6 yr; a methylation-only shift gives −30 yr at 0% functional change |
| `06_multidim.py` | 8-variable coupled ODE | arbitrary composite score; treatment 40–60 yr then stop | four-way peaks at 0.397, relapses by 0.25; synergy ≤ 2.4×10⁻⁴ |
| `07_reprogram.py` | 2-D potential, Langevin | three wells, identity boundary d = 0.45, arbitrary dose units | 61% of conditions safe; identity loss 0 → 1 across the dose boundary |
| `08_cancer.py` | multistage (m = 6) + Gompertz | 40% lifetime risk at r = 1; hazard ∝ r^m | doubling division rate multiplies risk 64×; “≤2× risk” allows r ≈ 1.17 |
| `09_survival.py` | 8 independent organ Gompertz hazards | death = first failure; healthspan = 70% still event-free | median 83 → 150 yr if ageing is halved; 1% survival to 200 needs floor < 0.023/yr |
| `10_evidence.py` | evidence matrix + log-rank formula | scores are the author’s reading; α = 0.05, power 0.8 | 1 of 9 has weak healthspan evidence; HR 0.8 needs 631 events ≈ 2,102 people; 11 yr minimum |
| `11_public_data.py` | published ITP summary + Gompertz translation | rounded ITP values; proportional mortality reduction | 5 of 8 agents significant in males, 2 in females; perfect transfer of 23% gives +3.1 yr |
| `12_repair.py` | repair inventory + coverage matrix | percentages only where a single published number exists | 10 systems, 6 with a single percentage; best coverage 12/20 points |

## Figures

| # | file | source |
|---|---|---|
| 1 | `fig-timeline.png` | `01` |
| 2 | `fig-species.png` | `02` |
| 3 | `fig-evolution.png` | `03` |
| 4 | `fig-hallmarks.png` | `04` |
| 5 | `fig-clocks.png` | `05` |
| 6 | `fig-multidim.png` | `06` |
| 7 | `fig-reprogram.png` | `07` |
| 8 | `fig-cancer.png` | `08` |
| 9 | `fig-survival.png` | `09` |
| 10 | `fig-evidence.png` | `10` |
| 11 | `fig-itp.png` | `11` |
| 12 | `fig-repair.png` | `12` |

`common/style.py` sets the palette and writes PNG (300 dpi) plus SVG;
`common/utils.py` holds paths, seeds and the JSON writer.

## Tests

```powershell
python experiments/tests/test_core.py
```

Twelve assertions: the allometric fit reproduces the stored R²; the evolution
model selects more maintenance at lower extrinsic mortality; the hallmark edge
count matches the curated list; the methylation-only intervention is 30 years
at zero function; the four-way ODE beats the control and relapses; the
reprogramming scan has a safe and an unsafe corner; the cancer calibration is
40% and the Monte Carlo matches it; halving ageing extends median lifespan and
healthspan is below lifespan; the log-rank formula reproduces 631 events; the
ITP translation gives more than two years; and the coverage score matches the
matrix. These are checks of formulas and stored values, not a re-run of the
expensive simulations.

## Seeds and determinism

Every script calls `set_seed(0)`; stochastic simulations use explicit
`numpy.random.default_rng` seeds. Results reproduce on the same machine. Small
differences can appear across numpy/BLAS versions and thread counts; the
article quotes only two or three significant digits.

## Limitations

- **Toy models.** The “damage state”, the eight-variable ODE, the potential
  landscape and the organ survival model are constructions, not fits to data.
  They are designed to expose logical possibilities and arithmetic, not to
  estimate any real therapy.
- **Curated inputs.** The hallmark network, the evidence matrix and the
  coverage matrix are the author’s reading of the literature. They are not
  systematic reviews and their counts are not effect sizes.
- **Species sample.** The 17 species are deliberately chosen to span the range;
  they are not a random sample, and maximum lifespan is not median lifespan.
- **Cancer model.** The multistage model ignores immune surveillance and
  tissue-specific mutation spectra, so it cannot estimate the cancer risk of any
  real regenerative therapy.
- **Survival model.** Eight independent Gompertz organs are a simplification;
  real deaths are correlated across organs, and “healthspan = 70% event-free”
  is a crude demographic proxy.
- **ITP values.** Rounded from published reports; the “perfect transfer”
  calculation assumes a proportional mortality reduction at every age, which no
  human trial has demonstrated.
- **No error bars.** One seed per configuration; differences below roughly 10%
  should not be over-read.

## Render the post

From the repository root:

```powershell
quarto render "posts/2026-11-12 Longevity From Immortality Elixirs to Anti-Aging Myths"
```

The setup chunk reads `experiments/outputs/metrics.json` with `jsonlite`, so run
the experiments before rendering. The file is strict JSON (NaN and infinities
are written as `null`).
