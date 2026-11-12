# From Molecules to Cells — reproduction package

Ten CPU-only Python scripts produce every number and every figure in
`../index.qmd`. Nothing is hand-drawn or hand-typed: `outputs/metrics.json` is
written by the scripts and read by the article.

- Article: `../index.qmd`
- Scripts: this directory
- Figures: `../figures/*.png`, `../figures/*.svg`, and one animation
  `fig-rd.gif`
- Numbers: `outputs/metrics.json`
- Tests: `tests/test_core.py`

## Environment

Python 3.10+; tested with Python 3.13 on Windows 11, numpy 2.x, scipy 1.x,
matplotlib 3.11, scikit-learn 1.9, Pillow. No GPU, no network at run time, no
pretrained weights.

```powershell
python -m pip install -r experiments/requirements.txt
```

## Run

From the **post directory** (the parent of `experiments/`):

```powershell
python experiments/run_all.py          # all ten, in order
python experiments/run_all.py 3 7      # only 03_projection.py and 07_spatial.py
```

Or one at a time:

```powershell
python experiments/01_cost.py            # MD cost model and feasibility frontier
python experiments/02_identifiability.py # hidden state, Bayes error, information
python experiments/03_projection.py      # Mori-Zwanzig kernel and closure error
python experiments/04_thermo.py          # driven cycle, flux, entropy production
python experiments/05_bifurcation.py     # toggle bifurcation, Gillespie switching
python experiments/06_error.py           # free-energy error propagation, Sobol
python experiments/07_spatial.py         # reaction-diffusion and the flask test
python experiments/08_ladder.py          # six models on one error-cost plane
python experiments/09_framework.py       # layer diagram and three gates
python experiments/10_quantum.py         # energy scales, tunnelling, decoherence
```

Nothing is expensive: the whole set is a few minutes on a laptop. `05` is the
slowest (Gillespie on a 4×4 grid of β and Ω), `08` trains small scikit-learn
models, `07` runs 96×96 PDE grids and writes the GIF.

## What each script assumes

| script | model | key assumptions | headline result |
|---|---|---|---|
| `01_cost.py` | `wall = N (T/Δt) / throughput` | GROMACS 2024 on one A100 (148 ns/day at 23,558 atoms) and Anton 3 (100 µs/day at 10⁶ atoms) as calibration; H100 = 2× A100; 150 B/atom; ideal weak scaling | one hour of *E. coli* all-atom ≈ 4.3×10⁹ years even on 10,000 ideal GPUs; syn3A cycle ≈ 1.5×10⁹ years |
| `02_identifiability.py` | bistable toggle + noisy readout of the hidden bias δ | Gaussian readout, uniform prior on δ, fate = sign(a−b) at t=40 | Bayes error of fate rises from 0.013 to 0.285 as readout noise grows; I(structure;fate) = 0.78 of H = 1.00 |
| `03_projection.py` | 2 slow + 8 fast linear variables, damped-rotation fast block | exact linear Mori-Zwanzig kernel `K(τ)=A_sf exp(A_ff τ) A_fs`; Euler quadrature for the memory integral | Markov closure error grows from 1e-4 to 0.16 as ε goes 0.01→1; memory model stays at 0.004 (quadrature bias) |
| `04_thermo.py` | 3-state cycle with affinity A | local detailed balance; k_B = 1; symmetric rates | steady-state concentrations are uniform at every drive; σ = J·A; scaling all rates by 25 leaves p unchanged and multiplies σ by 25 |
| `05_bifurcation.py` | toggle switch + Goodwin loop | Hill n = 4, K = 1; Gillespie with system size Ω | β_c = 1.008; switching rate peaks at criticality and falls with Ω; Hopf onset near p = 8 |
| `06_error.py` | ΔΔG → K_d → Hill cascade → fate threshold | RT = 0.592 kcal/mol at 298 K; Gaussian ΔG errors; Sobol with 512 base samples | a buffered cascade has zero phenotype sd up to 1 kcal/mol; a near-threshold cascade reaches 0.33; the buffered regime is dominated by τ, the critical one by K₁ and ΔG |
| `07_spatial.py` | bistable Schlögl reaction + Gaussian source | 96×96 grid, D = 0.05, explicit Euler, reflecting boundaries | same total input ignites at σ = 3 and does nothing at σ = 10; a 0-D model predicts nothing in both cases |
| `08_ladder.py` | 6 slow + 6 fast ODE "cell" | tanh regulation, τ_fast = 0.1, phenotype threshold at 0.5; 600 train / 400 test initial states | fine model MSE ≈ 0 with exact parameters, 0.200 with 15% parameter error — worse than the coarse model's 0.032 at a tenth of the cost; with all six states observed, parameter error moves the error from 3.6e-5 to 0.16 |
| `09_framework.py` | none (diagrams) | — | layer bookkeeping and the three gates |
| `10_quantum.py` | scale estimates | kT at 300 K; WKB with a 5 kcal/mol barrier; collisional decoherence model | kT = 0.026 eV; proton tunnelling 3.6e-5 vs deuteron 1e-6 at 0.5 Å; 0.1 nm coherence dies in 5.7e-15 s |

The cost model is the one place where external numbers enter. Both benchmarks
are published performance figures, not measurements made here; the derivation
of atom-steps/s is in the docstring of `01_cost.py`. Ideal weak scaling is
optimistic: real runs will be slower.

## Figures

| # | file | source |
|---|---|---|
| 1 | `fig-cost-landscape.png` | `01` |
| 2 | `fig-cost-intuition.png` | `01` |
| 3 | `fig-cost-memory.png` | `01` |
| 4 | `fig-cost-coarsegrain.png` | `01` |
| 5 | `fig-identifiability.png` | `02` |
| 6 | `fig-information.png` | `02` |
| 7 | `fig-projection.png` | `03` |
| 8 | `fig-thermo.png` | `04` |
| 9 | `fig-thermo-stochastic.png` | `04` |
| 10 | `fig-bifurcation.png` | `05` |
| 11 | `fig-noise.png` | `05` |
| 12 | `fig-error.png` | `06` |
| 13 | `fig-rd.png` + `fig-rd.gif` | `07` |
| 14 | `fig-spatial.png` | `07` |
| 15 | `fig-ladder.png` | `08` |
| 16 | `fig-quantum.png` | `10` |
| 17 | `fig-layers.png` | `09` |
| 18 | `fig-gates.png` | `09` |

`common/style.py` sets the palette and writes PNG (300 dpi) plus SVG;
`common/utils.py` holds paths, seeds and the JSON writer.

## Tests

```powershell
python experiments/tests/test_core.py
```

Nine assertions: the cost formula reproduces the stored number; the Bayes error
is bounded and the data-processing inequality holds; the Markov closure error
grows with ε; equilibrium has zero flux and uniform occupancy; the analytic
β_c matches the numerical bifurcation; ΔG = −1 kcal/mol gives K_d ≈ 0.18; the
narrow source ignites and the broad one does not; the fine model beats the
coarse one and the perturbed fine model loses to it; kT and the isotope effect
are right. These are checks of formulas and stored values, not a re-run of the
expensive simulations.

## Seeds and determinism

Every script calls `set_seed(0)`; stochastic simulations use explicit
`numpy.random.default_rng` seeds. Results reproduce on the same machine. Small
differences can appear across numpy/BLAS versions and thread counts; the
article quotes only two or three significant digits.

## Limitations

- **Toy models.** The "cell" in `08` is a twelve-dimensional ODE; the "virtual
  cell" of `02` is a two-gene toggle; the spatial system of `07` is a bistable
  reaction–diffusion equation. None of them is a claim about a real cell.
- **Cost model.** Calibrated to two published benchmarks, assumes ideal weak
  scaling and 2 fs steps. Real MD at 10⁹ atoms is slower and harder to
  initialise; the model gives the optimistic bound.
- **Equilibrium and linearity.** `03` is a linear system, where the
  Mori-Zwanzig kernel is exact. Nonlinear systems have state-dependent kernels;
  the qualitative conclusion (memory is needed when ε is not small) is expected
  to survive, but the numbers are not transferable.
- **One seed per configuration.** No error bars across seeds; differences below
  roughly 10 % should not be over-read.
- **Thermodynamics.** `04` uses local detailed balance for elementary steps; it
  does not model enzyme memory or measured apparent affinities.
- **Quantum section.** Order-of-magnitude estimates with a first-order
  collisional decoherence model; it does not rule out specific claims of
  quantum coherence in biology.
- **No comparison of raw losses across tasks.** Every comparison is within one
  task and one metric.

## Render the post

From the repository root:

```powershell
quarto render "posts/2026-11-12 From Molecules to Cells Can Microphysics Derive Life"
```

The setup chunk reads `experiments/outputs/metrics.json` with `jsonlite`, so run
the experiments before rendering. The file is strict JSON (NaN and infinities
are written as `null`).
