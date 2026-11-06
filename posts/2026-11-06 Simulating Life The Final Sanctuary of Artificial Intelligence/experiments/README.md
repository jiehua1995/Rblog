# Simulating Life: The Final Sanctuary of Artificial Intelligence

Reproduction package for the post `index.qmd`. Six CPU-only Python scripts produce
every number and every figure in the article. Nothing is hand-drawn or hand-typed:
`experiments/outputs/metrics.json` is written by the scripts and read by the article.

- Article: `../index.qmd`
- Scripts: this directory
- Figures: `../figures/*.png` and `../figures/*.svg` (PNG is what the page shows)
- Numbers: `experiments/outputs/metrics.json`

## Environment

Python 3.10+; tested on Python 3.13 (Windows 11) with numpy 2.x, matplotlib 3.11,
torch 2.x (CPU), torchvision 0.x, scikit-learn 1.9. No GPU, no API key, no pretrained
weights.

```powershell
python -m pip install -r experiments/requirements.txt
# if pip picks a CUDA build, use the CPU wheel instead:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run

From the **post directory** (the parent of `experiments/`):

```powershell
python experiments/run_all.py        # all six, in order
python experiments/run_all.py 2 3    # only 02_image.py and 03_physics.py
```

Or one at a time:

```powershell
python experiments/01_language.py       # artificial grammar, two GRUs, parsing test
python experiments/02_image.py          # 2-D VAE on Fashion-MNIST
python experiments/03_physics.py        # driven pendulum: black box vs structured
python experiments/04_virtual_cell.py   # ten-gene network: attractors, hysteresis
python experiments/05_perturbation.py   # knockout prediction, four-model ladder
python experiments/06_conceptual.py     # paradigm, four worlds, matrix, loop
```

`01`–`03` take a few minutes each on a laptop CPU; `04` and `05` are the slowest
(the perturbation experiment trains on 3,000 trajectories); `02` needs the
Fashion-MNIST download on first run, and falls back to `sklearn.datasets.load_digits`
(upsampled to 28×28) when `torchvision` or the network is unavailable.

Order does not matter for correctness — each script seeds its own RNG and merges its
own block into `metrics.json` — but `06_conceptual.py` should run last if you want the
conceptual figures regenerated after the experimental ones.

## What each script measures

| script | model | metric | key result |
|---|---|---|---|
| `01_language.py` | 1-layer GRU (hidden 32 / 96), next-token cross-entropy | grammaticality by an independent parser; log-probability margin for long-distance agreement; bits/token inside vs outside training lengths | 99.8 % / 98.8 % grammatical; agreement margin falls 6.31 → 0.42 nats as distance grows to 15–30 tokens; OOD cost rises to 4.79 bits/token |
| `02_image.py` | VAE, 2-D latent, MLP encoder/decoder | ELBO, reconstruction MSE, four OOD probes, latent geometry | val ELBO 270.2; recon MSE 0.0305; inversion costs 11× the in-distribution error |
| `03_physics.py` | damped driven pendulum; black-box increment predictor vs structured (acceleration + RK4) | 100- and 400-step rollout MSE, energy drift, Lyapunov exponent of the true system | black box is more accurate at every damping (e.g. 0.0060 vs 0.0366); structured model halves the energy drift (0.0646 vs 0.1461); λ ≈ 0.311 per time unit |
| `04_virtual_cell.py` | 10 genes, 20 edges, Hill activation | number of attractors, PCA variance, hysteresis gap | 4 attractors; PC1+PC2 = 86.1 %; hysteresis 1.66 |
| `05_perturbation.py` | persistence / MLP / graph net / mechanistic | MSE normalised by response variance, direction accuracy, long-horizon MSE, edge-damage test | held-out knockouts: graph net 0.1045 vs MLP 0.4787; double knockouts: 0.0990 vs 0.3291; 200-step horizon reverses it (MLP 0.1178 best); 30 % edge damage: 0.1255 |
| `06_conceptual.py` | none (diagrams) | — | paradigm ladder, four state spaces, 12×4 qualitative matrix, predict–explain–design–validate loop |

## Figures

| # | file | source |
|---|---|---|
| 1 | `fig-paradigm.png` | `06` |
| 2 | `fig-four-worlds.png` | `06` |
| 3 | `fig-language-curves.png` | `01` |
| 4 | `fig-language-generalization.png` | `01` |
| 5 | `fig-image-recon.png` | `02` |
| 6 | `fig-image-latent.png` | `02` |
| 7 | `fig-image-ood.png` | `02` |
| 8 | `fig-physics-rollout.png` | `03` |
| 9 | `fig-physics-generalize.png` | `03` |
| 10 | `fig-physics-chaos.png` | `03` |
| 11 | `fig-cell-network.png` | `04` |
| 12 | `fig-cell-dynamics.png` | `04` |
| 13 | `fig-cell-attractors.png` | `04` |
| 14 | `fig-cell-perturbation.png` | `05` |
| 15 | `fig-cell-ko.png` | `05` |
| 16 | `fig-cell-negative.png` | `05` |
| 17 | `fig-cross-domain.png` | `06` |
| 18 | `fig-loop.png` | `06` |

`common/style.py` sets the shared palette and exports PNG (300 dpi) plus SVG;
`common/utils.py` holds the seeds, paths and the JSON writer; `common/grn.py` is the
ten-gene model shared by `04` and `05`.

## Seeds and determinism

Every script calls `set_seed(0)` (Python `random`, NumPy, PyTorch). Data splits use
explicit `torch.Generator` seeds. PyTorch CPU kernels are deterministic for these
models, so reruns reproduce the numbers quoted in the article; small last-digit
differences can still appear across BLAS versions or thread counts
(`torch.set_num_threads(4)` in `02`).

## Limitations

- **Toy models.** The virtual cell has no membrane, metabolism, space, division or
  evolution. Its four attractors are a property of the equations we wrote, not a claim
  about any real cell. The same applies to the grammar (invented), the garment images
  (Fashion-MNIST) and the pendulum (a textbook system).
- **The structured models receive prior knowledge.** The graph network and the
  mechanistic model are given the true interaction matrix; their advantage measures
  the value of that prior, not what a model could discover on its own.
- **No error bars across seeds.** Each configuration is trained once, so differences
  smaller than roughly 10 % in the tables should not be read as reliable.
- **The 12×4 matrix is the author's qualitative rating** (1–3), not a measurement. The
  figure and the article both say so.
- **The negative result is kept.** The black box beats the structured model on
  short-horizon rollouts, and the graph network loses to the MLP at a 200-step horizon.
  Both are reported as they came out.
- **Different tasks, different losses.** Cross-experiment loss values are never
  compared; only within-task differences are used as evidence.
- **AlphaFold, AlphaGenome, STATE and the Virtual Cell Challenge are described from
  their papers and institutional pages**, not re-run here. Links are in the article's
  reference list.

## Render the post

From the repository root:

```powershell
quarto render "posts/2026-11-06 Simulating Life The Final Sanctuary of Artificial Intelligence"
```

The setup chunk reads `experiments/outputs/metrics.json` with `jsonlite`, so run the
experiments before rendering. `metrics.json` is valid strict JSON (NaN and infinities
are written as `null`).
