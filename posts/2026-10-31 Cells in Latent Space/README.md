# Cells in Latent Space — code and reproducibility notes

This directory holds the article, the simulator, eight R experiments, and the
results that the figures are drawn from. Everything runs in R; no Python step is
required.

```
index.qmd              the article (bilingual, Quarto)
R/00_setup.R           palette, PCA / diffusion map / metrics helpers
R/01_simulate_cells.R  the generative model (3,000 cells x 1,000 genes)
R/02_preprocessing.R   preprocessing ladder, split, metric bundle
R/04_autoencoders.R    R/torch modules (AE, VAE, VAE-NB) and training loop
experiments/E00_simulate.R      builds results/sim.rds
experiments/E01_pca.R           five preprocessing pipelines
experiments/E02_umap.R          3 x 3 UMAP grid + stability
experiments/E03_manifold.R      Swiss roll, PCA / UMAP / diffusion maps
experiments/E04_vae.R           AE and VAE training, latent-width sweep
experiments/E05_interpolation.R straight-line interpolation and its failures
experiments/E06_multiomics.R    unpaired RNA-protein alignment
experiments/E07_perturbation.R  state-dependent perturbation prediction
experiments/E08_real_data.R     10x PBMC3k (downloaded on first run)
results/               RDS files consumed by the article
figures/               standalone SVG copies of the main figures
data/                  10x PBMC3k tarball and cached count matrix
```

## Running

From this directory:

```bash
Rscript experiments/E00_simulate.R
Rscript experiments/E01_pca.R
Rscript experiments/E02_umap.R
Rscript experiments/E03_manifold.R
Rscript experiments/E04_vae.R
Rscript experiments/E05_interpolation.R
Rscript experiments/E06_multiomics.R
Rscript experiments/E07_perturbation.R
Rscript experiments/E08_real_data.R
```

E04 depends on nothing else and takes the longest (a few minutes on CPU);
E05 needs `results/vae_nb_d4.pt` written by E04. E08 downloads ~7.6 MB from
`cf.10xgenomics.com` on first run and then caches `data/pbmc3k_counts.rds`.

Then render the article from the repository root:

```bash
quarto render "posts/2026-10-31 Cells in Latent Space"
```

The article only reads the RDS files, so a render after the experiments finish
takes seconds.

## Dependencies

R 4.5 with: `ggplot2`, `scales`, `dplyr`, `tidyr`, `patchwork`, `plotly`,
`Matrix`, `irlba`, `FNN`, `igraph`, `uwot`, `torch` (CPU build), `knitr`,
`rmarkdown`, `ragg`. `MASS` ships with R.

`torch` needs its libtorch backend once per machine:
`torch::install_torch()`.

## Ground truth

The simulator stores the true factors (`t`, `a`, branch coordinate) alongside
the counts. Every metric in the article that mentions "truth" is computed by
comparing a learned embedding with those columns — never with a re-derived
estimate. The real-data section is explicitly different: PBMC3k has no ground
truth, and its labels come from marker scores plus k-means, which the article
states wherever they appear.

## Known limits

- The simulation is additive and low-dimensional by construction; it cannot
  show what happens when the true manifold is high-dimensional.
- The perturbation model is a synthetic linear-in-state response; real
  perturbation effects are not assumed to be linear in any latent coordinate.
- The PBMC3k annotation is marker-based and coarse (10 k-means clusters, 8
  marker sets); CD4/CD8 T cells are only partly resolved.
- Everything is CPU; no GPU timings are reported.
