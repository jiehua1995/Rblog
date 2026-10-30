# What Is Emergence? / 什么是涌现？

A bilingual (Chinese / English) post for the R Blog on emergence, built from three
reproducible simulation models and a set of macroscopic measurements.

- Rendered article: `index.qmd`
- Models: `R/`
- Experiments: `experiments/`
- Cached results: `results/*.rds` (read by the article; not tracked in git)
- Animations: `animations/*.gif`, `animations/eca-atlas.png` (generated, but tracked:
  they are part of the page)
- Lattice renderer: `python/render_animations.py` (numpy + PIL)

## Run order

Every R script is run from **this directory**, not from the repository root:

```powershell
Rscript experiments/E01_game_of_life.R      # ~8 s
Rscript experiments/E02_elementary_ca.R     # ~3 s
Rscript experiments/E03_vicsek_2d.R         # ~5 min  (three sweeps, 66 runs)
Rscript experiments/E04_vicsek_3d.R         # ~20 s
Rscript experiments/E05_cell_comm_pattern.R # ~10 s
Rscript experiments/E06_phase_scan.R        # ~3 min  (two phase diagrams)
Rscript experiments/E07_animations.R        # ~2 min  (Vicsek + pattern GIFs)
Rscript experiments/E08_vicsek_fluct.R      # ~3 min  (N = 120 … 1200, two phases)
Rscript experiments/E09_gs_detail.R         # ~10 s   (radial profiles for the kymograph)
Rscript experiments/E10_rule_space.R        # ~2 s    (all 256 elementary rules)
python python/render_animations.py          # ~1 min  (gol / gol-pair / eca GIFs, atlas)
```

Then render from the repository root:

```powershell
quarto render "posts/2026-10-30 What Is Emergence"
```

`E07` reads `results/e03_vicsek2d.rds` and `results/e05_patterns.rds`, so it must run
after `E03` and `E05`. `E02` reads `results/e01_gol.rds`. `E08`–`E10` are independent
of the rest and only need the files they write themselves.

The lattice animations are drawn in Python rather than with `ggplot`, because a
raster wants pixels: each cell becomes a solid block, so the window can be cropped
and the grid stays crisp. `E07` only handles the two point-cloud / field GIFs.

## Models

| file | what it is | key parameters |
|---|---|---|
| `R/01_ca_models.R` | Conway's Game of Life and elementary CA on a torus | lattice size, rule number, initial pattern |
| `R/02_vicsek_model.R` | Vicsek self-propelled particles, 2-D and 3-D | `N`, density `rho`, noise `eta`, radius `r` |
| `R/03_cell_communication.R` | Gray–Scott activator–inhibitor lattice | feed `F`, kill `k`, `Du`, `Dv` |
| `R/04_metrics.R` | macroscopic observables | — |
| `R/00_theme.R` | red/blue palette, GIF assembly, helpers | — |

The macroscopic observables are deliberately simple so they can be justified in the
prose: live fraction, block entropy (normalised Shannon entropy of 3×3 blocks),
connected components, frame-to-frame persistence, spatial autocorrelation, and
polarisation `Phi = |sum exp(i theta)| / N`.

## Notes on reproducibility

- All numbers in the article are inline R expressions reading the `.rds` files; none
  are typed by hand. Re-running the scripts changes them slightly on a different R
  version, and the text follows automatically.
- `results/` is gitignored (see the repository `.gitignore`); a fresh clone needs to
  re-run the experiments before it can render without the cached `_freeze`.
- `gganimate` and `gifski` are **not** used. The two field GIFs are assembled from
  `ragg` PNG frames with `magick::image_animate()`; the lattice GIFs are drawn cell
  by cell in `python/render_animations.py`.
- The site theme (`theme-blog.R`) is sourced from the repository root; `R/00_theme.R`
  resolves that path so the same file works from the post directory and from knitr.

## Known limits (also stated in the article)

- System sizes are small (`N = 300` for most Vicsek sweeps). The ordering transition
  is therefore a smooth crossover, and the finite-size panel (Figure 11c) is included
  precisely to show that small systems *look* more ordered. `E08` puts a number on it:
  in the disordered phase the order parameter shrinks roughly as `N^-1/2`.
- The Gray–Scott morphology labels (`uniform / spots / labyrinth / mixed / saturated`)
  are a coarse classifier based on connected components and signal fraction, not a
  proper morphological analysis.
- The "correlation length" is the first lag at which the horizontal autocorrelation
  drops below `1/e`; for oscillatory patterns (rule 250, fine stripes) this is 1 and
  should not be read as a physical length scale.
- No experimental data are used. Everything here is a model; the article says so
  explicitly wherever it draws a conclusion about real flocks, tissues or AI systems.
