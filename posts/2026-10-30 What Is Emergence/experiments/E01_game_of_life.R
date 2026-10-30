# =============================================================================
# E01_game_of_life.R — does a three-line rule produce a persistent object?
#
# Three initial conditions are run on the same 120 x 120 torus:
#   * a random soup (28% alive),
#   * the r-pentomino (5 cells, the classic long transient),
#   * a single glider (a travelling, self-maintaining structure).
#
# For each we record the live fraction, block entropy, number of components,
# largest-component share and frame-to-frame persistence.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")

t0 <- Sys.time()
steps <- 300; every <- 5

cfgs <- list(
  random     = gol_random(120, 0.28, seed = 11),
  r_pentomino = gol_pattern("r_pentomino", 120, at = c(60, 60)),
  glider     = gol_pattern("glider", 120, at = c(60, 60))
)

metrics <- list(); frames <- list()
for (nm in names(cfgs)) {
  sim <- gol_simulate(cfgs[[nm]], steps = steps, keep_every = every)
  frames[[nm]] <- sim
  prev <- NULL
  for (i in seq_along(sim)) {
    m <- sim[[i]]
    cp <- components(m)
    metrics[[length(metrics) + 1]] <- data.frame(
      system = nm, gen = (i - 1) * every,
      live = live_fraction(m),
      entropy = block_entropy(m),
      comps = cp[["n"]],
      largest_frac = cp[["largest_frac"]],
      persistence = if (is.null(prev)) NA else persistence(prev, m)
    )
    prev <- m
  }
}

mets <- do.call(rbind, metrics)

# six snapshots of the r-pentomino run for the figure
gens_wanted <- c(0, 30, 60, 120, 200, 300)
idx <- match(gens_wanted, unique(mets$gen))
snaps <- frames$r_pentomino[idx]

# ---- robustness: does the glider survive a single flipped cell? ------------
# The glider is a macroscopic object, so the interesting question is not
# whether it exists but whether it is robust.  Each cell of its 3 x 3 bounding
# box is flipped in turn, the system is run for 40 further generations, and we
# ask whether a single 5-cell structure is still there.
base <- gol_pattern("glider", 40, at = c(20, 20))
survive <- function(m, gens = 40) {
  for (i in seq_len(gens)) m <- gol_step(m)
  cp <- components(m)
  list(live = sum(m), comps = cp[["n"]])
}
robust <- do.call(rbind, lapply(1:3, function(r) {
  do.call(rbind, lapply(1:3, function(cc) {
    m <- base
    m[19 + r, 19 + cc] <- 1 - m[19 + r, 19 + cc]
    s <- survive(m)
    data.frame(flipped_row = r, flipped_col = cc,
               live = s$live, comps = s$comps,
               intact = s$live == 5 && s$comps == 1)
  }))
}))
robust$control <- "single flipped cell"
ctrl <- survive(base)
robust <- rbind(robust, data.frame(flipped_row = NA, flipped_col = NA,
                                   live = ctrl$live, comps = ctrl$comps,
                                   intact = ctrl$live == 5 && ctrl$comps == 1,
                                   control = "no perturbation"))

saveRDS(list(metrics = mets, snaps = snaps, gens = gens_wanted,
             random_final = frames$random[[length(frames$random)]],
             glider_final = frames$glider[[length(frames$glider)]],
             robust = robust, steps = steps, every = every),
        "results/e01_gol.rds")

cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
tail_mets <- subset(mets, gen == max(mets$gen))
print(tail_mets, row.names = FALSE, digits = 3)
cat("\nglider robustness:", sum(robust$intact[!is.na(robust$flipped_row)]), "of 9",
    "single-cell flips still give one 5-cell object; unperturbed live =",
    ctrl$live, "comps =", ctrl$comps, "\n")
