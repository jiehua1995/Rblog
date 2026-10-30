# =============================================================================
# E05_cell_comm_pattern.R — one signalling rule, three different tissues.
#
# Same equations, same lattice, same initial seed size; only the feed rate F
# and the kill rate k change.  The three regimes are
#   * nothing (uniform),
#   * spots,
#   * labyrinthine stripes.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/03_cell_communication.R")

cfg <- list(
  "uniform"  = c(F = 0.010, k = 0.060),
  "spots"    = c(F = 0.035, k = 0.065),
  "stripes"  = c(F = 0.030, k = 0.055)
)

t0 <- Sys.time()
out <- lapply(names(cfg), function(nm) {
  p <- cfg[[nm]]
  s <- simulate_gray_scott(n = 96, steps = 4000, F = p[["F"]], k = p[["k"]],
                           seed_size = 6, record = 200)
  list(name = nm, F = p[["F"]], k = p[["k"]],
       metrics = s$metrics, snaps = s$snaps, final = s$v)
})
names(out) <- names(cfg)

saveRDS(out, "results/e05_patterns.rds")
cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
for (nm in names(out)) {
  m <- out[[nm]]$metrics
  cat(sprintf("%-8s F=%.3f k=%.3f  final v>0.2: %.3f  patches %d  ac_len %d  class %s\n",
              nm, out[[nm]]$F, out[[nm]]$k, m$v_frac[nrow(m)],
              m$patches[nrow(m)], m$ac_len[nrow(m)], pattern_class(out[[nm]]$final)))
}
