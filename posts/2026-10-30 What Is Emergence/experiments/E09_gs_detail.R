# =============================================================================
# E09_gs_detail.R — the same rule seen as a space-time diagram.
#
# A snapshots figure shows that a pattern exists; a kymograph shows how it got
# there and whether it keeps moving.  The pattern grows outwards from a seed,
# so the informative cut is not a row of the lattice but the *radial* profile:
# for each snapshot the lattice is averaged over rings of constant radius,
# which collapses the angular symmetry and leaves one curve per time step.
#
# Two parameter pairs are stored: one that makes structure, one that does not.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/03_cell_communication.R")

# Mean of v on rings of constant radius, in nbins bins from the centre outwards.
radial_profile <- function(v, nbins = 40) {
  n <- nrow(v); c0 <- (n + 1) / 2
  g <- expand.grid(i = seq_len(n), j = seq_len(n))
  r <- sqrt((g$i - c0)^2 + (g$j - c0)^2)
  bin <- factor(pmin(nbins, ceiling(r / (max(r) / nbins))), levels = seq_len(nbins))
  as.vector(tapply(v, bin, mean))
}

t0 <- Sys.time()
n <- 96; steps <- 4000; every <- 20
cfgs <- list(
  "spots (F=0.035, k=0.065)" = c(F = 0.035, k = 0.065),
  "uniform (F=0.010, k=0.060)" = c(F = 0.010, k = 0.060)
)

out <- lapply(names(cfgs), function(nm) {
  p <- cfgs[[nm]]
  s <- simulate_gray_scott(n = n, steps = steps, F = p[["F"]], k = p[["k"]],
                           seed_size = 6, record = every)
  prof <- do.call(rbind, lapply(s$snaps, radial_profile))
  list(regime = nm, prof = prof, times = s$metrics$t, metrics = s$metrics,
       params = p)
})
names(out) <- names(cfgs)

saveRDS(list(cases = out, n = n, steps = steps, every = every),
        "results/e09_gs_detail.rds")

cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
for (nm in names(out)) {
  m <- out[[nm]]$metrics
  cat(sprintf("%-28s v_frac %.3f -> %.3f  patches %d -> %d  corr_len %.1f -> %.1f\n",
              nm, m$v_frac[1], m$v_frac[nrow(m)],
              m$patches[1], m$patches[nrow(m)],
              m$ac_len[1], m$ac_len[nrow(m)]))
}
