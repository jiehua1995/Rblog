# =============================================================================
# E06_phase_scan.R — where in parameter space does order appear?
#
# Two phase diagrams on the same red/blue language:
#   * Vicsek: mean polarisation over (noise eta) x (density rho),
#   * Gray-Scott: fraction of the lattice carrying signal over (feed F) x (kill k).
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/02_vicsek_model.R")
source("R/03_cell_communication.R")

t0 <- Sys.time()

# ---- Vicsek phase diagram --------------------------------------------------
etas <- seq(0, 4, by = 0.5)
rhos <- c(1, 2, 4, 8)
vic <- do.call(rbind, lapply(etas, function(e) {
  do.call(rbind, lapply(rhos, function(r) {
    sim <- simulate_vicsek(N = 300, rho = r, eta = e, steps = 250, seed = 3)
    data.frame(eta = e, rho = r, phi = steady_phi(sim, 100))
  }))
}))

# ---- Gray-Scott phase diagram ---------------------------------------------
Fs <- seq(0.010, 0.060, length.out = 9)
ks <- seq(0.040, 0.070, length.out = 9)
gs <- do.call(rbind, lapply(Fs, function(F) {
  do.call(rbind, lapply(ks, function(k) {
    s <- simulate_gray_scott(n = 72, steps = 2500, F = F, k = k, seed_size = 5)
    data.frame(F = F, k = k, v_frac = mean(s$v > 0.2),
               patches = patch_count(s$v, 0.2),
               class = pattern_class(s$v))
  }))
}))

saveRDS(list(vicsek = vic, grayscott = gs),
        "results/e06_phase.rds")

cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
cat("\nVicsek phase diagram (mean phi):\n")
print(xtabs(phi ~ eta + rho, aggregate(phi ~ eta + rho, vic, mean)), digits = 2)
cat("\nGray-Scott classes:\n")
print(table(gs$class))
