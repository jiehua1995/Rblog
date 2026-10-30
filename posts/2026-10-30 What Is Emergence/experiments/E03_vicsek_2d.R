# =============================================================================
# E03_vicsek_2d.R — from disordered motion to a flock, and back.
#
# Three sweeps, each averaged over the last 100 of 250 updates:
#   * noise eta at fixed density rho = 4,
#   * density rho at fixed noise eta = 2.5 (near the transition),
#   * system size N at four noise levels (finite-size broadening).
#
# Everything is stored so the article can plot curves, an animation and a
# snapshot without re-running the simulation.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/02_vicsek_model.R")

steps <- 250; tail_steps <- 100

# ---- noise sweep -----------------------------------------------------------
etas <- seq(0, 4.5, by = 0.25)
noise_sweep <- do.call(rbind, lapply(etas, function(e) {
  do.call(rbind, lapply(1:2, function(s) {
    sim <- simulate_vicsek(N = 300, rho = 4, eta = e, steps = steps, seed = s)
    data.frame(N = 300, eta = e, rho = 4, seed = s,
               phi = steady_phi(sim, tail_steps))
  }))
}))

# ---- density sweep ---------------------------------------------------------
rhos <- c(0.5, 1, 2, 3, 4, 6, 8, 12)
dens_sweep <- do.call(rbind, lapply(rhos, function(r) {
  do.call(rbind, lapply(1:2, function(s) {
    sim <- simulate_vicsek(N = 300, rho = r, eta = 2.5, steps = steps, seed = s)
    data.frame(N = 300, eta = 2.5, rho = r, seed = s,
               phi = steady_phi(sim, tail_steps))
  }))
}))

# ---- finite-size check -----------------------------------------------------
sizes <- c(120, 300, 700)
size_sweep <- do.call(rbind, lapply(sizes, function(n) {
  do.call(rbind, lapply(c(2.0, 2.5, 3.0, 3.5), function(e) {
    sim <- simulate_vicsek(N = n, rho = 4, eta = e, steps = steps, seed = 7)
    data.frame(N = n, eta = e, phi = steady_phi(sim, tail_steps))
  }))
}))

# ---- two representative trajectories ---------------------------------------
traj_cfg <- c("eta = 0.6" = 0.6, "eta = 3.4" = 3.4)
traj <- do.call(rbind, lapply(seq_along(traj_cfg), function(i) {
  sim <- simulate_vicsek(N = 300, rho = 4, eta = traj_cfg[i], steps = 200, seed = 3)
  data.frame(t = seq_along(sim$phi), phi = sim$phi, regime = names(traj_cfg)[i])
}))

# ---- animation: one run through the ordering transition --------------------
anim <- simulate_vicsek(N = 250, rho = 4, eta = 0.9, steps = 240, seed = 4,
                        record = 4)

# static snapshots of an ordered and a disordered run
snap_ordered  <- simulate_vicsek(N = 250, rho = 4, eta = 0.2, steps = 200, seed = 5)
snap_disorder <- simulate_vicsek(N = 250, rho = 4, eta = 4.2, steps = 200, seed = 5)

saveRDS(list(noise_sweep = noise_sweep, dens_sweep = dens_sweep,
             size_sweep = size_sweep, traj = traj,
             anim = anim$snaps, anim_L = anim$L,
             ordered = snap_ordered[c("x", "y", "theta", "L", "params")],
             disordered = snap_disorder[c("x", "y", "theta", "L", "params")],
             N = 300, steps = steps),
        "results/e03_vicsek2d.rds")

cat("noise sweep:\n"); print(aggregate(phi ~ eta, noise_sweep, mean), row.names = FALSE, digits = 3)
cat("\ndensity sweep:\n"); print(aggregate(phi ~ rho, dens_sweep, mean), row.names = FALSE, digits = 3)
cat("\nsize sweep:\n"); print(reshape(size_sweep[, c("N", "eta", "phi")], idvar = "eta",
                                      timevar = "N", direction = "wide"), row.names = FALSE, digits = 3)
