# =============================================================================
# E04_vicsek_3d.R — the same rule, one dimension higher.
#
# The point of the 3-D run is not that 3-D is prettier: it is that nothing in
# the rule refers to the dimension.  A flock can form in any space in which
# "average the heading of your neighbours" is defined.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/02_vicsek_model.R")

runs <- lapply(c(0.2, 0.8, 1.6), function(e) {
  sim <- simulate_vicsek_3d(N = 250, rho = 3, eta = e, radius = 1.2,
                            v = 0.5, steps = 200, seed = 2)
  u <- cbind(cos(sim$theta) * cos(sim$phi_ang),
             sin(sim$theta) * cos(sim$phi_ang),
             sin(sim$phi_ang))
  mean_u <- colMeans(u)
  alignment <- as.vector(u %*% (mean_u / sqrt(sum(mean_u^2))))
  data.frame(x = sim$pos[, 1], y = sim$pos[, 2], z = sim$pos[, 3],
             dx = u[, 1], dy = u[, 2], dz = u[, 3],
             alignment = alignment, eta = e)
})
names(runs) <- c("eta 0.2", "eta 0.8", "eta 1.6")

phi <- sapply(runs, function(d) {
  u <- cbind(d$dx, d$dy, d$dz)
  sqrt(sum(colSums(u)^2)) / nrow(u)
})

saveRDS(list(runs = runs, phi = phi, L = (250 / 3)^(1 / 3)),
        "results/e04_vicsek3d.rds")

print(round(phi, 3))
