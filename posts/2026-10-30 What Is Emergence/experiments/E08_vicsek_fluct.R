# =============================================================================
# E08_vicsek_fluct.R — what the noise actually does to the order parameter.
#
# Two questions that the phase curves cannot answer:
#   * in the disordered phase, does the apparent order shrink as 1/sqrt(N)?
#     (finite-size scaling: a small system looks ordered for free)
#   * how wide is the distribution of phi at a fixed parameter pair?
#
# Stored per time step so the article can plot both the histogram and the
# N dependence without re-running.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/02_vicsek_model.R")

t0 <- Sys.time()
steps <- 300; tail_steps <- 100
Ns <- c(120, 300, 700, 1200)
etas <- c(0.4, 4.2)          # ordered and disordered

series <- list()
for (e in etas) {
  for (n in Ns) {
    sim <- simulate_vicsek(N = n, rho = 4, eta = e, steps = steps, seed = 12)
    series[[length(series) + 1]] <- data.frame(
      N = n, eta = e, t = seq_along(sim$phi), phi = sim$phi)
  }
}
series <- do.call(rbind, series)

# summary over the stationary tail
summ <- do.call(rbind, lapply(split(series, list(series$eta, series$N)),
                              function(d) {
  tail_phi <- tail(d$phi, tail_steps)
  data.frame(N = d$N[1], eta = d$eta[1],
             mean_phi = mean(tail_phi), sd_phi = sd(tail_phi))
}))

saveRDS(list(series = series, summary = summ,
             Ns = Ns, etas = etas, steps = steps),
        "results/e08_vicsek_fluct.rds")

cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
print(summ[order(summ$eta, summ$N), ], row.names = FALSE, digits = 3)
