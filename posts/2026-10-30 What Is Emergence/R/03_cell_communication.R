# =============================================================================
# 03_cell_communication.R — pattern formation from local cell-cell signalling.
#
# Model: Gray-Scott, read as a two-signal conversation between neighbouring
# cells on a square lattice (torus).
#
#   du/dt = Du lap(u) - u v^2 + F (1 - u)
#   dv/dt = Dv lap(v) + u v^2 - (F + k) v
#
# u = a ubiquitous substrate that diffuses fast; v = an autocatalytic signal
# that a cell produces when it senses enough v around it.  A cell never sees
# the global pattern: it only reads its own u, v and the four neighbours.
#
# The two control parameters are the feed rate F (how fast substrate is
# replenished) and the kill rate k (how fast the signal is removed).  Together
# they decide whether nothing happens, spots appear, or stripes appear.
# =============================================================================

gray_scott_step <- function(u, v, Du = 0.16, Dv = 0.08, F = 0.035, k = 0.065,
                            dt = 1) {
  uvv <- u * v * v
  u <- u + dt * (Du * laplacian_periodic(u) - uvv + F * (1 - u))
  v <- v + dt * (Dv * laplacian_periodic(v) + uvv - (F + k) * v)
  list(u = u, v = v)
}

# A square of signal in the middle of a uniform substrate: the minimal
# perturbation that can grow into a pattern.
gray_scott_ic <- function(n = 96, seed_size = 8, noise = 0) {
  u <- matrix(1, n, n)
  v <- matrix(0, n, n)
  c0 <- n %/% 2
  v[(c0 - seed_size):(c0 + seed_size), (c0 - seed_size):(c0 + seed_size)] <- 0.5
  u[(c0 - seed_size):(c0 + seed_size), (c0 - seed_size):(c0 + seed_size)] <- 0.5
  if (noise > 0) {
    set.seed(1)
    u <- u + matrix(runif(n * n, -noise, noise), n, n)
    v <- pmax(0, v + matrix(runif(n * n, -noise, noise), n, n))
  }
  list(u = u, v = v)
}

simulate_gray_scott <- function(n = 96, steps = 4000, F = 0.035, k = 0.065,
                                Du = 0.16, Dv = 0.08, dt = 1,
                                seed_size = 8, noise = 0,
                                record = 0, ic = NULL) {
  ic <- if (is.null(ic)) gray_scott_ic(n, seed_size, noise) else ic
  u <- ic$u; v <- ic$v
  snaps <- list(); metrics <- list()
  for (t in seq_len(steps)) {
    s <- gray_scott_step(u, v, Du, Dv, F, k, dt)
    u <- s$u; v <- s$v
    if (record > 0 && t %% record == 0) {
      snaps[[length(snaps) + 1]] <- v
      metrics[[length(metrics) + 1]] <- data.frame(
        t = t,
        v_mean = mean(v),
        v_frac = mean(v > 0.2),
        patches = patch_count(v, 0.2),
        ac_len = corr_length(field_autocorr(v, 40))
      )
    }
  }
  list(u = u, v = v, snaps = snaps,
       metrics = if (length(metrics)) do.call(rbind, metrics) else NULL,
       params = list(n = n, F = F, k = k, Du = Du, Dv = Dv, steps = steps))
}

# Classify a finished lattice into "none" / "spots" / "stripes" using the
# number of patches relative to the domain area.  This is a coarse label for
# the phase diagram, not a precise morphological classifier.
pattern_class <- function(v, thr = 0.2) {
  frac <- mean(v > thr)
  if (frac < 0.01) return("uniform")
  if (frac > 0.85) return("saturated")
  cp <- components(matrix(as.integer(v > thr), nrow(v), ncol(v)))
  if (cp[["n"]] <= 3) "labyrinth" else if (cp[["n"]] >= 8) "spots" else "mixed"
}
