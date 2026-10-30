# =============================================================================
# 02_vicsek_model.R — the Vicsek model of self-propelled particles.
#
# Every particle does three things at each step:
#   1. look at all particles within radius r,
#   2. adopt (approximately) their mean heading,
#   3. move forward at constant speed on a torus.
#
# Nothing in that list mentions a flock.  The flock is what the interaction
# between particles produces, which is exactly the object of study.
#
# Convention: theta is in radians, noise eta is drawn uniformly from
# [-eta/2, eta/2].  radius = 1 and v = 0.5 set the natural units.
# =============================================================================

# Pairwise torus distance for a 2-D or 3-D point set.
torus_dist <- function(pos, L) {
  n <- nrow(pos)
  out <- matrix(0, n, n)
  for (k in seq_len(ncol(pos))) {
    d <- outer(pos[, k], pos[, k], "-")
    d <- d - L * round(d / L)
    out <- out + d^2
  }
  sqrt(out)
}

# One update step. Returns the new headings and positions.
vicsek_step <- function(x, y, theta, L, r, eta, v) {
  n <- length(theta)
  d <- torus_dist(cbind(x, y), L)
  adj <- d < r                       # includes the particle itself
  cx <- as.vector(adj %*% cos(theta))
  cy <- as.vector(adj %*% sin(theta))
  theta <- atan2(cy, cx) + eta * (runif(n) - 0.5)
  x <- (x + v * cos(theta)) %% L
  y <- (y + v * sin(theta)) %% L
  list(x = x, y = y, theta = theta)
}

#' @param N number of particles
#' @param rho density (particles per unit area); sets L = sqrt(N / rho)
#' @param eta angular noise in radians
#' @param radius interaction radius
#' @param steps number of updates
#' @param record if > 0, store positions every `record` steps
simulate_vicsek <- function(N = 400, rho = 4, eta = 0.5, radius = 1,
                            v = 0.5, steps = 300, seed = 1, record = 0) {
  set.seed(seed)
  L <- sqrt(N / rho)
  x <- runif(N, 0, L); y <- runif(N, 0, L)
  theta <- runif(N, 0, 2 * pi)
  phi <- numeric(steps)
  snaps <- list()
  for (t in seq_len(steps)) {
    s <- vicsek_step(x, y, theta, L, radius, eta, v)
    x <- s$x; y <- s$y; theta <- s$theta
    phi[t] <- order_parameter(theta)
    if (record > 0 && t %% record == 0) {
      snaps[[length(snaps) + 1]] <- data.frame(x = x, y = y, theta = theta, t = t)
    }
  }
  list(phi = phi, L = L, x = x, y = y, theta = theta, snaps = snaps,
       params = list(N = N, rho = rho, eta = eta, radius = radius, v = v))
}

# Steady-state polarisation: mean phi over the last `tail` steps.
steady_phi <- function(sim, tail = 100) mean(tail(sim$phi, tail))

# ---- 3-D variant -----------------------------------------------------------
vicsek_step_3d <- function(pos, theta, phi_ang, L, r, eta, v) {
  n <- nrow(pos)
  d <- torus_dist(pos, L)
  adj <- d < r
  # mean heading vector in Cartesian coordinates
  ux <- rowSums(adj * matrix(cos(theta) * cos(phi_ang), n, n, byrow = TRUE))
  uy <- rowSums(adj * matrix(sin(theta) * cos(phi_ang), n, n, byrow = TRUE))
  uz <- rowSums(adj * matrix(sin(phi_ang), n, n, byrow = TRUE))
  norm <- sqrt(ux^2 + uy^2 + uz^2); norm[norm == 0] <- 1
  ux <- ux / norm; uy <- uy / norm; uz <- uz / norm
  # noise: perturb the two angles
  theta <- atan2(uy, ux) + eta * (runif(n) - 0.5)
  phi_ang <- asin(pmax(-1, pmin(1, uz))) + eta * (runif(n) - 0.5)
  pos <- (pos + v * cbind(cos(theta) * cos(phi_ang),
                          sin(theta) * cos(phi_ang),
                          sin(phi_ang))) %% L
  list(pos = pos, theta = theta, phi_ang = phi_ang)
}

simulate_vicsek_3d <- function(N = 250, rho = 3, eta = 0.4, radius = 1.2,
                               v = 0.5, steps = 200, seed = 1) {
  set.seed(seed)
  L <- (N / rho)^(1 / 3)
  pos <- matrix(runif(3 * N, 0, L), N, 3)
  theta <- runif(N, 0, 2 * pi); phi_ang <- runif(N, -pi / 2, pi / 2)
  phi <- numeric(steps)
  for (t in seq_len(steps)) {
    s <- vicsek_step_3d(pos, theta, phi_ang, L, radius, eta, v)
    pos <- s$pos; theta <- s$theta; phi_ang <- s$phi_ang
    u <- cbind(cos(theta) * cos(phi_ang), sin(theta) * cos(phi_ang), sin(phi_ang))
    phi[t] <- sqrt(sum(colSums(u)^2)) / N
  }
  list(pos = pos, theta = theta, phi_ang = phi_ang, phi = phi, L = L,
       params = list(N = N, rho = rho, eta = eta, radius = radius))
}
