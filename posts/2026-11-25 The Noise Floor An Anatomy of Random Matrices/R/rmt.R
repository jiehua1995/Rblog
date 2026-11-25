# Random-matrix helpers: the Marchenko-Pastur law, spiked models,
# singular-value thresholds and shrinkage.

# ---- Marchenko-Pastur law ---------------------------------------------------
# For a p x n matrix with iid entries of variance sigma^2, the spectrum of
# S = (1/n) X X^T converges to a density on [lambda_-, lambda_+].
mp_bounds <- function(gamma, sigma = 1) {
  sigma^2 * (1 + c(-1, 1) * sqrt(gamma))^2
}

mp_density <- function(x, gamma, sigma = 1) {
  b <- mp_bounds(gamma, sigma)
  lo <- b[1]
  hi <- b[2]
  out <- rep(NA_real_, length(x))
  inside <- x >= lo & x <= hi & x > 0
  out[inside] <- sqrt((hi - x[inside]) * (x[inside] - lo)) /
    (2 * pi * sigma^2 * gamma * x[inside])
  out
}

# Fraction of the p eigenvalues that pile up at zero when gamma > 1.
mp_zero_mass <- function(gamma) pmax(0, 1 - 1 / gamma)

# ---- spectra of a data matrix ----------------------------------------------
# Standardise each feature (row) to mean 0, sd 1 so that sigma = 1 and the
# noise floor depends on gamma alone.
standardize_rows <- function(X) {
  mu <- rowMeans(X)
  s <- sqrt(rowMeans((X - mu)^2))
  keep <- is.finite(s) & s > 0
  X <- X[keep, , drop = FALSE]
  (X - mu[keep]) / s[keep]
}

# Eigenvalues of S = (1/n) X X^T, sorted decreasingly. When p > n we take the
# n x n Gram matrix instead: the non-zero eigenvalues are identical.
spectrum <- function(X, standardize = TRUE) {
  if (standardize) X <- standardize_rows(X)
  p <- nrow(X)
  n <- ncol(X)
  if (p <= n) {
    S <- tcrossprod(X) / n
    ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  } else {
    G <- crossprod(X) / n
    ev <- eigen(G, symmetric = TRUE, only.values = TRUE)$values
  }
  ev <- sort(pmax(ev, 0), decreasing = TRUE)
  list(values = ev, p = p, n = n, gamma = p / n)
}

# Overall permutation null: shuffle every entry. This keeps the marginal
# distribution of the standardised values but destroys all gene-gene and
# sample-gene structure. Returns a B x k matrix of the k largest eigenvalues.
perm_null_spectrum <- function(X, B = 100, k = 10, seed = 1) {
  set.seed(seed)
  X <- standardize_rows(X)
  p <- nrow(X)
  n <- ncol(X)
  v <- as.vector(X)
  k <- min(k, n)
  out <- matrix(NA_real_, B, k)
  for (b in seq_len(B)) {
    Xb <- standardize_rows(matrix(sample(v), p, n))
    G <- crossprod(Xb) / n
    ev <- sort(eigen(G, symmetric = TRUE, only.values = TRUE)$values,
               decreasing = TRUE)
    out[b, ] <- ev[seq_len(k)]
  }
  out
}

# ---- simulation -------------------------------------------------------------
sim_noise <- function(p, n, sigma = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  matrix(stats::rnorm(p * n, sd = sigma), p, n)
}

# Population covariance I + ell * u u^T, so one eigenvalue sits at 1 + ell.
sim_spike <- function(p, n, ell, sigma = 1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  u <- stats::rnorm(p)
  u <- u / sqrt(sum(u^2))
  Z <- matrix(stats::rnorm(p * n), p, n) * sigma
  X <- Z + (sqrt(1 + ell) - 1) * outer(u, as.vector(crossprod(u, Z)))
  list(X = X, u = u)
}

# Theoretical top eigenvalue of the spiked model.
spike_top <- function(gamma, ell, sigma = 1) {
  edge <- sigma^2 * (1 + sqrt(gamma))^2
  ifelse(ell <= sqrt(gamma), edge,
         sigma^2 * (1 + ell) * (1 + gamma / ell))
}

# Squared cosine between the top sample eigenvector and the true spike.
spike_cosine2 <- function(gamma, ell) {
  ifelse(ell <= sqrt(gamma), 0, (1 - gamma / ell^2) / (1 + gamma / ell))
}

# ---- denoising --------------------------------------------------------------
# Gavish-Donoho optimal hard threshold for singular values of an m x n matrix
# with noise sd sigma (m = min dim, N = max dim).
gd_tau <- function(p, n, sigma = 1) {
  m <- min(p, n)
  N <- max(p, n)
  beta <- m / N
  omega <- 0.56 * beta^3 - 0.95 * beta^2 + 1.82 * beta + 1.43
  omega * sqrt(N) * sigma
}

# Ledoit-Wolf shrinkage of the sample covariance toward mu * I.
lw_shrink <- function(Y) {
  Y <- Y - rowMeans(Y)
  p <- nrow(Y)
  n <- ncol(Y)
  S <- tcrossprod(Y) / n
  m <- sum(diag(S)) / p
  d2 <- sum((S - m * diag(p))^2) / p
  col_sq <- colSums(Y^2)
  b2bar <- (sum(col_sq^2) - n * sum(S^2)) / (n^2 * p)
  b2bar <- max(b2bar, 0)
  delta <- min(b2bar / d2, 1)
  list(cov = (1 - delta) * S + delta * m * diag(p),
       delta = delta, target = m)
}
