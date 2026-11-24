# From peptide intensities to protein numbers: MaxLFQ, Top3, iBAQ, NSAF, emPAI.
# All functions take peptides x runs matrices of raw intensities with NA allowed.

# Median-of-ratios run normalisation, the step MaxQuant applies before MaxLFQ.
median_normalize <- function(mat) {
  lg <- log2(mat)
  ref <- which.max(colSums(is.finite(lg)))
  off <- apply(lg, 2, function(v) stats::median(v - lg[, ref], na.rm = TRUE))
  off[!is.finite(off)] <- 0
  off <- off - stats::median(off)
  2^sweep(lg, 2, off, "-")
}

# MaxLFQ: pairwise peptide ratios -> weighted least squares on log intensities.
# Returns a list with the fitted profile and the pairwise deltas.
maxlfq <- function(mat, min_shared = 1) {
  lg <- log2(mat)
  nr <- nrow(lg)
  nc <- ncol(lg)
  if (nc == 1) return(list(profile = mat[, 1], deltas = NULL, weights = NULL))
  pairs <- utils::combn(nc, 2)
  delta <- rep(NA_real_, ncol(pairs))
  weight <- rep(0, ncol(pairs))
  for (j in seq_len(ncol(pairs))) {
    a <- pairs[1, j]; b <- pairs[2, j]
    r <- lg[, a] - lg[, b]
    ok <- is.finite(r)
    if (sum(ok) >= min_shared) {
      delta[j] <- stats::median(r[ok])
      weight[j] <- sum(ok)
    }
  }
  L <- matrix(0, nc, nc)
  rhs <- numeric(nc)
  for (j in seq_len(ncol(pairs))) {
    if (weight[j] == 0) next
    a <- pairs[1, j]; b <- pairs[2, j]
    L[a, a] <- L[a, a] + weight[j]
    L[b, b] <- L[b, b] + weight[j]
    L[a, b] <- L[a, b] - weight[j]
    L[b, a] <- L[b, a] - weight[j]
    rhs[a] <- rhs[a] + weight[j] * delta[j]
    rhs[b] <- rhs[b] - weight[j] * delta[j]
  }
  x <- numeric(nc)
  idx <- 2:nc
  A <- L[idx, idx, drop = FALSE] + diag(1e-8, length(idx))
  x[idx] <- solve(A, rhs[idx])
  mu <- mean(lg, na.rm = TRUE)
  x <- x - mean(x) + mu
  list(profile = 2^x,
       deltas = data.frame(a = pairs[1, ], b = pairs[2, ],
                           delta = delta, weight = weight))
}

# Sum of observed peptide intensities (what MaxQuant's Intensity column does).
sum_intensity <- function(mat) {
  apply(mat, 2, function(v) {
    if (!any(is.finite(v))) return(NA_real_)
    sum(v, na.rm = TRUE)
  })
}

# Geometric mean of observed peptides.
mean_log_intensity <- function(mat) {
  2^apply(log2(mat), 2, mean, na.rm = TRUE)
}

# Top3: mean of the three most intense observed peptides.
top3_intensity <- function(mat, n = 3) {
  apply(mat, 2, function(v) {
    v <- sort(v[is.finite(v)], decreasing = TRUE)
    if (!length(v)) return(NA_real_)
    mean(utils::head(v, n))
  })
}

# iBAQ: total peptide intensity divided by the theoretical observable peptides.
ibaq <- function(mat, n_obs) {
  if (!is.finite(n_obs) || n_obs <= 0) return(NA_real_)
  sum(mat, na.rm = TRUE) / n_obs
}

# NSAF from spectral counts and protein length.
nsaf <- function(count, length) {
  d <- count / length
  d / sum(d, na.rm = TRUE)
}

# emPAI from observed and observable peptide counts.
empai <- function(observed, observable) {
  10^(observed / observable) - 1
}

# iBAQ fraction -> copies per cell, given total protein mass per cell.
copy_number <- function(ibaq_fraction, total_mass_pg = 200, mw_kda = 50) {
  ibaq_fraction * total_mass_pg * 1e-12 / (mw_kda * 1000) * 6.02214076e23
}

# TMT reporter ion with a fraction of co-isolated background: ratio compression.
tmt_observed <- function(true_a, true_b, background = 0) {
  c(a = true_a + background, b = true_b + background)
}
