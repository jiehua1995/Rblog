# =============================================================================
# 04_metrics.R — macroscopic observables.
#
# The point of the whole article: a picture is not evidence.  These functions
# turn a lattice or a point cloud into numbers that can be compared across
# parameters.  Each one is deliberately simple so it can be justified in words.
# =============================================================================

# ---- binary lattices -------------------------------------------------------

# Fraction of occupied sites. Trivial, but it separates "dense" from "structured".
live_fraction <- function(m) mean(m > 0)

# Shannon entropy of the distribution of k x k blocks, normalised by k^2 bits
# (the maximum for binary blocks).  0 = uniform lattice, 1 = all blocks equally
# likely, which is what iid noise gives.
block_entropy <- function(m, k = 3) {
  n <- nrow(m); p <- ncol(m)
  codes <- integer(n * p)
  for (dr in 0:(k - 1)) {
    for (dc in 0:(k - 1)) {
      codes <- codes * 2L + as.integer(shift_torus(m, dr, dc))
    }
  }
  pr <- tabulate(codes + 1L) / length(codes)
  entropy_bits(pr) / (k^2)
}

# Connected components of occupied sites under 8-neighbourhood (4 for the 1-D
# case).  Returns count, largest component size and its share.
components <- function(m, eight = TRUE) {
  idx <- which(m > 0, arr.ind = TRUE)
  if (nrow(idx) == 0) return(c(n = 0, largest = 0, largest_frac = 0))
  if (nrow(idx) == 1) return(c(n = 1, largest = 1, largest_frac = 1 / sum(m > 0)))
  n <- nrow(m); p <- ncol(m)
  key <- function(r, c) (r - 1) * p + c
  cells <- key(idx[, 1], idx[, 2])
  present <- logical(n * p); present[cells] <- TRUE
  offs <- if (eight) expand.grid(dr = -1:1, dc = -1:1) else expand.grid(dr = 0, dc = -1:1)
  offs <- offs[!(offs$dr == 0 & offs$dc == 0), ]
  from <- integer(0); to <- integer(0)
  for (i in seq_len(nrow(offs))) {
    rr <- ((idx[, 1] - 1 + offs$dr[i]) %% n) + 1
    cc <- ((idx[, 2] - 1 + offs$dc[i]) %% p) + 1
    nb <- key(rr, cc)
    ok <- present[nb]
    from <- c(from, cells[ok]); to <- c(to, nb[ok])
  }
  g <- igraph::graph_from_edgelist(
    cbind(as.character(from), as.character(to)), directed = FALSE)
  comp <- igraph::components(g)
  sizes <- comp$csize
  c(n = comp$no, largest = max(sizes), largest_frac = max(sizes) / length(cells))
}

# Fraction of sites that keep their state from one step to the next.
persistence <- function(m1, m2) mean(m1 == m2)

# Spatial autocorrelation of a binary lattice at lags 1..max_lag (horizontal),
# i.e. the Pearson correlation between m and m shifted by h columns.
spatial_autocorr <- function(m, max_lag = 10) {
  vm <- matrix(as.vector(m) - mean(m), nrow(m), ncol(m))
  denom <- sum(vm^2)
  vapply(seq_len(max_lag),
         function(h) sum(vm * shift_torus(vm, 0, h)) / denom, numeric(1))
}

# Correlation length: first lag where autocorrelation drops below 1/e.
corr_length <- function(ac) {
  hit <- which(ac < exp(-1))[1]
  if (is.na(hit)) length(ac) else hit
}

# ---- continuous fields (reaction-diffusion) --------------------------------

field_autocorr <- function(f, max_lag = 30) {
  f <- f - mean(f)
  denom <- sum(f^2)
  vapply(seq_len(max_lag), function(h) sum(f * shift_torus(f, 0, h)) / denom, numeric(1))
}

# Number of connected "high" patches after thresholding at the mean.
patch_count <- function(f, thr = mean(f)) {
  components(matrix(as.integer(f > thr), nrow(f), ncol(f)))[["n"]]
}

# Rough compressibility: gzip size of the raw lattice over its raw size.
# Low values mean the pattern is predictable from a short description.
compressibility <- function(m) {
  raw <- as.raw(as.vector(m))
  length(memCompress(raw, "gzip")) / length(raw)
}

# ---- point clouds (Vicsek) -------------------------------------------------

# Polarisation order parameter: |sum of unit heading vectors| / N, in [0, 1].
order_parameter <- function(theta) Mod(sum(exp(1i * theta))) / length(theta)
