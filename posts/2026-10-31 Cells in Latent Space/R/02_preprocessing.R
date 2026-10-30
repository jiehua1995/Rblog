# =============================================================================
# 02_preprocessing.R — the preprocessing ladder used in Experiment 1
#
# Every step is a documented, reversible choice, so that the article can talk
# about *what each choice costs* rather than presenting "the" pipeline.
# =============================================================================

#' Residualise every gene on a cell-level covariate (intercept included).
regress_cov <- function(X, v) {
  q <- qr(cbind(1, v))
  qr.resid(q, as.matrix(X))
}

#' Remove a group mean per gene (simple, explicit batch centring).
center_by_group <- function(X, g) {
  X <- as.matrix(X)
  for (lev in unique(g)) {
    i <- g == lev
    X[i, ] <- sweep(X[i, , drop = FALSE], 2, colMeans(X[i, , drop = FALSE]), "-")
  }
  X
}

#' Select the n most variable genes on the log scale (sparse-aware).
top_hvg <- function(X, n = 400) {
  if (inherits(X, "Matrix")) {
    m <- Matrix::colMeans(X)
    v <- Matrix::colMeans(X^2) - m^2
  } else {
    v <- apply(X, 2, var)
  }
  order(v, decreasing = TRUE)[seq_len(n)]
}

#' Build an analysis-ready matrix.
#' @param depth regress out log library size (technical covariate).
#' @param batch centre each batch.
#' @param hvg number of highly variable genes to keep (0 = all).
#' @param scale z-score genes.
build_input <- function(sim, depth = TRUE, batch = TRUE, hvg = 0, scale = TRUE) {
  X <- logcpm(sim$counts)
  if (depth)  X <- regress_cov(X, log(sim$truth$lib))
  if (batch)  X <- center_by_group(X, sim$truth$batch)
  if (hvg > 0) X <- X[, top_hvg(X, hvg), drop = FALSE]
  if (scale)  X <- zscore(X)
  X
}

#' Split cells into training and test sets once, so every method sees the same
#' partition (no leakage through re-drawing a split per method).
make_split <- function(n, prop = 0.7, seed = 7) {
  set.seed(seed)
  tr <- sample(n, floor(prop * n))
  list(train = tr, test = setdiff(seq_len(n), tr))
}

#' Metric bundle used across experiments.
#' Correlations are maximised over the first five components (any of them may
#' carry a factor); neighbourhood metrics use the first three, because that is
#' the dimension a reader would actually plot.
embedding_metrics <- function(emb, truth, n_pc = 5, knn_dim = 3) {
  emb <- as.matrix(emb)
  d <- ncol(emb)
  kd <- seq_len(min(knn_dim, d))
  # the latent space is put on a common scale so that no single factor
  # dominates the Euclidean metric used for neighbourhood comparisons
  Z <- scale(as.matrix(truth[, c("Z1", "Z2", "Z3")]))
  rho <- function(v) {
    r <- cor(emb[, seq_len(min(d, n_pc)), drop = FALSE], v, method = "spearman")
    max(abs(r), na.rm = TRUE)
  }
  list(
    r_t      = rho(truth$t),
    r_a      = rho(truth$a),
    r_branch = rho(truth$s_branch),
    knn      = knn_preserve(Z, emb[, kd, drop = FALSE], k = 10),
    dist_cor = dist_cor(Z, emb[, kd, drop = FALSE])
  )
}
