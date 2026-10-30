# =============================================================================
# 00_setup.R — palette, theme overlay, and reusable numerical helpers
# Sourced by every experiment script and by the article's setup chunk.
# Run scripts with the post directory as the working directory.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(Matrix)
  library(FNN)
  library(dplyr)
  library(tidyr)
})

# ---- palette ---------------------------------------------------------------
# Article palette (teal / deep teal / gold / accent red / ink) extended with
# Okabe-Ito hues so that up to eight categories stay distinguishable in
# greyscale and for colour-vision-deficient readers.
cell_pal <- c(
  teal   = "#159A9C",
  deep   = "#176B78",
  gold   = "#D4A547",
  red    = "#D95D62",
  blue   = "#0072B2",
  purple = "#CC79A7",
  green  = "#009E73",
  orange = "#E69F00",
  sky    = "#56B4E9",
  grey   = "#7F7F7F",
  ink    = "#28323C",
  mist   = "#F7FAFA"
)

# sequential gradient: deep teal -> teal -> gold
grad_teal_gold <- c("#176B78", "#159A9C", "#8FC7B8", "#D4A547")
grad_diverging <- c("#176B78", "#8FC7B8", "#F7FAFA", "#E9A6A9", "#D95D62")

# ggplot scale helpers (keep call sites short and consistent)
scale_colour_cell <- function(...) {
  scale_colour_manual(values = unname(cell_pal), ...)
}
scale_fill_cell <- function(...) {
  scale_fill_manual(values = unname(cell_pal), ...)
}
scale_colour_t <- function(...) {
  scale_colour_gradientn(colours = grad_teal_gold, ...)
}

# transparent-background variant for ggsave() of the standalone SVG copies
save_svg <- function(plot, file, width = 8, height = 4.4) {
  dir.create("figures", showWarnings = FALSE)
  ggsave(file.path("figures", file), plot, width = width, height = height,
         device = "svg", bg = "transparent")
}

# ---- small utilities -------------------------------------------------------
fmt <- function(x, d = 2) formatC(x, format = "f", digits = d)
pct <- function(x, d = 1) paste0(formatC(100 * x, format = "f", digits = d), "%")

relu <- function(x) pmax(x, 0)

zscore <- function(X) {
  X <- as.matrix(X)
  s <- apply(X, 2, sd)
  s[s == 0] <- 1
  sweep(sweep(X, 2, colMeans(X), "-"), 2, s, "/")
}

#' log(1 + CPM) for a cells x genes count matrix.
#' The library size is a per-cell quantity, so the scaling must be applied
#' row-wise; `t(counts) / lib` silently divides by the wrong entries whenever
#' nrow(counts) is not a multiple of length(lib).
logcpm <- function(counts, lib = Matrix::rowSums(counts)) {
  if (inherits(counts, "Matrix")) {
    M <- methods::as(Matrix::Diagonal(x = 1 / lib) %*% counts, "CsparseMatrix")
    M@x <- log1p(M@x * 1e4)          # log1p(0) = 0, so zeros stay zero
    M
  } else {
    M <- as.matrix(counts)
    log1p(sweep(M, 1, lib, "/") * 1e4)
  }
}

# ---- linear algebra --------------------------------------------------------

#' PCA via SVD on the centred matrix.
#' @param X cells x genes, already transformed.
#' @param d number of components.
pca_svd <- function(X, d = 10, center = TRUE, scale_gene = FALSE) {
  X <- as.matrix(X)
  if (scale_gene) X <- zscore(X)
  if (center) X <- sweep(X, 2, colMeans(X), "-")
  tot <- sum(X^2)
  if (d > min(dim(X)) / 2) {
    s <- svd(X)
    d <- min(d, length(s$d))
    fit <- list(u = s$u[, seq_len(d), drop = FALSE],
                d = s$d[seq_len(d)],
                v = s$v[, seq_len(d), drop = FALSE])
  } else {
    fit <- irlba::irlba(X, nv = d, nu = d)
  }
  scores <- fit$u %*% diag(fit$d, nrow = d)
  rownames(scores) <- rownames(X)
  colnames(scores) <- paste0("PC", seq_len(d))
  list(scores = scores, rotation = fit$v, sdev = fit$d / sqrt(max(1, nrow(X) - 1)),
       var_explained = fit$d^2 / tot, total_ss = tot)
}

#' Relative reconstruction error of a d-dimensional linear projection,
#' evaluated on held-out rows using the training rotation.
pca_recon_error <- function(X_train, X_test, d = 3, scale_gene = FALSE) {
  X_train <- as.matrix(X_train); X_test <- as.matrix(X_test)
  mu <- colMeans(X_train)
  sdv <- if (scale_gene) apply(X_train, 2, sd) else rep(1, ncol(X_train))
  sdv[sdv == 0] <- 1
  Ztr <- sweep(sweep(X_train, 2, mu, "-"), 2, sdv, "/")
  Zte <- sweep(sweep(X_test, 2, mu, "-"), 2, sdv, "/")
  fit <- irlba::irlba(Ztr, nv = d, nu = d)
  V <- fit$v
  rec <- (Zte %*% V) %*% t(V)
  sqrt(sum((Zte - rec)^2) / sum(Zte^2))
}

#' Principal-angle style subspace agreement: mean squared canonical correlation
#' between two column spaces (1 = identical, 0 = orthogonal).
subspace_overlap <- function(A, B) {
  A <- qr.Q(qr(as.matrix(A))); B <- qr.Q(qr(as.matrix(B)))
  k <- min(ncol(A), ncol(B))
  s <- svd(crossprod(A, B))$d[seq_len(k)]
  mean(pmin(s, 1)^2)
}

#' Procrustes similarity between two configurations (rotation + reflection +
#' isotropic scaling allowed). Returns the R-statistic in [0, 1].
procrustes_sim <- function(A, B) {
  A <- as.matrix(A); B <- as.matrix(B)
  A <- sweep(A, 2, colMeans(A), "-"); B <- sweep(B, 2, colMeans(B), "-")
  A <- A / sqrt(sum(A^2)); B <- B / sqrt(sum(B^2))
  s <- svd(crossprod(A, B))$d
  sum(s)
}

# ---- neighbourhood metrics -------------------------------------------------

#' Fraction of the k nearest neighbours in the high-dimensional space that are
#' also among the k nearest neighbours in the embedding. 1 = perfect.
knn_preserve <- function(X_high, Y_low, k = 10, n_sub = 1500, seed = 1) {
  set.seed(seed)
  n <- nrow(X_high)
  idx <- if (n > n_sub) sample(n, n_sub) else seq_len(n)
  Kh <- FNN::get.knnx(X_high, X_high[idx, , drop = FALSE], k = k)$nn.index
  Kl <- FNN::get.knnx(Y_low,  Y_low[idx, , drop = FALSE],  k = k)$nn.index
  mean(vapply(seq_along(idx), function(i) length(intersect(Kh[i, ], Kl[i, ])) / k, 0))
}

#' Spearman correlation between pairwise distances in two spaces (sampled).
dist_cor <- function(X, Y, n_sub = 400, seed = 1) {
  set.seed(seed)
  n <- nrow(X)
  idx <- if (n > n_sub) sample(n, n_sub) else seq_len(n)
  dx <- as.vector(dist(X[idx, , drop = FALSE]))
  dy <- as.vector(dist(Y[idx, , drop = FALSE]))
  cor(dx, dy, method = "spearman")
}

#' Mean silhouette width for a two-class labelling given a distance matrix.
silhouette2 <- function(label, D) {
  lab <- as.character(label)
  u <- unique(lab)
  if (length(u) < 2) return(NA_real_)
  s <- numeric(length(lab))
  for (i in seq_along(lab)) {
    same <- lab == lab[i]
    same[i] <- FALSE
    a <- if (any(same)) mean(D[i, same]) else 0
    b <- min(vapply(setdiff(u, lab[i]), function(k) mean(D[i, lab == k]), 0))
    s[i] <- (b - a) / max(a, b)
  }
  mean(s)
}

#' kNN label purity: fraction of k nearest neighbours carrying the same label.
knn_purity <- function(Y, label, k = 15, n_sub = 1500, seed = 1) {
  set.seed(seed)
  n <- length(label)
  idx <- if (n > n_sub) sample(n, n_sub) else seq_len(n)
  K <- FNN::get.knnx(Y, Y[idx, , drop = FALSE], k = k + 1)$nn.index[, -1, drop = FALSE]
  mean(vapply(seq_along(idx), function(i) mean(label[K[i, ]] == label[idx[i]]), 0))
}

# ---- manifold learning -----------------------------------------------------

#' Diffusion map (Coifman & Lafon 2006), alpha = 1 normalisation.
#' @param X cells x features (Euclidean geometry is assumed).
#' @param k neighbours used for the sparse kernel.
#' @param ndim number of non-trivial eigenvectors to return.
dmaps <- function(X, k = 15, ndim = 3) {
  X <- as.matrix(X)
  n <- nrow(X)
  nn <- FNN::get.knn(X, k = k)
  eps <- 1e-12
  dk <- pmax(nn$nn.dist[, k], eps)
  i_idx <- rep(seq_len(n), k)
  j_idx <- as.vector(nn$nn.index)
  d_ij <- as.vector(nn$nn.dist)
  # adaptive Gaussian kernel: bandwidth = geometric mean of local scales
  w <- exp(-d_ij^2 / (dk[i_idx] * dk[j_idx] + eps))
  W <- Matrix::sparseMatrix(i = i_idx, j = j_idx, x = w, dims = c(n, n))
  W <- (W + Matrix::t(W)) / 2
  d <- Matrix::rowSums(W); d[d <= 0] <- eps
  Dinv <- Matrix::Diagonal(x = 1 / d)
  K <- Dinv %*% W %*% Dinv                    # alpha = 1
  v <- pmax(Matrix::rowSums(K), eps)
  Dv <- Matrix::Diagonal(x = 1 / sqrt(v))
  S <- as.matrix(Dv %*% K %*% Dv)             # symmetric conjugate
  ee <- eigen(S, symmetric = TRUE)
  ord <- order(ee$values, decreasing = TRUE)
  vals <- ee$values[ord]; vecs <- ee$vectors[, ord]
  psi <- vecs / sqrt(v)                        # right eigenvectors of K
  use <- seq_len(ndim) + 1                     # drop trivial component
  lam <- vals[use]
  emb <- psi[, use, drop = FALSE] %*% diag(lam / (1 - lam), nrow = ndim)
  colnames(emb) <- paste0("DC", seq_len(ndim))
  list(embedding = emb, eigenvalues = vals, vectors = psi, kernel = W)
}

#' Geodesic (graph) distances from a kNN graph, used as a proxy for the
#' intrinsic distance on a sampled manifold.
graph_geodesic <- function(X, k = 10) {
  nn <- FNN::get.knn(X, k = k)
  n <- nrow(X)
  edges <- data.frame(from = rep(seq_len(n), k), to = as.vector(nn$nn.index),
                      w = as.vector(nn$nn.dist))
  g <- igraph::graph_from_data_frame(edges, directed = FALSE, vertices = data.frame(name = seq_len(n)))
  igraph::distances(g, weights = igraph::E(g)$w)
}

# ---- intrinsic dimension ---------------------------------------------------

#' Levina-Bickel maximum-likelihood estimate of local intrinsic dimension.
#' m_k(x) = [ (1/(k-1)) sum_{j<k} log(r_k(x)/r_j(x)) ]^{-1}
intrinsic_dim_mle <- function(X, k = 10, n_sub = 1200, seed = 1, trim = 0.05) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X)
  idx <- if (n > n_sub) sample(n, n_sub) else seq_len(n)
  nn <- FNN::get.knnx(X, X[idx, , drop = FALSE], k = k + 1)
  d <- nn$nn.dist[, -1, drop = FALSE]        # drop the self match at distance 0
  rk <- d[, k]
  denom <- rowSums(log(rk / d[, seq_len(k - 1), drop = FALSE])) / (k - 1)
  m <- 1 / denom
  m <- m[is.finite(m)]
  mean(m, trim = trim)
}

#' kNN regression R^2 on held-out cells: how recoverable is a latent variable
#' from the embedding, without assuming linearity.
knn_reg_r2 <- function(X, y, split, k = 25) {
  X <- as.matrix(X); tr <- split$train; te <- split$test
  nn <- FNN::get.knnx(X[tr, , drop = FALSE], X[te, , drop = FALSE], k = k)
  pred <- rowMeans(matrix(y[tr][as.vector(nn$nn.index)], nrow = length(te)))
  1 - sum((y[te] - pred)^2) / sum((y[te] - mean(y[tr]))^2)
}

# ---- factorial helper used by the simulation -------------------------------

#' Sigmoid with a given midpoint and steepness.
sigm <- function(x, mid = 0.5, slope = 10) 1 / (1 + exp(-slope * (x - mid)))

#' Load a results object written by an experiment script.
read_res <- function(file) readRDS(file.path("results", file))
