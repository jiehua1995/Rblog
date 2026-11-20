# Normalisation methods used in the post.
# All functions take a matrix with features in rows and samples in columns.
# The self-implemented versions follow the published formulas; the wrappers
# around DESeq2 / edgeR / limma / vsn / scran / sctransform are used as
# official references in the cross-checks.

# ---------------------------------------------------------------- total count
cpm <- function(counts) {
  lib <- colSums(counts)
  sweep(counts, 2, lib, "/") * 1e6
}

tpm <- function(counts, length) {
  rate <- counts / length
  sweep(rate, 2, colSums(rate), "/") * 1e6
}

rpkm <- function(counts, length) {
  rate <- counts / length
  sweep(rate, 2, colSums(counts), "/") * 1e9
}

# ------------------------------------------------------------ distribution
upper_quartile <- function(x, p = 0.75) {
  q <- apply(x, 2, function(v) stats::quantile(v[v > 0], probs = p, na.rm = TRUE))
  sweep(x, 2, q, "/") * mean(colSums(x))
}

quantile_norm <- function(x) {
  out <- preprocessCore::normalize.quantiles(as.matrix(x))
  dimnames(out) <- dimnames(x)
  out
}

loess_norm <- function(logx) {
  limma::normalizeCyclicLoess(as.matrix(logx), method = "fast")
}

# --------------------------------------------------- reference-sample ratios
# DESeq2-style median-of-ratios size factors.
median_ratios <- function(counts, type = c("ratio", "poscounts")) {
  type <- match.arg(type)
  counts <- as.matrix(counts)
  if (type == "ratio") {
    logc <- log(counts)
    keep <- rowSums(!is.finite(logc)) == 0
    lg <- rowMeans(logc[keep, , drop = FALSE])
    sf <- apply(counts[keep, , drop = FALSE], 2,
                function(v) stats::median(v / exp(lg)))
  } else {
    logc <- log(counts)
    logc[!is.finite(logc)] <- 0
    lg <- rowSums(logc) / ncol(counts)
    sf <- apply(counts, 2, function(v) {
      ok <- v > 0
      exp(stats::median(log(v[ok]) - lg[ok]))
    })
  }
  sf / exp(mean(log(sf)))
}

# edgeR-style relative log expression factors.  RLE works on proportions,
# not raw counts: otherwise the factor absorbs the library size.
rle_factors <- function(counts) {
  counts <- as.matrix(counts)
  lib <- colSums(counts)
  logc <- log(sweep(counts, 2, lib, "/"))
  keep <- rowSums(!is.finite(logc)) == 0
  gm <- rowMeans(logc[keep, , drop = FALSE])
  f <- exp(apply(logc[keep, , drop = FALSE], 2,
                 function(v) stats::median(v - gm)))
  f / exp(mean(log(f)))
}

# edgeR-style trimmed mean of M values.
tmm_factors <- function(counts, logratio_trim = 0.3, sum_trim = 0.05) {
  counts <- as.matrix(counts)
  lib <- colSums(counts)
  uq <- apply(counts, 2, function(v) stats::quantile(v[v > 0], 0.75))
  ref <- which.min(abs(uq - mean(uq)))
  f <- rep(1, ncol(counts))
  for (j in seq_len(ncol(counts))) {
    if (j == ref) next
    cj <- counts[, j]; cr <- counts[, ref]
    Nj <- lib[j]; Nr <- lib[ref]
    ok <- cj > 0 & cr > 0
    M <- log2((cj[ok] / Nj) / (cr[ok] / Nr))
    A <- 0.5 * log2((cj[ok] / Nj) * (cr[ok] / Nr))
    loM <- stats::quantile(M, logratio_trim)
    hiM <- stats::quantile(M, 1 - logratio_trim)
    loA <- stats::quantile(A, sum_trim)
    hiA <- stats::quantile(A, 1 - sum_trim)
    keep <- M >= loM & M <= hiM & A >= loA & A <= hiA
    w <- (Nj - cj[ok]) / Nj / cj[ok] + (Nr - cr[ok]) / Nr / cr[ok]
    f[j] <- 2^stats::weighted.mean(M[keep], w[keep])
  }
  f / exp(mean(log(f)))
}

# RLE and TMM factors are relative to library size; the effective library
# size is lib * factor.  Dividing by the factor alone leaves depth in place.
rle_norm <- function(counts) {
  sweep(counts, 2, colSums(counts) * rle_factors(counts), "/") *
    mean(colSums(counts))
}

tmm_norm <- function(counts) {
  sweep(counts, 2, colSums(counts) * tmm_factors(counts), "/") *
    mean(colSums(counts))
}

# --------------------------------------------------------- variance stabilising
vsn_norm <- function(x) {
  out <- vsn::justvsn(as.matrix(x))
  out <- if (inherits(out, "ExpressionSet")) vsn::exprs(out) else as.matrix(out)
  dimnames(out) <- dimnames(x)
  out
}

vst_norm <- function(counts) {
  coldata <- data.frame(row.names = colnames(counts),
                        grp = factor(rep("a", ncol(counts))))
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = round(as.matrix(counts)),
                                        colData = coldata, design = ~1)
  SummarizedExperiment::assay(DESeq2::vst(dds, blind = TRUE))
}

# --------------------------------------------------------------- proteomics
median_center_log <- function(logx) {
  sweep(logx, 2, apply(logx, 2, stats::median, na.rm = TRUE), "-")
}

# Simplified MaxLFQ: pairwise peptide log-ratios, then least squares.
# The full MaxLFQ also uses a Delaunay triangulation to choose pairs; this
# teaching version uses every pair with at least two shared peptides.
maxlfq_group <- function(y) {
  ns <- ncol(y)
  ratio <- numeric(0)
  pair <- list()
  for (j in seq_len(ns - 1)) {
    for (k in seq(j + 1, ns)) {
      ok <- is.finite(y[, j]) & is.finite(y[, k]) & y[, j] > 0 & y[, k] > 0
      if (sum(ok) >= 2) {
        ratio <- c(ratio, stats::median(log2(y[ok, j] / y[ok, k])))
        pair[[length(pair) + 1]] <- c(j, k)
      }
    }
  }
  if (length(ratio) == 0) return(rep(NA_real_, ns))
  X <- matrix(0, length(ratio), ns)
  for (i in seq_along(ratio)) {
    X[i, pair[[i]][1]] <- 1
    X[i, pair[[i]][2]] <- -1
  }
  fit <- stats::lm.fit(X[, -ns, drop = FALSE], ratio)
  out <- c(fit$coefficients, 0)
  out - mean(out)
}

maxlfq <- function(peptides, groups) {
  out <- matrix(NA_real_, nlevels(factor(groups)), ncol(peptides))
  rownames(out) <- levels(factor(groups))
  colnames(out) <- colnames(peptides)
  for (g in rownames(out)) {
    out[g, ] <- maxlfq_group(peptides[groups == g, , drop = FALSE])
  }
  out
}

# -------------------------------------------------------- single-cell methods
scran_factors <- function(counts, clusters = NULL) {
  sce <- SingleCellExperiment::SingleCellExperiment(list(counts = counts))
  if (is.null(clusters)) clusters <- scran::quickCluster(sce)
  sce <- scran::computeSumFactors(sce, clusters = clusters)
  SingleCellExperiment::sizeFactors(sce)
}

sctransform_norm <- function(counts) {
  v <- sctransform::vst(as.matrix(counts), verbosity = 0)
  v$y
}

# ------------------------------------------------------------- compositional
clr <- function(x, pseudocount = 0.5) {
  lx <- log(x + pseudocount)
  sweep(lx, 1, rowMeans(lx), "-")
}

pareto <- function(x) {
  m <- colMeans(x, na.rm = TRUE)
  s <- apply(x, 2, stats::sd, na.rm = TRUE)
  sweep(sweep(x, 2, m, "-"), 2, sqrt(s), "/")
}

# -------------------------------------------------------------- batch models
combat_norm <- function(logx, batch, group = NULL) {
  mod <- if (is.null(group)) NULL else stats::model.matrix(~ factor(group))
  sva::ComBat(dat = as.matrix(logx), batch = batch, mod = mod, par.prior = TRUE)
}
