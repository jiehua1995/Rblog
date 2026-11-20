# Metrics shared by the experiments.

# Adjusted Rand index, implemented to avoid an extra dependency.
ari <- function(x, y) {
  tab <- table(x, y)
  n <- sum(tab)
  comb2 <- function(v) sum(v * (v - 1) / 2)
  index <- comb2(tab)
  a <- comb2(rowSums(tab))
  b <- comb2(colSums(tab))
  expected <- a * b / (n * (n - 1) / 2)
  max_index <- (a + b) / 2
  (index - expected) / (max_index - expected)
}

# Differential expression with limma on a log-scale matrix.
limma_de <- function(logx, group) {
  design <- stats::model.matrix(~ factor(group))
  fit <- limma::eBayes(limma::lmFit(as.matrix(logx), design))
  tab <- limma::topTable(fit, coef = 2, number = Inf, sort.by = "none")
  data.frame(id = rownames(tab), logFC = tab$logFC,
             p = tab$P.Value, adj = tab$adj.P.Val)
}

evaluate_de <- function(p, truth, alpha = 0.05) {
  called <- p < alpha
  tp <- sum(called & truth)
  fp <- sum(called & !truth)
  c(tpr = tp / max(1, sum(truth)),
    fdr = fp / max(1, sum(called)),
    n_called = sum(called))
}

lfc_metrics <- function(est, true, mask = rep(TRUE, length(est))) {
  e <- est[mask]; t <- true[mask]
  c(bias = mean(e - t), rmse = sqrt(mean((e - t)^2)),
    cor = stats::cor(e, t))
}

# Median within-group coefficient of variation across features.
sample_cv <- function(x, group) {
  groups <- split(seq_len(ncol(x)), group)
  cv <- vapply(seq_len(nrow(x)), function(i) {
    v <- vapply(groups, function(idx) {
      z <- x[i, idx]
      if (any(!is.finite(z))) return(NA_real_)
      stats::sd(z) / abs(mean(z))
    }, numeric(1))
    mean(v, na.rm = TRUE)
  }, numeric(1))
  stats::median(cv, na.rm = TRUE)
}

# PCA coordinates for samples, computed on scaled features.
pca_samples <- function(logx, group) {
  z <- t(scale(t(as.matrix(logx))))
  z[!is.finite(z)] <- 0
  pc <- stats::prcomp(t(z), center = TRUE, scale. = FALSE)
  data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2],
             group = factor(group), sample = colnames(logx))
}

jaccard <- function(a, b) {
  length(intersect(a, b)) / max(1, length(union(a, b)))
}
