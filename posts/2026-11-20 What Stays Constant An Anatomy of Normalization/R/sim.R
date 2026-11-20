# Simulators with known ground truth.
# Every simulator returns the observed matrix plus a truth table, so that
# bias, RMSE, FDR and TPR can be computed against the generating parameters.

sim_counts <- function(n = 2000, m = 12, prop_de = 0.10, fold = 1,
                       fold_sd = 0.35, depth_sd = 0.50, shift_all = 0,
                       length_bias = 0, len_mean = 1500,
                       de_dir = c("balanced", "up", "down"), seed = 1) {
  de_dir <- match.arg(de_dir)
  set.seed(seed)
  group <- rep(1:2, each = m / 2)
  base <- exp(stats::rnorm(n, 6, 1.3))
  disp <- 0.02 + 1 / exp(stats::rnorm(n, 1.8, 0.8))
  lfc <- stats::rnorm(n, 0, 0.15)
  lengths <- round(exp(stats::rnorm(n, log(len_mean), 0.45)))

  de <- sample.int(n, round(n * prop_de))
  mag <- abs(stats::rnorm(length(de), fold, fold_sd))
  sgn <- switch(de_dir,
                balanced = sample(c(-1, 1), length(de), TRUE),
                up = rep(1, length(de)),
                down = rep(-1, length(de)))
  lfc[de] <- lfc[de] + sgn * mag

  if (shift_all != 0) lfc <- lfc + shift_all

  size <- exp(stats::rnorm(m, 0, depth_sd))
  size <- size / exp(mean(log(size)))
  mu <- outer(base, size) * 2^outer(lfc, as.numeric(group == 2)) *
    (lengths / stats::median(lengths))^length_bias
  counts <- matrix(stats::rnbinom(n * m, mu = as.vector(mu), size = 1 / disp),
                   nrow = n, ncol = m)
  colnames(counts) <- paste0("S", seq_len(m))
  rownames(counts) <- paste0("G", seq_len(n))

  list(counts = counts,
       length = lengths,
       group = group,
       size = size,
       truth = data.frame(id = rownames(counts), base = base, lfc = lfc,
                          de = seq_len(n) %in% de))
}

sim_proteins <- function(n = 1500, m = 12, prop_de = 0.10, fold = 1,
                         fold_sd = 0.35, missing = 0.15, mnar = TRUE,
                         load_sd = 0.50, seed = 1) {
  set.seed(seed)
  group <- rep(1:2, each = m / 2)
  base <- stats::rnorm(n, 20, 1.5)
  lfc <- stats::rnorm(n, 0, 0.15)

  de <- sample.int(n, round(n * prop_de))
  lfc[de] <- lfc[de] +
    sample(c(-1, 1), length(de), TRUE) *
    abs(stats::rnorm(length(de), fold, fold_sd))

  load <- stats::rnorm(m, 0, load_sd)
  eta <- outer(base, rep(1, m)) +
    outer(lfc, as.numeric(group == 2)) +
    matrix(rep(load, each = n), n, m)
  obs <- eta + matrix(stats::rnorm(n * m, 0, 0.25), n, m)

  if (missing > 0) {
    p <- rep(missing, n * m)
    if (mnar) {
      z <- as.vector(obs)
      p <- stats::plogis(-(z - stats::quantile(z, 0.25)) / 0.8)
      p <- p * (missing / mean(p))
      p <- pmin(p, 0.90)
    }
    obs[matrix(stats::runif(n * m) < p, n, m)] <- NA
  }

  colnames(obs) <- paste0("S", seq_len(m))
  rownames(obs) <- paste0("P", seq_len(n))

  list(observed = obs,
       group = group,
       truth = data.frame(id = rownames(obs), base = base, lfc = lfc,
                          de = seq_len(n) %in% de))
}

sim_peptides <- function(n_prot = 120, pep_per = 4, m = 12, prop_de = 0.10,
                         missing = 0.10, seed = 1) {
  set.seed(seed)
  group <- rep(1:2, each = m / 2)
  prot_base <- stats::rnorm(n_prot, 20, 1.2)
  lfc <- stats::rnorm(n_prot, 0, 0.15)

  de <- sample.int(n_prot, round(n_prot * prop_de))
  lfc[de] <- lfc[de] +
    sample(c(-1, 1), length(de), TRUE) * abs(stats::rnorm(length(de), 1, 0.3))

  load <- stats::rnorm(m, 0, 0.4)
  pep_effect <- matrix(stats::rnorm(n_prot * pep_per, 0, 0.45),
                       n_prot, pep_per)

  ids <- rep(seq_len(n_prot), each = pep_per)
  pep_base <- as.vector(prot_base[ids] + pep_effect)
  pep_lfc <- as.vector(lfc[ids])
  mu <- outer(pep_base, rep(1, m)) +
    outer(pep_lfc, as.numeric(group == 2)) +
    matrix(rep(load, each = length(ids)), length(ids), m)
  y <- 2^(mu + matrix(stats::rnorm(length(ids) * m, 0, 0.20),
                      length(ids), m))

  p <- stats::plogis(-(log2(y) - stats::quantile(log2(y), 0.2)) / 0.9)
  p <- p * (missing / mean(p))
  y[matrix(stats::runif(length(y)) < p, nrow(y), ncol(y))] <- NA

  colnames(y) <- paste0("S", seq_len(m))
  rownames(y) <- paste0("pep", seq_len(nrow(y)))

  list(peptides = y, groups = ids, group = group,
       truth = data.frame(id = paste0("P", seq_len(n_prot)), lfc = lfc,
                          de = seq_len(n_prot) %in% de))
}

sim_metab <- function(n = 300, m = 12, prop_de = 0.20, dilution_sd = 0.40,
                      seed = 1) {
  set.seed(seed)
  group <- rep(1:2, each = m / 2)
  conc <- exp(stats::rnorm(n, 8, 1.5))
  lfc <- stats::rnorm(n, 0, 0.15)

  de <- sample.int(n, round(n * prop_de))
  lfc[de] <- lfc[de] +
    sample(c(-1, 1), length(de), TRUE) * abs(stats::rnorm(length(de), 0.8, 0.3))

  dilution <- exp(stats::rnorm(m, 0, dilution_sd))
  x <- outer(conc, dilution) * 2^outer(lfc, as.numeric(group == 2))
  x <- x * exp(matrix(stats::rnorm(n * m, 0, 0.15), n, m))

  colnames(x) <- paste0("S", seq_len(m))
  rownames(x) <- paste0("M", seq_len(n))

  list(x = x, group = group,
       truth = data.frame(id = rownames(x), lfc = lfc,
                          de = seq_len(n) %in% de))
}

# Single-cell counts with known cell types and cell-specific size factors.
sim_sc <- function(n_cells = 600, n_genes = 1200, n_types = 3, seed = 1) {
  set.seed(seed)
  type <- rep(seq_len(n_types), length.out = n_cells)
  base <- exp(stats::rnorm(n_genes, 2, 1.5))
  fold <- matrix(1, n_genes, n_types)
  for (k in seq_len(n_types)) {
    idx <- sample.int(n_genes, 60)
    fold[idx, k] <- 4
  }
  lib <- exp(stats::rnorm(n_cells, 0, 0.45))
  eta <- outer(base, lib) * fold[, type] *
    exp(matrix(stats::rnorm(n_genes * n_cells, 0, 0.25), n_genes, n_cells))
  counts <- matrix(stats::rnbinom(n_genes * n_cells, mu = as.vector(eta),
                                  size = 0.4),
                   n_genes, n_cells)
  rownames(counts) <- paste0("G", seq_len(n_genes))
  colnames(counts) <- paste0("C", seq_len(n_cells))
  list(counts = counts, type = type, lib = lib)
}
