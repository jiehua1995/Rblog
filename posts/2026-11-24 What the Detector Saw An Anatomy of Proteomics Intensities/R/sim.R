# Ground-truth simulators: peptide intensities from protein abundances, a
# dilution series with a detector ceiling, and TMT reporter ions with background.

# Build a peptides x runs intensity matrix plus the truth behind it.
sim_peptides <- function(n_prot = 300, n_run = 6, group = rep(1:2, each = n_run / 2),
                         prop_de = 0.2, lfc_sd = 1, load_sd = 0.35,
                         pep_sd = 0.55, noise_sd = 0.3, miss_rate = 0.08,
                         seed = 1) {
  set.seed(seed)
  base <- rnorm(n_prot, 8, 1.8)
  lfc <- ifelse(runif(n_prot) < prop_de, rnorm(n_prot, 0, lfc_sd), 0)
  n_pep <- sample(2:8, n_prot, replace = TRUE)
  pep_prot <- rep(seq_len(n_prot), n_pep)
  pep_effect <- rnorm(length(pep_prot), 0, pep_sd)
  loading <- rnorm(n_run, 0, load_sd)

  grid <- expand.grid(pep = seq_along(pep_prot), run = seq_len(n_run))
  grid$prot <- pep_prot[grid$pep]
  grid$group <- group[grid$run]
  mu <- base[grid$prot] +
    ifelse(grid$group == 2, lfc[grid$prot], 0) +
    pep_effect[grid$pep] +
    loading[grid$run]
  grid$intensity <- 2^(mu + rnorm(nrow(grid), 0, noise_sd))

  # MNAR: low-intensity peptides drop out more often.
  score <- log2(grid$intensity)
  cut <- stats::quantile(score, miss_rate)
  p_miss <- stats::plogis(-(score - cut) / 0.6)
  grid$intensity[runif(nrow(grid)) < p_miss] <- NA

  mat <- matrix(NA_real_, length(pep_prot), n_run)
  mat[cbind(grid$pep, grid$run)] <- grid$intensity
  prot_of_pep <- pep_prot
  truth <- data.frame(protein = seq_len(n_prot), base = base, lfc = lfc,
                      n_pep = n_pep)
  list(mat = mat, pep_protein = prot_of_pep, truth = truth,
       group = group, loading = loading)
}

# Protein-level estimate under one aggregation rule.
aggregate_proteins <- function(mat, pep_protein, method, n_obs = NULL) {
  n_prot <- max(pep_protein)
  out <- matrix(NA_real_, n_prot, ncol(mat))
  for (p in seq_len(n_prot)) {
    sub <- mat[pep_protein == p, , drop = FALSE]
    out[p, ] <- switch(
      method,
      sum = sum_intensity(sub),
      meanlog = mean_log_intensity(sub),
      maxlfq = maxlfq(sub)$profile,
      top3 = top3_intensity(sub)
    )
  }
  if (!is.null(n_obs)) out <- out / n_obs
  out
}

lfc_from_matrix <- function(prot_mat, group) {
  a <- rowMeans(prot_mat[, group == 1, drop = FALSE], na.rm = TRUE)
  b <- rowMeans(prot_mat[, group == 2, drop = FALSE], na.rm = TRUE)
  log2(b / a)
}

# A dilution series observed through a detector with a finite ceiling and LOD.
sim_dilution <- function(amounts = 2^seq(0, 12, by = 1), n_pep = 4,
                         ceiling = 2^11, lod = 2^1.2, noise = 0.18,
                         flyability_sd = 0.6, seed = 2) {
  set.seed(seed)
  pep_eff <- exp(rnorm(n_pep, 0, flyability_sd))
  grid <- expand.grid(amount = amounts, pep = seq_len(n_pep))
  grid$true <- grid$amount * pep_eff[grid$pep]
  grid$observed <- grid$true / (1 + grid$true / ceiling)
  grid$observed <- grid$observed * exp(rnorm(nrow(grid), 0, noise))
  grid$observed[grid$observed < lod] <- NA
  grid
}

# TMT: each reporter ion contains a fraction of co-isolated background.
sim_tmt <- function(ratios = 2^seq(-2, 2, length.out = 25),
                   background = c(0, 0.05, 0.2, 0.5), reps = 200, seed = 3) {
  set.seed(seed)
  out <- do.call(rbind, lapply(background, function(bg) {
    data.frame(
      true_ratio = ratios,
      background = bg,
      observed = ratios + bg * (1 + ratios) / (1 + bg) +
        rnorm(length(ratios) * reps, 0, 0.06),
      replicate = rep(seq_len(reps), each = length(ratios))
    )
  }))
  out
}
