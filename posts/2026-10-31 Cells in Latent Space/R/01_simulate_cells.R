# =============================================================================
# 01_simulate_cells.R — a generative model with a known ground truth
#
# Biological state of cell i:
#   t_i  differentiation time        ~ Beta(2, 2),        t in [0, 1]
#   b_i  branch commitment           ~ Bernoulli(0.5)
#   a_i  activation level            ~ Beta(2, 5),        a in [0, 1]
#   c_i  cell-cycle phase            ~ categorical, weakly t-dependent
#
# The branch is a *latent direction that only becomes visible late*:
#   s_i(t) = w(t) * (2 b_i - 1),   w(t) = sigmoid((t - 0.58) / 0.10)
# so early cells with different b are exchangeable by construction.
#
# Expression:
#   log mu_ig = log s_i + beta_0g + f_g(t_i, b_i, a_i, c_i) + gamma_{g, batch_i}
#   X_ig      ~ NB(mu_ig, theta_g)
#
# f_g is a sum of module effects: monotone, transient (pulse), branch,
# activation and cell-cycle programs.  Nothing in the generator is tuned to
# favour one particular dimensionality-reduction method.
# =============================================================================

suppressPackageStartupMessages({
  library(Matrix)
})

simulate_cells <- function(n_cells = 3000, n_genes = 1000, seed = 20261031,
                           n_batches = 3, batch_strength = 0.5,
                           dispersion_range = c(0.6, 20)) {
  set.seed(seed)

  # ---- latent state --------------------------------------------------------
  t <- rbeta(n_cells, 2, 2)
  b <- rbinom(n_cells, 1, 0.5)
  a <- rbeta(n_cells, 2, 4)
  w <- sigm(t, mid = 0.58, slope = 10)
  s_branch <- w * (2 * b - 1)                       # continuous branch coordinate

  phase_prob <- cbind(
    G1 = 0.55 + 0.15 * (1 - t),
    S  = 0.25,
    G2M = 0.20 - 0.15 * (1 - t)
  )
  phase_prob <- phase_prob / rowSums(phase_prob)
  phase <- apply(phase_prob, 1, function(p) sample(c("G1", "S", "G2M"), 1, prob = p))

  lib <- exp(rnorm(n_cells, 0, 0.35))               # library-size factor
  batch <- sample(seq_len(n_batches), n_cells, replace = TRUE,
                  prob = c(0.45, 0.35, 0.20))

  # ---- gene modules --------------------------------------------------------
  stopifnot(n_genes == 1000)
  module <- c(rep("housekeeping", 500),
              rep("differentiation", 200),
              rep("branch", 120),
              rep("activation", 120),
              rep("cellcycle", 60))
  module <- factor(module, levels = c("housekeeping", "differentiation", "branch",
                                      "activation", "cellcycle"))

  g <- data.frame(
    gene   = sprintf("G%04d", seq_len(n_genes)),
    module = module,
    beta0  = rnorm(n_genes, 2.2, 0.8),
    theta  = exp(runif(n_genes, log(dispersion_range[1]), log(dispersion_range[2]))),
    amp    = 0, mid = 0.5, slope = 10,
    sign_b = 0, phase_prog = NA_character_,
    stringsAsFactors = FALSE
  )

  # differentiation program: sigmoidal in t, with per-gene midpoint and sign
  idx_d   <- which(module == "differentiation")
  # 140 monotone genes, 60 transient "pulse" genes
  mono <- idx_d[1:140]; pulse <- idx_d[141:200]
  g$amp[mono]   <- rnorm(length(mono), 0, 1.1)
  g$mid[mono]   <- runif(length(mono), 0.2, 0.8)
  g$slope[mono] <- runif(length(mono), 7, 18)
  g$mid[pulse]  <- runif(length(pulse), 0.3, 0.7)
  g$amp[pulse]  <- rnorm(length(pulse), 0, 1.2)
  g$slope[pulse] <- runif(length(pulse), 0.05, 0.12)  # width of the pulse

  # branch program: only expressed after commitment
  idx_b <- which(module == "branch")
  g$amp[idx_b] <- abs(rnorm(length(idx_b), 0.9, 0.35))
  g$sign_b[idx_b] <- sample(c(-1, 1), length(idx_b), replace = TRUE)

  # activation program
  idx_a <- which(module == "activation")
  g$amp[idx_a] <- abs(rnorm(length(idx_a), 1.3, 0.35))

  # cell-cycle program: 20 genes per phase
  idx_c <- which(module == "cellcycle")
  g$phase_prog[idx_c] <- rep(c("G1", "S", "G2M"), each = 20)

  # batch effect on a random quarter of the genes
  batch_genes <- sample(n_genes, 250)
  gamma <- matrix(0, n_genes, n_batches)
  gamma[batch_genes, ] <- matrix(rnorm(length(batch_genes) * n_batches, 0, batch_strength),
                                 length(batch_genes), n_batches)
  gamma[, 1] <- 0  # batch 1 is the reference

  # ---- expected log expression --------------------------------------------
  # evaluated on a per-gene basis; n_cells x n_genes is small enough here
  F_mat <- matrix(0, n_cells, n_genes)

  # monotone / pulse differentiation genes
  F_mat[, mono] <- sapply(mono, function(j)
    g$amp[j] * sigm(t, g$mid[j], g$slope[j]))
  F_mat[, pulse] <- sapply(pulse, function(j)
    g$amp[j] * exp(-((t - g$mid[j]) / g$slope[j])^2))

  # branch genes: amplitude scales with commitment w(t)
  F_mat[, idx_b] <- sapply(idx_b, function(j)
    g$amp[j] * g$sign_b[j] * s_branch)

  # activation genes
  F_mat[, idx_a] <- sapply(idx_a, function(j) g$amp[j] * (a - 0.3))

  # cell-cycle genes
  for (ph in c("G1", "S", "G2M")) {
    jj <- idx_c[g$phase_prog[idx_c] == ph]
    F_mat[, jj] <- sapply(jj, function(j) 0.9 * (phase == ph) - 0.45)
  }

  gamma_mat <- gamma[, batch, drop = FALSE]           # n_genes x n_cells
  log_mu <- sweep(F_mat, 2, g$beta0, "+") +
    matrix(log(lib), n_cells, n_genes, byrow = FALSE) +
    t(gamma_mat)

  # ---- negative-binomial counts -------------------------------------------
  mu <- exp(log_mu)
  counts <- matrix(rnbinom(n_cells * n_genes, mu = as.vector(mu),
                           size = rep(g$theta, each = n_cells)),
                   n_cells, n_genes)
  counts <- methods::as(Matrix::Matrix(counts, sparse = TRUE), "dgCMatrix")
  dimnames(counts) <- list(sprintf("cell_%04d", seq_len(n_cells)), g$gene)

  truth <- data.frame(
    cell = colnames(counts), t = t, b = b, a = a, phase = phase,
    batch = factor(batch), lib = lib, s_branch = s_branch,
    Z1 = t, Z2 = a, Z3 = s_branch
  )

  modules <- list(
    housekeeping = which(module == "housekeeping"),
    mono = mono, pulse = pulse,
    branchA = idx_b[g$sign_b[idx_b] > 0], branchB = idx_b[g$sign_b[idx_b] < 0],
    activation = idx_a, cellcycle = idx_c
  )

  list(counts = counts, truth = truth, genes = g, modules = modules,
       gamma = gamma, batch = batch, log_mu = log_mu,
       params = list(n_cells = n_cells, n_genes = n_genes, seed = seed,
                     n_batches = n_batches, batch_strength = batch_strength))
}

#' Rebuild the expected log-expression matrix from stored parameters.
#' Keeps the saved object small and lets later scripts ask counterfactual
#' questions ("what would this cell look like on the other branch?").
sim_logmu <- function(sim, truth = sim$truth) {
  g <- sim$genes; m <- sim$modules
  n <- nrow(truth)
  F_mat <- matrix(0, n, nrow(g))
  F_mat[, m$mono]  <- sapply(m$mono, function(j) g$amp[j] * sigm(truth$t, g$mid[j], g$slope[j]))
  F_mat[, m$pulse] <- sapply(m$pulse, function(j) g$amp[j] * exp(-((truth$t - g$mid[j]) / g$slope[j])^2))
  F_mat[, m$branchA] <- sapply(m$branchA, function(j) g$amp[j] * g$sign_b[j] * truth$s_branch)
  F_mat[, m$branchB] <- sapply(m$branchB, function(j) g$amp[j] * g$sign_b[j] * truth$s_branch)
  F_mat[, m$activation] <- sapply(m$activation, function(j) g$amp[j] * (truth$a - 0.3))
  idx_c <- m$cellcycle
  for (ph in c("G1", "S", "G2M")) {
    jj <- idx_c[g$phase_prog[idx_c] == ph]
    F_mat[, jj] <- sapply(jj, function(j) 0.9 * (truth$phase == ph) - 0.45)
  }
  gam <- sim$gamma[, as.integer(truth$batch), drop = FALSE]
  sweep(F_mat, 2, g$beta0, "+") +
    matrix(log(truth$lib), n, nrow(g)) + t(gam)
}
