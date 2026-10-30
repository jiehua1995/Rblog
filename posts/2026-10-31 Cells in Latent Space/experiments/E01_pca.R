# =============================================================================
# E01_pca.R — can PCA recover the hidden cell state?
#
# Four pipelines are compared on the same cells, same split:
#   P1 logCPM, gene-centred           (no variance weighting)
#   P2 logCPM, z-scored               (each gene on the same footing)
#   P3 P2 + depth regressed out
#   P4 P3 + batch centred + 400 HVGs
#
# Metrics are computed on held-out cells wherever a fit is involved, and are
# always compared against the simulated truth (Z1 = t, Z2 = a, Z3 = branch).
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")

sim <- readRDS("results/sim.rds")
tr  <- sim$truth
split <- make_split(nrow(tr), prop = 0.7, seed = 7)

inputs <- list(
  P0_raw_counts     = list(depth = FALSE, batch = FALSE, hvg = 0,   scale = FALSE, raw = TRUE),
  P1_logcpm_centred = list(depth = FALSE, batch = FALSE, hvg = 0,   scale = FALSE),
  P2_logcpm_z       = list(depth = FALSE, batch = FALSE, hvg = 0,   scale = TRUE),
  P3_depth          = list(depth = TRUE,  batch = FALSE, hvg = 0,   scale = TRUE),
  P4_depth_batch_hvg = list(depth = TRUE, batch = TRUE,  hvg = 400, scale = TRUE)
)

metrics <- list()
scores  <- list()
ve_list <- list()

for (nm in names(inputs)) {
  a <- inputs[[nm]]
  if (isTRUE(a$raw)) {
    X <- as.matrix(sim$counts)          # deliberately un-normalised
  } else {
    X <- build_input(sim, depth = a$depth, batch = a$batch, hvg = a$hvg, scale = a$scale)
  }
  if (!a$scale) X <- sweep(X, 2, colMeans(X), "-")

  pc <- pca_svd(X, d = 10, center = TRUE, scale_gene = FALSE)
  sc <- pc$scores
  scores[[nm]] <- sc[, 1:5]
  ve_list[[nm]] <- pc$var_explained

  emb <- embedding_metrics(sc[, 1:5], tr)
  # recon error with the training rotation, evaluated on held-out cells
  rec <- pca_recon_error(X[split$train, ], X[split$test, ], d = 3)

  # technical covariate predictability from the first five PCs
  r2_of <- function(y) summary(lm(y ~ tr$batch))$r.squared
  batch_r2 <- r2_of(sc[, 1])
  batch_r2_5 <- max(sapply(1:5, function(j) r2_of(sc[, j])))

  metrics[[nm]] <- data.frame(
    pipeline = nm,
    ve3 = sum(pc$var_explained[1:3]),
    ve10 = sum(pc$var_explained[1:10]),
    rec_err3 = rec,
    r_t = emb$r_t, r_a = emb$r_a, r_branch = emb$r_branch,
    knn = emb$knn, dist_cor = emb$dist_cor,
    knnR2_t = knn_reg_r2(sc[, 1:5], tr$t, split),
    knnR2_a = knn_reg_r2(sc[, 1:5], tr$a, split),
    batch_r2_pc1 = batch_r2,
    batch_r2_pc15 = batch_r2_5,
    depth_r2_pc1 = summary(lm(sc[, 1] ~ log(tr$lib)))$r.squared,
    stringsAsFactors = FALSE
  )
}

metrics <- do.call(rbind, metrics)
rownames(metrics) <- NULL
print(metrics[, c("pipeline", "ve3", "rec_err3", "r_t", "r_a", "r_branch",
                  "knn", "batch_r2_pc1", "depth_r2_pc1")],
      digits = 3)

# ---- intrinsic dimension estimates ---------------------------------------
# The generative model has three continuous factors; the noiseless expectation
# is the benchmark the estimator should hit if noise did not interfere.
X2 <- build_input(sim, depth = FALSE, batch = FALSE, hvg = 0, scale = TRUE)
X3 <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 0, scale = TRUE)
id_est <- data.frame(
  data = c("observed, logCPM z-scored", "observed, depth+batch corrected",
           "noiseless expectation (log mu)", "true factors (t, a, branch)"),
  k10 = c(intrinsic_dim_mle(X2, 10), intrinsic_dim_mle(X3, 10),
          intrinsic_dim_mle(sim_logmu(sim), 10),
          intrinsic_dim_mle(as.matrix(tr[, c("Z1", "Z2", "Z3")]), 10)),
  k20 = c(intrinsic_dim_mle(X2, 20), intrinsic_dim_mle(X3, 20),
          intrinsic_dim_mle(sim_logmu(sim), 20),
          intrinsic_dim_mle(as.matrix(tr[, c("Z1", "Z2", "Z3")]), 20))
)
print(id_est, digits = 2)

# ---- PC-by-PC correlation with the three simulated factors ---------------
pc3 <- pca_svd(X3, d = 8)
pc_cor <- data.frame(
  PC = 1:8,
  var_explained = pc3$var_explained,
  t = sapply(1:8, function(k) cor(pc3$scores[, k], tr$t, method = "spearman")),
  a = sapply(1:8, function(k) cor(pc3$scores[, k], tr$a, method = "spearman")),
  branch = sapply(1:8, function(k) cor(pc3$scores[, k], tr$s_branch, method = "spearman"))
)
print(pc_cor, digits = 3)

# ---- variance explained by technical covariates, per gene ----------------
# how much of each gene's variance is explained by depth and by batch?
Xd <- as.matrix(logcpm(sim$counts))
var_depth <- apply(Xd, 2, function(y) summary(lm(y ~ log(tr$lib)))$r.squared)
var_batch <- apply(Xd, 2, function(y) summary(lm(y ~ tr$batch))$r.squared)

# ---- an oracle ceiling for neighbourhood preservation ----------------------
# What fraction of the true-neighbour structure is recoverable *at all* from
# this noisy matrix?  A supervised linear map onto (t, a, s) gives an optimistic
# upper bound, because it is allowed to see the answer.
Xo <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
Xm <- cbind(1, Xo)
Y  <- cbind(t = tr$t, a = tr$a, s_branch = tr$s_branch)
coef_or <- qr.coef(qr(Xm), Y)
oracle_emb <- Xm %*% coef_or
Zt <- scale(as.matrix(tr[, c("Z1", "Z2", "Z3")]))
oracle <- c(
  r2_t = 1 - sum((Y[, "t"] - oracle_emb[, "t"])^2) / sum((Y[, "t"] - mean(Y[, "t"]))^2),
  knn = knn_preserve(Zt, oracle_emb, k = 10),
  dist_cor = dist_cor(Zt, oracle_emb)
)
print(round(oracle, 3))

# ---- sensitivity: a stronger batch effect ---------------------------------
# The main simulation uses a deliberately modest batch shift.  Here the same
# generator is re-run with a three-fold stronger batch effect, to show at what
# point technical variation takes over the leading components.
sim_hi <- simulate_cells(batch_strength = 1.5)
tr_hi  <- sim_hi$truth
X_hi   <- build_input(sim_hi, depth = FALSE, batch = FALSE, hvg = 0, scale = TRUE)
X_hi_c <- build_input(sim_hi, depth = FALSE, batch = TRUE,  hvg = 0, scale = TRUE)
pc_hi   <- pca_svd(X_hi, 5)
pc_hi_c <- pca_svd(X_hi_c, 5)
batch_r2 <- function(sc, batch) max(sapply(1:5, function(j) summary(lm(sc[, j] ~ batch))$r.squared))
sens_batch <- data.frame(
  stage = c("batch effect present", "batch centred"),
  batch_r2_pc15 = c(batch_r2(pc_hi$scores, tr_hi$batch),
                    batch_r2(pc_hi_c$scores, tr_hi$batch)),
  r_t = c(max(abs(cor(pc_hi$scores, tr_hi$t, method = "spearman"))),
          max(abs(cor(pc_hi_c$scores, tr_hi$t, method = "spearman")))),
  ve1 = c(pc_hi$var_explained[1], pc_hi_c$var_explained[1])
)
print(sens_batch, digits = 3)

saveRDS(list(metrics = metrics, scores = scores, var_explained = ve_list,
             id_est = id_est, split = split, pc_cor = pc_cor,
             var_depth = var_depth, var_batch = var_batch,
             sens_batch = sens_batch, oracle = oracle, modules = sim$modules),
        "results/e01_pca.rds")
cat("saved results/e01_pca.rds\n")
