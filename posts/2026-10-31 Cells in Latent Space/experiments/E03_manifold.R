# =============================================================================
# E03_manifold.R — curved and branching trajectories
#
# Part A: a Swiss roll with a known parameterisation.  We can therefore compare
#         2-D Euclidean distance with graph-geodesic distance.
# Part B: the simulated bifurcation.  PCA, UMAP and diffusion maps are asked to
#         recover the ordering (t) and the split (branch) of the trajectory.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")
suppressPackageStartupMessages({library(uwot); library(igraph)})

set.seed(2026)

# ---- Part A: Swiss roll ----------------------------------------------------
n_sw <- 1200
u <- runif(n_sw); v <- runif(n_sw)
tt <- 1.5 * pi * (1 + 2 * u)
swiss <- cbind(x = tt * cos(tt), y = 21 * v, z = tt * sin(tt))
swiss_truth <- data.frame(u = u, curve = tt)

swiss_pca <- pca_svd(swiss, d = 2)$scores
swiss_umap <- uwot::umap(swiss, n_neighbors = 15, min_dist = 0.1, seed = 42,
                         n_sgd_threads = 1, verbose = FALSE)
swiss_dm <- dmaps(swiss, k = 10, ndim = 2)$embedding

geo <- graph_geodesic(swiss, k = 10)

# compare distances on a 600-cell subsample (full matrices, same index order)
set.seed(5)
si <- sort(sample(n_sw, 600))
geo_sub <- geo[si, si]
ok <- is.finite(as.vector(geo_sub))
geo_vec <- as.vector(geo_sub)[ok]
d2 <- function(E) as.vector(as.matrix(dist(E[si, , drop = FALSE])))[ok]

swiss_metrics <- data.frame(
  method = c("PCA", "UMAP", "Diffusion map"),
  geo_cor = c(cor(d2(swiss_pca), geo_vec, method = "spearman"),
              cor(d2(swiss_umap), geo_vec, method = "spearman"),
              cor(d2(swiss_dm), geo_vec, method = "spearman"))
)
print(swiss_metrics, digits = 3)
cat("intrinsic dimension of the Swiss roll (MLE):",
    round(intrinsic_dim_mle(swiss, k = 10, n_sub = 600), 2), "\n")

swiss_out <- list(data = swiss, truth = swiss_truth, pca = swiss_pca,
                  umap = swiss_umap, dm = swiss_dm, metrics = swiss_metrics)

# ---- Part B: simulated bifurcation ----------------------------------------
sim <- readRDS("results/sim.rds")
tr <- sim$truth
X <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
pc50 <- pca_svd(X, d = 50)$scores

emb_pca <- pca_svd(X, d = 2)$scores
emb_umap <- uwot::umap(pc50, n_neighbors = 15, min_dist = 0.3, seed = 42,
                       n_sgd_threads = 1, verbose = FALSE)
dm <- dmaps(pc50, k = 15, ndim = 3)
emb_dm <- dm$embedding
cat("diffusion eigenvalues:", round(dm$eigenvalues[1:6], 3), "\n")

Z <- scale(as.matrix(tr[, c("Z1", "Z2", "Z3")]))
post <- tr$t > 0.75
Dpost <- as.matrix(dist(Z))[post, post]

score <- function(emb, label) {
  e <- as.matrix(emb)[, 1:min(2, ncol(emb)), drop = FALSE]
  data.frame(
    method = label,
    r_t = max(abs(cor(e, tr$t, method = "spearman"))),
    r_a = max(abs(cor(e, tr$a, method = "spearman"))),
    sil_branch = silhouette2(tr$b[post], as.matrix(dist(e[post, ]))),
    knn = knn_preserve(Z, as.matrix(emb)[, 1:min(3, ncol(emb))], k = 10),
    dist_cor = dist_cor(Z, e)
  )
}
traj_metrics <- rbind(
  score(emb_pca, "PCA (2 PC)"),
  score(emb_umap, "UMAP"),
  score(emb_dm, "Diffusion map (3)")
)
traj_metrics$sil_branch_true <- silhouette2(tr$b[post], Dpost)
print(traj_metrics, digits = 3)

# pseudotime quality: diffusion components are ordered by graph scale, not by
# time, so the "pseudotime" is a choice.  We report every component and then
# pick, with hindsight, the one that tracks t best (an oracle choice).
dc_cor <- sapply(1:3, function(k) cor(emb_dm[, k], tr$t, method = "spearman"))
best <- which.max(abs(dc_cor))
pseudotime <- emb_dm[, best]
if (dc_cor[best] < 0) pseudotime <- -pseudotime
rho_t <- abs(dc_cor[best])
cat("DC-t correlations:", round(dc_cor, 3), " best DC:", best, "\n")

# does the pseudotime ordering keep branch identity separate?  Within cells of
# similar pseudotime, branch labels should not be mixed once commitment is done.
late <- tr$t > 0.75
mix <- mean(vapply(seq_len(20), function(i) {
  target <- quantile(pseudotime[late], probs = i / 21)
  near <- late & abs(pseudotime - target) < 0.1 * diff(range(pseudotime))
  if (sum(near) < 5) return(NA_real_)
  mean(tr$b[near])
}, 0), na.rm = TRUE)
cat(sprintf("DM pseudotime rho(t)=%.3f, branch mixing in late windows=%.2f (0.5 = fully mixed)\n",
            rho_t, mix))

saveRDS(list(swiss = swiss_out,
             sim = list(pca = emb_pca, umap = emb_umap, dm = emb_dm,
                        eigenvalues = dm$eigenvalues, metrics = traj_metrics,
                        pseudotime = pseudotime, rho_t = rho_t, mix = mix),
             truth = tr),
        "results/e03_manifold.rds")
cat("saved results/e03_manifold.rds\n")
