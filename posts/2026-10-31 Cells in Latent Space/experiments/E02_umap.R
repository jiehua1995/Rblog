# =============================================================================
# E02_umap.R — how stable is a UMAP picture?
#
# 3 x 3 grid over n_neighbors and min_dist, plus three random seeds and one
# 80% subsample.  Every embedding is scored against the simulated truth, so
# "looks nice" and "preserves structure" are kept as separate quantities.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")
suppressPackageStartupMessages(library(uwot))

sim <- readRDS("results/sim.rds")
tr  <- sim$truth

# analysis-ready input: technical covariates removed, then 50 PCs
X <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
pc50 <- pca_svd(X, d = 50)$scores

run_umap <- function(X, nn, md, seed = 42) {
  uwot::umap(X, n_neighbors = nn, min_dist = md, n_components = 2,
             metric = "euclidean", seed = seed, n_sgd_threads = 1,
             verbose = FALSE)
}

grid <- expand.grid(n_neighbors = c(5, 15, 50), min_dist = c(0.05, 0.3, 0.8))
grid$seed <- 42

Z <- scale(as.matrix(tr[, c("Z1", "Z2", "Z3")]))
post <- tr$t > 0.7
D_true <- dist(Z)

emb_list <- list()
metrics <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  set.seed(grid$seed[i])
  emb <- run_umap(pc50, grid$n_neighbors[i], grid$min_dist[i], grid$seed[i])
  key <- sprintf("nn%02d_md%03d", grid$n_neighbors[i], round(100 * grid$min_dist[i]))
  emb_list[[key]] <- emb
  metrics[[i]] <- data.frame(
    key = key,
    n_neighbors = grid$n_neighbors[i],
    min_dist = grid$min_dist[i],
    knn = knn_preserve(Z, emb, k = 10),
    dist_cor = dist_cor(Z, emb),
    r_t = max(abs(cor(emb, tr$t, method = "spearman"))),
    sil_branch = silhouette2(tr$b[post], as.matrix(dist(emb[post, ]))),
    sil_branch_true = silhouette2(tr$b[post], as.matrix(D_true)[post, post]),
    kmax = max(emb),
    stringsAsFactors = FALSE
  )
  cat(sprintf("nn=%2d md=%.2f  knn=%.3f  r_t=%.2f  sil=%.2f\n",
              grid$n_neighbors[i], grid$min_dist[i],
              metrics[[i]]$knn, metrics[[i]]$r_t, metrics[[i]]$sil_branch))
}
metrics <- do.call(rbind, metrics)

# ---- seed stability --------------------------------------------------------
seeds <- c(1, 2, 3)
seed_emb <- lapply(seeds, function(s) run_umap(pc50, 15, 0.3, s))
seed_sim <- c(
  procrustes_sim(seed_emb[[1]], seed_emb[[2]]),
  procrustes_sim(seed_emb[[1]], seed_emb[[3]]),
  procrustes_sim(seed_emb[[2]], seed_emb[[3]])
)

# ---- subsample stability ---------------------------------------------------
set.seed(11)
sub <- sort(sample(nrow(pc50), floor(0.8 * nrow(pc50))))
emb_sub <- run_umap(pc50[sub, ], 15, 0.3, 42)
emb_full <- emb_list[["nn15_md030"]]
sub_sim <- procrustes_sim(emb_full[sub, ], emb_sub)

# ---- how much does the 50-PC pre-reduction matter? -------------------------
emb_raw <- run_umap(X, 15, 0.3, 42)
raw_vs_pc <- procrustes_sim(emb_raw, emb_full)
raw_metrics <- data.frame(
  key = "genes_direct", n_neighbors = 15, min_dist = 0.3,
  knn = knn_preserve(Z, emb_raw, k = 10),
  dist_cor = dist_cor(Z, emb_raw),
  r_t = max(abs(cor(emb_raw, tr$t, method = "spearman"))),
  sil_branch = silhouette2(tr$b[post], as.matrix(dist(emb_raw[post, ]))),
  sil_branch_true = silhouette2(tr$b[post], as.matrix(D_true)[post, post]),
  kmax = max(emb_raw)
)
emb_list[["genes_direct"]] <- emb_raw
metrics <- rbind(metrics, raw_metrics)

stability <- list(seed_sim = seed_sim, subsample_sim = sub_sim,
                  raw_vs_pc = raw_vs_pc)

print(metrics, digits = 3)
cat("seed similarity:", round(seed_sim, 3), "\n")
cat("subsample similarity:", round(sub_sim, 3), " raw vs PC:", round(raw_vs_pc, 3), "\n")

saveRDS(list(metrics = metrics, embeddings = emb_list, stability = stability,
             grid = grid, truth = tr[, c("t", "a", "b", "s_branch", "batch")],
             post = post),
        "results/e02_umap.rds")
cat("saved results/e02_umap.rds\n")
