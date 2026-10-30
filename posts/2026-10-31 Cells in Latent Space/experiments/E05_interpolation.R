# =============================================================================
# E05_interpolation.R — what happens between two cells?
#
# The Gaussian VAE decoder maps a latent point to a full expression profile
# (in the standardised, depth- and batch-corrected space).  We walk a straight
# line between two encoded cells and ask three questions:
#   1. does the decoded profile stay near the observed data cloud?
#   2. does it respect mutually exclusive branch programmes?
#   3. how much longer is the straight line than the graph path?
#
# The negative-binomial VAE is deliberately *not* used here: its latent space
# tracks global expression rather than cell state (see E04), which is itself an
# interesting result but a bad substrate for an interpolation demo.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")
source("R/04_autoencoders.R")

sim <- readRDS("results/sim.rds")
tr  <- sim$truth
X   <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
e4  <- readRDS("results/e04_vae.rds")
vae <- torch_load("results/vae_d4.pt")

Z <- encode_matrix(vae, X, 4)
cat("latent correlations with truth:\n")
print(round(sapply(seq_len(4), function(j) c(
  t = cor(Z[, j], tr$t, method = "spearman"),
  a = cor(Z[, j], tr$a, method = "spearman"),
  br = cor(Z[, j], tr$s_branch, method = "spearman"))), 2))

# ---- decode back to the log-CPM scale --------------------------------------
mu_x <- e4$X_meta$mu
sd_x <- e4$X_meta$sd
decode_logcpm <- function(P) {
  D <- decode_matrix(vae, P)
  sweep(sweep(D, 2, sd_x, "*"), 2, mu_x, "+")
}

# ---- choose a pair of cells ------------------------------------------------
pick <- function(t_target, b_target) {
  cand <- which(abs(tr$t - t_target) < 0.04 & tr$b == b_target)
  cand[which.min(abs(tr$a[cand] - 0.25))]
}
iA <- pick(0.12, 1); iB <- pick(0.92, 1); iC <- pick(0.92, 0)
cat(sprintf("A: t=%.2f b=%d  B: t=%.2f b=%d  C: t=%.2f b=%d\n",
            tr$t[iA], tr$b[iA], tr$t[iB], tr$b[iB], tr$t[iC], tr$b[iC]))

alpha <- seq(0, 1, length.out = 41)
# ---- reference: observed profiles ------------------------------------------
# The decoder lives in the standardised, corrected space, so the reference must
# be that same matrix (not raw log-CPM: per-gene standardisation changes the
# correlation between genes).
Lobs <- X
Lobs <- Lobs - rowMeans(Lobs)                 # centre each cell across genes
# X columns are ordered by variance, so module indices must be mapped to the
# columns of X before they can be used as a selector
mod_idx <- function(m) match(sim$genes$gene[m], colnames(X))
mA <- mod_idx(sim$modules$branchA)
mB <- mod_idx(sim$modules$branchB)
real_modA <- rowMeans(Lobs[, mA])
real_modB <- rowMeans(Lobs[, mB])

nearest_dist <- function(v) {
  # Euclidean distance to the closest real cell, expressed in units of the
  # typical nearest-neighbour distance among real cells (so 1.0 = "as close as
  # a real cell is to its own neighbour")
  min(sqrt(colSums((t(Lobs) - v)^2))) / dnn_real
}

dnn_real <- mean(FNN::get.knn(Lobs, k = 1)$nn.dist)
baseline_nn <- 1

paths <- list()
for (nm in c("A_B_same_branch", "A_C_cross_branch")) {
  ii <- if (nm == "A_B_same_branch") c(iA, iB) else c(iA, iC)
  P <- (1 - alpha) %o% Z[ii[1], ] + alpha %o% Z[ii[2], ]
  L <- decode_logcpm(P)
  L <- L - rowMeans(L)
  paths[[nm]] <- list(
    logmean = L, alpha = alpha,
    off_manifold = apply(L, 1, nearest_dist),
    modA = rowMeans(L[, mA]),
    modB = rowMeans(L[, mB]),
    t_interp = (1 - alpha) * tr$t[ii[1]] + alpha * tr$t[ii[2]],
    cells = ii
  )
}

# ---- does the decoded path match the real intermediate states? -------------
# Compare each decoded profile with the mean observed profile of real cells at
# the interpolated time, and calibrate against two real cells at similar t.
cor_true_path <- function(path) {
  sapply(seq_along(alpha), function(k) {
    w <- dnorm(tr$t, mean = path$t_interp[k], sd = 0.05)
    w <- w / sum(w)
    cor(path$logmean[k, ], colSums(Lobs * w))
  })
}
for (nm in names(paths)) paths[[nm]]$cor_true <- cor_true_path(paths[[nm]])

set.seed(1)
ii <- sample(nrow(Lobs), 400)
cc <- cor(t(Lobs[ii, ]))
dd <- abs(outer(tr$t[ii], tr$t[ii], "-"))
sel <- upper.tri(cc) & dd < 0.05
ref_cor <- mean(cc[sel])
cor_true <- paths$A_B_same_branch$cor_true

# ---- mutually exclusive programmes ----------------------------------------
real_contrast <- real_modA - real_modB
late <- tr$t > 0.8
abs_real <- abs(real_contrast[late])
thr <- quantile(abs_real, 0.05)
Lx <- paths$A_C_cross_branch
contrast_x <- Lx$modA - Lx$modB
impossible <- mean(abs(contrast_x) < thr)
cat(sprintf("real cells (t>0.8): |A-B| 5th pct = %.2f; interpolated path below it: %.0f%%\n",
            thr, 100 * impossible))

# ---- straight line versus graph path --------------------------------------
nn <- FNN::get.knn(Z, k = 15)
edges <- data.frame(from = rep(1:nrow(Z), 15), to = as.vector(nn$nn.index),
                    w = as.vector(nn$nn.dist))
g <- igraph::graph_from_data_frame(edges, directed = FALSE,
                                   vertices = data.frame(name = seq_len(nrow(Z))))
d_graph <- igraph::distances(g, weights = igraph::edge_attr(g, "w"))
geo_ratio <- c(
  A_B = d_graph[iA, iB] / sqrt(sum((Z[iA, ] - Z[iB, ])^2)),
  A_C = d_graph[iA, iC] / sqrt(sum((Z[iA, ] - Z[iC, ])^2))
)

# ---- PCA-space interpolation, for comparison ------------------------------
# A 4-component PCA is also a latent space, and its decoder reconstructs the
# branch programmes far more faithfully than the VAE's; the same walk is
# therefore a cleaner test of the geometry.
pca4 <- pca_svd(X, d = 4)
path_pca <- function(i1, i2) {
  Pp <- (1 - alpha) %o% pca4$scores[i1, ] + alpha %o% pca4$scores[i2, ]
  Rp <- Pp %*% t(pca4$rotation)
  Rp <- Rp - rowMeans(Rp)
  list(modA = rowMeans(Rp[, mA]),
       modB = rowMeans(Rp[, mB]), logmean = Rp,
       off = apply(Rp, 1, nearest_dist))
}
pca_AC <- path_pca(iA, iC)
pca_AB <- path_pca(iA, iB)
impossible_pca <- mean(abs(pca_AC$modA - pca_AC$modB) < thr)

# ---- admissibility at matched pseudotime ----------------------------------
# Comparing the whole path with *terminal* cells is unfair: at t = 0.1 real
# cells also have almost no branch contrast, so any path would look "impossible"
# near its start.  The fair test is whether the decoded profile is consistent
# with real cells at the same interpolated time, in the branch the path is
# heading towards.
t_pca <- (1 - alpha) * tr$t[iA] + alpha * tr$t[iC]
contrast_pca <- pca_AC$modA - pca_AC$modB
contrast_vae <- paths$A_C_cross_branch$modA - paths$A_C_cross_branch$modB
t_vae <- paths$A_C_cross_branch$t_interp

band_at <- function(tq, b_target, probs = c(0.05, 0.5, 0.95), half = 0.06) {
  ref <- which(tr$b == b_target & abs(tr$t - tq) < half)
  if (length(ref) < 20) return(rep(NA_real_, length(probs)))
  as.numeric(quantile(real_contrast[ref], probs))
}

adm_check <- function(tp, cp, b_target) {
  B <- t(sapply(tp, band_at, b_target = b_target))
  list(inside = cp >= B[, 1] & cp <= B[, 3],
       lo = B[, 1], hi = B[, 3], med = B[, 2],
       t = tp, contrast = cp)
}
adm_pca <- adm_check(t_pca, contrast_pca, tr$b[iC])
adm_vae <- adm_check(t_vae, contrast_vae, tr$b[iC])

t_grid <- seq(0.1, 0.95, length.out = 60)
band_grid <- data.frame(
  t  = t_grid,
  lo = sapply(t_grid, function(q) band_at(q, tr$b[iC])[1]),
  hi = sapply(t_grid, function(q) band_at(q, tr$b[iC])[3])
)
Pp <- (1 - alpha) %o% pca4$scores[iA, ] + alpha %o% pca4$scores[iB, ]
Rp <- Pp %*% t(pca4$rotation)
off_pca <- apply(Rp, 1, function(v) 1 - max(cor(v, t(Lobs))))
cor_true_pca <- sapply(seq_along(alpha), function(k) {
  w <- dnorm(tr$t, mean = (1 - alpha[k]) * tr$t[iA] + alpha[k] * tr$t[iB], sd = 0.05)
  w <- w / sum(w)
  cor(pca_AB$logmean[k, ], colSums(Lobs * w))
})

out <- list(paths = paths, cor_true = cor_true, ref_cor = ref_cor,
            impossible = impossible, impossible_pca = impossible_pca,
            pca_AB = pca_AB, pca_AC = pca_AC, cor_true_pca = cor_true_pca,
            thr = thr, baseline_nn = baseline_nn,
            adm_pca = adm_pca, adm_vae = adm_vae, band_grid = band_grid,
            geo_ratio = geo_ratio, off_pca = off_pca, Z = Z,
            real_modA = real_modA, real_modB = real_modB,
            cells = c(A = iA, B = iB, C = iC),
            modules = sim$modules, truth = tr)
saveRDS(out, "results/e05_interp.rds")

cat(sprintf("geodesic / straight-line: A-B %.2f, A-C %.2f\n", geo_ratio[1], geo_ratio[2]))
cat(sprintf("off-manifold (1 - r) at alpha=0.5: same branch %.3f, cross branch %.3f, baseline %.3f\n",
            paths$A_B_same_branch$off_manifold[21],
            paths$A_C_cross_branch$off_manifold[21], baseline_nn))
cat(sprintf("mean corr with the real intermediate profile: %.3f (two real cells at similar t: %.3f)\n",
            mean(cor_true), ref_cor))
cat(sprintf("inside the matched-t band: 4-PC %.0f%%, VAE %.0f%%\n",
            100 * mean(adm_pca$inside, na.rm = TRUE),
            100 * mean(adm_vae$inside, na.rm = TRUE)))
cat(sprintf("distance to the nearest real cell (x mean NN): 4-PC max %.2f, VAE max %.2f\n",
            max(pca_AC$off), max(paths$A_C_cross_branch$off_manifold)))
