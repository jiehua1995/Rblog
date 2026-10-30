# =============================================================================
# E06_multiomics.R — aligning two modalities without paired cells
#
# RNA (1000 genes) and protein (60 markers) are generated from the same latent
# state.  Most cells are measured in one assay only; 60 "anchor" cells are
# measured in both, mimicking a spike-in or a small multiome subset.
#
# The task is retrieval: given a protein cell, which RNA cell has the same
# state?  Four strategies are compared, plus a counter-example in which the
# marginal distributions are identical but the biology is reversed.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")

sim <- readRDS("results/sim.rds")
tr  <- sim$truth
set.seed(4242)

n_rna <- 1500; n_prot <- 800; n_anchor <- 60
anchor_cells <- sample(nrow(tr), n_anchor)
rna_cells  <- c(anchor_cells, sample(setdiff(seq_len(nrow(tr)), anchor_cells), n_rna))
prot_cells <- c(anchor_cells, sample(setdiff(seq_len(nrow(tr)), c(rna_cells, anchor_cells)), n_prot))
stopifnot(length(intersect(rna_cells, prot_cells)) == n_anchor)

Xrna_all <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
Xrna <- Xrna_all[rna_cells, ]

# ---- protein modality ------------------------------------------------------
# 60 markers: 20 differentiation, 15 activation, 15 branch, 10 housekeeping.
# A modality-specific technical factor eta affects half of the markers, so the
# two assays are not related by a rotation alone.
n_prot <- 60
prot_type <- c(rep("diff", 20), rep("act", 15), rep("branch", 15), rep("hk", 10))
make_protein <- function(truth, flip_branch = FALSE, seed = 1) {
  set.seed(seed)
  eta <- rnorm(nrow(truth), 0, 0.6)
  eta_load <- c(rep(0.8, 30), rep(0, 30))
  B <- matrix(rnorm(n_prot * 3, 0, 0.8), n_prot, 3)
  s <- if (flip_branch) -truth$s_branch else truth$s_branch
  f <- cbind(1.3 * sigm(truth$t, 0.5, 6) - 0.65, truth$a - 0.3, s)
  mu <- sweep(f %*% t(B), 2, rnorm(n_prot, 3, 0.5), "+")
  mu <- mu + outer(eta, eta_load)
  mu[, prot_type == "hk"] <- 3 + outer(eta, eta_load)[, prot_type == "hk"] * 0.3
  obs <- mu + matrix(rnorm(length(mu), 0, 0.35), nrow(mu))
  colnames(obs) <- sprintf("P%02d", seq_len(n_prot))
  obs
}
Pro      <- make_protein(tr[prot_cells, ], flip_branch = FALSE, seed = 11)
Pro_flip <- make_protein(tr[prot_cells, ], flip_branch = TRUE,  seed = 11)

p_rna <- pca_svd(Xrna, d = 6)
p_pro <- pca_svd(Pro, d = 6)
p_pro_flip <- pca_svd(Pro_flip, d = 6)
anchor_idx <- match(anchor_cells, prot_cells)
target_rna <- p_rna$scores[, 1:3]

# ---- alignment methods -----------------------------------------------------
align_anchor <- function(P, flip = FALSE) {
  Xa <- P$scores[anchor_idx, 1:5, drop = FALSE]
  Ya <- p_rna$scores[match(anchor_cells, rna_cells), 1:5, drop = FALSE]
  W <- solve(crossprod(Xa) + 1e-2 * diag(5), crossprod(Xa, Ya))
  P$scores[, 1:5, drop = FALSE] %*% W
}

align_coral <- function(P) {
  Xs <- P$scores[, 1:3, drop = FALSE]; Xt <- target_rna
  Cx <- cov(Xs) + 1e-6 * diag(3); Ct <- cov(Xt) + 1e-6 * diag(3)
  w <- with(eigen(Cx), vectors %*% diag(1 / sqrt(values)) %*% t(vectors))
  cc <- with(eigen(Ct), vectors %*% diag(sqrt(values)) %*% t(vectors))
  sweep((Xs - rep(colMeans(Xs), each = nrow(Xs))) %*% w %*% cc, 2, -colMeans(Xt), "+")
}

align_mnn <- function(P, k = 10) {
  Xs <- P$scores[, 1:3, drop = FALSE]; Xt <- target_rna
  k1 <- FNN::get.knnx(Xt, Xs, k = k)$nn.index
  k2 <- FNN::get.knnx(Xs, Xt, k = k)$nn.index
  pairs <- which(vapply(seq_len(nrow(Xs)), function(i) any(k2[k1[i, ], 1] == i), TRUE))
  A <- cbind(1, Xs[pairs, ]); Ya <- Xt[k1[pairs, 1], ]
  W <- solve(crossprod(A) + 1e-2 * diag(4), crossprod(A, Ya))
  sweep(Xs %*% W[-1, ], 2, W[1, ], "+")
}

# ---- evaluation: retrieval of the matching RNA cell ------------------------
random_dt <- mean(abs(outer(tr$t[prot_cells], tr$t[rna_cells], "-")))

evaluate_alignment <- function(E, truth_prot, label) {
  nn <- FNN::get.knnx(target_rna, as.matrix(E)[, 1:3], k = 1)$nn.index[, 1]
  t_nn <- tr$t[rna_cells][nn]
  dt <- abs(t_nn - truth_prot$t)
  br_p <- truth_prot$s_branch
  br_r <- tr$s_branch[rna_cells][nn]
  sel <- abs(br_p) > 0.5 & abs(br_r) > 0.5
  agree <- if (sum(sel) > 20) mean(sign(br_p[sel]) == sign(br_r[sel])) else NA_real_
  data.frame(method = label, match_dt = mean(dt),
             dt_ratio = mean(dt) / random_dt,
             branch_agree = agree, stringsAsFactors = FALSE)
}

mmd_rbf <- function(A, B, n_sub = 400) {
  set.seed(1)
  A <- as.matrix(A); B <- as.matrix(B)
  if (nrow(A) > n_sub) A <- A[sample(nrow(A), n_sub), ]
  if (nrow(B) > n_sub) B <- B[sample(nrow(B), n_sub), ]
  D <- as.matrix(dist(rbind(A, B)))
  K <- exp(-D^2 / median(D^2))
  nA <- nrow(A)
  ia <- seq_len(nA); ib <- setdiff(seq_len(nrow(K)), ia)
  mean(K[ia, ia]) + mean(K[ib, ib]) - 2 * mean(K[ia, ib])
}

settings <- list(
  list(tag = "matched",  P = p_pro,      flip = FALSE),
  list(tag = "reversed", P = p_pro_flip, flip = TRUE)
)
non_anchor <- setdiff(seq_along(prot_cells), anchor_idx)

rows <- list(); emb <- list()
for (s in settings) {
  E_none   <- s$P$scores[, 1:3]
  E_anchor <- align_anchor(s$P)[, 1:3]
  E_coral  <- align_coral(s$P)
  E_mnn    <- align_mnn(s$P)
  methods <- list(none = E_none, anchors = E_anchor, CORAL = E_coral, MNN = E_mnn)
  for (m in names(methods)) {
    r <- evaluate_alignment(methods[[m]][non_anchor, , drop = FALSE],
                            tr[prot_cells[non_anchor], ], m)
    r$scenario <- s$tag
    r$mmd <- mmd_rbf(methods[[m]], target_rna)
    rows[[paste(s$tag, m)]] <- r
    emb[[paste(s$tag, m)]] <- methods[[m]]
  }
}
res <- do.call(rbind, rows)
res <- res[, c("scenario", "method", "match_dt", "dt_ratio", "branch_agree", "mmd")]
rownames(res) <- NULL
print(res, digits = 3)

# how often does an MNN pair actually connect cells with similar t?
k1 <- FNN::get.knnx(target_rna, p_pro$scores[, 1:3], k = 1)$nn.index[, 1]
mnn_acc <- mean(abs(tr$t[prot_cells] - tr$t[rna_cells][k1]) < 0.1)
raw_dt <- mean(abs(tr$t[prot_cells] - tr$t[rna_cells][k1]))
cat(sprintf("nearest-RNA retrieval before alignment: mean |dt| = %.3f (random %.3f)\n",
            raw_dt, random_dt))

saveRDS(list(res = res, embeddings = emb, random_dt = random_dt,
             p_rna = p_rna$scores[, 1:3], p_pro = p_pro$scores[, 1:3],
             truth_prot = tr[prot_cells, ], truth_rna = tr[rna_cells, ],
             n_anchor = n_anchor, non_anchor = non_anchor),
        "results/e06_multiomics.rds")
cat("saved results/e06_multiomics.rds\n")
