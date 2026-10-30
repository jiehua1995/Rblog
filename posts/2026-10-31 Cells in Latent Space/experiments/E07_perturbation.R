# =============================================================================
# E07_perturbation.R — state-dependent perturbation responses
#
# Three perturbations act on disjoint gene modules with different dependence on
# the cell state.  A fourth perturbation is never observed.  Methods:
#   fixed    : one average response vector per perturbation
#   kNN      : average response of neighbouring cells with the same perturbation
#   interact : ridge regression with state x perturbation interactions
# Evaluation is always on held-out cells, never on the cells used for fitting.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")

sim <- readRDS("results/sim.rds")
tr  <- sim$truth
g   <- sim$genes
X   <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
pca4 <- pca_svd(X, d = 4)
Z    <- pca4$scores                        # learned 4-D state representation
set.seed(777)

# ---- perturbation design ---------------------------------------------------
# each perturbation targets 80 genes, disjoint from the branch modules
cand <- setdiff(seq_len(1000), c(sim$modules$branchA, sim$modules$branchB))
mods <- list(); used <- integer(0)
for (u in c("u1", "u2", "u3", "u4")) {
  pick <- sample(setdiff(cand, used), 80)
  used <- c(used, pick)
  mods[[u]] <- pick
}

h <- function(u, t, a) switch(u,
  u1 = 0.2 + 1.6 * t,           # late-acting
  u2 = 1.6 - 2.4 * t,           # switches sign along the trajectory
  u3 = 0.4 + 1.2 * t + 1.0 * a, # late-acting, modulated by activation
  u4 = 0.3 + 1.5 * t)           # never observed

# Perturbed libraries are sequenced three times deeper than the controls; this
# only reduces counting noise, the expected log-CPM profile is unchanged.
depth_mult <- 3

# Each perturbed cell is paired with an independent control draw from exactly
# the same expected profile; this is the cleanest version of a matched control
# and it removes any bias from neighbourhood averaging.
draw_perturbed <- function(u, cells, amp = 2.0) {
  d <- matrix(0, length(cells), 1000)
  d[, mods[[u]]] <- -amp * h(u, tr$t[cells], tr$a[cells])
  mu0 <- exp(sim$log_mu[cells, , drop = FALSE]) * depth_mult
  mu1 <- exp(sim$log_mu[cells, , drop = FALSE] + d) * depth_mult
  th <- rep(g$theta, each = length(cells))
  cnt0 <- matrix(rnbinom(length(mu0), mu = as.vector(mu0), size = th),
                 length(cells), 1000)
  cnt1 <- matrix(rnbinom(length(mu1), mu = as.vector(mu1), size = th),
                 length(cells), 1000)
  list(counts = cnt1, control = cnt0, delta_true = d, cells = cells, u = u)
}

n_u <- 500
obs <- list(
  u1 = draw_perturbed("u1", sample(3000, n_u)),
  u2 = draw_perturbed("u2", sample(which(tr$t < 0.5), n_u)),
  u3 = draw_perturbed("u3", sample(3000, n_u)),
  # held-out late cells for u2: the perturbation was profiled early only
  u2_late = draw_perturbed("u2", sample(which(tr$t > 0.7), 200))
)
obs_u4 <- draw_perturbed("u4", sample(3000, 300))

# ---- observed response: perturbed profile minus its matched control --------
delta_obs <- function(o) logcpm(o$counts) - logcpm(o$control)
D <- lapply(obs, delta_obs)
train_us <- c("u1", "u2", "u3")

split_u <- lapply(D[train_us], function(d) {
  n <- nrow(d); te <- sample(n, round(0.3 * n))
  list(train = setdiff(seq_len(n), te), test = te)
})

# ---- methods ---------------------------------------------------------------
fit_fixed <- function(u) colMeans(D[[u]][split_u[[u]]$train, , drop = FALSE])

predict_knn <- function(u, cells_query, k = 25) {
  tr_idx <- split_u[[u]]$train
  nn <- FNN::get.knnx(Z[obs[[u]]$cells[tr_idx], , drop = FALSE],
                      Z[cells_query, , drop = FALSE], k = k)$nn.index
  Dsub <- D[[u]][tr_idx, , drop = FALSE]
  t(apply(nn, 1, function(ii) colMeans(Dsub[ii, , drop = FALSE])))
}

# design: 1, four state coordinates, three perturbation dummies,
# and all state x perturbation products (4 x 3)
design <- function(z, u) {
  U <- c(u1 = u == "u1", u2 = u == "u2", u3 = u == "u3")
  c(1, z, U, outer(z, U))
}

train_X <- do.call(rbind, lapply(train_us, function(u) {
  do.call(rbind, lapply(split_u[[u]]$train, function(i) design(Z[obs[[u]]$cells[i], ], u)))
}))
train_Y <- do.call(rbind, lapply(train_us, function(u) D[[u]][split_u[[u]]$train, , drop = FALSE]))
ridge_W <- solve(crossprod(train_X) + 0.5 * diag(ncol(train_X)), crossprod(train_X, train_Y))

predict_interact <- function(u, cells_query) {
  Xn <- do.call(rbind, lapply(cells_query, function(cc) design(Z[cc, ], u)))
  Xn %*% ridge_W
}

# ---- evaluation ------------------------------------------------------------
resp_genes <- unique(unlist(mods))
evaluate <- function(pred, obs_delta, genes = resp_genes) {
  p <- pred[, genes, drop = FALSE]; o <- obs_delta[, genes, drop = FALSE]
  cos_s <- mean(rowSums(p * o) / (sqrt(rowSums(p^2)) * sqrt(rowSums(o^2)) + 1e-12))
  r2 <- 1 - sum((p - o)^2) / sum((o - mean(o))^2)
  mag <- mean(sqrt(rowSums(p^2)) / (sqrt(rowSums(o^2)) + 1e-12))
  c(cos = cos_s, r2 = r2, magnitude = mag)
}

rows <- list()
cat(sprintf("u2 fixed module mean %.2f ; u2 late observed module mean %.2f ; u2 early observed %.2f\n",
            mean(fit_fixed("u2")[mods$u2]), mean(D$u2_late[, mods$u2]),
            mean(D$u2[, mods$u2])))
for (u in train_us) {
  te <- split_u[[u]]$test
  od <- D[[u]][te, , drop = FALSE]
  rows[[paste(u, "fixed")]] <- data.frame(scenario = u, method = "fixed",
    t(evaluate(matrix(fit_fixed(u), length(te), 1000, byrow = TRUE), od)))
  rows[[paste(u, "knn")]] <- data.frame(scenario = u, method = "kNN",
    t(evaluate(predict_knn(u, obs[[u]]$cells[te]), od)))
  rows[[paste(u, "int")]] <- data.frame(scenario = u, method = "interaction",
    t(evaluate(predict_interact(u, obs[[u]]$cells[te]), od)))
}

# extrapolation inside a seen perturbation: u2 was only measured at t < 0.5,
# but the same perturbation is now queried on cells with t > 0.7
od <- D$u2_late
qz <- obs$u2_late$cells
rows[["u2late_fixed"]] <- data.frame(scenario = "u2 (late, extrapolation)",
  method = "fixed", t(evaluate(matrix(fit_fixed("u2"), length(qz), 1000, byrow = TRUE), od)))
rows[["u2late_knn"]] <- data.frame(scenario = "u2 (late, extrapolation)",
  method = "kNN", t(evaluate(predict_knn("u2", qz), od)))
rows[["u2late_int"]] <- data.frame(scenario = "u2 (late, extrapolation)",
  method = "interaction", t(evaluate(predict_interact("u2", qz), od)))

# unseen perturbation: only the interaction model can even be evaluated
du4 <- delta_obs(obs_u4)
test_u4 <- sample(nrow(du4), 60)
rows[["u4_int"]] <- data.frame(scenario = "u4 (unseen)", method = "interaction",
  t(evaluate(predict_interact("u0", obs_u4$cells[test_u4]), du4[test_u4, , drop = FALSE])))
rows[["u4_null"]] <- data.frame(scenario = "u4 (unseen)", method = "null (0)",
  t(evaluate(matrix(0, length(test_u4), 1000), du4[test_u4, , drop = FALSE])))

res <- do.call(rbind, rows); rownames(res) <- NULL
print(res, digits = 3)

# ---- response direction over a grid of cell states (for the vector field) ---
qs1 <- quantile(Z[, 1], seq(0.08, 0.92, length.out = 9))
qs2 <- quantile(Z[, 2], c(0.2, 0.5, 0.8))
vfield <- expand.grid(z1 = qs1, z2 = qs2, u = c("u1", "u2"), stringsAsFactors = FALSE)
vfield$dx <- NA_real_; vfield$dy <- NA_real_
proj2 <- pca4$rotation[, 1:2]
gene_sd <- apply(X, 2, sd)
for (i in seq_len(nrow(vfield))) {
  z <- c(vfield$z1[i], vfield$z2[i], mean(Z[, 3]), mean(Z[, 4]))
  v <- as.vector(design(z, vfield$u[i]) %*% ridge_W)
  # put the log-CPM response on the standardised scale used by the PCA
  d_lat <- as.vector((v / gene_sd) %*% proj2)
  vfield$dx[i] <- d_lat[1]; vfield$dy[i] <- d_lat[2]
}

# ---- expression similarity versus functional similarity --------------------
# Two cells can be neighbours in expression space and still respond differently
# to the same perturbation.  Quantify that with the u1 data.
set.seed(3)
ii <- sample(seq_len(nrow(D$u1)), 300)
d_state <- as.matrix(dist(Z[obs$u1$cells[ii], ]))
d_resp  <- as.matrix(dist(D$u1[ii, resp_genes]))
up <- upper.tri(d_state)
nn10 <- FNN::get.knnx(Z[obs$u1$cells[ii], ], Z[obs$u1$cells[ii], ], k = 11)$nn.index[, -1]
func <- c(
  cor_state_response = cor(d_state[up], d_resp[up], method = "spearman"),
  resp_diff_neighbours = mean(vapply(seq_along(ii), function(i) mean(d_resp[i, nn10[i, ]]), 0)),
  resp_diff_random = mean(d_resp[up])
)
print(round(func, 3))

saveRDS(list(res = res, vfield = vfield, resp_genes = resp_genes, func = func,
             modules = mods, truth = tr, Z = Z, pca_rotation = pca4$rotation),
        "results/e07_perturbation.rds")
cat("saved results/e07_perturbation.rds\n")
