# A deliberately small, readable CART implementation.
# It is used only where the point is the split rule itself; all large-scale
# experiments use ranger's C++ backend.

cart_fit <- function(X, y, task = c("regression", "classification"),
                     max_depth = 20, min_node = 5,
                     mtry = ncol(X), max_candidates = 20, seed = 1) {
  task <- match.arg(task)
  X <- as.matrix(X)
  y <- as.numeric(y)
  p <- ncol(X)
  set.seed(seed)

  impurity <- function(idx) {
    if (task == "regression") {
      length(idx) * var(y[idx])
    } else {
      pr <- mean(y[idx])
      2 * length(idx) * pr * (1 - pr)
    }
  }

  best_split <- function(idx, feats) {
    parent <- impurity(idx)
    best <- list(gain = 0)
    for (j in feats) {
      xj <- X[idx, j]
      ux <- unique(xj)
      if (length(ux) < 2) next
      if (length(ux) > max_candidates) {
        qs <- unique(quantile(xj, probs = seq(0, 1, length.out = max_candidates + 1)))
      } else {
        qs <- sort(ux)
      }
      cand <- (head(qs, -1) + tail(qs, -1)) / 2
      for (s in cand) {
        left <- xj <= s
        nl <- sum(left)
        nr <- length(idx) - nl
        if (nl < min_node || nr < min_node) next
        gain <- parent - impurity(idx[left]) - impurity(idx[!left])
        if (gain > best$gain) {
          best <- list(gain = gain, var = j, split = s,
                       left = idx[left], right = idx[!left])
        }
      }
    }
    best
  }

  grow <- function(idx, depth) {
    node <- list(leaf = TRUE, n = length(idx),
                 value = if (task == "regression") mean(y[idx]) else mean(y[idx]))
    if (depth >= max_depth || length(idx) < 2 * min_node) return(node)
    feats <- if (mtry < p) sample.int(p, mtry) else seq_len(p)
    sp <- best_split(idx, feats)
    if (sp$gain <= 1e-10) return(node)
    node <- list(leaf = FALSE, n = length(idx), var = sp$var,
                 split = sp$split, gain = sp$gain,
                 left = grow(sp$left, depth + 1),
                 right = grow(sp$right, depth + 1))
    node
  }

  structure(list(tree = grow(seq_along(y), 0), task = task),
            class = "cart")
}

cart_predict_one <- function(node, x) {
  while (!node$leaf) {
    node <- if (x[node$var] <= node$split) node$left else node$right
  }
  node$value
}

cart_predict <- function(fit, X) {
  X <- as.matrix(X)
  apply(X, 1, function(x) cart_predict_one(fit$tree, x))
}

# Bootstrap forest used for the small pedagogical figures.
rf_boot <- function(X, y, B, mtry, task = "regression",
                    max_depth = 20, min_node = 5, seed = 1) {
  n <- nrow(X)
  set.seed(seed)
  trees <- vector("list", B)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    trees[[b]] <- cart_fit(X[idx, , drop = FALSE], y[idx], task = task,
                           max_depth = max_depth, min_node = min_node,
                           mtry = mtry, seed = seed + b)
  }
  structure(list(trees = trees, task = task), class = "rf_boot")
}

rf_boot_predict <- function(fit, X) {
  X <- as.matrix(X)
  preds <- vapply(fit$trees, function(tr) cart_predict(tr, X), numeric(nrow(X)))
  rowMeans(preds)
}

# Mean pairwise correlation between tree predictions, estimated on a sample
# of test points to keep the cost linear in B.
tree_correlation <- function(pred_mat, n_points = 200, n_trees = 60) {
  pred_mat <- as.matrix(pred_mat)
  if (ncol(pred_mat) > n_trees) {
    pred_mat <- pred_mat[, sample.int(ncol(pred_mat), n_trees), drop = FALSE]
  }
  if (nrow(pred_mat) > n_points) {
    pred_mat <- pred_mat[sample.int(nrow(pred_mat), n_points), , drop = FALSE]
  }
  R <- cor(pred_mat)
  mean(R[upper.tri(R)])
}

# Pointwise covariance between two trees: cov(T_b(x), T_b'(x)) at a fixed x,
# estimated across forests trained on independently drawn datasets. Averaging
# covariances (rather than correlations) avoids letting low-variance test
# points dominate.
pointwise_cov <- function(pred_list, n_pairs = 20, n_points = 120) {
  R <- length(pred_list)
  B <- ncol(pred_list[[1]])
  n <- nrow(pred_list[[1]])
  pts <- sample.int(n, min(n_points, n))
  pairs <- matrix(sample.int(B, 2 * n_pairs, replace = TRUE), ncol = 2)
  pairs <- pairs[pairs[, 1] != pairs[, 2], , drop = FALSE]
  vals <- numeric(nrow(pairs))
  for (k in seq_len(nrow(pairs))) {
    A <- vapply(pred_list, function(P) P[pts, pairs[k, 1]], numeric(length(pts)))
    Bm <- vapply(pred_list, function(P) P[pts, pairs[k, 2]], numeric(length(pts)))
    vals[k] <- mean(diag(cov(t(A), t(Bm))))
  }
  mean(vals)
}

pointwise_rho <- function(pred_list, sigma2, n_pairs = 20, n_points = 120) {
  pointwise_cov(pred_list, n_pairs, n_points) / sigma2
}

# Variance of a single tree at a test point, pooling bootstrap replicates and
# independently drawn training sets.
pooled_sigma2 <- function(pred_list) {
  arr <- simplify2array(pred_list)  # n_test x B x R
  mean(apply(arr, 1, function(slice) var(as.vector(slice))))
}
