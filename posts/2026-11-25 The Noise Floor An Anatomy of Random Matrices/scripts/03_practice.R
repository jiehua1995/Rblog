# E03 -- practical consequences: how many components, hard thresholding,
# shrinkage, and what denoising buys on a real matrix.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("../../theme-blog.R")
source("R/rmt.R")
source("R/io.R")

# ---- figure 13: three rules for choosing k --------------------------------
# X = sum_j s_j u_j v_j' + noise, with every s_j above the noise edge.
sim_rank <- function(p, n, k, s) {
  Z <- matrix(stats::rnorm(p * n), p, n)
  if (k > 0) {
    U <- qr.Q(qr(matrix(stats::rnorm(p * k), p, k)))
    V <- qr.Q(qr(matrix(stats::rnorm(n * k), n, k)))
    Z <- Z + U %*% (diag(s, k) %*% t(V))
  }
  Z
}

p13 <- 200
n13 <- 500
k_true <- c(0, 1, 2, 3, 5)
reps13 <- 40
s13 <- seq(140, 80, length.out = 5)

# the permutation null depends on (p, n) only, so compute it once
null13 <- perm_null_spectrum(sim_rank(p13, n13, 0, numeric(0)),
                             B = 100, k = 10, seed = 99)
q95_13 <- stats::quantile(null13[, 1], probs = 0.95)

choose_k <- function(X, q95) {
  ev <- spectrum(X, standardize = FALSE)$values
  k_null <- sum(ev > q95)
  k_kaiser <- sum(ev > mean(ev))
  drops <- ev[-length(ev)] - ev[-1]
  k_elbow <- which.max(drops)
  c(null = k_null, kaiser = k_kaiser, elbow = k_elbow)
}

res13 <- lapply(k_true, function(kk) {
  t(replicate(reps13, {
    X <- sim_rank(p13, n13, kk, if (kk == 0) numeric(0) else s13[seq_len(kk)])
    choose_k(X, q95_13)
  }))
})

acc13 <- do.call(rbind, lapply(seq_along(k_true), function(i) {
  r <- res13[[i]]
  data.frame(k_true = k_true[i],
             rule = c("null", "kaiser", "elbow"),
             k_hat = unname(colMeans(r)))
}))
print(acc13)

fig13 <- ggplot(acc13, aes(k_true, k_hat, colour = rule)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = blog_pal[["grey"]], linewidth = 0.4) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 2.4) +
  scale_colour_manual(values = c(null = blog_pal[["blue"]],
                                 kaiser = blog_pal[["orange"]],
                                 elbow = blog_pal[["purple"]]),
                      labels = c(null = "null edge (95%)", kaiser = "Kaiser",
                                 elbow = "elbow")) +
  labs(x = "true number of signals / 真实信号数",
       y = "mean estimated k / 估计的 k 的均值", colour = NULL)
save_fig(fig13, "fig13_choose_k", width = 8.4, height = 4.4)

save_metrics("e10", acc13)

# ---- figure 14: Gavish-Donoho hard threshold ------------------------------
p14 <- 200
n14 <- 500
r14 <- 3
s14 <- c(80, 60, 45)
tau_gd <- gd_tau(p14, n14)
taus <- seq(0.05, 1.6, length.out = 40) * tau_gd
reps14 <- 12

set.seed(301)
mats14 <- lapply(seq_len(reps14), function(i) {
  U <- qr.Q(qr(matrix(stats::rnorm(p14 * r14), p14, r14)))
  V <- qr.Q(qr(matrix(stats::rnorm(n14 * r14), n14, r14)))
  S <- U %*% (diag(s14) %*% t(V))
  list(S = S, X = S + matrix(stats::rnorm(p14 * n14), p14, n14))
})

mse14 <- lapply(taus, function(tau) {
  hard <- mean(vapply(mats14, function(m) {
    d <- svd(m$X)
    keep <- d$d > tau
    Xh <- d$u[, keep, drop = FALSE] %*%
      (diag(d$d[keep], sum(keep)) %*% t(d$v[, keep, drop = FALSE]))
    mean((Xh - m$S)^2)
  }, numeric(1)))
  soft <- mean(vapply(mats14, function(m) {
    d <- svd(m$X)
    Xs <- d$u %*% (diag(pmax(d$d - tau, 0)) %*% t(d$v))
    mean((Xs - m$S)^2)
  }, numeric(1)))
  tibble(tau = tau, hard = hard, soft = soft)
}) %>% do.call(rbind, .)

best14 <- mse14$tau[which.min(mse14$hard)]
fig14 <- ggplot(mse14, aes(tau / tau_gd)) +
  geom_line(aes(y = hard, colour = "hard"), linewidth = 0.8) +
  geom_line(aes(y = soft, colour = "soft"), linewidth = 0.8) +
  geom_vline(xintercept = 1, linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.4) +
  geom_vline(xintercept = best14 / tau_gd, linetype = "dotted",
             colour = blog_pal[["vermillion"]], linewidth = 0.4) +
  scale_colour_manual(values = c(hard = blog_pal[["blue"]],
                                 soft = blog_pal[["orange"]]),
                      labels = c(hard = "hard threshold", soft = "soft threshold")) +
  labs(x = expression(tau / tau[GD] * " (relative to the Gavish-Donoho threshold)"),
       y = "mean squared error / 均方误差", colour = NULL)
save_fig(fig14, "fig14_hard_threshold", width = 8, height = 4.4)

save_metrics("e11", list(
  gd_tau = tau_gd, gd_ratio_best = best14 / tau_gd,
  mse_at_gd = mse14$hard[which.min(abs(mse14$tau / tau_gd - 1))],
  mse_best = min(mse14$hard), mse_at_zero = mse14$hard[1]
))

# ---- figure 15: shrinkage --------------------------------------------------
p15 <- 100
n15 <- 60
grid15 <- seq(0, 1, by = 0.02)
reps15 <- 20
# population covariance: three signals on top of an identity bulk
Sig <- diag(p15)
diag(Sig)[1:3] <- c(4, 3, 2)

set.seed(401)
risk15 <- rowMeans(replicate(reps15, {
  Y <- t(chol(Sig)) %*% matrix(stats::rnorm(p15 * n15), p15, n15)
  S <- tcrossprod(Y - rowMeans(Y)) / n15
  m <- sum(diag(S)) / p15
  vapply(grid15, function(d) {
    C <- (1 - d) * S + d * m * diag(p15)
    sum((C - Sig)^2)
  }, numeric(1))
}))

Y15 <- t(chol(Sig)) %*% matrix(stats::rnorm(p15 * n15), p15, n15)
S15 <- tcrossprod(Y15 - rowMeans(Y15)) / n15
lw15 <- lw_shrink(Y15)

ev15 <- tibble(
  value = c(eigen(S15, symmetric = TRUE, only.values = TRUE)$values,
            eigen(lw15$cov, symmetric = TRUE, only.values = TRUE)$values,
            eigen(Sig, symmetric = TRUE, only.values = TRUE)$values),
  what = factor(rep(c("sample", "shrunk", "truth"), each = p15),
                levels = c("sample", "shrunk", "truth"))
)

risks <- tibble(delta = grid15, risk = risk15)
fig15a <- ggplot(risks, aes(delta, risk)) +
  geom_line(colour = blog_pal[["blue"]], linewidth = 0.8) +
  geom_vline(xintercept = lw15$delta, linetype = "dashed",
             colour = blog_pal[["vermillion"]], linewidth = 0.5) +
  geom_vline(xintercept = grid15[which.min(risk15)], linetype = "dotted",
             colour = blog_pal[["green"]], linewidth = 0.5) +
  labs(x = expression(delta * " (shrinkage toward " * mu * I * ")"),
       y = "Frobenius risk / 弗罗贝尼乌斯风险")

fig15b <- ggplot(ev15, aes(value, fill = what)) +
  geom_histogram(position = "identity", bins = 25, alpha = 0.55, colour = NA) +
  scale_fill_manual(values = c(sample = blog_pal[["grey"]],
                               shrunk = blog_pal[["green"]],
                               truth = blog_pal[["vermillion"]])) +
  labs(x = "eigenvalue / 特征值", y = "count / 计数", fill = NULL)

fig15 <- fig15a | fig15b
save_fig(fig15, "fig15_shrinkage", width = 9.2, height = 4.2)

save_metrics("e12", list(
  lw_delta = lw15$delta,
  risk_at_lw = risk15[which.min(abs(grid15 - lw15$delta))],
  risk_at_oracle = min(risk15), risk_at_zero = risk15[1],
  oracle_delta = grid15[which.min(risk15)]
))

# ---- figure 16: the floor as a function of the number of samples -----------
air <- readRDS("data/airway.rds")
p_air <- nrow(standardize_rows(log2(t(t(air$counts) / colSums(air$counts)) * 1e6 + 1)))
ns <- 2:8
floor_df <- tibble(
  n = ns,
  floor = vapply(ns, function(k) mp_bounds(p_air / k)[2] / p_air, numeric(1))
)
# the observed PC1 share is computed in 02_real.R and stored in metrics.json
obs_share <- jsonlite::fromJSON("outputs/metrics.json")$e06$airway_pc1_share

fig16 <- ggplot(floor_df, aes(n, floor)) +
  geom_col(fill = blog_pal[["grey"]], width = 0.6) +
  geom_hline(yintercept = obs_share, linetype = "dashed",
             colour = blog_pal[["blue"]], linewidth = 0.6) +
  annotate("text", x = 2.6, y = obs_share, vjust = -0.6, hjust = 0,
           label = sprintf("observed PC1 = %.1f%%", 100 * obs_share),
           colour = blog_pal[["blue"]], size = 3.3) +
  scale_x_continuous(breaks = ns) +
  labs(x = "number of samples n / 样本数",
       y = "noise floor as a share of variance / 噪声地板占总方差的比例")
save_fig(fig16, "fig16_floor_vs_n", width = 8, height = 4.4)

save_metrics("e13", list(n = ns, floor = floor_df$floor,
                         pc1_share = obs_share,
                         min_n_for_pc1 = min(ns[floor_df$floor < obs_share])))

message("03_practice done")
