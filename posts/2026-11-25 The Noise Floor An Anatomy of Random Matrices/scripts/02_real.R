# E02 -- the same floor in three real datasets.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(Matrix)
})
source("../../theme-blog.R")
source("R/rmt.R")
source("R/io.R")

# ---- helper ----------------------------------------------------------------
# Number of leading eigenvalues above the 95th percentile of the null.
parallel_k <- function(obs, null, level = 0.95) {
  q <- apply(null, 2, stats::quantile, probs = level)
  k <- 0
  for (i in seq_along(q)) {
    if (obs[i] > q[i]) k <- i else break
  }
  k
}

# ---- airway: bulk RNA-seq, 8 samples ---------------------------------------
air <- readRDS("data/airway.rds")
counts <- air$counts
cpm <- t(t(counts) / colSums(counts)) * 1e6
x_air <- log2(cpm + 1)

s_air <- spectrum(x_air)
p_air <- s_air$p
n_air <- s_air$n
g_air <- s_air$gamma
b_air <- mp_bounds(g_air)
# row-centring leaves one exact zero eigenvalue, so only n - 1 are informative
obs_air <- s_air$values[seq_len(n_air - 1)]
null_air <- perm_null_spectrum(x_air, B = 200, k = n_air - 1, seed = 12)
k_air <- parallel_k(obs_air, null_air)

dat_air <- tibble(
  index = seq_along(obs_air),
  share = obs_air / p_air,
  null95 = apply(null_air, 2, stats::quantile, probs = 0.95) / p_air,
  null50 = apply(null_air, 2, stats::median) / p_air
)

fig07 <- ggplot(dat_air, aes(index, share)) +
  geom_col(fill = blog_pal[["blue"]], width = 0.65) +
  geom_point(aes(y = null95), colour = blog_pal[["vermillion"]], size = 2) +
  geom_line(aes(y = null95), colour = blog_pal[["vermillion"]],
            linewidth = 0.4, linetype = "dashed") +
  geom_hline(yintercept = b_air[2] / p_air, linetype = "dotted",
             colour = blog_pal[["grey"]], linewidth = 0.4) +
  scale_x_continuous(breaks = seq_along(obs_air)) +
  labs(x = "component / 主成分序号",
       y = "share of total variance / 占总方差的比例")
save_fig(fig07, "fig07_airway_scree", width = 8, height = 4.4)

null_long <- as.data.frame(null_air) / p_air
names(null_long) <- seq_len(ncol(null_long))
null_long <- pivot_longer(null_long, everything(), names_to = "index",
                          values_to = "share") %>%
  mutate(index = as.integer(index))

fig08 <- ggplot(null_long, aes(factor(index), share)) +
  geom_boxplot(outlier.size = 0.4, linewidth = 0.3, fill = blog_pal[["sky"]],
               alpha = 0.35, colour = blog_pal[["grey"]]) +
  geom_point(data = dat_air, aes(factor(index), share),
             colour = blog_pal[["blue"]], size = 2.6) +
  labs(x = "component / 主成分序号",
       y = "share of total variance / 占总方差的比例")
save_fig(fig08, "fig08_airway_parallel", width = 8, height = 4.4)

# the payoff: PC1 carries a treatment effect
d_air <- svd(standardize_rows(x_air))
# X = U D V', so the coordinates of the samples on the axes are X' U = V D
scores <- d_air$v[, 1:2] %*% diag(d_air$d[1:2])
pca_df <- tibble(
  PC1 = scores[, 1],
  PC2 = scores[, 2],
  dex = factor(air$meta$dex, levels = c("untrt", "trt")),
  cell = air$meta$cell
)
pc1_dex_cor <- cor(scores[, 1], as.numeric(air$meta$dex == "trt"))

fig09 <- ggplot(pca_df, aes(PC1, PC2, colour = dex)) +
  geom_point(size = 3) +
  scale_colour_manual(values = c(untrt = blog_pal[["blue"]],
                                 trt = blog_pal[["vermillion"]])) +
  labs(x = sprintf("PC1 (%.1f%% of variance)", 100 * obs_air[1] / p_air),
       y = sprintf("PC2 (%.1f%% of variance)", 100 * obs_air[2] / p_air),
       colour = NULL)
save_fig(fig09, "fig09_airway_pca", width = 7, height = 4.4)

save_metrics("e06", list(
  airway_p = p_air, airway_n = n_air, airway_gamma = g_air,
  airway_lam_plus = b_air[2],
  airway_floor_share = b_air[2] / p_air,
  airway_pc1_share = obs_air[1] / p_air,
  airway_pc2_share = obs_air[2] / p_air,
  airway_pc1_ratio = obs_air[1] / b_air[2],
  airway_k_parallel = k_air,
  airway_null95_share = dat_air$null95[1],
  airway_pc1_dex_cor = pc1_dex_cor,
  airway_observed_share = obs_air / p_air
))

# ---- PBMC 3k: single cell --------------------------------------------------
mtx <- as(readRDS("data/pbmc.rds"), "CsparseMatrix")
lib <- Matrix::colSums(mtx)
x <- mtx
x@x <- log1p(x@x / rep(lib, diff(x@p)) * 1e4)
mu <- Matrix::rowMeans(x)
x2 <- x
x2@x <- x2@x^2
v <- Matrix::rowMeans(x2) - mu^2
hvg <- order(v, decreasing = TRUE)[1:2000]
set.seed(7)
cells <- sample(ncol(x), 1200)
x_sc <- as.matrix(x[hvg, cells])

s_sc <- spectrum(x_sc)
p_sc <- s_sc$p
n_sc <- s_sc$n
g_sc <- s_sc$gamma
b_sc <- mp_bounds(g_sc)
obs_sc <- s_sc$values
null_sc <- perm_null_spectrum(x_sc, B = 30, k = 25, seed = 13)
k_sc <- parallel_k(obs_sc[seq_len(25)], null_sc)

top_sc <- tibble(
  index = seq_len(15),
  value = obs_sc[seq_len(15)],
  null95 = apply(null_sc, 2, stats::quantile, probs = 0.95)[seq_len(15)]
)

fig10a <- ggplot(top_sc, aes(index, value)) +
  geom_hline(yintercept = b_sc[2], linetype = "dotted",
             colour = blog_pal[["grey"]], linewidth = 0.4) +
  geom_point(aes(y = null95), colour = blog_pal[["vermillion"]], size = 1.8) +
  geom_line(aes(y = null95), colour = blog_pal[["vermillion"]],
            linewidth = 0.4, linetype = "dashed") +
  geom_point(aes(y = value), colour = blog_pal[["blue"]], size = 2.2) +
  geom_line(aes(y = value), colour = blog_pal[["blue"]], linewidth = 0.5) +
  scale_y_log10() +
  scale_x_continuous(breaks = c(1, 5, 10, 15)) +
  labs(x = "component / 主成分序号", y = "eigenvalue / 特征值")

bulk_sc <- tibble(value = obs_sc)
xgrid <- seq(1e-3, b_sc[2] * 1.02, length.out = 500)

fig10b <- ggplot(bulk_sc, aes(log10(value))) +
  geom_histogram(aes(y = after_stat(density)), bins = 60,
                 fill = blog_pal[["sky"]], colour = NA, alpha = 0.55) +
  geom_line(data = tibble(x = xgrid,
                          y = mp_density(xgrid, g_sc) * xgrid * log(10)),
            aes(x = log10(x), y = y), colour = blog_pal[["vermillion"]],
            linewidth = 0.7) +
  coord_cartesian(xlim = log10(c(1e-3, b_sc[2] * 1.1))) +
  labs(x = expression(log[10] * "(eigenvalue)"), y = "density / 密度")

fig10 <- fig10a | fig10b
save_fig(fig10, "fig10_pbmc_spectrum", width = 9.2, height = 4.2)

save_metrics("e07", list(
  pbmc_p = p_sc, pbmc_n = n_sc, pbmc_gamma = g_sc,
  pbmc_lam_plus = b_sc[2], pbmc_floor_share = b_sc[2] / p_sc,
  pbmc_pc1_share = obs_sc[1] / p_sc, pbmc_pc1_ratio = obs_sc[1] / b_sc[2],
  pbmc_k_parallel = k_sc, pbmc_k_capped_at_25 = k_sc == 25
))

# ---- CPTAC: 6 runs ---------------------------------------------------------
cpt <- readRDS("data/cptac.rds")
s_cp <- spectrum(cpt$x)
p_cp <- s_cp$p
n_cp <- s_cp$n
g_cp <- s_cp$gamma
b_cp <- mp_bounds(g_cp)
obs_cp <- s_cp$values[seq_len(n_cp - 1)]
null_cp <- perm_null_spectrum(cpt$x, B = 200, k = n_cp - 1, seed = 14)
k_cp <- parallel_k(obs_cp, null_cp)

dat_cp <- tibble(
  index = seq_along(obs_cp),
  share = obs_cp / p_cp,
  null95 = apply(null_cp, 2, stats::quantile, probs = 0.95) / p_cp
)

fig11 <- ggplot(dat_cp, aes(index, share)) +
  geom_col(fill = blog_pal[["purple"]], width = 0.6) +
  geom_point(aes(y = null95), colour = blog_pal[["vermillion"]], size = 2) +
  geom_hline(yintercept = b_cp[2] / p_cp, linetype = "dotted",
             colour = blog_pal[["grey"]], linewidth = 0.4) +
  scale_x_continuous(breaks = seq_along(obs_cp)) +
  labs(x = "component / 主成分序号",
       y = "share of total variance / 占总方差的比例")
save_fig(fig11, "fig11_cptac_scree", width = 8, height = 4.4)

save_metrics("e08", list(
  cptac_p = p_cp, cptac_n = n_cp, cptac_gamma = g_cp,
  cptac_lam_plus = b_cp[2], cptac_floor_share = b_cp[2] / p_cp,
  cptac_pc1_share = obs_cp[1] / p_cp, cptac_pc1_ratio = obs_cp[1] / b_cp[2],
  cptac_k_parallel = k_cp
))

# ---- figure 12: observed PC1 against the floor, three datasets -------------
pc1_obs <- c(obs_air[1] / p_air, obs_sc[1] / p_sc, obs_cp[1] / p_cp)
pc1_floor <- c(b_air[2] / p_air, b_sc[2] / p_sc, b_cp[2] / p_cp)
summ <- tibble(
  data = rep(c("airway\n24.7k x 8", "PBMC 3k\n2k x 1.2k", "CPTAC\n1.1k x 6"),
             each = 2),
  what = rep(c("observed", "floor"), times = 3),
  share = as.vector(rbind(pc1_obs, pc1_floor))
)

fig12 <- ggplot(summ, aes(data, share, fill = what)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.62) +
  scale_fill_manual(values = c(observed = blog_pal[["blue"]],
                               floor = blog_pal[["grey"]]),
                    labels = c(floor = "noise floor", observed = "observed PC1")) +
  labs(x = NULL, y = "share of total variance / 占总方差的比例", fill = NULL)
save_fig(fig12, "fig12_three_datasets", width = 8, height = 4.4)

save_metrics("e09", list(
  pc1_observed = pc1_obs, pc1_floor = pc1_floor,
  labels = c("airway", "PBMC 3k", "CPTAC")
))

message("02_real done")
