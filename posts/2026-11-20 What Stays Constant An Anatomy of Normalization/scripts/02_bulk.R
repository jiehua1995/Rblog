# E02-E04, E08, E10: depth, composition, length, downstream overlap and the
# global-shift counterexample. Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("../../theme-blog.R")
source("R/norm.R")
source("R/sim.R")
source("R/metrics.R")
source("R/io.R")

bulk_methods <- function(counts, len) {
  list(
    CPM = log2(cpm(counts) + 1),
    TPM = log2(tpm(counts, len) + 1),
    UQ  = log2(upper_quartile(counts) + 1),
    MoR = log2(sweep(counts, 2, median_ratios(counts, "poscounts"), "/") + 1),
    RLE = log2(rle_norm(counts) + 1),
    TMM = log2(tmm_norm(counts) + 1),
    QN  = log2(quantile_norm(counts) + 1)
  )
}

summarise_method <- function(logx, sim) {
  d <- limma_de(logx, sim$group)
  m <- match(sim$truth$id, d$id)
  est <- d$logFC[m]
  truth <- sim$truth$lfc
  de <- sim$truth$de
  ev <- evaluate_de(d$adj[m], de)
  c(rmse_de = sqrt(mean((est[de] - truth[de])^2)),
    bias_de = mean(est[de] - truth[de]),
    bias_unchanged = mean(est[!de] - truth[!de]),
    tpr = ev[["tpr"]], fdr = ev[["fdr"]], n_called = ev[["n_called"]])
}

# ------------------------------------------------- E02: depth only
rows <- list()
for (r in 1:5) {
  sim <- sim_counts(n = 2000, m = 12, prop_de = 0.10, depth_sd = 0.60,
                    seed = 100 + r)
  ms <- bulk_methods(sim$counts, sim$length)
  for (m in names(ms)) {
    rows[[length(rows) + 1]] <- data.frame(
      rep = r, method = m, t(summarise_method(ms[[m]], sim)))
  }
}
e02 <- bind_rows(rows)

p1 <- ggplot(e02, aes(reorder(method, rmse_de), rmse_de)) +
  geom_boxplot(fill = blog_pal[["blue"]], colour = "grey20", width = 0.6) +
  labs(x = NULL, y = "RMSE of log2 fold change (DE genes)")

p2 <- ggplot(e02, aes(reorder(method, tpr), tpr)) +
  geom_boxplot(fill = blog_pal[["vermillion"]], colour = "grey20", width = 0.6) +
  labs(x = NULL, y = "TPR at BH < 0.05")

fig_e02 <- p1 | p2
save_fig(fig_e02, "fig04_depth", width = 9.6, height = 3.8)

# ------------------------------------------------- E03: composition
grid <- c(0.05, 0.10, 0.20, 0.35, 0.50)
rows <- list()
for (pd in grid) {
  for (r in 1:3) {
    sim <- sim_counts(n = 2000, m = 12, prop_de = pd, depth_sd = 0.35,
                      de_dir = "up", seed = 300 + r)
    ms <- bulk_methods(sim$counts, sim$length)
    for (m in names(ms)) {
      s <- summarise_method(ms[[m]], sim)
      rows[[length(rows) + 1]] <- data.frame(
        method = m, prop_de = pd, bias_unchanged = s[["bias_unchanged"]])
    }
  }
}
e03 <- bind_rows(rows)

fig_e03 <- ggplot(e03, aes(prop_de, bias_unchanged, colour = method)) +
  stat_summary(fun = mean, geom = "line", linewidth = 0.6) +
  stat_summary(fun = mean, geom = "point", size = 1.8) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
  scale_colour_manual(values = unname(blog_pal[c("blue", "vermillion", "green",
                                                 "purple", "orange", "sky",
                                                 "grey")])) +
  labs(x = "fraction of genes with a true effect",
       y = "mean bias on unchanged genes (log2)") +
  theme(legend.position = "bottom")
save_fig(fig_e03, "fig05_composition", width = 8, height = 4.6)

# ------------------------------------------------- E04: length bias
sim <- sim_counts(n = 3000, m = 12, prop_de = 0.10, depth_sd = 0.40,
                  length_bias = 0.60, seed = 404)
ms <- bulk_methods(sim$counts, sim$length)
d4 <- bind_rows(lapply(c("CPM", "TPM"), function(m) {
  data.frame(method = m, expr = rowMeans(ms[[m]]), length = sim$length)
}))

fig_e04 <- ggplot(d4, aes(log10(length), expr)) +
  geom_point(size = 0.25, alpha = 0.25, colour = blog_pal[["blue"]]) +
  geom_smooth(method = "lm", se = FALSE, colour = blog_pal[["vermillion"]],
              linewidth = 0.6) +
  facet_wrap(~ method, nrow = 1) +
  labs(x = "log10 gene length", y = "mean log2 expression")
save_fig(fig_e04, "fig06_length", width = 8, height = 3.6)

length_cor <- sapply(c("CPM", "TPM"), function(m) {
  stats::cor(rowMeans(ms[[m]]), log10(sim$length))
})

# ------------------------------------------------- E08: real airway overlap
air <- readRDS("data/airway.rds")
real <- list(
  CPM = log2(cpm(air$counts) + 1),
  TPM = log2(tpm(air$counts, air$length) + 1),
  UQ  = log2(upper_quartile(air$counts) + 1),
  MoR = log2(sweep(air$counts, 2, median_ratios(air$counts, "poscounts"), "/") + 1),
  RLE = log2(rle_norm(air$counts) + 1),
  TMM = log2(tmm_norm(air$counts) + 1),
  QN  = log2(quantile_norm(air$counts) + 1),
  VST = vst_norm(air$counts)
)

de_sets <- lapply(real, function(logx) {
  d <- limma_de(logx, air$meta$dex)
  d$id[d$adj < 0.05 & abs(d$logFC) > 1]
})
n_de <- data.frame(method = names(de_sets),
                   n = vapply(de_sets, length, integer(1)))

jac <- matrix(NA, length(de_sets), length(de_sets),
              dimnames = list(names(de_sets), names(de_sets)))
for (i in seq_along(de_sets)) {
  for (j in seq_along(de_sets)) {
    jac[i, j] <- jaccard(de_sets[[i]], de_sets[[j]])
  }
}
jac_long <- as.data.frame(as.table(jac))
names(jac_long) <- c("m1", "m2", "jaccard")

p_heat <- ggplot(jac_long, aes(m1, m2, fill = jaccard)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", jaccard)), size = 2.6) +
  scale_fill_gradient(low = "white", high = blog_pal[["blue"]]) +
  labs(x = NULL, y = NULL, fill = "Jaccard") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_bar <- ggplot(n_de, aes(reorder(method, n), n)) +
  geom_col(fill = blog_pal[["vermillion"]], width = 0.65) +
  coord_flip() +
  labs(x = NULL, y = "DE genes (BH < 0.05, |log2FC| > 1)")

fig_e08 <- p_heat | p_bar
save_fig(fig_e08, "fig07_overlap", width = 9.6, height = 4.4)

# ------------------------------------------- E10: the global-shift counterexample
sim <- sim_counts(n = 2000, m = 12, prop_de = 0.0, shift_all = 0.5,
                  depth_sd = 0.30, seed = 1010)
# Add 50 spike-in features with no group effect.
set.seed(1010)
spike <- matrix(stats::rnbinom(50 * 12,
                               mu = as.vector(outer(exp(stats::rnorm(50, 7, 0.5)),
                                                    sim$size)),
                               size = 20), 50, 12)
spike <- spike / colSums(spike) * (0.02 * mean(colSums(sim$counts)) * sim$size)
rownames(spike) <- paste0("SP", seq_len(nrow(spike)))
colnames(spike) <- colnames(sim$counts)
counts_sp <- rbind(sim$counts, spike)
len_sp <- c(sim$length, rep(1000, 50))
ms <- bulk_methods(counts_sp, len_sp)
ms$SpikeIn <- log2(sweep(counts_sp, 2, colSums(spike), "/") * 1e6 + 1)

rows <- lapply(names(ms), function(m) {
  d <- limma_de(ms[[m]], sim$group)
  idx <- match(sim$truth$id, d$id)
  data.frame(method = m, mean_lfc = mean(d$logFC[idx]),
             true_lfc = 0.5)
})
e10 <- bind_rows(rows)

fig_e10 <- ggplot(e10, aes(reorder(method, mean_lfc), mean_lfc)) +
  geom_col(fill = blog_pal[["green"]], width = 0.65) +
  geom_hline(yintercept = 0.5, linetype = 2, colour = blog_pal[["vermillion"]]) +
  coord_flip() +
  labs(x = NULL, y = "estimated global log2 shift (truth = 0.5)")
save_fig(fig_e10, "fig10_global_shift", width = 7.2, height = 3.6)

# ------------------------------------------------------------------- metrics
save_metrics("e02", list(rows = e02))
save_metrics("e03", list(rows = e03))
save_metrics("e04", list(length_cor = length_cor))
save_metrics("e08", list(n_de = n_de, jaccard = jac))
save_metrics("e10", list(rows = e10))
