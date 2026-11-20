# E01: what the raw data look like, and what six normalisations do to one
# RNA-seq experiment. Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("../../theme-blog.R")
source("R/norm.R")
source("R/metrics.R")
source("R/io.R")

air <- readRDS("data/airway.rds")
prot <- readRDS("data/lfq.rds")
pbmc <- as.matrix(readRDS("data/pbmc.rds"))
cll <- readRDS("data/cll.rds")
metab <- readRDS("data/metab.rds")

# ---------------------------------------------------------------- helpers
long_mat <- function(mat, n = 12, prefix = "") {
  idx <- unique(round(seq(1, ncol(mat), length.out = min(n, ncol(mat)))))
  d <- as.data.frame(mat[, idx, drop = FALSE])
  d$feature <- rownames(d)
  pivot_longer(d, -feature, names_to = "sample", values_to = "value")
}

panel_box <- function(df, fill, title, ylab) {
  ggplot(df, aes(sample, value)) +
    geom_boxplot(fill = fill, colour = "grey20", linewidth = 0.25,
                 outlier.size = 0.2) +
    labs(title = title, x = NULL, y = ylab) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))
}

# ------------------------------------------------------- figure 1: raw data
# Low-count features are filtered for display only; the article says so.
keep_air <- rowSums(cpm(air$counts) > 1) >= 2
keep_sc  <- rowSums(pbmc > 0) >= 0.10 * ncol(pbmc)
keep_cll <- rowMeans(cll$expr > 6) >= 0.50

c_air <- log2(cpm(air$counts) + 1)
p_air <- panel_box(long_mat(c_air[keep_air, ], 8), blog_pal[["blue"]],
                   "A  RNA-seq (airway)", "log2 CPM")

p_prot <- panel_box(long_mat(log2(prot$x), 12), blog_pal[["vermillion"]],
                    "B  LFQ proteomics (dilution series)", "log2 intensity")

p_sc <- panel_box(long_mat(log2(cpm(pbmc) + 1)[keep_sc, ], 12),
                  blog_pal[["green"]],
                  "C  single cell (PBMC 3k)", "log2 CPM")

p_cll <- panel_box(long_mat(log2(cll$expr + 1)[keep_cll, ], 12),
                   blog_pal[["purple"]],
                   "D  microarray (CLL)", "log2 intensity")

p_met <- panel_box(long_mat(metab$x, 12), blog_pal[["orange"]],
                   "E  metabolomics (sacurine)", "log10 intensity")

fig1 <- (p_air | p_prot | p_sc) / (p_cll | p_met | plot_spacer()) +
  plot_layout(heights = c(1, 1))
save_fig(fig1, "fig01_raw_data", width = 10, height = 6.2)

# ------------------------------------- figure 2: six normalisations on airway
methods <- list(
  CPM = log2(cpm(air$counts) + 1),
  TPM = log2(tpm(air$counts, air$length) + 1),
  UQ  = log2(upper_quartile(air$counts) + 1),
  MoR = log2(sweep(air$counts, 2, median_ratios(air$counts, "poscounts"), "/") + 1),
  TMM = log2(tmm_norm(air$counts) + 1),
  QN  = log2(quantile_norm(air$counts) + 1),
  VST = vst_norm(air$counts)
)

methods <- lapply(methods, function(m) m[keep_air, , drop = FALSE])

long_methods <- bind_rows(lapply(names(methods), function(m) {
  d <- long_mat(methods[[m]], 8)
  d$method <- m
  d
}))
print(unique(long_methods$sample))

fig2 <- ggplot(long_methods, aes(sample, value)) +
  geom_boxplot(fill = blog_pal[["sky"]], colour = "grey20",
               linewidth = 0.22, outlier.size = 0.15) +
  facet_wrap(~ method, nrow = 2, scales = "free_y") +
  labs(x = NULL, y = "log2 expression") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6))
save_fig(fig2, "fig02_methods_airway", width = 10, height = 5.4)

# --------------------------------------------------- figure 3: PCA by method
pca_one <- function(logx, method) {
  v <- apply(logx, 1, stats::var)
  top <- order(v, decreasing = TRUE)[seq_len(min(2000, length(v)))]
  z <- t(scale(t(logx[top, , drop = FALSE])))
  z[!is.finite(z)] <- 0
  pc <- stats::prcomp(t(z))
  data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2],
             cell = factor(air$meta$cell), dex = factor(air$meta$dex),
             method = method)
}

pca_all <- bind_rows(lapply(names(methods), function(m) pca_one(methods[[m]], m)))
lib <- colSums(air$counts)

pca_tab <- bind_rows(lapply(names(methods), function(m) {
  p <- pca_all[pca_all$method == m, ]
  data.frame(method = m,
             pc1_lib_cor = abs(stats::cor(p$PC1, log2(lib))),
             cell_r2 = summary(stats::lm(PC1 ~ cell, data = p))$r.squared,
             dex_r2 = summary(stats::lm(PC1 ~ dex, data = p))$r.squared)
}))

pca_long <- pivot_longer(pca_tab, c(pc1_lib_cor, cell_r2),
                         names_to = "metric", values_to = "value")
fig3 <- ggplot(pca_long, aes(reorder(method, value), value, fill = metric)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  facet_wrap(~ metric, nrow = 1, scales = "free_x",
             labeller = as_labeller(c(cell_r2 = "cell-line R²",
                                      pc1_lib_cor = "|cor(PC1, depth)|"))) +
  scale_fill_manual(values = unname(blog_pal[c("blue", "vermillion")])) +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme(legend.position = "none")
save_fig(fig3, "fig03_pca_airway", width = 8.6, height = 3.6)

# ------------------------------------------------------------------- metrics
lib <- colSums(air$counts)
raw_stats <- list(
  airway_dim = dim(air$counts),
  airway_lib_range = range(lib),
  airway_zero_fraction = mean(air$counts == 0),
  airway_median_cor = median(cor(c_air)),
  prot_dim = dim(prot$x),
  prot_median_cor = median(cor(log2(prot$x), use = "pairwise.complete.obs"),
                           na.rm = TRUE),
  pbmc_dim = dim(pbmc),
  pbmc_zero_fraction = mean(pbmc == 0),
  cll_dim = dim(cll$expr),
  metab_dim = dim(metab$x)
)

save_metrics("e01", list(raw = raw_stats, pca = pca_tab,
                         methods = names(methods)))
