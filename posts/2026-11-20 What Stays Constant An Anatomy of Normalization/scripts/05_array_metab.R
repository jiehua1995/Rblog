# E07: microarray and metabolomics. Distribution matching on CLL, and
# scaling/CLR on urine metabolomics.

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

pca_samples <- function(m, label, extra = NULL) {
  pc <- stats::prcomp(scale(t(m)))
  d <- data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2], method = label)
  if (!is.null(extra)) d <- cbind(d, extra)
  d
}

# ------------------------------------------------------------------- CLL
cll <- readRDS("data/cll.rds")
x <- cll$expr
v <- apply(x, 1, stats::var)
top <- order(v, decreasing = TRUE)[seq_len(3000)]
xs <- x[top, , drop = FALSE]
raw <- log2(xs)

cll_methods <- list(
  raw = raw,
  quantile = quantile_norm(raw),
  vsn = vsn_norm(xs),
  loess = loess_norm(raw)
)

long_cll <- bind_rows(lapply(names(cll_methods), function(k) {
  d <- as.data.frame(cll_methods[[k]])
  d$probe <- rownames(d)
  d <- pivot_longer(d, -probe, names_to = "sample", values_to = "value")
  d$method <- k
  d
}))

p_dist <- ggplot(long_cll, aes(sample, value)) +
  geom_boxplot(fill = blog_pal[["purple"]], colour = "grey20",
               linewidth = 0.22, outlier.size = 0.1) +
  facet_wrap(~ method, nrow = 2, scales = "free_y") +
  labs(x = NULL, y = "log2 intensity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5))

pca_cll <- bind_rows(lapply(names(cll_methods), function(k) {
  pca_samples(cll_methods[[k]], k)
}))
p_pca_cll <- ggplot(pca_cll, aes(PC1, PC2)) +
  geom_point(size = 0.8, colour = blog_pal[["blue"]]) +
  facet_wrap(~ method, nrow = 2, scales = "free")

fig12 <- p_dist / p_pca_cll + plot_layout(heights = c(1.2, 1))
save_fig(fig12, "fig12_microarray", width = 9.6, height = 6.8)

cll_cor <- sapply(cll_methods, function(m) median(stats::cor(m)))

# -------------------------------------------------------------- metabolomics
metab <- readRDS("data/metab.rds")
mx <- metab$x
meta <- metab$meta

met_methods <- list(
  raw = mx,
  median = sweep(mx, 2, apply(mx, 2, stats::median), "-"),
  quantile = quantile_norm(mx),
  pareto = pareto(mx),
  clr = clr(10^mx)
)

pca_met <- bind_rows(lapply(names(met_methods), function(k) {
  pca_samples(met_methods[[k]], k, data.frame(gender = meta$gender))
}))
p_pca_met <- ggplot(pca_met, aes(PC1, PC2, colour = gender)) +
  geom_point(size = 0.9, alpha = 0.7) +
  facet_wrap(~ method, nrow = 2, scales = "free") +
  scale_colour_manual(values = unname(blog_pal[c("blue", "vermillion")])) +
  labs(colour = "gender")

met_tab <- bind_rows(lapply(names(met_methods), function(k) {
  m <- met_methods[[k]]
  pc <- stats::prcomp(scale(t(m)))$x[, 1:2]
  gender_r2 <- summary(stats::lm(pc[, 1] ~ meta$gender))$r.squared +
    summary(stats::lm(pc[, 2] ~ meta$gender))$r.squared
  age_cor <- median(abs(apply(m, 1, function(v) stats::cor(v, meta$age))))
  data.frame(method = k, gender_r2 = gender_r2, age_cor = age_cor)
}))

p_met_tab <- ggplot(pivot_longer(met_tab, c(gender_r2, age_cor),
                                 names_to = "metric", values_to = "value"),
                    aes(reorder(method, value), value, fill = metric)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = unname(blog_pal[c("orange", "sky")])) +
  labs(x = NULL, y = NULL, fill = NULL)

fig13 <- p_pca_met / p_met_tab + plot_layout(heights = c(2, 1))
save_fig(fig13, "fig13_metabolomics", width = 9.6, height = 6.4)

save_metrics("e07", list(
  cll_cor = cll_cor,
  metab = met_tab,
  cll_dim = dim(xs),
  metab_dim = dim(mx)
))
