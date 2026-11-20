# E05 and E09: missing values in label-free proteomics, and the cost of the
# two-level normalisation (peptide -> protein -> sample).

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

impute_min <- function(x, width = 1.0) {
  for (j in seq_len(ncol(x))) {
    m <- stats::quantile(x[, j], 0.05, na.rm = TRUE)
    x[is.na(x[, j]), j] <- m - width
  }
  x
}

# ------------------------------------------------------- E05: missing values
sim <- sim_proteins(n = 1500, m = 12, prop_de = 0.10, missing = 0.25,
                    mnar = TRUE, seed = 505)
obs <- log2(sim$observed)
complete <- rowSums(is.na(obs)) == 0

# MNAR curve: missing fraction against abundance decile.
abund <- rowMeans(obs, na.rm = TRUE)
mis <- rowMeans(is.na(obs))
ok <- is.finite(abund)
bins <- cut(abund[ok], breaks = quantile(abund[ok], probs = seq(0, 1, 0.1)),
            include.lowest = TRUE)
curve <- data.frame(bin = bins, missing = mis[ok]) |>
  group_by(bin) |>
  summarise(missing = mean(missing), .groups = "drop") |>
  mutate(mid = seq_len(n()))

p_mnar <- ggplot(curve, aes(mid, missing)) +
  geom_col(fill = blog_pal[["vermillion"]], width = 0.7) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%")) +
  labs(x = "abundance decile (low to high)", y = "missing fraction")

# Four strategies.
strategies <- list(
  "median (complete)" = median_center_log(obs[complete, , drop = FALSE]),
  "median + min (all)" = median_center_log(impute_min(obs)),
  "quantile + min (all)" = quantile_norm(impute_min(obs)),
  "VSN (complete)" = vsn_norm(2^obs[complete, , drop = FALSE])
)

eval_strategy <- function(logx, ids) {
  d <- limma_de(logx, sim$group)
  idx <- match(ids, d$id)
  truth <- sim$truth[match(ids, sim$truth$id), ]
  ev <- evaluate_de(d$adj[idx], truth$de)
  c(rmse_de = sqrt(mean((d$logFC[idx][truth$de] - truth$lfc[truth$de])^2)),
    tpr = ev[["tpr"]], fdr = ev[["fdr"]], n_used = length(ids))
}

rows <- lapply(names(strategies), function(k) {
  ids <- if (grepl("complete", k)) sim$truth$id[complete] else sim$truth$id
  data.frame(method = k, t(eval_strategy(strategies[[k]], ids)))
})
e05 <- bind_rows(rows)

p_eval <- ggplot(e05, aes(reorder(method, rmse_de), rmse_de)) +
  geom_col(fill = blog_pal[["blue"]], width = 0.65) +
  geom_text(aes(label = paste0("n=", n_used)), hjust = -0.1, size = 2.8) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  coord_flip() +
  labs(x = NULL, y = "RMSE of log2 fold change (DE proteins)")

fig_e05 <- p_mnar | p_eval
save_fig(fig_e05, "fig08_missing", width = 9.6, height = 4.0)

# ------------------------------------------------ real LFQ: dilution series
lfq <- readRDS("data/lfq.rds")
x <- lfq$x
design <- lfq$design
ok_real <- rowSums(is.na(x)) == 0
x <- x[ok_real, , drop = FALSE]
logx <- log2(x)

real <- list(
  raw = logx,
  median = median_center_log(logx),
  quantile = quantile_norm(logx),
  vsn = vsn_norm(x)
)

long_real <- bind_rows(lapply(names(real), function(k) {
  d <- as.data.frame(real[[k]])
  d$peptide <- rownames(d)
  d <- pivot_longer(d, -peptide, names_to = "sample", values_to = "value")
  d$method <- k
  d
}))

fig_e09a <- ggplot(long_real, aes(sample, value)) +
  geom_boxplot(fill = blog_pal[["sky"]], colour = "grey20",
               linewidth = 0.22, outlier.size = 0.15) +
  facet_wrap(~ method, nrow = 2, scales = "free_y") +
  labs(x = NULL, y = "log2 intensity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5))

# Technical precision within the nine triplicates, on the log2 scale.
sd_within <- sapply(real, function(m) {
  median(apply(m, 1, function(v) {
    g <- split(v, design$group)
    mean(sapply(g, stats::sd))
  }))
})
batch_r2 <- sapply(real, function(m) {
  pc <- stats::prcomp(scale(t(m)))$x[, 1]
  summary(stats::lm(pc ~ design$batch))$r.squared
})

real_tab <- data.frame(method = names(real), sd_within = sd_within,
                       batch_r2 = batch_r2)

real_long <- pivot_longer(real_tab, c(sd_within, batch_r2),
                          names_to = "metric", values_to = "value")
p_real <- ggplot(real_long, aes(reorder(method, value), value, fill = metric)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = unname(blog_pal[c("blue", "vermillion")])) +
  labs(x = NULL, y = NULL, fill = NULL)

fig_e09 <- fig_e09a / p_real + plot_layout(heights = c(2, 1))
save_fig(fig_e09, "fig09_lfq_real", width = 10, height = 6.2)

# --------------------------------------------- E09: peptide -> protein order
pep <- sim_peptides(n_prot = 120, pep_per = 4, m = 12, prop_de = 0.10,
                    missing = 0.10, seed = 909)
pep_log <- log2(pep$peptides)

lfq <- maxlfq(pep$peptides, pep$groups)
rownames(lfq) <- paste0("P", seq_len(nrow(lfq)))
keep <- rownames(lfq)[is.finite(rowMeans(lfq))]
lfq <- lfq[keep, , drop = FALSE]

mean_pep <- sapply(seq_len(ncol(pep_log)), function(j) {
  tapply(pep_log[, j], pep$groups, mean, na.rm = TRUE)
})
sum_pep <- sapply(seq_len(ncol(pep$peptides)), function(j) {
  tapply(pep$peptides[, j], pep$groups, sum, na.rm = TRUE)
})
rownames(mean_pep) <- paste0("P", seq_len(nrow(mean_pep)))
rownames(sum_pep) <- paste0("P", seq_len(nrow(sum_pep)))

protein_matrices <- list(
  MaxLFQ = median_center_log(lfq),
  "mean log2 peptide" = median_center_log(mean_pep[keep, , drop = FALSE]),
  "sum intensity" = median_center_log(log2(sum_pep[keep, , drop = FALSE] + 1))
)
truth <- pep$truth[match(keep, pep$truth$id), ]

rows <- lapply(names(protein_matrices), function(k) {
  d <- limma_de(protein_matrices[[k]], pep$group)
  idx <- match(truth$id, d$id)
  data.frame(method = k,
             rmse_de = sqrt(mean((d$logFC[idx][truth$de] -
                                    truth$lfc[truth$de])^2)),
             bias_de = mean(d$logFC[idx][truth$de] -
                              truth$lfc[truth$de]))
})
e09 <- bind_rows(rows)

fig_e09b <- ggplot(e09, aes(reorder(method, rmse_de), rmse_de)) +
  geom_col(fill = blog_pal[["purple"]], width = 0.6) +
  coord_flip() +
  labs(x = NULL, y = "RMSE of protein log2 fold change")
save_fig(fig_e09b, "fig12_maxlfq", width = 7, height = 3.2)

# ------------------------------------------------------------------- metrics
save_metrics("e05", list(rows = e05, missing_fraction = mean(is.na(obs)),
                         complete_fraction = mean(complete)))
save_metrics("e09", list(rows = e09, real = real_tab,
                         complete_peptides = sum(ok_real),
                         total_peptides = length(ok_real)))
