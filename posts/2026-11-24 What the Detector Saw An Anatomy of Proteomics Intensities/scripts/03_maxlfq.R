# E03/E04/E05 -- peptide ratios to protein numbers: a worked MaxLFQ example,
# a check against MaxQuant's own LFQ column, and two failure modes.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("R/io.R")
source("R/quant.R")
source("R/sim.R")
source("../../theme-blog.R")

dat <- readRDS("outputs/cptac.rds")
prot <- dat$prot
pep <- dat$pep
runs <- dat$runs
group <- dat$group
clean <- !prot$reverse & !prot$contaminant

# ---- fig09: a two-protein, three-run worked example --------------------
set.seed(11)
loading <- c(0, 1, 2)
pep_true <- c(A1 = 10, A2 = 9.4, A3 = 8.8, B1 = 9.2, B2 = 8.6, B3 = 8.1)
ex <- outer(pep_true, loading, "+") + matrix(rnorm(18, 0, 0.05), 6, 3)
rownames(ex) <- names(pep_true)
colnames(ex) <- paste0("run", 1:3)
ex["A1", "run2"] <- NA   # one peptide drops out of one run

ex_long <- as.data.frame(ex) |>
  tibble::rownames_to_column("peptide") |>
  pivot_longer(-peptide, names_to = "run", values_to = "log2")

p_ex <- ggplot(ex_long, aes(run, peptide, fill = log2)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = ifelse(is.na(log2), "missing", sprintf("%.1f", log2))),
            size = 2.8) +
  scale_fill_gradientn(colours = blog_gradient, name = expression(log[2]),
                       na.value = "grey90") +
  scale_y_discrete(limits = rev(names(pep_true))) +
  labs(x = NULL, y = NULL, title = "A: six peptides, three runs",
       subtitle = "run 2 loaded 2x; A1 missing in run 2")

fit_a <- maxlfq(2^ex[1:3, ])
fit_b <- maxlfq(2^ex[4:6, ])
naive <- log2(colSums(2^ex, na.rm = TRUE))
naive <- naive - mean(naive) + mean(loading)
prof <- data.frame(
  run = colnames(ex),
  loading = loading,
  maxlfq = (log2(fit_a$profile) + log2(fit_b$profile)) / 2,
  naive = naive
)
prof$maxlfq <- prof$maxlfq - mean(prof$maxlfq) + mean(loading)

p_prof <- ggplot(prof, aes(run)) +
  geom_point(aes(y = loading, colour = "true loading"), size = 2.6) +
  geom_point(aes(y = maxlfq, colour = "MaxLFQ"), size = 2.6) +
  geom_point(aes(y = naive, colour = "naive sum"), size = 2.6) +
  geom_line(aes(y = loading, group = 1, colour = "true loading"), linewidth = 0.6) +
  geom_line(aes(y = maxlfq, group = 1, colour = "MaxLFQ"), linewidth = 0.6) +
  geom_line(aes(y = naive, group = 1, colour = "naive sum"), linewidth = 0.6) +
  scale_colour_manual(values = c(MaxLFQ = blog_pal[["blue"]],
                                 `naive sum` = blog_pal[["orange"]],
                                 `true loading` = blog_pal[["vermillion"]]),
                      name = NULL) +
  labs(x = NULL, y = expression(log[2]*" scale"),
       title = "B: what the algorithm recovers",
       subtitle = "peptide ratios survive, absolute levels do not")

save_fig(p_ex + p_prof, "fig09_maxlfq_example", width = 8.6, height = 4.2)

# ---- real data: peptide matrix -----------------------------------------
source("R/chem.R")
pep_acc <- parse_accession(pep$leading)
pep_ok <- !is.na(pep_acc) & pep_acc %in% prot$acc
pep_mat <- pep$intensity[pep_ok, ]
rownames(pep_mat) <- pep$sequence[pep_ok]
pep_norm <- median_normalize(pep_mat)
pep_prot <- match(pep_acc[pep_ok], prot$acc)

ids <- prot$id

# self-implemented protein profiles
self_lfq <- matrix(NA_real_, nrow(prot$intensity), ncol(pep_mat))
self_sum <- self_lfq
self_top3 <- self_lfq
for (i in seq_len(nrow(prot$intensity))) {
  idx <- which(pep_prot == i)
  if (!length(idx)) next
  sub_mat <- pep_norm[idx, , drop = FALSE]
  self_lfq[i, ] <- maxlfq(sub_mat)$profile
  self_sum[i, ] <- sum_intensity(sub_mat)
  self_top3[i, ] <- top3_intensity(sub_mat)
}

n_pep_prot <- tabulate(pep_prot, nbins = nrow(prot))
two_pep <- n_pep_prot >= 2
mq_lfq <- log2(prot$lfq[clean, ])
our_lfq <- log2(self_lfq[clean, ])
mq_c <- mq_lfq - rowMeans(mq_lfq, na.rm = TRUE)
our_c <- our_lfq - rowMeans(our_lfq, na.rm = TRUE)
ok <- is.finite(mq_c) & is.finite(our_c) & two_pep[clean]

scat <- data.frame(mq = mq_c[ok], ours = our_c[ok])

p_lfq <- ggplot(scat, aes(mq, ours)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey60", linetype = "dashed") +
  geom_point(size = 0.35, alpha = 0.3, colour = blog_pal[["blue"]]) +
  labs(x = expression("MaxQuant LFQ, within-protein centered ("*log[2]*")"),
       y = expression("this post's MaxLFQ, centered ("*log[2]*")"),
       title = "Re-implementing MaxLFQ tracks the official column",
       subtitle = sprintf("r = %.3f on %s protein-run pairs with 2+ peptides",
                          cor(scat$mq, scat$ours), format(nrow(scat), big.mark = ",")))

save_fig(p_lfq, "fig10_maxlfq_real", width = 7.6, height = 4.6)

# ---- fold-change benchmark on the spike-in truth ------------------------
lfc_of <- function(m) {
  a <- rowMeans(m[, group == 1, drop = FALSE], na.rm = TRUE)
  b <- rowMeans(m[, group == 2, drop = FALSE], na.rm = TRUE)
  b - a
}
methods <- list(
  Intensity = log2(prot$intensity[clean, ]),
  `LFQ (MaxQuant)` = log2(prot$lfq[clean, ]),
  `LFQ (this post)` = log2(self_lfq[clean, ]),
  `sum of peptides` = log2(self_sum[clean, ]),
  Top3 = log2(self_top3[clean, ])
)
bench <- do.call(rbind, lapply(names(methods), function(nm) {
  m <- methods[[nm]]
  eligible <- rowSums(is.finite(m[, group == 1, drop = FALSE])) >= 2 &
    rowSums(is.finite(m[, group == 2, drop = FALSE])) >= 2
  data.frame(protein = ids[clean], ups = prot$ups[clean], method = nm,
             lfc = lfc_of(m), eligible = eligible)
}))

lfc_tab <- bench |>
  filter(eligible, is.finite(lfc)) |>
  mutate(method = factor(method, levels = names(methods)))

p_lfc <- ggplot(lfc_tab, aes(method, lfc, fill = ups)) +
  geom_hline(yintercept = 0, colour = "grey70", linetype = "dotted") +
  geom_hline(yintercept = log2(0.74 / 0.25), colour = blog_pal[["vermillion"]],
             linetype = "dashed") +
  geom_violin(alpha = 0.5, linewidth = 0.3, scale = "width") +
  geom_boxplot(width = 0.12, outlier.size = 0.3, linewidth = 0.3,
               fill = "white") +
  scale_fill_manual(values = c(`FALSE` = blog_pal[["grey"]],
                               `TRUE` = blog_pal[["orange"]]),
                    labels = c("yeast background", "UPS1 spike-in"),
                    name = NULL) +
  labs(x = NULL, y = expression(log[2]*" fold change (B / A)"),
       title = "What each column does to a known 2.96x change",
       subtitle = "dashed line = expected spike-in ratio, dotted = no change") +
  coord_cartesian(ylim = c(-3, 4))

save_fig(p_lfc, "fig11_lfc_methods", width = 9, height = 4.6)

# ---- simulation: missingness and aggregation ---------------------------
grid <- expand.grid(miss = c(0, 0.05, 0.1, 0.2, 0.35), seed = 1:10)
sim_res <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i) {
  s <- sim_peptides(n_prot = 200, n_run = 6, prop_de = 0.2,
                    miss_rate = grid$miss[i], seed = grid$seed[i])
  out <- lapply(c("sum", "meanlog", "maxlfq"), function(m) {
    est <- aggregate_proteins(s$mat, s$pep_protein, m)
    eligible <- rowSums(is.finite(est[, s$group == 1, drop = FALSE])) >= 2 &
      rowSums(is.finite(est[, s$group == 2, drop = FALSE])) >= 2
    lfc <- lfc_from_matrix(est, s$group)
    data.frame(miss = grid$miss[i], seed = grid$seed[i], method = m,
               bias = lfc - s$truth$lfc, eligible = eligible)
  })
  bind_rows(out)
}))

sim_sum <- sim_res |>
  group_by(miss, method) |>
  summarise(rmse = sqrt(mean(bias[eligible]^2, na.rm = TRUE)),
            retained = sum(eligible) / n(), .groups = "drop") |>
  mutate(method = factor(method, levels = c("sum", "meanlog", "maxlfq"),
                         labels = c("sum of peptides", "mean log", "MaxLFQ")))

p_sim <- ggplot(sim_sum, aes(miss, rmse, colour = method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_colour_manual(values = pal[c(2, 1, 3)], name = NULL) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "peptides missing (MNAR)", y = expression("RMSE of "*log[2]*" fold change"),
       title = "Aggregation rules differ mainly in what survives",
       subtitle = "RMSE among proteins still quantified in 2+ runs per group")

p_ret <- ggplot(sim_sum, aes(miss, retained, colour = method)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_colour_manual(values = pal[c(2, 1, 3)], name = NULL) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "peptides missing (MNAR)", y = "proteins retained",
       title = "The naive sum loses proteins first",
       subtitle = "200 simulated proteins, 10 seeds per point")

save_fig(p_sim + p_ret, "fig12_missing_sim", width = 9.2, height = 4.2)

# ---- shared peptides distort ratios ------------------------------------
# Protein A is unchanged; protein B doubles. A shared peptide carries
# A + B, so assigning it to A imports part of B's change.
razor_res <- do.call(rbind, lapply(seq(0, 1, by = 0.2), function(frac) {
  set.seed(21)
  a <- 100
  b1 <- 100
  b2 <- 200
  shared1 <- a + b1
  shared2 <- a + b2
  exclusive <- a
  est1 <- (1 - frac) * exclusive + frac * shared1
  est2 <- (1 - frac) * exclusive + frac * shared2
  data.frame(shared = frac,
             bias = log2(est2 / est1),
             naive_bias = log2(est2 / est1))
}))

p_share <- ggplot(razor_res, aes(shared, naive_bias)) +
  geom_hline(yintercept = 0, colour = "grey70", linetype = "dotted") +
  geom_line(colour = blog_pal[["vermillion"]], linewidth = 0.9) +
  geom_point(size = 2.2, colour = blog_pal[["vermillion"]]) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "fraction of protein A's peptides shared with B",
       y = expression("observed "*log[2]*" fold change of A"),
       title = "Shared peptides import the neighbour's change",
       subtitle = "A is unchanged, B doubles; truth is exactly zero")

save_fig(p_share, "fig13_shared_peptides", width = 8, height = 4.2)

# ---- metrics ------------------------------------------------------------
bench_sum <- bench |>
  group_by(method) |>
  summarise(
    n_eligible = sum(eligible),
    n_ups = sum(eligible & ups),
    ups_error = stats::median(abs(lfc[eligible & ups] - log2(0.74 / 0.25)), na.rm = TRUE),
    yeast_error = stats::median(abs(lfc[eligible & !ups]), na.rm = TRUE),
    .groups = "drop"
  )

mu_mq <- rowMeans(mq_lfq, na.rm = TRUE)
mu_our <- rowMeans(our_lfq, na.rm = TRUE)
keep_mu <- is.finite(mu_mq) & is.finite(mu_our)

save_metrics("e03", list(
  example_true_loading = loading,
  example_recovered = as.numeric(round(prof$maxlfq, 3)),
  real_lfq_cor = cor(scat$mq, scat$ours),
  real_lfq_rank = cor(mu_mq[keep_mu], mu_our[keep_mu], method = "spearman"),
  real_lfq_n = nrow(scat),
  real_lfq_slope = unname(coef(lm(ours ~ mq, data = scat))[2]),
  bench = as.data.frame(bench_sum),
  sim_rmse = as.data.frame(sim_sum) |> mutate(method = as.character(method)),
  shared_bias_full = razor_res$naive_bias[razor_res$shared == 1]
))
message("E03 done")
