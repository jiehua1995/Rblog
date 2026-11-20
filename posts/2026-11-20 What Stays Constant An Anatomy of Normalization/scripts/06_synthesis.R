# Summary: the same seven normalisers across four violations of their
# assumptions, plus a spike-in reference.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})
source("../../theme-blog.R")
source("R/norm.R")
source("R/sim.R")
source("R/metrics.R")
source("R/io.R")

methods_of <- function(counts, len) {
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

add_spikes <- function(sim, frac = 0.02, seed = 1) {
  set.seed(seed)
  sp <- matrix(stats::rnbinom(50 * ncol(sim$counts),
                              mu = as.vector(outer(exp(stats::rnorm(50, 7, 0.5)),
                                                   sim$size)),
                              size = 20), 50, ncol(sim$counts))
  sp <- sp / colSums(sp) * (frac * mean(colSums(sim$counts)) * sim$size)
  rownames(sp) <- paste0("SP", seq_len(nrow(sp)))
  colnames(sp) <- colnames(sim$counts)
  list(counts = rbind(sim$counts, sp), len = c(sim$length, rep(1000, 50)),
       spike = sp)
}

scenarios <- list(
  depth = sim_counts(prop_de = 0.10, depth_sd = 0.80, seed = 11),
  composition = sim_counts(prop_de = 0.40, de_dir = "up", depth_sd = 0.30,
                           seed = 12),
  length = sim_counts(prop_de = 0.10, length_bias = 0.60, depth_sd = 0.30,
                      seed = 13),
  global = sim_counts(prop_de = 0, shift_all = 0.50, depth_sd = 0.30,
                      seed = 14)
)

rows <- list()
for (s in names(scenarios)) {
  sim <- scenarios[[s]]
  sp <- add_spikes(sim, seed = 100 + which(names(scenarios) == s))
  ms <- methods_of(sp$counts, sp$len)
  ms$SpikeIn <- log2(sweep(sp$counts, 2, colSums(sp$spike), "/") * 1e6 + 1)
  for (m in names(ms)) {
    d <- limma_de(ms[[m]], sim$group)
    idx <- match(sim$truth$id, d$id)
    rows[[length(rows) + 1]] <- data.frame(
      scenario = s, method = m,
      rmse = sqrt(mean((d$logFC[idx] - sim$truth$lfc)^2)))
  }
}
res <- bind_rows(rows)

scaled <- res |>
  group_by(scenario) |>
  mutate(scaled = (rmse - min(rmse)) / (max(rmse) - min(rmse))) |>
  ungroup()

fig <- ggplot(scaled, aes(scenario, reorder(method, rmse), fill = scaled)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.2f", rmse)), size = 2.8) +
  scale_fill_gradient(low = "white", high = blog_pal[["vermillion"]]) +
  labs(x = NULL, y = NULL, fill = "RMSE\n(0 = best)") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

save_fig(fig, "fig14_scenarios", width = 8.2, height = 4.4)
save_metrics("e11", list(rows = res))
