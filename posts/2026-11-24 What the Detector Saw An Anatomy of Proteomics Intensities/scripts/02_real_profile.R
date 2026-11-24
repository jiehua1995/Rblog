# E02 -- what the real matrices look like: dynamic range, run-to-run
# reproducibility, missingness that depends on abundance, and peptide spread.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("R/io.R")
source("../../theme-blog.R")

dat <- readRDS("outputs/cptac.rds")
prot <- dat$prot
pep <- dat$pep
runs <- dat$runs
group <- dat$group
clean <- !prot$reverse & !prot$contaminant

pep_log <- log2(pep$intensity)
prot_log <- log2(prot$intensity[clean, ])

long <- function(m, what) {
  as.data.frame(m) |>
    mutate(feature = row_number()) |>
    pivot_longer(-feature, names_to = "run", values_to = "value") |>
    mutate(level = what)
}
pep_long <- long(pep_log, "peptides")
prot_long <- long(prot_log, "protein groups")

p_dist <- ggplot(bind_rows(pep_long, prot_long), aes(value, colour = run, fill = run)) +
  geom_density(alpha = 0.08, linewidth = 0.6, na.rm = TRUE) +
  facet_wrap(~ level, scales = "free") +
  scale_colour_manual(values = setNames(c(pal, blog_pal[["orange"]])[1:6], runs),
                      breaks = runs, name = NULL) +
  scale_fill_manual(values = setNames(c(pal, blog_pal[["orange"]])[1:6], runs),
                    breaks = runs, name = NULL) +
  labs(x = expression(log[2]*" intensity"), y = "density",
       title = "Six runs, two levels of aggregation")

save_fig(p_dist, "fig06_distributions", width = 9, height = 4)

# ---- missingness vs abundance ------------------------------------------
miss_tab <- function(m, label) {
  mu <- rowMeans(m, na.rm = TRUE)
  miss <- rowMeans(is.na(m))
  data.frame(mean_log2 = mu, miss = miss, level = label) |>
    filter(is.finite(mean_log2))
}
miss_all <- bind_rows(miss_tab(pep_log, "peptides"),
                      miss_tab(prot_log, "protein groups"))

p_miss <- ggplot(miss_all, aes(mean_log2, miss)) +
  geom_point(alpha = 0.25, size = 0.5, colour = blog_pal[["blue"]]) +
  geom_smooth(method = "loess", span = 0.8, se = FALSE,
              colour = blog_pal[["vermillion"]], linewidth = 0.9) +
  facet_wrap(~ level, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = expression("mean "*log[2]*" intensity"), y = "fraction missing",
       title = "Missing values are not random",
       subtitle = "the weaker the signal, the more often it disappears")

save_fig(p_miss, "fig07_missing", width = 9, height = 4)

# ---- peptides of the same protein disagree ------------------------------
counts <- table(pep$leading)
top <- names(sort(counts, decreasing = TRUE))[1:8]
sub_df <- pep[pep$leading %in% top, ]
pep_long2 <- as.data.frame(log2(sub_df$intensity)) |>
  mutate(protein = sub_df$leading, peptide = sub_df$sequence) |>
  pivot_longer(-c(protein, peptide), names_to = "run", values_to = "value")
clean_acc <- function(x) {
  vapply(strsplit(x, "\\|"), function(p) {
    p <- gsub("^CON__", "", p[!p %in% c("sp", "tr", "CON__")])
    if (!length(p)) return(NA_character_)
    sub("ups$", "", sub("-[0-9]+$", "", p[1]))
  }, character(1))
}

pep_long2 <- pep_long2 |>
  group_by(protein) |>
  mutate(dev = value - mean(value, na.rm = TRUE)) |>
  ungroup() |>
  mutate(protein = clean_acc(protein))

p_spread <- ggplot(pep_long2, aes(reorder(protein, dev, median), dev)) +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
  geom_boxplot(outlier.size = 0.4, linewidth = 0.3, fill = "transparent",
               colour = blog_pal[["grey"]]) +
  geom_jitter(width = 0.15, size = 0.6, alpha = 0.5,
              colour = blog_pal[["blue"]]) +
  labs(x = NULL, y = expression(log[2]*" deviation from protein mean"),
       title = "Peptides from one protein are not interchangeable")

save_fig(p_spread, "fig08_peptide_spread", width = 9, height = 4)

# ---- metrics ------------------------------------------------------------
cv <- apply(prot$intensity[clean, group == 1], 1, function(v) {
  sd(v, na.rm = TRUE) / mean(v, na.rm = TRUE)
})
q <- apply(pep_log, 2, quantile, probs = c(0.05, 0.95), na.rm = TRUE)
save_metrics("e02", list(
  pep_dynamic_range_log2 = as.numeric(quantile(pep_log, 0.99, na.rm = TRUE) -
                                        quantile(pep_log, 0.01, na.rm = TRUE)),
  prot_dynamic_range_log2 = as.numeric(quantile(prot_log, 0.99, na.rm = TRUE) -
                                         quantile(prot_log, 0.01, na.rm = TRUE)),
  pep_missing = mean(is.na(pep_log)),
  prot_missing = mean(is.na(prot_log)),
  median_cv_a = median(cv, na.rm = TRUE),
  cor_run_median = median(cor(pep_log, use = "pairwise.complete.obs"), na.rm = TRUE),
  peptide_dev_sd = sd(pep_long2$dev, na.rm = TRUE),
  run_median_range = range(colMeans(pep_log, na.rm = TRUE))
))
message("E02 done")
