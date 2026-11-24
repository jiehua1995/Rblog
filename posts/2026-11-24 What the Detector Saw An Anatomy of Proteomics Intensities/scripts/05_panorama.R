# E07 -- the rest of the family: TMT reporter ions, DIA, and spectral counting.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("R/io.R")
source("R/sim.R")
source("../../theme-blog.R")

dat <- readRDS("outputs/cptac.rds")
prot <- dat$prot
group <- dat$group
clean <- !prot$reverse & !prot$contaminant

# ---- fig19: isobaric reporter ions compress ratios ---------------------
ratios <- 10^seq(-0.8, 0.8, length.out = 80)
bg <- c(0, 0.25, 0.5, 1, 2)
tmt <- expand.grid(true_ratio = ratios, background = bg)
tmt$observed <- (tmt$true_ratio + tmt$background) / (1 + tmt$background)
tmt$background <- factor(tmt$background, levels = bg,
                         labels = c("0 (MS3)", "0.25", "0.5", "1", "2"))

p_tmt <- ggplot(tmt, aes(true_ratio, observed, colour = background)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.85) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c(blog_pal[["grey"]], pal[c(3, 2, 1, 6)]),
                      name = "co-isolated background") +
  labs(x = "true ratio between two channels", y = "reporter ion ratio",
       title = "Isobaric labelling pulls ratios toward one",
       subtitle = "dashed line = perfect accuracy")

comp <- data.frame(background = bg,
                   ratio = (4 + bg) / (1 + bg) / 4)
p_comp <- ggplot(comp, aes(factor(background), ratio)) +
  geom_col(fill = blog_pal[["sky"]], width = 0.55) +
  geom_text(aes(label = sprintf("%.0f%%", 100 * ratio)), vjust = -0.4, size = 3.2) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1.1)) +
  labs(x = "co-isolated background", y = "fraction of a 4x ratio that survives",
       title = "How much of the truth is left")

save_fig(p_tmt / p_comp, "fig19_tmt", width = 7.8, height = 6.2)

# ---- fig20: DDA samples, DIA watches ----------------------------------
ab <- 10^seq(-1, 6, length.out = 120)
dda <- data.frame(abundance = ab,
                  p = 0.85 * plogis((log10(ab) - 2.6) / 0.75),
                  mode = "DDA (stochastic)")
dia <- data.frame(abundance = ab,
                  p = plogis((log10(ab) - 1.9) / 0.25),
                  mode = "DIA (deterministic)")
p_det <- bind_rows(dda, dia)

p_mode <- ggplot(p_det, aes(abundance, p, colour = mode)) +
  geom_line(linewidth = 0.9) +
  scale_x_log10(labels = scales::label_log()) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1)) +
  scale_colour_manual(values = pal[c(2, 1)], name = NULL) +
  labs(x = "peptide abundance", y = "probability of being quantified",
       title = "Two ways to lose a peptide",
       subtitle = "DDA samples; DIA watches every window")

# real DDA missingness for comparison
pep_long <- expand.grid(i = seq_len(nrow(dat$pep)), run = seq_along(dat$runs))
pep_long$intensity <- as.vector(dat$pep$intensity)
pep_long$miss <- is.na(pep_long$intensity)
pep_mean <- rowMeans(log2(dat$pep$intensity), na.rm = TRUE)
real_miss <- data.frame(mean_log2 = pep_mean,
                        miss = rowMeans(is.na(dat$pep$intensity))) |>
  filter(is.finite(mean_log2))

p_real <- ggplot(real_miss, aes(mean_log2, miss)) +
  geom_point(size = 0.4, alpha = 0.2, colour = blog_pal[["blue"]]) +
  geom_smooth(method = "loess", span = 0.8, se = FALSE,
              colour = blog_pal[["vermillion"]], linewidth = 0.9) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = expression("peptide "*log[2]*" intensity"), y = "fraction missing",
       title = "The same curve in the real data",
       subtitle = "CPTAC spike-in, six runs")

save_fig(p_mode + p_real, "fig20_dda_dia", width = 9.2, height = 4.2)

# ---- fig21: spectral counting ----------------------------------------
msms <- prot$msms_run[clean, ]
msms[msms <= 0] <- NA
mean_msms <- rowMeans(msms, na.rm = TRUE)
mean_lfq <- rowMeans(log2(prot$lfq[clean, ]), na.rm = TRUE)
ok <- is.finite(mean_msms) & is.finite(mean_lfq)

sc_df <- data.frame(count = mean_msms[ok], lfq = mean_lfq[ok],
                    ups = prot$ups[clean][ok])

p_sc <- ggplot(sc_df, aes(count, lfq, colour = ups)) +
  geom_point(size = 0.7, alpha = 0.5) +
  scale_x_log10(breaks = c(1, 3, 10, 30), labels = c("1", "3", "10", "30")) +
  scale_colour_manual(values = c(`FALSE` = blog_pal[["grey"]],
                                 `TRUE` = blog_pal[["orange"]]),
                      labels = c("yeast", "UPS1"), name = NULL) +
  labs(x = "MS/MS count per run (mean)", y = expression(log[2]*" LFQ intensity"),
       title = "Spectral counting is quantised",
       subtitle = "one spectrum, one vote, no intensity")

lfc_sc <- log2(rowMeans(msms[, group == 2], na.rm = TRUE)) -
  log2(rowMeans(msms[, group == 1], na.rm = TRUE))
lfc_ref <- log2(rowMeans(prot$lfq[clean, group == 2], na.rm = TRUE)) -
  log2(rowMeans(prot$lfq[clean, group == 1], na.rm = TRUE))
cmp <- data.frame(
  method = factor(c("LFQ", "spectral count"), levels = c("LFQ", "spectral count")),
  ups = c(median(abs(lfc_ref[prot$ups[clean]] - log2(0.74 / 0.25)), na.rm = TRUE),
          median(abs(lfc_sc[prot$ups[clean]] - log2(0.74 / 0.25)), na.rm = TRUE))
)

p_err <- ggplot(cmp, aes(method, ups, fill = method)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = sprintf("%.2f", ups)), vjust = -0.4, size = 3.4) +
  scale_fill_manual(values = c(LFQ = pal[3], `spectral count` = pal[2]),
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = expression("median |error| in "*log[2]*" fold change"),
       title = "Cost of counting spectra",
       subtitle = "UPS1 proteins, expected change 1.57")

save_fig(p_sc / p_err, "fig21_spectral_count", width = 7.8, height = 6.2)

# ---- metrics ----------------------------------------------------------
save_metrics("e07", list(
  tmt_surviving_4x = (4 + bg) / (1 + bg) / 4,
  dda_max = 0.85,
  sc_vs_lfq_cor = cor(log10(sc_df$count), sc_df$lfq, use = "complete.obs"),
  sc_ups_error = cmp$ups[2],
  lfq_ups_error = cmp$ups[1]
))
message("E07 done")
