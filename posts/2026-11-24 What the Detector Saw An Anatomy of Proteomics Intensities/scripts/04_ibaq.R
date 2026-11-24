# E05/E06 -- iBAQ: what the denominator is, why length bias shrinks,
# how it compares with Top3, spectral counting, NSAF and emPAI.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("R/io.R")
source("R/chem.R")
source("R/quant.R")
source("R/sim.R")
source("../../theme-blog.R")

dat <- readRDS("outputs/cptac.rds")
prot <- dat$prot
clean <- !prot$reverse & !prot$contaminant

# ---- fig14: theoretical observable peptides vs protein length ----------
d14 <- data.frame(length = prot$length[clean],
                  n_obs = prot$n_obs[clean],
                  ups = prot$ups[clean]) |>
  filter(is.finite(n_obs), length > 0)

p_obs <- ggplot(d14, aes(length, n_obs, colour = ups)) +
  geom_point(size = 0.7, alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = blog_pal[["vermillion"]], linewidth = 0.8) +
  scale_x_log10(breaks = c(100, 300, 1000, 3000),
                labels = c("100", "300", "1,000", "3,000")) +
  scale_y_log10(breaks = c(1, 3, 10, 30, 100, 300),
                labels = c("1", "3", "10", "30", "100", "300")) +
  scale_colour_manual(values = c(`FALSE` = blog_pal[["blue"]],
                                 `TRUE` = blog_pal[["orange"]]),
                      labels = c("yeast", "UPS1"), name = NULL) +
  labs(x = "protein length (aa)", y = "observable tryptic peptides",
       title = "The iBAQ denominator is a length proxy")

p_hist <- ggplot(d14, aes(n_obs, fill = ups)) +
  geom_histogram(bins = 45, position = "identity", alpha = 0.55,
                 colour = "white", linewidth = 0.15) +
  scale_fill_manual(values = c(`FALSE` = blog_pal[["blue"]],
                               `TRUE` = blog_pal[["orange"]]),
                    labels = c("yeast", "UPS1"), name = NULL) +
  labs(x = "observable peptides per protein", y = "proteins",
       title = "Denominator spans two orders of magnitude")

save_fig(p_obs + p_hist, "fig14_observable", width = 9.2, height = 4.2)

# ---- fig15: length bias before and after iBAQ --------------------------
# Equal true abundance by construction: only the number of observable
# peptides varies, so any slope in Intensity is a technical artefact.
set.seed(51)
n15 <- 500
d15 <- data.frame(
  length = round(10^rnorm(n15, log10(420), 0.35))
)
d15$n_obs <- pmax(1, round(d15$length / 26 * exp(rnorm(n15, 0, 0.12))))
pep_int <- exp(rnorm(n15, 0, 0.5))
d15$intensity <- d15$n_obs * pep_int
d15$ibaq <- d15$intensity / d15$n_obs

fit1 <- lm(log10(intensity) ~ log10(length), data = d15)
fit2 <- lm(log10(ibaq) ~ log10(length), data = d15)

p_int <- ggplot(d15, aes(length, intensity)) +
  geom_point(size = 0.6, alpha = 0.35, colour = blog_pal[["blue"]]) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = blog_pal[["vermillion"]], linewidth = 0.9) +
  scale_x_log10(breaks = c(100, 300, 1000, 3000),
                labels = c("100", "300", "1,000", "3,000")) +
  scale_y_log10(breaks = c(1, 10, 100, 1000),
                labels = c("1", "10", "100", "1,000")) +
  labs(x = "protein length (aa)", y = "summed peptide intensity",
       title = sprintf("Intensity: slope %.2f", coef(fit1)[2]))

p_ib <- ggplot(d15, aes(length, ibaq)) +
  geom_point(size = 0.6, alpha = 0.35, colour = blog_pal[["green"]]) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = blog_pal[["vermillion"]], linewidth = 0.9) +
  scale_x_log10(breaks = c(100, 300, 1000, 3000),
                labels = c("100", "300", "1,000", "3,000")) +
  scale_y_log10(breaks = c(0.03, 0.1, 0.3, 1, 3),
                labels = c("0.03", "0.1", "0.3", "1", "3")) +
  labs(x = "protein length (aa)", y = "summed intensity / observable peptides",
       title = sprintf("iBAQ: slope %.2f", coef(fit2)[2]))

save_fig(p_int + p_ib, "fig15_ibaq_length", width = 9.2, height = 4.2)

# ---- fig16: readouts against a known abundance ------------------------
set.seed(41)
n <- 700
abund <- 10^rnorm(n, 6, 1.3)
len <- round(10^rnorm(n, log10(420), 0.3))
n_obs <- pmax(1, round(len / 26))
n_det <- pmin(n_obs, rpois(n, pmin(25, 0.6 + abund / 1e5)))
sc <- rpois(n, pmax(0.01, abund / 4e4))
pep_per_prot <- pmax(1, round(pmin(n_obs, 1 + abund / 5e4)))
raw <- numeric(n)
top3 <- numeric(n)
for (i in seq_len(n)) {
  k <- pep_per_prot[i]
  pep <- rlnorm(k, log(abund[i]) - 3, 0.5)
  raw[i] <- sum(pep)
  top3[i] <- mean(sort(pep, decreasing = TRUE)[1:min(3, k)])
}
sim <- data.frame(
  true = abund,
  raw = raw,
  ibaq = raw / n_obs,
  top3 = top3,
  sc = sc,
  nsaf = nsaf(sc, len),
  empai = empai(n_det, n_obs)
)
sim[sim <= 0] <- NA
sim <- sim |>
  pivot_longer(-true, names_to = "readout", values_to = "observed") |>
  mutate(readout = factor(readout, levels = c("raw", "ibaq", "top3", "sc", "nsaf", "empai"),
                          labels = c("raw sum", "iBAQ", "Top3", "spectral count",
                                     "NSAF", "emPAI")))

p_read <- ggplot(sim, aes(true, observed)) +
  geom_point(size = 0.35, alpha = 0.25, colour = blog_pal[["blue"]]) +
  facet_wrap(~ readout, scales = "free_y", nrow = 2) +
  scale_x_log10(labels = scales::label_log()) +
  scale_y_log10(labels = scales::label_log()) +
  labs(x = "true protein abundance (arbitrary)", y = "reported value",
       title = "Six readouts, one truth")

save_fig(p_read, "fig16_readouts", width = 9.4, height = 5.4)

# ---- fig17: a per-protein flyability error is invisible to iBAQ -------
fly <- expand.grid(sd_fly = c(0.3, 0.7, 1.1), seed = 1:8)
fly_res <- do.call(rbind, lapply(seq_len(nrow(fly)), function(i) {
  set.seed(100 + fly$seed[i])
  n <- 400
  truth <- 10^rnorm(n, 6, 1)
  n_obs <- pmax(1, round(runif(n, 5, 40)))
  flyability <- exp(rnorm(n, 0, fly$sd_fly[i]))
  obs <- truth * flyability
  data.frame(sd_fly = fly$sd_fly[i], seed = fly$seed[i],
             bias = log10(obs / truth))
}))

p_fly <- ggplot(fly_res, aes(factor(sd_fly), bias)) +
  geom_hline(yintercept = 0, colour = "grey70", linetype = "dotted") +
  geom_boxplot(width = 0.45, outlier.size = 0.3, linewidth = 0.3,
               fill = "transparent", colour = blog_pal[["grey"]]) +
  labs(x = expression("heterogeneity of per-protein flyability ("*sigma*")"),
       y = expression(log[10]*" iBAQ error"),
       title = "iBAQ trusts every protein to ionise equally",
       subtitle = "the error is systematic, not random")

save_fig(p_fly, "fig17_flyability", width = 8, height = 4.2)

# ---- fig18: from iBAQ to copies per cell ------------------------------
mean_int <- rowMeans(prot$intensity[clean, ], na.rm = TRUE)
d18_real <- data.frame(length = prot$length[clean],
                       intensity = mean_int,
                       n_obs = prot$n_obs[clean],
                       mw = prot$mw_kda[clean],
                       id = prot$id[clean]) |>
  filter(is.finite(intensity), is.finite(n_obs), intensity > 0)
d18 <- d18_real |>
  mutate(ibaq = intensity / n_obs, frac = ibaq / sum(ibaq)) |>
  arrange(desc(frac)) |>
  slice(1:10)
mass_grid <- c(100, 200, 400)
copies <- expand.grid(i = seq_len(nrow(d18)), mass = mass_grid)
copies$value <- copy_number(d18$frac[copies$i], total_mass_pg = copies$mass,
                            mw_kda = d18$mw[copies$i])
p_id <- parse_accession(d18$id)
copies$protein <- factor(p_id[copies$i], levels = unique(p_id))

p_cn <- ggplot(copies, aes(value, protein, colour = factor(mass))) +
  geom_line(aes(group = protein), colour = "grey75", linewidth = 0.5) +
  geom_point(size = 2.2) +
  scale_x_log10(breaks = c(1e7, 1e8, 1e9),
                labels = c("10M", "100M", "1G")) +
  scale_colour_manual(values = pal[c(1, 2, 3)], name = "protein mass per cell (pg)") +
  labs(x = "estimated copies per cell", y = NULL,
       title = "Copy numbers inherit every assumption",
       subtitle = "same iBAQ, three guesses at total protein mass")

save_fig(p_cn, "fig18_copy_number", width = 8.4, height = 4.6)

# ---- metrics ------------------------------------------------------------
save_metrics("e05", list(
  nobs_vs_length_slope = unname(coef(lm(log10(n_obs) ~ log10(length), data = d14))[2]),
  nobs_median = median(d14$n_obs),
  nobs_range = range(d14$n_obs),
  intensity_length_slope = unname(coef(fit1)[2]),
  ibaq_length_slope = unname(coef(fit2)[2]),
  intensity_length_r2 = summary(fit1)$r.squared,
  ibaq_length_r2 = summary(fit2)$r.squared,
  readout_spearman = sim |>
    group_by(readout) |>
    summarise(rho = cor(true, observed, method = "spearman", use = "complete.obs"),
              .groups = "drop") |>
    as.data.frame(),
  readout_dynamic_range = sim |>
    group_by(readout) |>
    summarise(range_log10 = diff(range(log10(observed), na.rm = TRUE)), .groups = "drop") |>
    as.data.frame(),
  flyability_bias_sd = fly_res |>
    group_by(sd_fly) |>
    summarise(sd_bias = sd(bias), .groups = "drop") |>
    as.data.frame(),
  copy_number_range = range(copies$value)
))
message("E05 done")
