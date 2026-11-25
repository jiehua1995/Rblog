# E01 -- the limiting spectrum: Marchenko-Pastur, edge fluctuations,
# the BBP transition, and how the detection limit moves with n.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
})
source("../../theme-blog.R")
source("R/rmt.R")
source("R/io.R")

# ---- figure 1: pure noise, three aspect ratios ------------------------------
cases <- tibble(gamma_lab = c("gamma == 0.25", "gamma == 2", "gamma == 6"),
                gamma = c(0.25, 2, 6),
                p = c(500, 2000, 6000),
                n = c(2000, 1000, 1000))

spec_df <- pmap_dfr(cases, function(gamma_lab, gamma, p, n) {
  s <- spectrum(sim_noise(p, n, seed = 11), standardize = FALSE)
  tibble(gamma_lab = gamma_lab, value = s$values)
})

curve_df <- pmap_dfr(cases, function(gamma_lab, gamma, p, n) {
  b <- mp_bounds(gamma)
  x <- seq(b[1] * 1.02 + 1e-6, b[2] * 0.98, length.out = 400)
  tibble(gamma_lab = gamma_lab, x = x, y = mp_density(x, gamma))
})

edge_df <- pmap_dfr(cases, function(gamma_lab, gamma, p, n) {
  tibble(gamma_lab = gamma_lab, x = mp_bounds(gamma))
})

fig01 <- ggplot(spec_df, aes(value)) +
  geom_histogram(aes(y = after_stat(density)), bins = 45,
                 fill = blog_pal[["sky"]], colour = NA, alpha = 0.55) +
  geom_line(data = curve_df, aes(x, y), colour = blog_pal[["vermillion"]],
            linewidth = 0.7) +
  geom_vline(data = edge_df, aes(xintercept = x), linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.35) +
  facet_wrap(~ gamma_lab, scales = "free", labeller = label_parsed) +
  labs(x = "eigenvalue of the sample covariance / 样本协方差特征值",
       y = "density / 密度")
save_fig(fig01, "fig01_mp_noise", width = 9.2, height = 3.6)

save_metrics("e01", list(
  lam_lo_g0.25 = mp_bounds(0.25)[1], lam_hi_g0.25 = mp_bounds(0.25)[2],
  lam_lo_g2 = mp_bounds(2)[1], lam_hi_g2 = mp_bounds(2)[2],
  lam_lo_g6 = mp_bounds(6)[1], lam_hi_g6 = mp_bounds(6)[2],
  observed_max_g0.25 = max(spec_df$value[spec_df$gamma_lab == "gamma == 0.25"]),
  observed_max_g6 = max(spec_df$value[spec_df$gamma_lab == "gamma == 6"])
))

# ---- figure 2: the family of MP densities ----------------------------------
gammas <- c(0.1, 0.25, 0.5, 2, 4, 6, 9)
dens_df <- map_dfr(gammas, function(g) {
  x <- 10^seq(log10(1e-3), log10(mp_bounds(g)[2]), length.out = 500)
  tibble(gamma = factor(g), x = x, y = pmin(mp_density(x, g), 1.5))
})

fig02 <- ggplot(dens_df, aes(x, y, colour = gamma, group = gamma)) +
  geom_line(linewidth = 0.6) +
  scale_x_log10() +
  scale_colour_manual(values = pal) +
  labs(x = "eigenvalue (log scale) / 特征值（对数轴）",
       y = "density (clipped at 1.5) / 密度（截断于 1.5）", colour = "gamma")
save_fig(fig02, "fig02_mp_family", width = 8, height = 4.4)

# ---- figure 3: how wide is the edge? --------------------------------------
sizes <- c(50, 100, 200, 400, 800)
reps <- 100
edge_sd <- vapply(sizes, function(n) {
  set.seed(100 + n)
  v <- replicate(reps, max(spectrum(sim_noise(n, n), standardize = FALSE)$values))
  sd(v)
}, numeric(1))

edge_df2 <- tibble(n = sizes, sd = edge_sd)
fit <- lm(log(sd) ~ log(n), data = edge_df2)
slope <- unname(coef(fit)[2])

fig03 <- ggplot(edge_df2, aes(n, sd)) +
  geom_point(colour = blog_pal[["blue"]], size = 2.2) +
  geom_line(colour = blog_pal[["blue"]], linewidth = 0.5) +
  scale_x_log10(breaks = sizes) +
  scale_y_log10() +
  annotate("text", x = sizes[3], y = min(edge_sd) * 1.6, hjust = 0.5,
           label = sprintf("fitted slope = %.2f\n(theory: -2/3)", slope),
           colour = blog_pal[["vermillion"]], size = 3.4) +
  labs(x = "matrix size n (square matrix) / 矩阵维度",
       y = expression(sd(lambda[max]) * " / 最大特征值的标准差"))
save_fig(fig03, "fig03_edge_scaling", width = 7, height = 4.4)

save_metrics("e03", list(edge_scaling_slope = slope, edge_sizes = sizes,
                         edge_sd = edge_sd))

# ---- figures 4 and 5: the BBP transition ----------------------------------
gamma <- 4
p <- 1000
n <- 250
ells <- seq(0.2, 6, by = 0.2)
reps_bbp <- 15

bbp <- map_dfr(ells, function(ell) {
  set.seed(round(1000 * ell) + 7)
  obs <- replicate(reps_bbp, {
    sp <- sim_spike(p, n, ell)
    X <- sp$X
    G <- crossprod(X) / n
    e <- eigen(G, symmetric = TRUE)
    V <- X %*% e$vectors[, 1]
    V <- V / sqrt(sum(V^2))
    c(top = e$values[1], cos2 = sum(V * sp$u)^2)
  })
  tibble(ell = ell, top = mean(obs["top", ]), cos2 = mean(obs["cos2", ]))
})

thr <- sqrt(gamma)
bbp <- bbp %>%
  mutate(top_theory = spike_top(gamma, ell),
         cos2_theory = spike_cosine2(gamma, ell))

fig04 <- ggplot(bbp, aes(ell)) +
  geom_hline(yintercept = mp_bounds(gamma)[2], linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.35) +
  geom_vline(xintercept = thr, linetype = "dotted",
             colour = blog_pal[["grey"]], linewidth = 0.35) +
  geom_point(aes(y = top), colour = blog_pal[["blue"]], size = 1.4) +
  geom_line(aes(y = top_theory), colour = blog_pal[["vermillion"]],
            linewidth = 0.7) +
  annotate("text", x = thr, y = mp_bounds(gamma)[2] - 0.45,
           hjust = -0.08, label = "threshold", colour = blog_pal[["grey"]],
           size = 3.2) +
  labs(x = expression("signal strength " * ell * " / 信号强度"),
       y = expression(lambda[1] * " / 第一特征值"))
save_fig(fig04, "fig04_bbp_eigenvalue", width = 8, height = 4.4)

fig05 <- ggplot(bbp, aes(ell)) +
  geom_vline(xintercept = thr, linetype = "dotted",
             colour = blog_pal[["grey"]], linewidth = 0.35) +
  geom_point(aes(y = cos2), colour = blog_pal[["green"]], size = 1.4) +
  geom_line(aes(y = cos2_theory), colour = blog_pal[["vermillion"]],
            linewidth = 0.7) +
  labs(x = expression("signal strength " * ell * " / 信号强度"),
       y = expression(cos^2 * " between PC1 and the true signal / 与真信号的平方余弦"))
save_fig(fig05, "fig05_bbp_cosine", width = 8, height = 4.4)

save_metrics("e04", list(
  bbp_gamma = gamma, bbp_threshold = thr,
  bbp_top_at_ell_1 = bbp$top[which.min(abs(bbp$ell - 1))],
  bbp_top_theory_at_ell_1 = spike_top(gamma, 1),
  bbp_top_at_ell_5 = bbp$top[which.min(abs(bbp$ell - 5))],
  bbp_top_theory_at_ell_5 = spike_top(gamma, 5),
  bbp_cos2_at_ell_1 = bbp$cos2[which.min(abs(bbp$ell - 1))],
  bbp_cos2_at_ell_5 = bbp$cos2[which.min(abs(bbp$ell - 5))]
))

# ---- figure 6: the detection limit moves with gamma -----------------------
gammas6 <- c(0.5, 1, 2, 4, 8)
ells6 <- seq(0, 10, by = 0.05)
det_df <- map_dfr(gammas6, function(g) {
  tibble(gamma = factor(g), ell = ells6,
         ratio = spike_top(g, ells6) / mp_bounds(g)[2])
})
thr_df <- tibble(gamma = factor(gammas6), thr = sqrt(gammas6))

fig06 <- ggplot(det_df, aes(ell, ratio, colour = gamma)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.3) +
  geom_point(data = thr_df, aes(x = thr, y = 1), size = 2, shape = 4,
             colour = blog_pal[["grey"]]) +
  scale_colour_manual(values = pal[seq_along(gammas6)]) +
  labs(x = expression("signal strength " * ell * " / 信号强度"),
       y = expression(lambda[1] / lambda["+"]), colour = "gamma")
save_fig(fig06, "fig06_detection_limit", width = 8, height = 4.4)

save_metrics("e05", list(
  detection_gammas = gammas6,
  detection_thresholds = sqrt(gammas6)
))

message("01_theory done")
