# E01 -- the physics of the number: isotope envelope, chromatographic peak,
# XIC integration, AGC-limited accumulation and ESI competition.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})
source("R/io.R")
source("R/chem.R")
source("R/peak.R")
source("../../theme-blog.R")
# ---- fig01: one peptide in RT x m/z space -------------------------------
seq1 <- "LVNELTEFAK"
charge <- 2
m0 <- (peptide_mass(seq1) + 1.007276 * charge) / charge
rt_grid <- seq(18, 22, length.out = 240)
surf <- peak_surface(1e6, seq1, charge = charge, apex = 20, sigma = 0.25,
                     mz_window = 2.2, n_mz = 700, rt = rt_grid)
xic <- xic_from_surface(surf, m0 - 0.05, m0 + 2.6)

p_map <- ggplot(surf, aes(rt, mz, fill = log10(intensity + 1))) +
  geom_raster() +
  scale_fill_gradientn(colours = c("#56B4E900", blog_gradient),
                       name = expression(log[10]*" intensity")) +
  labs(x = "retention time (min)", y = "m/z",
       title = "One peptide, several isotope peaks")

p_xic <- ggplot(xic, aes(rt, intensity)) +
  geom_area(fill = blog_pal[["sky"]], alpha = 0.45) +
  geom_line(colour = blog_pal[["blue"]], linewidth = 0.7) +
  labs(x = "retention time (min)", y = "extracted ion current",
       title = "The same data as a chromatogram")

save_fig(p_map + p_xic, "fig01_peak", width = 9.2, height = 4.2)

# ---- fig02: isotope envelopes depend on composition --------------------
seqs <- c(short = "DLGEEHFK",
          medium = "YICDNQDTISSK",
          long = "VFDEFKPLVEEPQNLIK")
env <- do.call(rbind, lapply(names(seqs), function(nm) {
  e <- isotope_envelope(seqs[[nm]], n_peaks = 8)
  e$peptide <- sprintf("%s (%d aa, M = %.0f Da)", nm, nchar(seqs[[nm]]),
                       peptide_mass(seqs[[nm]]))
  e
}))

p_env <- ggplot(env, aes(offset, p, colour = peptide)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.6) +
  scale_colour_manual(values = pal[c(1, 2, 3)], name = NULL) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "isotope peak (nominal mass units above monoisotopic)",
       y = "share of the envelope",
       title = "Why one peptide is several peaks")

save_fig(p_env, "fig02_isotopes", width = 8.6, height = 4.2)

# ---- fig03: how much of the peak do you integrate? --------------------
rt_seq <- rt_grid
sigma <- 0.25
apex <- 20
isotopes <- 1:5
widths <- c(0.5, 1, 1.5, 2, 3, 4)
surf5 <- peak_surface(1e6, seq1, charge = charge, apex = apex, sigma = sigma,
                      mz_window = 5.4, n_mz = 1600, rt = rt_grid)
full_area <- xic_area(xic_from_surface(surf5, m0 - 0.4, m0 + 5.4),
                      min(rt_seq), max(rt_seq))
area_tab <- expand.grid(isotope = isotopes, width = widths)
area_tab$fraction <- mapply(function(k, w) {
  hi <- m0 + (k - 1) * 1.003355 + 0.4
  xic <- xic_from_surface(surf5, m0 - 0.2, hi)
  xic_area(xic, apex - w * sigma, apex + w * sigma) / full_area
}, area_tab$isotope, area_tab$width)

p_int <- ggplot(area_tab, aes(width, fraction, colour = factor(isotope))) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c(pal, blog_pal[["grey"]]), name = "isotopes summed") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                     limits = c(0, 1.02)) +
  labs(x = expression("integration window around apex ("*sigma*")"),
       y = "share of true peak area",
       title = "Peak area is an integration choice",
       subtitle = "narrow window, one isotope: most of the signal is discarded")

save_fig(p_int, "fig03_integration", width = 8.2, height = 4.2)

# ---- fig04: AGC target sets the absolute scale ------------------------
rates <- data.frame(ion_rate = c(2e4, 1e5, 5e5),
                    label = factor(c("low abundance", "medium", "high"),
                                   levels = c("low abundance", "medium", "high")))
agc <- merge(rates, data.frame(agc_target = 10^seq(3, 7, length.out = 80)))
agc$signal <- agc_signal(agc$ion_rate, agc$agc_target, max_injection = 0.1)

p_agc <- ggplot(agc, aes(agc_target, signal, colour = label, group = label)) +
  geom_line(linewidth = 0.85) +
  scale_x_log10(breaks = 10^(3:7), labels = scales::label_log()) +
  scale_y_log10(breaks = 10^(3:7), labels = scales::label_log()) +
  scale_colour_manual(values = pal[c(1, 2, 3)], name = NULL) +
  labs(x = "AGC target (ions per scan)", y = "measured intensity (arbitrary units)",
       title = "Same amount, different number",
       subtitle = "below the injection-time limit the scale is set by the instrument")

save_fig(p_agc, "fig04_agc", width = 8, height = 4.2)

# ---- fig05: saturation compresses ratios ------------------------------
ratios <- 10^seq(-1, 2, length.out = 80)
supp <- expand.grid(true_ratio = ratios, capacity = c(3, 10, 30, Inf))
supp$observed <- mapply(function(r, cp) {
  load <- 1 + r
  saturating_signal(r, cp, load = load) / saturating_signal(1, cp, load = load)
}, supp$true_ratio, supp$capacity)

supp$cap_label <- factor(supp$capacity, levels = c(3, 10, 30, Inf),
                         labels = c("3", "10", "30", "unlimited"))

p_sup <- ggplot(supp, aes(true_ratio, observed, colour = cap_label)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey55") +
  geom_line(linewidth = 0.85) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c(pal[c(2, 1, 6)], blog_pal[["grey"]]),
                      name = "ion capacity") +
  labs(x = "true ratio (peptide B / peptide A)", y = "observed ratio",
       title = "Saturation compresses ratios",
       subtitle = "dashed line = perfect transmission")

save_fig(p_sup, "fig05_suppression", width = 8, height = 4.2)

# ---- metrics ----------------------------------------------------------
save_metrics("e01", list(
  peptide = seq1,
  peptide_mass = peptide_mass(seq1),
  precursor_mz = m0,
  area_2sigma_5isotopes = area_tab$fraction[area_tab$width == 2 & area_tab$isotope == 5],
  area_1sigma_1isotope = area_tab$fraction[area_tab$width == 1 & area_tab$isotope == 1],
  area_2sigma_1isotope = area_tab$fraction[area_tab$width == 2 & area_tab$isotope == 1],
  auc_agc_low_max = 2e4 * 0.1,
  saturated_ratio_10_cap10 = supp$observed[supp$capacity == 10][
    which.min(abs(supp$true_ratio[supp$capacity == 10] - 10))],
  isotopologue_share_mono = isotope_envelope(seqs[["medium"]], 8)$p[1]
))
message("E01 done")
