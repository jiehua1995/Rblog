# E04 -- what the three plotting systems do with the same 120 points, and what
# one save costs. Run from the post directory: Rscript scripts/04_plots.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(jsonlite)
  library(microbenchmark)
})
source("R/io.R")
source("../../theme-blog.R")

small <- fread("data/small.csv")
small[, cat := factor(cat, levels = c("alpha", "beta", "gamma"))]
py <- fromJSON("outputs/python.json")

# ---- the bare default look of ggplot2 --------------------------------------
# theme_blog() is in force globally, so strip it back to the shipped default
draw_default <- function() {
  ggplot(small, aes(x, y, colour = cat)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    labs(x = "x", y = "y") +
    theme_grey()
}

fig07 <- draw_default()
save_fig(fig07, "fig07_ggplot_default", width = 6.2, height = 4.4)

# ---- one line of faceting ---------------------------------------------------
brk <- quantile(small$x, c(0, 1 / 3, 2 / 3, 1))
small[, bin := cut(x, brk, labels = c("low", "mid", "high"), include.lowest = TRUE)]

fig08 <- ggplot(small, aes(x, y)) +
  geom_point(size = 1) +
  facet_wrap(~ cat + bin, ncol = 3) +
  labs(x = "x", y = "y") +
  theme_grey(base_size = 9)
save_fig(fig08, "fig08_ggplot_facet", width = 7.0, height = 5.2)

# ---- what one save costs ----------------------------------------------------
t_gg <- bench_min({
  p <- ggplot(small, aes(x, y, colour = cat)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    labs(x = "x", y = "y") +
    theme_grey()
  suppressMessages(ggsave("figures/_tmp_gg.png", p, width = 4.6, height = 3.4,
                          dpi = 200, bg = "transparent"))
})

save_metrics("e04_plots", list(
  n_points = nrow(small),
  ggplot_save_s = t_gg,
  mpl_save_s = py$render$mpl_s,
  plotnine_save_s = py$render$plotnine_s,
  ggplot_over_mpl = t_gg / py$render$mpl_s,
  plotnine_over_ggplot = py$render$plotnine_s / t_gg
))

cat("04_plots done\n")
