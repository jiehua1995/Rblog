source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n <- 300
x <- runif(n, 0, 1)
xt <- seq(-0.6, 1.6, length.out = 500)

funs <- list(
  quadratic = function(z) z^2,
  sine = function(z) sin(2 * pi * z)
)

out <- list()
plots <- list()
for (nm in names(funs)) {
  f <- funs[[nm]]
  y <- f(x) + rnorm(n, 0, 0.15)
  rf <- ranger(x = data.frame(x = x), y = y, num.trees = 500, mtry = 1,
               min.node.size = 1, seed = SEED)
  pr_rf <- predict(rf, data = data.frame(x = xt))$predictions
  fit_lm <- lm(y ~ x)
  pr_lm <- predict(fit_lm, newdata = data.frame(x = xt))

  d <- data.frame(x = xt, truth = f(xt), RF = pr_rf, Linear = pr_lm)
  long <- d |>
    pivot_longer(c(RF, Linear), names_to = "model", values_to = "pred")

  out[[nm]] <- data.frame(
    function_name = nm,
    rf_inside = sqrt(mean((pr_rf[xt >= 0 & xt <= 1] - f(xt[xt >= 0 & xt <= 1]))^2)),
    rf_outside = sqrt(mean((pr_rf[xt < 0 | xt > 1] - f(xt[xt < 0 | xt > 1]))^2)),
    lm_inside = sqrt(mean((pr_lm[xt >= 0 & xt <= 1] - f(xt[xt >= 0 & xt <= 1]))^2)),
    lm_outside = sqrt(mean((pr_lm[xt < 0 | xt > 1] - f(xt[xt < 0 | xt > 1]))^2))
  )

  plots[[nm]] <- ggplot(long, aes(x, pred, colour = model)) +
    geom_vline(xintercept = c(0, 1), linetype = 2, colour = blog_pal[["grey"]]) +
    geom_line(aes(group = model), linewidth = 0.8) +
    geom_line(data = d, aes(x, truth), colour = "black", linewidth = 0.7,
              linetype = 3, inherit.aes = FALSE) +
    scale_colour_manual(values = c(RF = blog_pal[["vermillion"]],
                                   Linear = blog_pal[["blue"]])) +
    coord_cartesian(ylim = c(-1.2, 2.8)) +
    labs(x = "x", y = "Prediction", colour = NULL,
         title = ifelse(nm == "quadratic", "y = x²", "y = sin(2πx)")) +
    theme(legend.position = "bottom", plot.title = element_text(size = 10))
}
tab <- bind_rows(out)

p <- plots[[1]] + plots[[2]] + plot_layout(nrow = 1, guides = "collect") &
  theme(legend.position = "bottom")
save_fig(p, "fig10_extrapolation", width = 9.4, height = 4.4)

write_metrics(list(
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  rf_outside_worst = max(tab$rf_outside),
  lm_outside_worst = max(tab$lm_outside)
), "e10_extrapolation")
