source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "forest.R"))

set.seed(SEED)
test <- friedman_data(500, p = 10, noise = 1)

mtrys <- c(1, 2, 3, 5, 10)
B <- 60
R <- 80

out <- list()
for (m in mtrys) {
  set.seed(SEED + m)
  P_list <- vector("list", R)
  mean_pred <- matrix(NA_real_, R, nrow(test$X))
  for (r in seq_len(R)) {
    d <- friedman_data(300, p = 10, noise = 1)
    rf <- ranger(x = d$X, y = d$y, num.trees = B, mtry = m,
                 min.node.size = 1, replace = TRUE, sample.fraction = 1,
                 seed = SEED + 1000 * m + r)
    P <- predict(rf, data = test$X, predict.all = TRUE)$predictions
    P_list[[r]] <- P
    mean_pred[r, ] <- rowMeans(P)
  }
  sigma2 <- pooled_sigma2(P_list)
  rho <- pointwise_rho(P_list, sigma2 = sigma2, n_pairs = 20, n_points = 100)
  var_formula <- rho * sigma2 + (1 - rho) * sigma2 / B
  var_empirical <- mean(apply(mean_pred, 2, var))
  out[[as.character(m)]] <- list(
    mtry = m,
    rho = rho,
    sigma2 = sigma2,
    var_formula = var_formula,
    var_empirical = var_empirical,
    se_empirical = var_empirical * sqrt(2 / (R - 1)),
    ratio = var_formula / var_empirical,
    floor_share = rho * sigma2 / var_formula
  )
}

tab <- bind_rows(lapply(out, as.data.frame))
tab$mtry <- factor(tab$mtry, levels = mtrys)

p1 <- ggplot(tab, aes(mtry, rho)) +
  geom_line(aes(group = 1), colour = blog_pal[["blue"]], linewidth = 0.8) +
  geom_point(colour = blog_pal[["blue"]], size = 2) +
  labs(x = "mtry", y = expression(rho), title = "Pointwise tree correlation") +
  theme(plot.title = element_text(size = 9))

p2 <- ggplot(tab, aes(mtry, sigma2)) +
  geom_line(aes(group = 1), colour = blog_pal[["vermillion"]], linewidth = 0.8) +
  geom_point(colour = blog_pal[["vermillion"]], size = 2) +
  labs(x = "mtry", y = expression(sigma^2), title = "Single-tree variance") +
  theme(plot.title = element_text(size = 9))

long <- tab |>
  select(mtry, `Formula` = var_formula, `Repeated forests` = var_empirical,
         se = se_empirical) |>
  pivot_longer(c(`Formula`, `Repeated forests`), names_to = "method",
               values_to = "variance")

p3 <- ggplot(long, aes(mtry, variance, colour = method)) +
  geom_line(aes(group = method), linewidth = 0.8) +
  geom_point(size = 2) +
  geom_errorbar(data = subset(long, method == "Repeated forests"),
                aes(ymin = variance - se, ymax = variance + se),
                width = 0.12, linewidth = 0.4) +
  scale_colour_manual(values = c(Formula = blog_pal[["blue"]],
                                 `Repeated forests` = blog_pal[["vermillion"]])) +
  labs(x = "mtry", y = "Variance of forest mean", colour = NULL,
       title = "Formula vs simulation") +
  theme(legend.position = "bottom",
        plot.title = element_text(size = 9))

p <- p1 + p2 + p3 + plot_layout(nrow = 1)
save_fig(p, "fig02_variance_formula", width = 9.6, height = 3.8)

write_metrics(list(
  B = B, R = R,
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  min_rho = min(tab$rho), max_rho = max(tab$rho),
  min_ratio = min(tab$ratio), max_ratio = max(tab$ratio),
  min_floor = min(tab$floor_share), max_floor = max(tab$floor_share)
), "e02_variance")
