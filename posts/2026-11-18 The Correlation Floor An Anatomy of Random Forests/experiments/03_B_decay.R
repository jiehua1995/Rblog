source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "forest.R"))

set.seed(SEED)
test <- friedman_data(2000, p = 10, noise = 1)
mtry <- 3
Bs <- c(1, 2, 5, 10, 20, 50, 100, 200)
R <- 20

res <- list()
P_list <- vector("list", R)
for (B in Bs) {
  set.seed(SEED + B)
  mse <- numeric(R)
  mean_pred <- matrix(NA_real_, R, nrow(test$X))
  for (r in seq_len(R)) {
    d <- friedman_data(300, p = 10, noise = 1)
    rf <- ranger(x = d$X, y = d$y, num.trees = B, mtry = mtry,
                 min.node.size = 1, replace = TRUE, sample.fraction = 1,
                 seed = SEED + 7919 * B + r)
    pr <- predict(rf, data = test$X)$predictions
    mean_pred[r, ] <- pr
    mse[r] <- mean((pr - test$y)^2)
    if (B == max(Bs)) {
      P_list[[r]] <- predict(rf, data = test$X, predict.all = TRUE)$predictions
    }
  }
  res[[as.character(B)]] <- data.frame(
    B = B,
    mse = mean(mse),
    var_forest = mean(apply(mean_pred, 2, var))
  )
}
tab <- bind_rows(res)

# Reference values at the largest forest, used to draw the theoretical 1/B curve.
sigma2 <- pooled_sigma2(P_list)
rho <- pointwise_rho(P_list, sigma2 = sigma2, n_pairs = 20, n_points = 100)
curve <- data.frame(B = 10^seq(0, log10(max(Bs)), length.out = 120))
curve$var_theory <- rho * sigma2 + (1 - rho) * sigma2 / curve$B
curve$asymptote <- rho * sigma2

p1 <- ggplot(tab, aes(B, mse)) +
  geom_line(colour = blog_pal[["vermillion"]], linewidth = 0.8) +
  geom_point(colour = blog_pal[["vermillion"]], size = 2) +
  scale_x_log10() +
  labs(x = "Number of trees B (log scale)", y = "Test MSE",
       title = "Error stops falling") +
  theme(plot.title = element_text(size = 9))

p2 <- ggplot() +
  geom_line(data = curve, aes(B, var_theory), colour = blog_pal[["blue"]],
            linewidth = 0.8) +
  geom_hline(yintercept = unique(curve$asymptote), linetype = 2,
             colour = blog_pal[["grey"]]) +
  geom_point(data = tab, aes(B, var_forest), colour = blog_pal[["vermillion"]],
             size = 2) +
  scale_x_log10() +
  labs(x = "Number of trees B (log scale)", y = "Variance of forest mean",
       title = expression(paste("Converges to ", rho, sigma^2))) +
  theme(plot.title = element_text(size = 9))

p <- p1 + p2 + plot_layout(nrow = 1)
save_fig(p, "fig03_B_decay", width = 9.2, height = 3.8)

write_metrics(list(
  mtry = mtry,
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  rho = rho, sigma2 = sigma2,
  asymptote = rho * sigma2,
  var_at_B200 = tab$var_forest[tab$B == max(Bs)],
  mse_at_B1 = tab$mse[tab$B == 1],
  mse_at_B200 = tab$mse[tab$B == max(Bs)]
), "e03_B")
