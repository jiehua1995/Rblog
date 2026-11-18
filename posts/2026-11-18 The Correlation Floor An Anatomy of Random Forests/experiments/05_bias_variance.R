source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n_train <- 300
R <- 30
mtrys <- c(1, 2, 3, 5, 10)
leaves <- c(1, 5, 20)
B <- 100

test <- friedman_data(600, p = 10, noise = 0)
set.seed(SEED + 5)
y_test <- test$f + rnorm(nrow(test$X), 0, 1)

res <- list()
for (leaf in leaves) {
  for (m in mtrys) {
    P <- matrix(NA_real_, R, nrow(test$X))
    mse_rep <- numeric(R)
    for (r in seq_len(R)) {
      set.seed(SEED + 100000 * leaf + 1000 * m + r)
      d <- friedman_data(n_train, p = 10, noise = 1)
      rf <- ranger(x = d$X, y = d$y, num.trees = B, mtry = m,
                   min.node.size = leaf, replace = TRUE, sample.fraction = 1,
                   seed = SEED + 100000 * leaf + 1000 * m + r)
      pr <- predict(rf, data = test$X)$predictions
      P[r, ] <- pr
      mse_rep[r] <- mean((pr - y_test)^2)
    }
    mu <- colMeans(P)
    bias2 <- mean((mu - test$f)^2)
    variance <- mean(apply(P, 2, var))
    res[[paste(leaf, m, sep = "_")]] <- data.frame(
      leaf = leaf, mtry = m, bias2 = bias2, variance = variance,
      noise = 1, mse_decomp = bias2 + variance + 1,
      mse_empirical = mean(mse_rep)
    )
  }
}
tab <- bind_rows(res)
tab$mtry <- factor(tab$mtry, levels = mtrys)
tab$leaf <- factor(tab$leaf, levels = leaves,
                   labels = paste0("min.node.size = ", leaves))

long <- tab |>
  select(leaf, mtry, `Bias²` = bias2, `Variance` = variance) |>
  pivot_longer(c(`Bias²`, `Variance`), names_to = "component", values_to = "value")

p1 <- ggplot(long, aes(mtry, value, colour = component)) +
  geom_line(aes(group = component), linewidth = 0.8) +
  geom_point(size = 1.8) +
  facet_wrap(~leaf, nrow = 1) +
  scale_y_log10() +
  scale_colour_manual(values = c(`Bias²` = blog_pal[["vermillion"]],
                                 Variance = blog_pal[["blue"]])) +
  labs(x = "mtry", y = "Error component (log scale)", colour = NULL) +
  theme(legend.position = "none")

p2 <- ggplot(tab, aes(mtry, mse_decomp)) +
  geom_line(aes(group = leaf, colour = leaf), linewidth = 0.8) +
  geom_point(aes(colour = leaf), size = 1.8) +
  geom_hline(yintercept = 1, linetype = 2, colour = blog_pal[["grey"]]) +
  scale_y_log10() +
  scale_colour_manual(values = c(blog_pal[["green"]], blog_pal[["orange"]],
                                 blog_pal[["purple"]])) +
  labs(x = "mtry", y = "Bias² + variance + noise (log scale)", colour = NULL,
       title = "Total error") +
  theme(legend.position = "bottom", plot.title = element_text(size = 9))

p <- p1 / p2 + plot_layout(heights = c(1.1, 1), guides = "collect") &
  theme(legend.position = "bottom")
save_fig(p, "fig05_bias_variance", width = 9.0, height = 6.6)

write_metrics(list(
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  min_variance = min(tab$variance),
  max_variance = max(tab$variance),
  min_bias2 = min(tab$bias2),
  max_bias2 = max(tab$bias2),
  mse_range = c(min(tab$mse_decomp), max(tab$mse_decomp)),
  max_gap = max(abs(tab$mse_decomp - tab$mse_empirical))
), "e05_bias_variance")
