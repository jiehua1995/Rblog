source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n_test <- 5000

Xt <- matrix(runif(n_test * 3, -2, 2), n_test, 3)
colnames(Xt) <- paste0("x", 1:3)
et <- 1.4 * Xt[, 1] - 1.2 * Xt[, 2] + 1.1 * sin(2 * Xt[, 3])
pt <- plogis(et)
yt <- rbinom(n_test, 1, pt)

settings <- expand.grid(n_train = c(200, 3000), leaf = c(1, 10, 50, 100))
curves <- list()
scores <- list()
for (i in seq_len(nrow(settings))) {
  n_train <- settings$n_train[i]
  leaf <- settings$leaf[i]
  set.seed(SEED + 1000 * n_train + leaf)
  X <- matrix(runif(n_train * 3, -2, 2), n_train, 3)
  colnames(X) <- paste0("x", 1:3)
  eta <- 1.4 * X[, 1] - 1.2 * X[, 2] + 1.1 * sin(2 * X[, 3])
  y <- factor(rbinom(n_train, 1, plogis(eta)))

  rf <- ranger(x = X, y = y, num.trees = 500, mtry = 2,
               min.node.size = leaf, probability = TRUE,
               seed = SEED + 1000 * n_train + leaf)
  pr <- predict(rf, data = Xt)$predictions[, "1"]
  br <- cut(pr, breaks = seq(0, 1, 0.1), include.lowest = TRUE)
  key <- paste(n_train, leaf)
  curves[[key]] <- data.frame(
    n_train = n_train, leaf = leaf,
    bin = as.numeric(tapply(pr, br, mean)),
    observed = as.numeric(tapply(yt, br, mean))
  )
  scores[[key]] <- data.frame(
    n_train = n_train, leaf = leaf,
    brier = mean((pr - yt)^2),
    logloss = -mean(yt * log(pmax(pr, 1e-6)) + (1 - yt) * log(pmax(1 - pr, 1e-6))),
    mean_pred = mean(pr), sd_pred = sd(pr)
  )
}
curves <- bind_rows(curves) |> drop_na()
scores <- bind_rows(scores)
curves$leaf <- factor(curves$leaf)
scores$leaf <- factor(scores$leaf, levels = sort(unique(settings$leaf)))
curves$n_train <- factor(curves$n_train, levels = c(200, 3000),
                         labels = c("n = 200", "n = 3000"))
scores$n_train <- factor(scores$n_train, levels = c(200, 3000),
                         labels = c("n = 200", "n = 3000"))

p1 <- ggplot(curves, aes(bin, observed, colour = leaf)) +
  geom_abline(linetype = 2, colour = blog_pal[["grey"]]) +
  geom_line(aes(group = leaf), linewidth = 0.7) +
  geom_point(size = 1.6) +
  facet_wrap(~n_train, nrow = 1) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_colour_manual(values = c(blog_pal[["blue"]], blog_pal[["green"]],
                                 blog_pal[["orange"]], blog_pal[["vermillion"]])) +
  labs(x = "Predicted probability", y = "Observed frequency",
       colour = "min.node.size") +
  theme(legend.position = "bottom")

p2 <- ggplot(scores, aes(leaf, brier, fill = n_train)) +
  geom_col(position = position_dodge(width = 0.76), width = 0.68) +
  scale_fill_manual(values = c(`n = 200` = blog_pal[["vermillion"]],
                               `n = 3000` = blog_pal[["blue"]])) +
  labs(x = "min.node.size", y = "Brier score", fill = NULL) +
  theme(legend.position = "bottom")

p <- p1 / p2 + plot_layout(heights = c(1.2, 1))
save_fig(p, "fig09_calibration", width = 9.2, height = 6.6)

write_metrics(list(
  scores = lapply(seq_len(nrow(scores)), function(i) as.list(scores[i, ])),
  curves = lapply(seq_len(nrow(curves)), function(i) as.list(curves[i, ])),
  brier_best = min(scores$brier),
  brier_worst = max(scores$brier),
  brier_small_n_leaf1 = scores$brier[scores$n_train == "n = 200" & scores$leaf == 1],
  brier_large_n_leaf1 = scores$brier[scores$n_train == "n = 3000" & scores$leaf == 1]
), "e09_calibration")
