source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "forest.R"))

set.seed(SEED)
n <- 300
X <- cbind(x1 = runif(n), x2 = runif(n))
y <- 4 * (X[, 1] - 0.5)^2 + 2 * X[, 2] + rnorm(n, 0, 0.2)

fit <- cart_fit(X, y, task = "regression", max_depth = 3,
                min_node = 20, mtry = 2, max_candidates = 30, seed = SEED)

acc <- new.env()
acc$leaves <- list()
acc$splits <- list()
walk <- function(node, xlo, xhi, ylo, yhi, depth = 1) {
  if (node$leaf) {
    acc$leaves[[length(acc$leaves) + 1]] <-
      data.frame(xmin = xlo, xmax = xhi, ymin = ylo, ymax = yhi,
                 value = node$value, n = node$n, depth = depth)
  } else {
    acc$splits[[length(acc$splits) + 1]] <-
      data.frame(var = node$var, split = node$split, gain = node$gain,
                 n = node$n, depth = depth)
    if (node$var == 1) {
      walk(node$left, xlo, node$split, ylo, yhi, depth + 1)
      walk(node$right, node$split, xhi, ylo, yhi, depth + 1)
    } else {
      walk(node$left, xlo, xhi, ylo, node$split, depth + 1)
      walk(node$right, xlo, xhi, node$split, yhi, depth + 1)
    }
  }
}
walk(fit$tree, 0, 1, 0, 1)
leaves <- bind_rows(acc$leaves)
splits <- bind_rows(acc$splits)

p <- ggplot() +
  geom_rect(data = leaves, aes(xmin = xmin, xmax = xmax, ymin = ymin,
                               ymax = ymax, fill = value),
            colour = "white", linewidth = 0.5) +
  geom_point(data = data.frame(X, y = y), aes(x1, x2), size = 0.45,
             colour = "black", alpha = 0.35) +
  geom_rect(data = leaves, aes(xmin = xmin, xmax = xmax, ymin = ymin,
                               ymax = ymax),
            fill = NA, colour = "black", linewidth = 0.25) +
  scale_fill_gradientn(colours = c(blog_pal[["sky"]], blog_pal[["blue"]],
                                   blog_pal[["purple"]], blog_pal[["vermillion"]]),
                       name = "Leaf mean") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(x = expression(x[1]), y = expression(x[2])) +
  theme(legend.position = "right",
        axis.ticks = element_blank())

save_fig(p, "fig14_cart", width = 7.6, height = 5.2)

write_metrics(list(
  n_leaves = nrow(leaves),
  max_depth = max(splits$depth),
  splits = lapply(seq_len(nrow(splits)), function(i) as.list(splits[i, ])),
  leaves = lapply(seq_len(nrow(leaves)), function(i) as.list(leaves[i, ])),
  train_rmse = sqrt(mean((cart_predict(fit, X) - y)^2)),
  root_gain = splits$gain[splits$depth == 1]
), "e14_cart")
