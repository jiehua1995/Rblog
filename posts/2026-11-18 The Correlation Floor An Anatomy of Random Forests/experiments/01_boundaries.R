source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

# Two interleaving half circles, a classic non-linear boundary.
make_moons <- function(n, noise = 0.12) {
  n1 <- ceiling(n / 2)
  n2 <- n - n1
  t1 <- runif(n1, 0, pi)
  t2 <- runif(n2, 0, pi)
  x1 <- cbind(cos(t1), sin(t1))
  x2 <- cbind(1 - cos(t2), 1 - sin(t2) - 0.5)
  X <- rbind(x1, x2)
  X <- X + matrix(rnorm(n * 2, 0, noise), n, 2)
  list(X = X, y = factor(c(rep(0, n1), rep(1, n2))))
}

set.seed(SEED)
d <- make_moons(400)
grid <- expand.grid(V1 = seq(-1.8, 2.8, length.out = 120),
                    V2 = seq(-1.4, 1.6, length.out = 120))
colnames(grid) <- c("V1", "V2")
colnames(d$X) <- c("V1", "V2")

cfg <- expand.grid(B = c(1, 3, 10, 100), mtry = c(1, 2))
res <- vector("list", nrow(cfg))
for (i in seq_len(nrow(cfg))) {
  set.seed(SEED + i)
  rf <- ranger(x = d$X, y = d$y, num.trees = cfg$B[i], mtry = cfg$mtry[i],
               min.node.size = 1, replace = TRUE, sample.fraction = 1,
               probability = TRUE, seed = SEED + i)
  pr <- predict(rf, data = grid)$predictions[, "1"]
  pred_train <- predict(rf, data = d$X)$predictions[, "1"]
  res[[i]] <- data.frame(grid,
                         prob = pr,
                         B = cfg$B[i],
                         mtry = cfg$mtry[i],
                         train_err = mean((pred_train > 0.5) != (d$y == "1")))
}
res <- bind_rows(res)
res$panel <- factor(paste0("B = ", res$B, ", mtry = ", res$mtry),
                    levels = paste0("B = ", c(1, 1, 3, 3, 10, 10, 100, 100),
                                    ", mtry = ", c(1, 2, 1, 2, 1, 2, 1, 2)))

p <- ggplot(res, aes(V1, V2)) +
  geom_raster(aes(fill = prob), interpolate = TRUE) +
  geom_point(data = data.frame(d$X, y = d$y), aes(V1, V2, colour = y),
             size = 0.55, alpha = 0.75, inherit.aes = FALSE) +
  facet_wrap(~panel, ncol = 4) +
  scale_fill_gradientn(colours = c("white", blog_pal[["sky"]], blog_pal[["blue"]]),
                       limits = c(0, 1), name = "P(y = 1)") +
  scale_colour_manual(values = c("0" = blog_pal[["orange"]], "1" = "black"),
                      guide = "none") +
  coord_equal() +
  labs(x = expression(x[1]), y = expression(x[2])) +
  theme(legend.position = "bottom",
        legend.key.width = unit(1.6, "lines"),
        axis.text = element_blank(),
        axis.ticks = element_blank())

save_fig(p, "fig01_boundaries", width = 9.2, height = 6.2)

write_metrics(list(
  train_error = lapply(seq_len(nrow(cfg)), function(i) list(
    B = cfg$B[i], mtry = cfg$mtry[i], error = res$train_err[res$B == cfg$B[i] & res$mtry == cfg$mtry[i]][1]
  )),
  n_train = nrow(d$X)
), "e01_boundaries")
