source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n <- 1000
x <- runif(n)
sig <- 0.2 + 0.8 * x
y <- sin(2 * pi * x) + sig * rnorm(n)
xt <- seq(0.01, 0.99, length.out = 300)
yt <- sin(2 * pi * xt) + (0.2 + 0.8 * xt) * rnorm(300)

rf <- ranger(x = data.frame(x = x), y = y, num.trees = 1000, mtry = 1,
             min.node.size = 5, seed = SEED)
P <- predict(rf, data = data.frame(x = xt), predict.all = TRUE)$predictions

rf_q <- ranger(x = data.frame(x = x), y = y, num.trees = 1000, mtry = 1,
               min.node.size = 5, quantreg = TRUE, seed = SEED)
rf_q50 <- ranger(x = data.frame(x = x), y = y, num.trees = 1000, mtry = 1,
                 min.node.size = 50, quantreg = TRUE, seed = SEED)

levels <- c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
cov <- width <- cov_q <- width_q <- cov_q50 <- width_q50 <- numeric(length(levels))
for (i in seq_along(levels)) {
  a <- levels[i]
  lo <- apply(P, 1, quantile, probs = (1 - a) / 2)
  hi <- apply(P, 1, quantile, probs = 1 - (1 - a) / 2)
  cov[i] <- mean(yt >= lo & yt <= hi)
  width[i] <- mean(hi - lo)

  pq <- predict(rf_q, data = data.frame(x = xt), type = "quantiles",
                quantiles = c((1 - a) / 2, 1 - (1 - a) / 2))$predictions
  cov_q[i] <- mean(yt >= pq[, 1] & yt <= pq[, 2])
  width_q[i] <- mean(pq[, 2] - pq[, 1])

  pq50 <- predict(rf_q50, data = data.frame(x = xt), type = "quantiles",
                  quantiles = c((1 - a) / 2, 1 - (1 - a) / 2))$predictions
  cov_q50[i] <- mean(yt >= pq50[, 1] & yt <= pq50[, 2])
  width_q50[i] <- mean(pq50[, 2] - pq50[, 1])
}
qtab <- data.frame(nominal = levels, coverage_naive = cov, width_naive = width,
                   coverage_qrf = cov_q, width_qrf = width_q,
                   coverage_qrf50 = cov_q50, width_qrf50 = width_q50)

qlong <- qtab |>
  select(nominal, `Tree quantiles` = coverage_naive,
         `QRF leaf 5` = coverage_qrf,
         `QRF leaf 50` = coverage_qrf50) |>
  pivot_longer(-nominal, names_to = "method", values_to = "coverage")

p1 <- ggplot(qlong, aes(nominal, coverage, colour = method)) +
  geom_abline(linetype = 2, colour = blog_pal[["grey"]]) +
  geom_line(aes(group = method), linewidth = 0.8) +
  geom_point(size = 2) +
  coord_equal(xlim = c(0.45, 1), ylim = c(0.45, 1)) +
  scale_colour_manual(values = c(`Tree quantiles` = blog_pal[["vermillion"]],
                                 `QRF leaf 5` = blog_pal[["orange"]],
                                 `QRF leaf 50` = blog_pal[["blue"]])) +
  labs(x = "Nominal coverage", y = "Empirical coverage",
       colour = NULL, title = "Prediction intervals") +
  theme(legend.position = "bottom", legend.text = element_text(size = 8),
        plot.title = element_text(size = 9))

wlong <- qtab |>
  select(nominal, `Tree quantiles` = width_naive,
         `QRF leaf 5` = width_qrf,
         `QRF leaf 50` = width_qrf50) |>
  pivot_longer(-nominal, names_to = "method", values_to = "width")

p2 <- ggplot(wlong, aes(nominal, width, colour = method)) +
  geom_line(aes(group = method), linewidth = 0.8) +
  geom_point(size = 2) +
  scale_colour_manual(values = c(`Tree quantiles` = blog_pal[["vermillion"]],
                                 `QRF leaf 5` = blog_pal[["orange"]],
                                 `QRF leaf 50` = blog_pal[["blue"]])) +
  labs(x = "Nominal coverage", y = "Mean interval width",
       colour = NULL) +
  theme(legend.position = "none")

# T-learner on a simulated heterogeneous treatment effect.
n <- 3000
X <- matrix(runif(n * 3), n, 3)
colnames(X) <- paste0("x", 1:3)
e <- plogis(2 * X[, 1] - 1)
Tt <- rbinom(n, 1, e)
mu0 <- 2 * sin(2 * pi * X[, 1]) + X[, 2]
tau <- 1 + 2 * X[, 1]
y2 <- mu0 + tau * Tt + rnorm(n, 0, 0.5)

rf0 <- ranger(x = X[Tt == 0, , drop = FALSE], y = y2[Tt == 0],
              num.trees = 500, mtry = 2, min.node.size = 3, seed = SEED)
rf1 <- ranger(x = X[Tt == 1, , drop = FALSE], y = y2[Tt == 1],
              num.trees = 500, mtry = 2, min.node.size = 3, seed = SEED)

Xt <- matrix(runif(2000 * 3), 2000, 3)
colnames(Xt) <- paste0("x", 1:3)
tau_true <- 1 + 2 * Xt[, 1]
tau_hat <- predict(rf1, data = Xt)$predictions -
  predict(rf0, data = Xt)$predictions

bin <- cut(Xt[, 1], breaks = seq(0, 1, 0.1), include.lowest = TRUE)
cal <- data.frame(
  x1 = as.numeric(tapply(Xt[, 1], bin, mean)),
  true = as.numeric(tapply(tau_true, bin, mean)),
  pred = as.numeric(tapply(tau_hat, bin, mean))
)

p3 <- ggplot(data.frame(x1 = Xt[, 1], tau_hat = tau_hat), aes(x1, tau_hat)) +
  geom_point(alpha = 0.12, size = 0.5, colour = blog_pal[["grey"]]) +
  geom_line(data = data.frame(x1 = Xt[, 1], true = tau_true) |>
              arrange(x1), aes(x1, true), colour = "black", linewidth = 0.8) +
  geom_point(data = cal, aes(x1, pred), colour = blog_pal[["vermillion"]],
             size = 2.2) +
  geom_line(data = cal, aes(x1, pred), colour = blog_pal[["vermillion"]],
            linewidth = 0.7) +
  labs(x = expression(x[1]), y = expression(hat(tau)(x)),
       title = "T-learner: CATE recovery") +
  theme(plot.title = element_text(size = 9))

p <- (p1 | p2) / p3 + plot_layout(heights = c(1, 1.15))
save_fig(p, "fig13_variants", width = 8.6, height = 6.8)

write_metrics(list(
  quantile = lapply(seq_len(nrow(qtab)), function(i) as.list(qtab[i, ])),
  coverage_error_naive = max(abs(qtab$coverage_naive - qtab$nominal)),
  coverage_error_qrf = max(abs(qtab$coverage_qrf - qtab$nominal)),
  coverage_error_qrf50 = max(abs(qtab$coverage_qrf50 - qtab$nominal)),
  width_ratio_95 = qtab$width_qrf50[qtab$nominal == 0.95] /
    qtab$width_naive[qtab$nominal == 0.95],
  cate_cor = cor(tau_hat, tau_true),
  cate_rmse = sqrt(mean((tau_hat - tau_true)^2)),
  cate_cal_max_gap = max(abs(cal$pred - cal$true))
), "e13_variants")
