source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
ns <- c(10, 20, 50, 100, 200, 500, 1000, 2000)
frac <- vapply(ns, function(n) {
  mean(replicate(400, length(unique(sample.int(n, n, replace = TRUE)))) / n)
}, numeric(1))

boot <- data.frame(n = ns, observed = frac, theory = 1 - (1 - 1 / ns)^ns)

p1 <- ggplot(boot, aes(n, observed)) +
  geom_line(colour = blog_pal[["blue"]], linewidth = 0.8) +
  geom_point(colour = blog_pal[["blue"]], size = 2) +
  geom_line(aes(y = theory), colour = blog_pal[["vermillion"]],
            linetype = 2, linewidth = 0.7) +
  geom_hline(yintercept = 1 - exp(-1), colour = blog_pal[["grey"]],
             linetype = 3) +
  scale_x_log10() +
  labs(x = "n (log scale)", y = "Share of distinct observations in a bootstrap",
       title = "Why 0.632") +
  theme(plot.title = element_text(size = 9))

set.seed(SEED + 1)
d <- friedman_data(400, p = 8, noise = 1)
rf <- ranger(x = d$X, y = d$y, num.trees = 300, mtry = 3,
             min.node.size = 1, replace = TRUE, sample.fraction = 1,
             keep.inbag = TRUE, seed = SEED + 1)
P <- predict(rf, data = d$X, predict.all = TRUE)$predictions
inbag <- do.call(cbind, rf$inbag.counts)

Bs <- seq(1, ncol(P), by = 5)
oob <- vapply(Bs, function(b) {
  W <- inbag[, seq_len(b), drop = FALSE] == 0
  num <- rowSums(P[, seq_len(b), drop = FALSE] * W)
  den <- rowSums(W)
  ok <- den > 0
  mean((num[ok] / den[ok] - d$y[ok])^2)
}, numeric(1))

oob_tab <- data.frame(B = Bs, mse = oob)

p2 <- ggplot(oob_tab, aes(B, mse)) +
  geom_line(colour = blog_pal[["green"]], linewidth = 0.8) +
  labs(x = "Number of trees B", y = "Cumulative OOB MSE",
       title = "OOB converges too") +
  theme(plot.title = element_text(size = 9))

p <- p1 + p2 + plot_layout(nrow = 1)
save_fig(p, "fig04_bootstrap_oob", width = 9.2, height = 3.8)

write_metrics(list(
  boot = lapply(seq_len(nrow(boot)), function(i) as.list(boot[i, ])),
  limit = 1 - exp(-1),
  oob_start = oob_tab$mse[1],
  oob_end = oob_tab$mse[nrow(oob_tab)],
  oob_drop = oob_tab$mse[1] / oob_tab$mse[nrow(oob_tab)]
), "e04_bootstrap")
