source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)

time_fit <- function(n, p, B, reps = 3) {
  X <- matrix(runif(n * p), n, p)
  colnames(X) <- paste0("x", seq_len(p))
  y <- rnorm(n)
  tms <- numeric(reps)
  for (r in seq_len(reps)) {
    tms[r] <- system.time({
      ranger(x = X, y = y, num.trees = B, mtry = max(1, floor(sqrt(p))),
             min.node.size = 1, seed = SEED + r)
    })[["elapsed"]]
  }
  median(tms)
}

grid_n <- expand.grid(n = c(300, 600, 1200, 2400, 4800), p = 10, B = 200)
grid_p <- expand.grid(n = 1000, p = c(5, 10, 20, 50, 100), B = 200)
grid_B <- expand.grid(n = 1000, p = 10, B = c(50, 100, 200, 400, 800))

res_n <- data.frame(value = grid_n$n,
                    time = mapply(time_fit, grid_n$n, grid_n$p, grid_n$B))
res_p <- data.frame(value = grid_p$p,
                    time = mapply(time_fit, grid_p$n, grid_p$p, grid_p$B))
res_B <- data.frame(value = grid_B$B,
                    time = mapply(time_fit, grid_B$n, grid_B$p, grid_B$B))

slope <- function(d) unname(coef(lm(log(time) ~ log(value), data = d))[2])
s_n <- slope(res_n); s_p <- slope(res_p); s_B <- slope(res_B)

mk <- function(d, lab, s) {
  ggplot(d, aes(value, time)) +
    geom_line(colour = blog_pal[["blue"]], linewidth = 0.8) +
    geom_point(colour = blog_pal[["blue"]], size = 2) +
    scale_x_log10() + scale_y_log10() +
    labs(x = lab, y = "Median fit time (s, log scale)",
         title = sprintf("slope = %.2f", s)) +
    theme(plot.title = element_text(size = 9))
}

p <- mk(res_n, "n (log scale)", s_n) +
  mk(res_p, "p (log scale)", s_p) +
  mk(res_B, "B (log scale)", s_B) +
  plot_layout(nrow = 1)
save_fig(p, "fig11_cost", width = 9.6, height = 3.8)

write_metrics(list(
  slope_n = s_n, slope_p = s_p, slope_B = s_B,
  time_n_4800 = res_n$time[res_n$value == 4800],
  time_p_100 = res_p$time[res_p$value == 100],
  time_B_800 = res_B$time[res_B$value == 800],
  table_n = lapply(seq_len(nrow(res_n)), function(i) as.list(res_n[i, ])),
  table_p = lapply(seq_len(nrow(res_p)), function(i) as.list(res_p[i, ])),
  table_B = lapply(seq_len(nrow(res_B)), function(i) as.list(res_B[i, ]))
), "e11_cost")
