# E01 -- what a loop costs per iteration, and what vectorisation actually buys.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(microbenchmark)
  library(Rcpp)
})
source("../../theme-blog.R")
source("R/io.R")
source("R/bench.R")
sourceCpp("R/kernels.cpp", cacheDir = "R/.rcpp-cache")

# ---- figure 1: the same inner product, five ways ---------------------------
f_loop <- function(x, y) {
  s <- 0
  for (i in seq_along(x)) s <- s + x[i] * y[i]
  s
}
f_vapply <- function(x, y) sum(vapply(seq_along(x), function(i) x[i] * y[i], 0))

ns <- c(1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
set.seed(1)

res <- lapply(ns, function(n) {
  x <- rnorm(n)
  y <- rnorm(n)
  tibble(
    n = n,
    loop = bench_ns(function() f_loop(x, y)) / n,
    vapply = bench_ns(function() f_vapply(x, y)) / n,
    product = bench_ns(function() sum(x * y)) / n,
    crossprod = bench_ns(function() drop(crossprod(x, y))) / n,
    cpp = bench_ns(function() sum_cpp(x, y)) / n
  )
}) %>% do.call(rbind, .)

fig01 <- res %>%
  pivot_longer(-n, names_to = "how", values_to = "ns_per_element") %>%
  mutate(how = factor(how, levels = c("loop", "vapply", "product",
                                      "crossprod", "cpp"),
                      labels = c("R for loop", "vapply", "sum(x*y)",
                                 "crossprod", "Rcpp loop"))) %>%
  ggplot(aes(n, ns_per_element, colour = how)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.4) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c("R for loop" = blog_pal[["vermillion"]],
                                 "vapply" = blog_pal[["orange"]],
                                 "sum(x*y)" = blog_pal[["blue"]],
                                 "crossprod" = blog_pal[["green"]],
                                 "Rcpp loop" = blog_pal[["purple"]])) +
  labs(x = "vector length n / 向量长度",
       y = "time per element (ns) / 每个元素耗时（纳秒）", colour = NULL)
save_fig(fig01, "fig01_inner_product", width = 8.4, height = 4.4)

save_metrics("e01", list(
  n = res$n,
  loop_ns = res$loop,
  vapply_ns = res$vapply,
  product_ns = res$product,
  crossprod_ns = res$crossprod,
  cpp_ns = res$cpp,
  ratio_loop_over_crossprod_1e6 = res$loop[res$n == 1e6] / res$crossprod[res$n == 1e6],
  ratio_vapply_over_loop_1e6 = res$vapply[res$n == 1e6] / res$loop[res$n == 1e6]
))

# ---- figure 2: per-iteration cost of R-level operations --------------------
n2 <- 1e5
x2 <- rnorm(n2)
df2 <- data.frame(x = x2)
lt2 <- as.list(x2)
m2 <- matrix(x2, ncol = 10)
f_identity <- function(v) v

ops <- list(
  "empty loop" = function() {
    for (i in seq_len(n2)) {}
  },
  "s <- s + 1" = function() {
    s <- 0
    for (i in seq_len(n2)) s <- s + 1
    s
  },
  "s <- s + x[i]" = function() {
    s <- 0
    for (i in seq_len(n2)) s <- s + x2[i]
    s
  },
  "s <- s + x[[i]]" = function() {
    s <- 0
    for (i in seq_len(n2)) s <- s + lt2[[i]]
    s
  },
  "s <- s + df$x[i]" = function() {
    s <- 0
    for (i in seq_len(n2)) s <- s + df2$x[i]
    s
  },
  "s <- s + f(x[i])" = function() {
    s <- 0
    for (i in seq_len(n2)) s <- s + f_identity(x2[i])
    s
  }
)

op_res <- tibble(
  op = names(ops),
  ns = vapply(ops, function(f) bench_ns(f, times = 5), numeric(1)) / n2
) %>% mutate(op = factor(op, levels = op))

fig02 <- ggplot(op_res, aes(ns, op)) +
  geom_col(fill = blog_pal[["blue"]], width = 0.6) +
  geom_text(aes(label = sprintf("%.0f ns", ns)), hjust = -0.15, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(x = "cost per iteration (ns) / 每次迭代耗时（纳秒）", y = NULL)
save_fig(fig02, "fig02_iteration_cost", width = 8, height = 3.8)

save_metrics("e02", list(op = as.character(op_res$op), ns = op_res$ns))

message("01_loop done")
