# E04 -- the cost of temporaries, the noise in a benchmark, and when compiled
# code is worth the call overhead.
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

# ---- figure 7: three ways to combine three vectors -------------------------
n7 <- 4e6
set.seed(7)
x7 <- rnorm(n7)
y7 <- rnorm(n7)
z7 <- rnorm(n7)

f_chain <- function() x7^2 + y7^2 + z7^2
f_loop <- function() {
  out <- numeric(n7)
  for (i in seq_len(n7)) out[i] <- x7[i]^2 + y7[i]^2 + z7[i]^2
  out
}
f_cpp <- function() hypot3(x7, y7, z7)

# measure the extra heap each version needs, over the idle baseline
base7 <- peak_mb(function() invisible(NULL))
peak_delta <- function(f) peak_mb(f) - base7

mem_time <- tibble(
  how = c("x^2 + y^2 + z^2", "R loop into out", "Rcpp into out"),
  peak_mb = c(peak_delta(f_chain), peak_delta(f_loop), peak_delta(f_cpp)),
  ms = c(bench_ns(f_chain, times = 5), bench_ns(f_loop, times = 3),
         bench_ns(f_cpp, times = 10)) / 1e6
) %>% mutate(how = factor(how, levels = how))

p7a <- ggplot(mem_time, aes(peak_mb, how)) +
  geom_col(fill = blog_pal[["orange"]], width = 0.6) +
  geom_text(aes(label = sprintf("%.0f MB", peak_mb)), hjust = -0.12, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "peak R heap (MB) / 峰值内存（MB）", y = NULL)

p7b <- ggplot(mem_time, aes(ms, how)) +
  geom_col(fill = blog_pal[["blue"]], width = 0.6) +
  geom_text(aes(label = sprintf("%.0f ms", ms)), hjust = -0.12, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "time (ms) / 耗时（毫秒）", y = NULL)

fig07 <- p7a + p7b
save_fig(fig07, "fig07_temporaries", width = 9.4, height = 3.6)

save_metrics("e07", list(how = as.character(mem_time$how),
                         peak_mb = mem_time$peak_mb, ms = mem_time$ms,
                         chain_over_cpp_mem = mem_time$peak_mb[1] / mem_time$peak_mb[3],
                         chain_over_cpp_time = mem_time$ms[1] / mem_time$ms[3]))

# ---- figure 8: one operation, 300 timings ---------------------------------
n8 <- 1e6
set.seed(8)
x8 <- rnorm(n8)
y8 <- rnorm(n8)
for (i in 1:3) crossprod(x8, y8)
t8 <- microbenchmark::microbenchmark(crossprod(x8, y8), times = 300)$time

fig08a <- tibble(run = seq_along(t8), ns = t8) %>%
  ggplot(aes(run, ns)) +
  geom_line(colour = blog_pal[["blue"]], linewidth = 0.35) +
  geom_hline(yintercept = median(t8), linetype = "dashed",
             colour = blog_pal[["vermillion"]], linewidth = 0.5) +
  labs(x = "repetition / 第几次重复", y = "time (ns) / 耗时（纳秒）")

fig08b <- ggplot(tibble(ns = t8), aes(ns)) +
  geom_histogram(bins = 40, fill = blog_pal[["sky"]], colour = NA, alpha = 0.6) +
  geom_vline(xintercept = median(t8), linetype = "dashed",
             colour = blog_pal[["vermillion"]], linewidth = 0.5) +
  labs(x = "time (ns) / 耗时（纳秒）", y = "count / 次数")

fig08 <- fig08a | fig08b
save_fig(fig08, "fig08_noise", width = 9.4, height = 3.8)

save_metrics("e08", list(n = n8, min_ns = min(t8), median_ns = median(t8),
                         max_ns = max(t8), p95_ns = as.numeric(quantile(t8, 0.95)),
                         max_over_min = max(t8) / min(t8)))

# ---- figure 9: fixed call cost, and where it stops mattering --------------
ns9 <- c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384)
set.seed(9)

call_cost <- lapply(ns9, function(n) {
  xx <- rnorm(n)
  tibble(
    n = n,
    builtin = bench_ns(function() sum(xx), times = 20),
    cpp = bench_ns(function() sum1(xx), times = 20),
    cpp_ptr = bench_ns(function() sum1_ptr(xx), times = 20),
    loop = bench_ns(function() {
      s <- 0
      for (i in seq_len(n)) s <- s + xx[i]
      s
    }, times = 20)
  )
}) %>% do.call(rbind, .)

fig09 <- call_cost %>%
  pivot_longer(c(builtin, cpp, cpp_ptr, loop), names_to = "how", values_to = "ns") %>%
  mutate(how = factor(how, levels = c("loop", "cpp", "cpp_ptr", "builtin"),
                      labels = c("R for loop", "Rcpp sum1", "Rcpp raw pointer",
                                 "sum()"))) %>%
  ggplot(aes(n, ns / 1000, colour = how)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.4) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c("R for loop" = blog_pal[["vermillion"]],
                                 "Rcpp sum1" = blog_pal[["purple"]],
                                 "Rcpp raw pointer" = blog_pal[["green"]],
                                 "sum()" = blog_pal[["blue"]])) +
  labs(x = "vector length n / 向量长度",
       y = "time per call (µs) / 每次调用耗时（微秒）", colour = NULL)
save_fig(fig09, "fig09_call_cost", width = 8.4, height = 4.4)

save_metrics("e09", list(n = call_cost$n, builtin_us = call_cost$builtin / 1000,
                         cpp_us = call_cost$cpp / 1000,
                         cpp_ptr_us = call_cost$cpp_ptr / 1000,
                         loop_us = call_cost$loop / 1000,
                         cpp_us_at_n1 = call_cost$cpp[1] / 1000,
                         builtin_us_at_n1 = call_cost$builtin[1] / 1000,
                         ptr_over_cpp_at_16384 =
                           call_cost$cpp_ptr[call_cost$n == 16384] /
                           call_cost$cpp[call_cost$n == 16384]))

# ---- machine facts, recorded once -----------------------------------------
mi <- machine()
save_metrics("e00", mi)

message("04_method done")
