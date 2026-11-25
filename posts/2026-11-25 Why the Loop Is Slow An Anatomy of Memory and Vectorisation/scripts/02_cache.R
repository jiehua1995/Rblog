# E02 -- the memory hierarchy: random access, and the cost of walking a
# column-major matrix the wrong way.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(microbenchmark)
  library(Rcpp)
})
source("../../theme-blog.R")
source("R/io.R")
source("R/bench.R")
sourceCpp("R/kernels.cpp", cacheDir = "R/.rcpp-cache")

# ---- figure 3: pointer chasing across the cache hierarchy ------------------
sizes <- c(8e3, 3.2e4, 1.28e5, 5.12e5, 2.05e6, 8.2e6, 3.3e7, 1.3e8, 2.6e8)
K <- 2e6
set.seed(2)

chase <- lapply(sizes, function(bytes) {
  n <- as.integer(bytes / 8)
  x <- rnorm(n)
  idx <- sample.int(n, K, replace = TRUE)
  tibble(
    size_kb = bytes / 1024,
    random = bench_ns(function() sum_at(x, idx), times = 5) / K,
    sequential = bench_ns(function() sum(x), times = 5) / n
  )
}) %>% do.call(rbind, .)

fig03 <- chase %>%
  pivot_longer(c(random, sequential), names_to = "how", values_to = "ns") %>%
  mutate(how = factor(how, levels = c("random", "sequential"),
                      labels = c("random access", "sequential scan"))) %>%
  ggplot(aes(size_kb, ns, colour = how)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  geom_vline(xintercept = c(48, 512, 12288), linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.3) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c("random access" = blog_pal[["vermillion"]],
                                 "sequential scan" = blog_pal[["blue"]])) +
  annotate("text", x = c(48, 512, 12288), y = max(chase$random) * 1.5,
           label = c("L1", "L2", "L3"), colour = blog_pal[["grey"]], size = 3,
           vjust = -0.3) +
  labs(x = "array size (KB) / 数组大小（KB）",
       y = "time per access (ns) / 每次访问耗时（纳秒）", colour = NULL)
save_fig(fig03, "fig03_cache_hierarchy", width = 8.4, height = 4.4)

save_metrics("e03", list(size_kb = chase$size_kb, random_ns = chase$random,
                         sequential_ns = chase$sequential,
                         random_over_seq_max = max(chase$random / chase$sequential),
                         random_ns_8kb = chase$random[1],
                         random_ns_260mb = chase$random[nrow(chase)]))

# ---- figure 4: row-major vs column-major traversal -------------------------
ns4 <- c(100, 200, 400, 800, 1200, 2000, 3000, 4000)
set.seed(3)

trav <- lapply(ns4, function(n) {
  m <- matrix(rnorm(n * n), n, n)
  tibble(
    n = n,
    mb = n * n * 8 / 1024^2,
    col_major = bench_ns(function() sum_col_major(m), times = 5) / (n * n),
    row_major = bench_ns(function() sum_row_major(m), times = 5) / (n * n)
  )
}) %>% do.call(rbind, .)

fig04 <- trav %>%
  pivot_longer(c(col_major, row_major), names_to = "order", values_to = "ns") %>%
  mutate(order = factor(order, levels = c("col_major", "row_major"),
                        labels = c("column-major (memory order)",
                                   "row-major (against memory)"))) %>%
  ggplot(aes(mb, ns, colour = order)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  geom_vline(xintercept = 12, linetype = "dashed",
             colour = blog_pal[["grey"]], linewidth = 0.3) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c("column-major (memory order)" = blog_pal[["blue"]],
                                 "row-major (against memory)" = blog_pal[["vermillion"]])) +
  annotate("text", x = 12, y = min(trav$col_major) * 0.6, label = "L3",
           colour = blog_pal[["grey"]], size = 3, hjust = -0.2) +
  labs(x = "matrix size (MB) / 矩阵大小（MB）",
       y = "time per element (ns) / 每个元素耗时（纳秒）", colour = NULL)
save_fig(fig04, "fig04_traversal", width = 8.4, height = 4.4)

save_metrics("e04", list(n = trav$n, mb = trav$mb, col_major_ns = trav$col_major,
                         row_major_ns = trav$row_major,
                         ratio_at_4000 = trav$row_major[trav$n == 4000] /
                           trav$col_major[trav$n == 4000]))

message("02_cache done")
