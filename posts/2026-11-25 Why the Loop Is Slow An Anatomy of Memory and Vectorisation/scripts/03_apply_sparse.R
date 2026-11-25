# E03 -- apply()'s family, and where sparse matrices start to win.
# Run from the post directory.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(Matrix)
  library(microbenchmark)
})
source("../../theme-blog.R")
source("R/io.R")
source("R/bench.R")

# ---- figure 5: four ways to average a 2000 x 2000 matrix -------------------
n5 <- 2000
set.seed(5)
m5 <- matrix(rnorm(n5 * n5), n5, n5)

apply_res <- tibble(
  how = c("colMeans", "apply(m, 2, mean)", "rowMeans", "apply(m, 1, mean)"),
  ns = c(
    bench_ns(function() colMeans(m5), times = 10),
    bench_ns(function() apply(m5, 2, mean), times = 5),
    bench_ns(function() rowMeans(m5), times = 10),
    bench_ns(function() apply(m5, 1, mean), times = 5)
  ) / n5
)

fig05 <- apply_res %>%
  mutate(how = factor(how, levels = rev(how))) %>%
  ggplot(aes(ns, how)) +
  geom_col(fill = blog_pal[["green"]], width = 0.6) +
  geom_text(aes(label = sprintf("%.0f ns", ns)), hjust = -0.12, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "time per column or row (ns) / 每列或每行耗时（纳秒）", y = NULL)
save_fig(fig05, "fig05_apply", width = 8, height = 3.6)

save_metrics("e05", list(how = apply_res$how, ns = apply_res$ns,
                         apply_col_over_colmeans = apply_res$ns[2] / apply_res$ns[1],
                         apply_row_over_rowmeans = apply_res$ns[4] / apply_res$ns[3]))

# ---- figure 6: sparse against dense, as a function of density --------------
n6 <- 2000
dens <- c(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5, 0.75, 1)
set.seed(6)

sparse_res <- lapply(dens, function(d) {
  nnz <- round(n6 * n6 * d)
  i <- sample.int(n6, nnz, replace = TRUE)
  j <- sample.int(n6, nnz, replace = TRUE)
  s <- sparseMatrix(i = i, j = j, x = rnorm(nnz), dims = c(n6, n6))
  s <- as(s, "CsparseMatrix")
  dm <- as.matrix(s)
  tibble(
    density = d,
    sparse = bench_ns(function() Matrix::colSums(s), times = 7),
    dense = bench_ns(function() base::colSums(dm), times = 7),
    sparse_mb = as.numeric(object.size(s)) / 1024^2,
    dense_mb = as.numeric(object.size(dm)) / 1024^2
  )
}) %>% do.call(rbind, .)

fig06 <- sparse_res %>%
  pivot_longer(c(sparse, dense), names_to = "how", values_to = "ns") %>%
  mutate(how = factor(how, levels = c("dense", "sparse"),
                      labels = c("dense matrix", "sparse dgCMatrix"))) %>%
  ggplot(aes(density, ns / 1e6, colour = how)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.5) +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c("dense matrix" = blog_pal[["grey"]],
                                 "sparse dgCMatrix" = blog_pal[["purple"]])) +
  labs(x = "fraction of non-zero entries / 非零元素比例",
       y = "time for column sums (ms) / 列求和耗时（毫秒）", colour = NULL)
save_fig(fig06, "fig06_sparse", width = 8.4, height = 4.4)

cross6 <- with(sparse_res,
               stats::approx(sparse - dense, density, xout = 0)$y)
save_metrics("e06", list(density = sparse_res$density,
                         sparse_ns = sparse_res$sparse,
                         dense_ns = sparse_res$dense,
                         sparse_mb = sparse_res$sparse_mb,
                         dense_mb = sparse_res$dense_mb,
                         crossover_density = cross6))

message("03_apply_sparse done")
