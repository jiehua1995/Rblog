# E02 -- the same five operations in four implementations, plus what a column
# costs in memory. Run from the post directory: Rscript scripts/02_speed.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
  library(microbenchmark)
})
source("R/io.R")
source("../../theme-blog.R")

dt <- fread("data/bench.csv")
py <- fromJSON("outputs/python.json")
v <- dt$val

lookup <- data.table(grp = unique(dt$grp))
lookup[, w := seq(1, 2, length.out = .N)]
d2 <- dt[, .(cat, bin = cut(x, 5), val)]

# ---- timings: fastest of five, seconds --------------------------------------
sp <- list(
  sum_loop    = bench_min({ s <- 0; for (i in seq_along(v)) s <- s + v[i]; s }),
  sum_base    = bench_min(sum(v)),
  sum_dplyr   = bench_min(summarise(dt, s = sum(val))),
  group_base  = bench_min(tapply(v, dt$grp, mean)),
  group_dt    = bench_min(dt[, .(m = mean(val)), by = grp]),
  group_dplyr = bench_min(summarise(dt, m = mean(val), .by = grp)),
  sort_base   = bench_min(sort(v)),
  sort_dt     = bench_min(dt[order(val)]),
  sort_dplyr  = bench_min(arrange(dt, val)),
  merge_base  = bench_min(merge(dt, lookup, by = "grp", all.x = TRUE)),
  merge_dt    = bench_min(lookup[dt, on = "grp"]),
  merge_dplyr = bench_min(left_join(dt, lookup, by = "grp")),
  pivot_base  = bench_min(tapply(d2$val, list(d2$cat, d2$bin), mean)),
  pivot_dt    = bench_min(dcast(d2, cat ~ bin, value.var = "val", fun.aggregate = mean)),
  pivot_dplyr = bench_min(d2 |>
                            summarise(m = mean(val), .by = c(cat, bin)) |>
                            pivot_wider(names_from = bin, values_from = m))
)

tasks <- c("sum", "group\nmean", "sort", "left\njoin", "pivot")
speed <- data.frame(
  task = factor(rep(tasks, each = 4), levels = tasks),
  impl = factor(c("R loop", "R base", "dplyr", "pandas",
                  rep(c("R base", "data.table", "dplyr", "pandas"), 4)),
                levels = c("R loop", "R base", "data.table", "dplyr", "pandas")),
  s = c(sp$sum_loop, sp$sum_base, sp$sum_dplyr, py$speed$np_sum_s,
        sp$group_base, sp$group_dt, sp$group_dplyr, py$speed$groupby_mean_s,
        sp$sort_base, sp$sort_dt, sp$sort_dplyr, py$speed$sort_s,
        sp$merge_base, sp$merge_dt, sp$merge_dplyr, py$speed$merge_s,
        sp$pivot_base, sp$pivot_dt, sp$pivot_dplyr, py$speed$pivot_s)
)
speed$ms <- speed$s * 1000
speed$label <- ifelse(speed$ms < 1, sprintf("%.2f", speed$ms), sprintf("%.1f", speed$ms))

save_metrics("e02_speed", speed)
save_metrics("e02_ratios", list(
  n = nrow(dt),
  sum_pandas_over_base = py$speed$np_sum_s / sp$sum_base,
  sum_loop_over_sum = sp$sum_loop / sp$sum_base,
  group_best_over_worst = min(sp$group_dt, sp$group_dplyr, py$speed$groupby_mean_s) /
    sp$group_base,
  sort_best_over_worst = min(sp$sort_dt, sp$sort_dplyr, py$speed$sort_s) / sp$sort_base,
  merge_best_over_worst = min(sp$merge_dt, sp$merge_dplyr, py$speed$merge_s) / sp$merge_base,
  pivot_best_over_worst = min(sp$pivot_dt, sp$pivot_dplyr, py$speed$pivot_s) / sp$pivot_base,
  py_sum_ms = py$speed$np_sum_s * 1000,
  r_sum_ms = sp$sum_base * 1000
))

# ---- figure: five tasks, four implementations ------------------------------
impl_cols <- c("R loop" = "#D55E00", "R base" = "#0072B2",
               "data.table" = "#009E73", "dplyr" = "#CC79A7", "pandas" = "#E69F00")

fig03 <- ggplot(speed, aes(impl, ms, colour = impl)) +
  geom_point(size = 2.8) +
  geom_text(aes(label = label), vjust = -1.1, size = 2.5, colour = "grey20") +
  facet_grid(~ task, scales = "free_x", space = "free_x") +
  scale_colour_manual(values = impl_cols, guide = "none") +
  scale_y_log10(expand = expansion(mult = c(0.12, 0.28))) +
  labs(x = NULL, y = "毫秒 / milliseconds, log scale (fastest of 5)",
       title = "同一件事的四种写法 / one task, four implementations") +
  theme_blog(base_size = 10) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1),
        strip.text = element_text(size = 9))

save_fig(fig03, "fig03_speed", width = 9.6, height = 3.8)

# ---- memory: character against factor/category ------------------------------
mem_r <- data.table(
  col = rep(c("grp (1000 水平)", "cat (3 水平)"), each = 2),
  impl = rep(c("R character", "R factor"), 2),
  mb = c(as.numeric(object.size(dt$grp)), as.numeric(object.size(factor(dt$grp))),
         as.numeric(object.size(dt$cat)), as.numeric(object.size(factor(dt$cat)))) / 1024^2
)
mem_py <- data.table(
  col = rep(c("grp (1000 水平)", "cat (3 水平)"), each = 2),
  impl = rep(c("pandas object", "pandas category"), 2),
  mb = c(py$memory$grp_object, py$memory$grp_category,
         py$memory$cat_object, py$memory$cat_category) / 1024^2
)
mem <- rbind(mem_r, mem_py)
mem$col <- factor(mem$col, levels = c("grp (1000 水平)", "cat (3 水平)"))
mem$impl <- factor(mem$impl, levels = c("R character", "R factor",
                                        "pandas object", "pandas category"))

frame_r <- as.numeric(object.size(dt)) / 1024^2
frame_r_fct <- as.numeric(object.size(transform(dt, grp = factor(grp), cat = factor(cat)))) / 1024^2

save_metrics("e02_memory", list(
  mem = mem,
  frame_datatable_mb = frame_r,
  frame_factor_mb = frame_r_fct,
  frame_pandas_mb = py$memory$frame_object / 1024^2,
  frame_pandas_category_mb = py$memory$frame_category / 1024^2
))

fig04 <- ggplot(mem, aes(impl, mb, fill = impl)) +
  geom_col(width = .72) +
  geom_text(aes(label = sprintf("%.2f", mb)), vjust = -0.35, size = 2.7, colour = "grey20") +
  facet_wrap(~ col, nrow = 1) +
  scale_fill_manual(values = c("R character" = "#0072B2", "R factor" = "#56B4E9",
                               "pandas object" = "#D55E00", "pandas category" = "#E69F00"),
                    guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(x = NULL, y = "MB", title = "一列字符串要多少内存 / what one string column costs") +
  theme_blog(base_size = 10) +
  theme(axis.text.x = element_text(angle = 40, hjust = 1))

save_fig(fig04, "fig04_memory", width = 8.4, height = 3.8)

cat("02_speed done\n")
