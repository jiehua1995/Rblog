# E01 -- what the same expression means in two languages.
# Run from the post directory: Rscript scripts/01_semantics.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(jsonlite)
})
source("R/io.R")
source("../../theme-blog.R")

dt <- fread("data/bench.csv")
x <- dt$x

# ---- indexing ---------------------------------------------------------------
save_metrics("e01_index", list(
  r_second_to_fourth = round(x[2:4], 6),
  r_first = x[1],
  r_last = x[length(x)],
  r_drop_first_head = round(x[-1][1:3], 6),
  r_len = length(x)
))

# ---- missing values ---------------------------------------------------------
v_na <- c(1, NA, 3)
v_nan <- c(1, NaN, 3)
as_chr <- function(z) {
  if (length(z) == 0) return(character(0))
  ifelse(is.na(z), ifelse(is.nan(z), "NaN", "NA"), format(z, trim = TRUE))
}
save_metrics("e01_missing", list(
  na_sum = as_chr(sum(v_na)),
  na_sum_rm = as_chr(sum(v_na, na.rm = TRUE)),
  na_mean = as_chr(mean(v_na)),
  na_mean_rm = as_chr(mean(v_na, na.rm = TRUE)),
  na_max = as_chr(max(v_na)),
  na_max_rm = as_chr(max(v_na, na.rm = TRUE)),
  nan_sum = as_chr(sum(v_nan)),
  nan_sum_rm = as_chr(sum(v_nan, na.rm = TRUE)),
  nan_mean_rm = as_chr(mean(v_nan, na.rm = TRUE)),
  is_na_of_nan = is.na(NaN),
  is_nan_of_na = is.nan(NA),
  sort_na = as_chr(sort(v_na)),
  sort_nan = as_chr(sort(v_nan)),
  which_na = which(is.na(c(NA, NaN, 1)))
))

# ---- copy on modify ---------------------------------------------------------
a <- c(1, 2, 3)
b <- a
b[1] <- 99
f <- function(z) { z[1] <- 0; z }
invisible(f(a))
save_metrics("e01_copy", list(
  parent_after_child_write = a[1],
  child_after_write = b[1],
  parent_after_function_call = a[1]
))

# ---- recycling --------------------------------------------------------------
wmsg <- NULL
rec3 <- withCallingHandlers(
  (1:3) + c(1, 2),
  warning = function(w) {
    wmsg <<- conditionMessage(w)
    invokeRestart("muffleWarning")
  }
)
save_metrics("e01_recycle", list(
  len4 = c(1, 2, 3, 4) + c(1, 2),
  len3 = rec3,
  len3_warning = if (is.null(wmsg)) NA_character_ else wmsg,
  bool_recycled = (1:3)[c(TRUE, FALSE, TRUE, FALSE)]
))

# ---- three dialects of group-by ---------------------------------------------
agg_dt <- dt[, .(count = .N, mean = mean(val), sd = sd(val)), by = grp][order(grp)]
agg_dplyr <- dt |>
  dplyr::summarise(count = dplyr::n(), mean = mean(val), sd = sd(val), .by = grp) |>
  dplyr::arrange(grp) |>
  as.data.table()
agg_py <- fread("outputs/py_groupby.csv")
setnames(agg_py, "std", "sd")

setkey(agg_dt, grp); setkey(agg_dplyr, grp); setkey(agg_py, grp)
save_metrics("e01_groupby", list(
  rows = nrow(agg_dt),
  max_abs_diff_mean_dt_dplyr = max(abs(agg_dt$mean - agg_dplyr$mean)),
  max_abs_diff_mean_dt_py = max(abs(agg_dt$mean - agg_py$mean)),
  max_abs_diff_sd_dt_py = max(abs(agg_dt$sd - agg_py$sd)),
  same_counts = all(agg_dt$count == agg_py$count),
  # a tidyverse alternative that gives a different *shape*, not a different value
  rows_tibble = nrow(dplyr::count(dt, grp)),
  first_mean = agg_dt$mean[1]
))

# ---- figure: two index systems ---------------------------------------------
n_show <- 8
is_slice <- seq_len(n_show) %in% 2:4
ridx <- data.frame(pos = seq_len(n_show), lab = as.character(seq_len(n_show)),
                   role = ifelse(is_slice, "slice",
                                 ifelse(seq_len(n_show) == 1, "drop", "plain")))
vals <- data.frame(pos = seq_len(n_show), lab = paste0("x", seq_len(n_show)),
                   role = ifelse(is_slice, "slice", "plain"))
pidx <- data.frame(pos = seq_len(n_show), lab = as.character(0:(n_show - 1)),
                   role = ifelse(is_slice, "slice",
                                 ifelse(seq_len(n_show) == n_show, "last", "plain")))
cols <- c(plain = "grey92", slice = "#009E73", drop = "#D55E00", last = "#E69F00")

fig01 <- ggplot() +
  geom_tile(data = ridx, aes(pos, 3, fill = role), width = .9, height = .5,
            colour = "grey35", linewidth = .3) +
  geom_text(data = ridx, aes(pos, 3, label = lab), size = 3.2) +
  geom_tile(data = vals, aes(pos, 2, fill = role), width = .9, height = .5,
            colour = "grey35", linewidth = .3) +
  geom_text(data = vals, aes(pos, 2, label = lab), size = 3.2) +
  geom_tile(data = pidx, aes(pos, 1, fill = role), width = .9, height = .5,
            colour = "grey35", linewidth = .3) +
  geom_text(data = pidx, aes(pos, 1, label = lab), size = 3.2) +
  annotate("text", x = 0.35, y = 3, label = "R", fontface = "bold", hjust = 1, size = 4) +
  annotate("text", x = 0.35, y = 1, label = "Python", fontface = "bold", hjust = 1, size = 4) +
  annotate("text", x = 0.35, y = 2, label = "值", hjust = 1, size = 3.2, colour = "grey35") +
  annotate("segment", x = 1.55, xend = 3.45, y = 3.32, yend = 3.32, linewidth = .4) +
  annotate("segment", x = 1.55, xend = 1.55, y = 3.32, yend = 3.27, linewidth = .4) +
  annotate("segment", x = 3.45, xend = 3.45, y = 3.32, yend = 3.27, linewidth = .4) +
  annotate("text", x = 2.5, y = 3.55, label = "x[2:4]", size = 3.2) +
  annotate("segment", x = 1.55, xend = 3.45, y = 0.68, yend = 0.68, linewidth = .4) +
  annotate("segment", x = 1.55, xend = 1.55, y = 0.68, yend = 0.73, linewidth = .4) +
  annotate("segment", x = 3.45, xend = 3.45, y = 0.68, yend = 0.73, linewidth = .4) +
  annotate("text", x = 2.5, y = 0.45, label = "x[1:4]", size = 3.2) +
  scale_fill_manual(values = cols, breaks = c("slice", "drop", "last"),
                    labels = c("同三个元素 / same three", "R: x[-1] 删掉它", "Python: x[-1] 是它"),
                    name = NULL) +
  coord_cartesian(xlim = c(-1.05, 8.6), ylim = c(0.3, 3.75)) +
  theme_void(base_size = 11) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.9, "lines"))

save_fig(fig01, "fig01_index", width = 7.2, height = 3.4)

# ---- figure: recycling against broadcasting --------------------------------
rec <- rbind(
  data.frame(row = "a", pos = 1:4, lab = c("1", "2", "3", "4"), kind = "own"),
  data.frame(row = "b", pos = 1:4, lab = c("1", "2", "1", "2"),
             kind = c("own", "own", "copy", "copy")),
  data.frame(row = "a + b", pos = 1:4, lab = c("2", "4", "4", "6"), kind = "result")
)
rec$y <- 4 - as.numeric(factor(rec$row, levels = c("a", "b", "a + b")))

fig02 <- ggplot(rec) +
  geom_tile(aes(pos, y, fill = kind, alpha = kind), width = .86, height = .62,
            colour = "grey35", linewidth = .3) +
  geom_text(aes(pos, y, label = lab), size = 3.6) +
  annotate("text", x = 0.42, y = 3, label = "a", fontface = "bold", hjust = 1, size = 4) +
  annotate("text", x = 0.42, y = 2, label = "b", fontface = "bold", hjust = 1, size = 4) +
  annotate("text", x = 0.42, y = 1, label = "a + b", fontface = "bold", hjust = 1, size = 4) +
  annotate("text", x = 0.42, y = 2.45, label = "+", hjust = 1, size = 4, colour = "grey35") +
  annotate("text", x = 0.42, y = 1.45, label = "=", hjust = 1, size = 4, colour = "grey35") +
  annotate("text", x = 4.65, y = 2, label = "b 被自动重复到长度 4",
           hjust = 0, size = 3.2, colour = "#D55E00") +
  annotate("text", x = 4.65, y = 1.55, label = "Python: 直接报错，不猜你的意思",
           hjust = 0, size = 3.2, colour = "grey35") +
  scale_fill_manual(values = c(own = "#0072B2", copy = "#0072B2", result = "#009E73"),
                    guide = "none") +
  scale_alpha_manual(values = c(own = 1, copy = .3, result = 1), guide = "none") +
  coord_cartesian(xlim = c(0.2, 8.4), ylim = c(0.55, 3.5)) +
  theme_void(base_size = 11)

save_fig(fig02, "fig02_recycle", width = 7.2, height = 3.0)

cat("01_semantics done\n")
