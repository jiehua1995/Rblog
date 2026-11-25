# E03 -- do the two languages compute the same numbers, and when do they not.
# Run from the post directory: Rscript scripts/03_numeric.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(microbenchmark)
})
source("R/io.R")
source("../../theme-blog.R")

dt <- fread("data/bench.csv")
py <- fromJSON("outputs/python.json")

# ---- least squares ----------------------------------------------------------
fit <- lm(y ~ x + z, data = dt)
cf <- summary(fit)$coefficients
term <- gsub("[()]", "", rownames(cf))
py_term <- py$ols$names

ols <- data.table(
  term = term,
  coef_r = cf[, "Estimate"],
  se_r = cf[, "Std. Error"],
  t_r = cf[, "t value"],
  coef_py = py$ols$coef,
  se_py = py$ols$bse,
  t_py = py$ols$tvalues
)
ols[, coef_abs := abs(coef_r - coef_py)]
ols[, coef_rel := coef_abs / abs(coef_r)]
ols[, se_rel := abs(se_r - se_py) / se_r]
r2 <- c(summary(fit)$r.squared, py$ols$rsquared)

save_metrics("e03_ols", list(
  table = ols,
  names_match = identical(term, py_term),
  r2_r = r2[1],
  r2_py = r2[2],
  r2_rel = abs(r2[1] - r2[2]) / r2[1],
  n = nrow(dt)
))

# ---- Welch-corrected? no: the textbook two-sample t ------------------------
a <- dt$x[dt$cat == "alpha"]
b <- dt$x[dt$cat == "beta"]
tt <- t.test(a, b, var.equal = TRUE)
save_metrics("e03_ttest", list(
  r_t = unname(tt$statistic),
  r_p = tt$p.value,
  py_t = py$ttest$t,
  py_p = py$ttest$p,
  abs_diff_t = abs(unname(tt$statistic) - py$ttest$t),
  abs_diff_p = abs(tt$p.value - py$ttest$p),
  n_alpha = length(a),
  n_beta = length(b)
))

# ---- summation order --------------------------------------------------------
s <- fread("data/sums.csv")$s
py_scale <- fromJSON("outputs/py_sums_scale.json")
ns <- py_scale$n
REPS <- 20L

sum_loop <- function(z) {
  tot <- 0
  for (i in seq_along(z)) tot <- tot + z[i]
  tot
}
kahan <- function(z) {
  tot <- 0
  comp <- 0
  for (i in seq_along(z)) {
    y <- z[i] - comp
    t <- tot + y
    comp <- (t - tot) - y
    tot <- t
  }
  tot
}

sc <- data.table(n = ns, loop = NA_real_, rsum = NA_real_, kahan = NA_real_)
for (j in seq_along(ns)) {
  n <- ns[j]
  ex <- as.numeric(py_scale$exact[j, ])
  e_loop <- e_sum <- e_ka <- numeric(REPS)
  for (r in 0:(REPS - 1L)) {
    pre <- s[(r * 20000L + 1L):(r * 20000L + n)]
    e_loop[r + 1L] <- abs(sum_loop(pre) - ex[r + 1L]) / abs(ex[r + 1L])
    e_sum[r + 1L] <- abs(sum(pre) - ex[r + 1L]) / abs(ex[r + 1L])
    e_ka[r + 1L] <- abs(kahan(pre) - ex[r + 1L]) / abs(ex[r + 1L])
  }
  sc[j, `:=`(loop = median(e_loop), rsum = median(e_sum), kahan = median(e_ka))]
}
sc[, py_loop := py_scale$naive_rel]
sc[, py_numpy := py_scale$numpy_rel]
sc[, py_kahan := py_scale$kahan_rel]

t_loop_r <- bench_min(sum_loop(s)) / length(s) * 1e9
t_sum_r <- bench_min(sum(s)) / length(s) * 1e9
t_loop_py <- py_scale$loop_s[length(ns)] / ns[length(ns)] * 1e9
t_np_py <- py_scale$numpy_s[length(ns)] / ns[length(ns)] * 1e9

cancel <- fread("data/cancel.csv")$t
save_metrics("e03_sums", list(
  scale = sc,
  eps = .Machine$double.eps,
  ns_per_element = list(r_loop = t_loop_r, r_sum = t_sum_r,
                        py_loop = t_loop_py, py_numpy = t_np_py),
  full = list(
    n = length(s),
    exact = py$sums$exact,
    r_sum = sum(s),
    r_loop = sum_loop(s),
    r_kahan = kahan(s),
    py_naive = py$sums$naive,
    py_numpy = py$sums$numpy,
    r_sum_rel = abs(sum(s) - py$sums$exact) / abs(py$sums$exact),
    r_loop_rel = abs(sum_loop(s) - py$sums$exact) / abs(py$sums$exact),
    r_kahan_rel = abs(kahan(s) - py$sums$exact) / abs(py$sums$exact),
    py_naive_rel = py$sums$naive_rel,
    py_numpy_rel = py$sums$numpy_rel,
    py_kahan_rel = py$sums$kahan_rel
  ),
  cancel = list(
    n = length(cancel),
    r_sum = sum(cancel),
    r_loop = sum_loop(cancel),
    r_kahan = kahan(cancel),
    exact = py$cancel$exact,
    py_naive = py$cancel$naive,
    py_numpy = py$cancel$numpy,
    py_kahan = py$cancel$kahan
  )
))

# ---- figure: coefficients, and how far apart they really are ---------------
lab_terms <- c("截距 / intercept", "x", "z")
long <- rbind(
  data.frame(term = lab_terms, impl = "R  lm()", est = ols$coef_r, se = ols$se_r),
  data.frame(term = lab_terms, impl = "Python statsmodels", est = ols$coef_py, se = ols$se_py)
)
long$term <- factor(long$term, levels = lab_terms)
long$impl <- factor(long$impl, levels = c("R  lm()", "Python statsmodels"))

p1 <- ggplot(long, aes(est, term)) +
  geom_point(aes(shape = impl, colour = impl, size = impl)) +
  scale_shape_manual(values = c("R  lm()" = 1, "Python statsmodels" = 3), name = NULL) +
  scale_size_manual(values = c("R  lm()" = 4.6, "Python statsmodels" = 1.7), name = NULL) +
  scale_colour_manual(values = c("R  lm()" = "#0072B2",
                                 "Python statsmodels" = "#D55E00"), name = NULL) +
  labs(x = "估计值 / estimate（±1.96 SE 只有 0.004，比点还小）", y = NULL,
       subtitle = "a. 加号落在圆圈正中间 / the plus sits inside the circle") +
  theme_blog(base_size = 10)

diff <- rbind(
  data.frame(q = lab_terms, what = "系数 / coefficient", rel = ols$coef_rel),
  data.frame(q = lab_terms, what = "标准误 / std. error", rel = ols$se_rel),
  data.frame(q = "R²", what = "系数 / coefficient", rel = abs(r2[1] - r2[2]) / r2[1])
)
diff$what <- factor(diff$what, levels = c("系数 / coefficient", "标准误 / std. error"))
diff$draw <- pmax(diff$rel, .Machine$double.eps / 10)

p2 <- ggplot(diff, aes(draw, q)) +
  geom_vline(xintercept = .Machine$double.eps, linetype = "dashed", colour = "grey55") +
  geom_point(size = 2.6, colour = "#009E73") +
  geom_text(aes(label = ifelse(rel == 0, "0", sprintf("%.1e", rel))),
            vjust = -1.1, size = 2.4, colour = "grey25") +
  annotate("text", x = .Machine$double.eps, y = 0.5, label = "eps",
           hjust = -0.3, size = 2.6, colour = "grey45") +
  facet_wrap(~ what, nrow = 1) +
  scale_x_log10(limits = c(1e-17, 1e-12), labels = scales::label_scientific()) +
  labs(x = "|R − Python| / |R|", y = NULL,
       subtitle = "b. 差别在哪里 / where the difference lives") +
  theme_blog(base_size = 10)

fig05 <- p1 + p2 + plot_layout(widths = c(1.05, 1.35))
save_fig(fig05, "fig05_ols", width = 9.6, height = 3.7)

# ---- figure: summation error against n -------------------------------------
floor_err <- .Machine$double.eps / 2
curve <- rbind(
  data.frame(n = rep(ns, 2),
             who = rep(c("R for 循环", "Python for 循环"), each = length(ns)),
             err = c(sc$loop, sc$py_loop)),
  data.frame(n = ns, who = "R sum() / numpy / Kahan",
             err = pmax(c(sc$rsum, sc$py_numpy, sc$py_kahan), floor_err))
)

fig06 <- ggplot(curve, aes(n, err, colour = who, linetype = who)) +
  geom_line(linewidth = .7) +
  geom_point(size = 1.4) +
  annotate("text", x = 3e5, y = .Machine$double.eps * 1.6,
           label = "精确到 1 ulp 以内", hjust = 1, size = 2.8, colour = "grey40") +
  scale_x_log10(labels = scales::label_number()) +
  scale_y_log10(labels = scales::label_scientific()) +
  scale_colour_manual(values = c("R for 循环" = "#0072B2",
                                 "Python for 循环" = "#D55E00",
                                 "R sum() / numpy / Kahan" = "#009E73"), name = NULL) +
  scale_linetype_manual(values = c("R for 循环" = "solid",
                                   "Python for 循环" = "dashed",
                                   "R sum() / numpy / Kahan" = "solid"), name = NULL) +
  labs(x = "n（20 个窗口的中位数 / median of 20 windows）",
       y = "相对误差 / relative error",
       title = "误差只跟算法有关，跟语言无关 / the algorithm decides, not the language") +
  theme_blog(base_size = 10)

save_fig(fig06, "fig06_sums", width = 7.6, height = 4.2)

cat("03_numeric done\n")
