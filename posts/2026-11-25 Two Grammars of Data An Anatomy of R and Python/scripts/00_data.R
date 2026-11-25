# E00 -- one dataset both languages read, so that every comparison runs on the
# same bits. Written with full precision so R and Python parse identical
# doubles.
# Run from the post directory: Rscript scripts/00_data.R

suppressPackageStartupMessages({
  library(data.table)
})
source("R/io.R")
ensure_dir("data")
ensure_dir("figures")
ensure_dir("outputs")

set.seed(2026)
n <- 200000
grp <- sprintf("g%03d", sample.int(1000, n, replace = TRUE))
cat3 <- sample(c("alpha", "beta", "gamma"), n, replace = TRUE)
x <- rnorm(n)
z <- rnorm(n)
y <- 1 + 2 * x - 1.5 * z + rnorm(n)
val <- exp(0.4 * x) * runif(n, 0.5, 1.5)

dt <- data.table(id = seq_len(n), grp = grp, cat = cat3,
                 x = x, y = y, z = z, val = val)

# fwrite writes the shortest representation that round-trips exactly
fwrite(dt, "data/bench.csv")

# a second file for the summation-order experiment: one realistic column with a
# wide range of magnitudes. Positive only, so the exact sum is well defined.
ns <- 1000000
s1 <- exp(rnorm(ns, 0, 2))
fwrite(data.table(s = s1), "data/sums.csv")

# the classic cancellation case: adding 1 a thousand times to 1e17 loses all of
# them, because the spacing of doubles at 1e17 is 16
cancel <- c(rep(1e16, 10), rep(1, 1000), rep(-1e16, 10))
fwrite(data.table(t = cancel), "data/cancel.csv")

# the 120-row sample both languages plot, so the figures show the same points
set.seed(7)
small <- dt[sort(sample.int(n, 120))]
fwrite(small, "data/small.csv")

save_metrics("e00", list(
  n = n,
  n_groups = length(unique(grp)),
  n_cats = 3,
  bytes = file.size("data/bench.csv"),
  cols = names(dt),
  sums_n = ns,
  small_n = nrow(small)
))

cat("wrote", nrow(dt), "rows to data/bench.csv\n")
