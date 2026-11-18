# Shared setup for every experiment script in this post.
# Resolves paths relative to the script itself so the post renders from any machine.

.this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
exp_dir <- dirname(.this_file)
post_dir <- dirname(exp_dir)
repo_root <- normalizePath(file.path(post_dir, "..", ".."))
fig_dir <- file.path(post_dir, "figures")
out_dir <- file.path(exp_dir, "outputs")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

source(file.path(repo_root, "theme-blog.R"))

suppressPackageStartupMessages({
  library(ranger)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

SEED <- 20261118
set.seed(SEED)

save_fig <- function(p, name, width = 8, height = 4.4, dpi = 200) {
  ggsave(file.path(fig_dir, paste0(name, ".png")), p,
         width = width, height = height, dpi = dpi, bg = "transparent")
  ggsave(file.path(fig_dir, paste0(name, ".svg")), p,
         width = width, height = height, bg = "transparent",
         device = svglite::svglite)
}

write_metrics <- function(x, key) {
  f <- file.path(out_dir, "metrics.json")
  all <- if (file.exists(f)) fromJSON(f, simplifyVector = FALSE) else list()
  all[[key]] <- x
  write_json(all, f, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

# Friedman #1 regression surface, x in [0, 1]^p
friedman_data <- function(n, p = 10, noise = 1) {
  X <- matrix(runif(n * p), n, p)
  colnames(X) <- paste0("x", seq_len(p))
  f <- friedman_truth(X)
  list(X = X, y = f + rnorm(n, 0, noise), f = f)
}

friedman_truth <- function(X) {
  10 * sin(pi * X[, 1] * X[, 2]) +
    20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] +
    5 * X[, 5]
}
