# Small helpers for paths, figures, the shared metrics file and timing.

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  invisible(path)
}

save_fig <- function(plot, name, width = 8, height = 4.4) {
  ensure_dir("figures")
  ggplot2::ggsave(file.path("figures", paste0(name, ".png")), plot,
                  width = width, height = height, dpi = 220,
                  bg = "transparent")
  invisible(plot)
}

# fastest of `times` runs, in seconds. microbenchmark times in nanoseconds, so
# this resolves microseconds where proc.time() would not. The expression is
# captured with substitute() and handed over as a list, otherwise microbenchmark
# would re-evaluate an already-forced promise and report zero.
bench_min <- function(expr, times = 5) {
  e <- substitute(expr)
  t <- microbenchmark::microbenchmark(list = list(e), times = times)
  min(as.numeric(t$time)) / 1e9
}

save_metrics <- function(key, values, file = "metrics.json") {
  ensure_dir("outputs")
  path <- file.path("outputs", file)
  all <- if (file.exists(path)) {
    jsonlite::fromJSON(path, simplifyVector = FALSE)
  } else {
    list()
  }
  clean <- function(x) {
    if (is.data.frame(x)) {
      x[] <- lapply(x, clean)
      return(x)
    }
    if (is.list(x)) return(lapply(x, clean))
    if (is.numeric(x)) {
      x[!is.finite(x)] <- NA
      return(x)
    }
    x
  }
  all[[key]] <- clean(values)
  jsonlite::write_json(all, path, auto_unbox = TRUE, pretty = TRUE,
                       digits = 10, na = "null")
  invisible(path)
}
