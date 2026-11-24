# Paths, figure export and the single metrics file shared by every script.

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

# Merge one experiment's results into outputs/metrics.json.
save_metrics <- function(key, values) {
  ensure_dir("outputs")
  path <- file.path("outputs", "metrics.json")
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
                       digits = 8, na = "null")
  invisible(path)
}

# Log2 with zeros mapped to NA: MaxQuant writes 0 for "not quantified".
log2_na <- function(x, floor = 1) {
  x[!is.finite(x) | x <= floor] <- NA
  log2(x)
}
