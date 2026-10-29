# What does this machine have? Run from the repository root:
#   Rscript --no-init-file .trae/skills/rblog-new-post/scripts/probe-env.R
# --no-init-file skips .Rprofile, which references renv/activate.R on a fresh clone.

ver <- function(p) {
  d <- tryCatch(utils::packageDescription(p), error = function(e) NA)
  if (length(d) == 1 && is.na(d)) "MISSING" else as.character(d$Version)
}

cat("--- R ---\n")
cat("R:        ", R.version.string, "\n", sep = "")
cat("Rscript:  ", file.path(R.home("bin"), "Rscript"), "\n", sep = "")
cat("libPaths: ", paste(.libPaths(), collapse = ", "), "\n", sep = "")

tool <- function(cmd, args = "--version") {
  path <- Sys.which(cmd)
  if (!nzchar(path)) return("MISSING")
  out <- tryCatch(system2(path, args, stdout = TRUE, stderr = TRUE),
                  error = function(e) conditionMessage(e))
  paste0(path, " (", trimws(paste(out, collapse = " ")), ")")
}

cat("\n--- tools ---\n")
cat("quarto:   ", tool("quarto"), "\n", sep = "")
cat("git:      ", tool("git"), "\n", sep = "")
cat("bash:     ", Sys.which("bash"), "\n", sep = "")

cat("\n--- R packages ---\n")
for (p in c("knitr", "rmarkdown", "ggplot2", "scales", "ragg", "dplyr", "tidyr",
            "reticulate", "torch", "jsonlite")) {
  cat(sprintf("%-12s %s\n", p, ver(p)))
}

cat("\n--- python ---\n")
py <- Sys.getenv("RETICULATE_PYTHON")
cat("RETICULATE_PYTHON: ",
    if (nzchar(py)) py else "(unset, reticulate will auto-discover)", "\n", sep = "")

for (cmd in c("python", "python3", "py")) {
  path <- Sys.which(cmd)
  cat(sprintf("%-18s %s\n", cmd, if (nzchar(path)) path else "-"))
}

cand <- if (nzchar(py)) py else Sys.which("python")
if (nzchar(cand)) {
  code <- "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
  res <- tryCatch(system2(cand, c("-c", shQuote(code)), stdout = TRUE, stderr = TRUE),
                  error = function(e) structure(conditionMessage(e), status = 1L))
  status <- attr(res, "status")
  cat("torch/numpy:      ",
      if (is.null(status)) res[1] else "not importable in this interpreter",
      "\n", sep = "")
}

cat("\n--- notes ---\n")
if (!nzchar(Sys.which("quarto"))) cat("- install the Quarto CLI before rendering\n")
if (identical(ver("torch"), "MISSING")) cat("- R torch is optional; install with install.packages(\"torch\") then torch::install_torch()\n")
cat("- machine-specific settings belong in .Renviron or .Rprofile, never in a post\n")
