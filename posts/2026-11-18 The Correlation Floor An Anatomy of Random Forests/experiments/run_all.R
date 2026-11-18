# Run every experiment in order and rebuild outputs/metrics.json from scratch.
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

metrics_file <- file.path(out_dir, "metrics.json")
if (file.exists(metrics_file)) file.remove(metrics_file)

scripts <- c(
  "00_timeline.R", "01_boundaries.R", "02_variance_formula.R",
  "03_B_decay.R", "04_bootstrap_oob.R", "05_bias_variance.R",
  "06_grid.R", "07_oob_dependence.R", "08_importance.R",
  "09_calibration.R", "10_extrapolation.R", "11_cost.R",
  "12_benchmark.R", "13_variants.R", "14_cart.R"
)

for (s in scripts) {
  cat("Running", s, "\n")
  source(file.path(exp_dir, s), local = new.env())
}
cat("All experiments finished.\n")
