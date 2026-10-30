# =============================================================================
# E00_simulate.R — build the reference data set and check its ground truth
#   Rscript experiments/E00_simulate.R     (run from the post directory)
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")

t0 <- Sys.time()
sim <- simulate_cells()
cat("simulation time:", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")

saveRDS(sim, "results/sim.rds")

# ---- sanity checks ---------------------------------------------------------
X <- logcpm(sim$counts)
tr <- sim$truth
cat("cells x genes:", paste(dim(X), collapse = " x "), "\n")
cat("zero fraction:", round(mean(X == 0), 3), "\n")
cat("median library size:", median(Matrix::rowSums(sim$counts)), "\n")

pc <- pca_svd(X, d = 5, scale_gene = TRUE)
cat("variance explained:", paste(round(pc$var_explained[1:5], 3), collapse = " "), "\n")
for (k in 1:5) {
  r <- c(cor(pc$scores[, k], tr$t, method = "spearman"),
         cor(pc$scores[, k], tr$a, method = "spearman"),
         cor(pc$scores[, k], tr$s_branch, method = "spearman"))
  cat(sprintf("PC%d  rho(t)=% .2f  rho(a)=% .2f  rho(branch)=% .2f\n", k, r[1], r[2], r[3]))
}

# batch contamination in PC1/PC2
cat("batch R2 on PC1-3:",
    round(summary(lm(pc$scores[, 1] ~ tr$batch))$r.squared, 3), "\n")

# module separation: does the simulated signal separate modules?
mk <- function(idx) rowMeans(X[, idx, drop = FALSE])
cat("cor(mean mono genes, t) =", round(cor(mk(sim$modules$mono), tr$t, method = "spearman"), 3), "\n")
cat("cor(mean branchA genes, s_branch) =",
    round(cor(mk(sim$modules$branchA), tr$s_branch, method = "spearman"), 3), "\n")
