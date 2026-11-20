# E06: single-cell size factors versus library size, and the effect on
# clustering. Real PBMC data plus a simulation with known cell types.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})
source("../../theme-blog.R")
source("R/norm.R")
source("R/sim.R")
source("R/metrics.R")
source("R/io.R")

pca_df <- function(m, label, lib) {
  z <- t(scale(t(m)))
  z[!is.finite(z)] <- 0
  pc <- stats::prcomp(t(z))
  data.frame(PC1 = pc$x[, 1], PC2 = pc$x[, 2], method = label,
             lib = log10(lib))
}

cluster_ari <- function(m, k, truth = NULL) {
  z <- scale(t(m))
  z[!is.finite(z)] <- 0
  cl <- stats::kmeans(z, centers = k, nstart = 10)$cluster
  if (is.null(truth)) cl else ari(cl, truth)
}

# ---------------------------------------------------------------- real PBMC
pbmc <- as.matrix(readRDS("data/pbmc.rds"))
pbmc <- pbmc[rowSums(pbmc > 0) >= 10, ]
set.seed(1)
pbmc <- pbmc[, sample(ncol(pbmc), 400)]
lib <- colSums(pbmc)
cp <- log2(cpm(pbmc) + 1)
v <- apply(cp, 1, stats::var)
top <- order(v, decreasing = TRUE)[seq_len(1500)]
sub <- pbmc[top, , drop = FALSE]

sf_scran <- scran_factors(sub)
sct <- sctransform_norm(sub)
cpm_sub <- log2(cpm(sub) + 1)

real_pca <- bind_rows(
  pca_df(cpm_sub, "CPM", lib),
  pca_df(log2(sweep(sub, 2, sf_scran, "/") + 1), "scran", lib),
  pca_df(sct, "sctransform", lib)
)

p_pca <- ggplot(real_pca, aes(PC1, PC2, colour = lib)) +
  geom_point(size = 0.8) +
  facet_wrap(~ method, nrow = 1, scales = "free") +
  scale_colour_gradient(low = blog_pal[["sky"]], high = blog_pal[["vermillion"]]) +
  labs(colour = "log10 library", x = "PC1", y = "PC2")

p_sf <- ggplot(data.frame(lib = lib, sf = sf_scran), aes(lib, sf)) +
  geom_point(size = 0.8, colour = blog_pal[["blue"]], alpha = 0.6) +
  scale_x_log10() + scale_y_log10() +
  labs(x = "library size", y = "scran size factor")

cl_real <- list(
  CPM = cluster_ari(cpm_sub, 5),
  scran = cluster_ari(log2(sweep(sub, 2, sf_scran, "/") + 1), 5),
  sct = cluster_ari(sct, 5)
)
ari_real <- c(CPM_scran = ari(cl_real$CPM, cl_real$scran),
              CPM_sct = ari(cl_real$CPM, cl_real$sct),
              scran_sct = ari(cl_real$scran, cl_real$sct))

# ------------------------------------------------------- simulated truth
sc <- sim_sc(n_cells = 600, n_genes = 1200, n_types = 3, seed = 6)
cnt <- sc$counts
lib_s <- colSums(cnt)
cp_s <- log2(cpm(cnt) + 1)
sf_s <- scran_factors(cnt)
sct_s <- sctransform_norm(cnt)

sim_ari <- c(
  CPM = cluster_ari(cp_s, 3, sc$type),
  scran = cluster_ari(log2(sweep(cnt, 2, sf_s, "/") + 1), 3, sc$type),
  sctransform = cluster_ari(sct_s, 3, sc$type)
)

p_ari <- ggplot(data.frame(method = names(sim_ari), ari = sim_ari),
                aes(reorder(method, ari), ari)) +
  geom_col(fill = blog_pal[["green"]], width = 0.6) +
  coord_flip() +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = "ARI against known cell types")

fig <- (p_sf | p_ari) / p_pca + plot_layout(heights = c(1, 1.2))
save_fig(fig, "fig11_singlecell", width = 10, height = 6.4)

save_metrics("e06", list(
  n_cells = ncol(pbmc), n_genes = nrow(pbmc),
  sf_lib_cor = stats::cor(log(lib), log(sf_scran)),
  ari_real = ari_real,
  ari_sim = sim_ari
))
