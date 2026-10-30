# =============================================================================
# E08_real_data.R — repetition of the key analyses on public 10x PBMC data
#
# Data: 10x Genomics, "3k PBMCs from a Healthy Donor" (v1 chemistry), filtered
# gene-barcode matrix.  Downloaded once into data/ and cached.
# There is no ground truth here: the labels below come from marker scores and
# k-means, and every statement in the article about them says so.
# =============================================================================
source("R/00_setup.R")
source("R/02_preprocessing.R")
suppressPackageStartupMessages({library(uwot); library(igraph)})

url <- "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
tar_file <- "data/pbmc3k_filtered_gene_bc_matrices.tar.gz"
cache <- "data/pbmc3k_counts.rds"

if (!file.exists(cache)) {
  if (!file.exists(tar_file)) {
    cat("downloading", url, "\n")
    utils::download.file(url, tar_file, mode = "wb", quiet = TRUE)
  }
  if (!dir.exists(file.path("data", "filtered_gene_bc_matrices"))) {
    utils::untar(tar_file, exdir = "data")
  }
  base <- file.path("data", "filtered_gene_bc_matrices", "hg19")
  mtx <- Matrix::readMM(file.path(base, "matrix.mtx"))
  features <- read.delim(file.path(base, "genes.tsv"), header = FALSE,
                         stringsAsFactors = FALSE)
  barcodes <- readLines(file.path(base, "barcodes.tsv"))
  counts <- methods::as(mtx, "CsparseMatrix")      # genes x cells
  counts <- methods::as(Matrix::t(counts), "CsparseMatrix")   # -> cells x genes
  colnames(counts) <- features$V2
  rownames(counts) <- barcodes
  counts <- counts[, !duplicated(colnames(counts))]
  saveRDS(counts, cache)
}
counts <- readRDS(cache)
cat("PBMC matrix:", paste(dim(counts), collapse = " x "), "(cells x genes)\n")

# ---- QC --------------------------------------------------------------------
mito <- grepl("^MT-", colnames(counts))
qc <- data.frame(
  n_gene = Matrix::rowSums(counts > 0),
  n_count = Matrix::rowSums(counts),
  mito_frac = Matrix::rowSums(counts[, mito, drop = FALSE]) / Matrix::rowSums(counts)
)
keep <- qc$n_gene >= 200 & qc$mito_frac < 0.1
counts <- counts[keep, ]
qc <- qc[keep, ]
cat(sprintf("after QC: %d cells, median %d genes/cell, median mito %.1f%%\n",
            nrow(counts), median(qc$n_gene), 100 * median(qc$mito_frac)))

counts <- counts[, Matrix::colSums(counts > 0) >= 3]

# ---- normalisation and embedding ------------------------------------------
L <- logcpm(counts)                     # cells x genes, sparse
hvg <- top_hvg(L, 1000)
X <- zscore(as.matrix(L[, hvg]))
pca <- pca_svd(X, d = 30)
pc <- pca$scores
cat("variance explained PC1-10:", round(sum(pca$var_explained[1:10]), 3),
    " PC1:", round(pca$var_explained[1], 3), "\n")

set.seed(42)
um <- uwot::umap(pc, n_neighbors = 15, min_dist = 0.3, seed = 42,
                 n_sgd_threads = 1, verbose = FALSE)
dm <- dmaps(pc, k = 15, ndim = 3)$embedding

# ---- marker-based annotation ----------------------------------------------
markers <- list(
  "CD4 T"       = c("IL7R", "CD3D", "CD3E", "LTB"),
  "CD8 T"       = c("CD8A", "CD8B", "CCL5", "GZMK"),
  "B"           = c("MS4A1", "CD79A", "CD79B", "TCL1A"),
  "NK"          = c("GNLY", "NKG7", "GZMB", "KLRD1"),
  "CD14 Mono"   = c("LYZ", "CD14", "S100A8", "S100A9"),
  "FCGR3A Mono" = c("FCGR3A", "MS4A7", "LILRB1", "CDKN1C"),
  "DC"          = c("FCER1A", "CST3", "CLEC10A"),
  "Platelet"    = c("PPBP", "PF4")
)
markers <- lapply(markers, function(m) intersect(m, colnames(counts)))
mg <- unique(unlist(markers))
Lm <- as.matrix(logcpm(counts[, mg, drop = FALSE]))
score <- sapply(markers, function(m) rowMeans(zscore(Lm[, m, drop = FALSE])))

km <- kmeans(pc[, 1:10], centers = 10, nstart = 20, iter.max = 100)
cluster <- factor(km$cluster)
cluster_label <- sapply(split(seq_len(nrow(score)), cluster), function(ii) {
  names(which.max(colMeans(score[ii, , drop = FALSE])))
})
lab_raw <- cluster_label[as.character(cluster)]
# two clusters may resolve to the same cell type; keep them distinct in the plot
dup <- duplicated(cluster_label)
nm <- cluster_label
nm[dup] <- paste0(nm[dup], " (2)")
label <- factor(nm[as.character(cluster)])
cat("cluster sizes:\n"); print(table(label))

score_tab <- as.data.frame(round(aggregate(score, by = list(cluster = cluster), mean)[, -1], 2))
score_tab <- cbind(cluster = levels(cluster), score_tab)
print(score_tab, row.names = FALSE)

# ---- does the embedding keep distances? -----------------------------------
set.seed(3)
si <- sample(nrow(pc), 500)
d_pca <- as.vector(as.matrix(dist(pc[si, 1:30])))
d_um  <- as.vector(as.matrix(dist(um[si, ])))
d_dm  <- as.vector(as.matrix(dist(dm[si, ])))
dist_keep <- c(umap = cor(d_pca, d_um, method = "spearman"),
               dmaps = cor(d_pca, d_dm, method = "spearman"))

cent_pca <- aggregate(pc[, 1:30], by = list(label), mean)
cent_um  <- aggregate(um, by = list(label), mean)
cent_dm  <- aggregate(dm, by = list(label), mean)
c_pca <- as.vector(as.matrix(dist(cent_pca[, -1])))
c_um  <- as.vector(as.matrix(dist(cent_um[, -1])))
c_dm  <- as.vector(as.matrix(dist(cent_dm[, -1])))
centroid_keep <- c(umap = cor(c_pca, c_um, method = "spearman"),
                   dmaps = cor(c_pca, c_dm, method = "spearman"))

purity <- knn_purity(pc[, 1:10], as.character(label), k = 15)
cat("kNN label purity in PC space:", round(purity, 3), "\n")
cat("distance preservation:", round(dist_keep, 3), "\n")
cat("centroid distance preservation:", round(centroid_keep, 3), "\n")

saveRDS(list(umap = um, dmaps = dm, pcs = pc[, 1:10], label = label,
             score = score, qc = qc, dist_keep = dist_keep,
             centroid_keep = centroid_keep, purity = purity,
             id_est = intrinsic_dim_mle(X, k = 10),
             n_cells = ncol(counts), var_explained = pca$var_explained,
             markers = markers, d_pca = d_pca, d_um = d_um),
        "results/e08_real.rds")
cat("saved results/e08_real.rds\n")
