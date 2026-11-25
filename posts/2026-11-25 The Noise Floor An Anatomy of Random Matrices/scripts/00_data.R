# E00 -- load the three real datasets and cache them under data/.
# Run from the post directory: Rscript scripts/00_data.R

suppressPackageStartupMessages({
  library(Matrix)
  library(airway)
})

source("R/io.R")
ensure_dir("data")

# ---- bulk RNA-seq: airway (8 samples, dexamethasone) ------------------------
data(airway, package = "airway")
counts <- SummarizedExperiment::assay(airway, "counts")
meta <- as.data.frame(colData(airway))[, c("cell", "dex")]
# drop genes that are essentially silent so that the spectrum is not dominated
# by a few thousand all-zero rows
keep <- rowSums(counts > 0) >= 4
saveRDS(list(counts = counts[keep, ], meta = meta), "data/airway.rds")
cat("airway:", sum(keep), "genes x", ncol(counts), "samples\n")

# ---- single cell: 10x PBMC 3k ----------------------------------------------
base <- "data/filtered_gene_bc_matrices/hg19"
tar_path <- "data/pbmc3k.tar.gz"
if (!dir.exists(base)) {
  if (!file.exists(tar_path)) {
    message("downloading pbmc3k")
    utils::download.file(
      "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
      tar_path, quiet = TRUE, mode = "wb")
  }
  utils::untar(tar_path, exdir = "data")
}
mtx <- readMM(file.path(base, "matrix.mtx"))
mtx <- as(mtx, "CsparseMatrix")
genes <- read.delim(file.path(base, "genes.tsv"), header = FALSE)
barcodes <- read.delim(file.path(base, "barcodes.tsv"), header = FALSE)
rownames(mtx) <- genes$V2
colnames(mtx) <- barcodes$V1
# genes detected in at least 10 cells and cells with at least 500 counts
mtx <- mtx[rowSums(mtx > 0) >= 10, ]
mtx <- mtx[, colSums(mtx) >= 500]
saveRDS(mtx, "data/pbmc.rds")
cat("pbmc:", nrow(mtx), "genes x", ncol(mtx), "cells\n")

# ---- label-free proteomics: CPTAC spike-in (6 runs) ------------------------
files <- c(proteins = "proteinGroups.txt",
           annotation = "experimental_annotation.txt")
base_url <- "https://raw.githubusercontent.com/statOmics/msqrob2data/main/dda/cptacAvsB_lab3/"
for (f in names(files)) {
  path <- file.path("data", files[[f]])
  if (!file.exists(path)) {
    message("downloading ", files[[f]])
    utils::download.file(paste0(base_url, files[[f]]), path, quiet = TRUE)
  }
}
pg <- read.delim(file.path("data", files[["proteins"]]), check.names = FALSE,
                 stringsAsFactors = FALSE, quote = "")
ann <- read.delim(file.path("data", files[["annotation"]]),
                  check.names = FALSE, stringsAsFactors = FALSE)
runs <- sub("^Intensity[.]", "", ann$quantCols)
intensity <- as.matrix(pg[, paste0("Intensity ", runs)])
colnames(intensity) <- runs
intensity[intensity <= 0] <- NA
clean <- pg$Reverse != "+" & pg$`Potential contaminant` != "+"
intensity <- intensity[clean, ]
# keep protein groups quantified in every run so that the matrix is complete
complete <- rowSums(is.na(intensity)) == 0
cptac <- log2(intensity[complete, ])
rownames(cptac) <- pg$`Majority protein IDs`[clean][complete]
saveRDS(list(x = cptac, group = ifelse(ann$treatment == "A", "A", "B")),
        "data/cptac.rds")
cat("cptac:", nrow(cptac), "proteins x", ncol(cptac), "runs\n")

save_metrics("e00", list(
  airway_genes = nrow(counts[keep, ]),
  airway_samples = ncol(counts),
  pbmc_genes = nrow(mtx),
  pbmc_cells = ncol(mtx),
  cptac_proteins = nrow(cptac),
  cptac_runs = ncol(cptac)
))
