# Load the four real datasets once and cache them under data/.
# Run from the post directory: Rscript scripts/00_data.R

suppressPackageStartupMessages({
  library(Matrix)
  library(airway)
  library(pRolocdata)
  library(CLL)
})

dir.create("data", showWarnings = FALSE)

# ---- bulk RNA-seq: airway (paired dexamethasone experiment) -----------------
data(airway, package = "airway")
counts <- assay(airway, "counts")
lengths <- vapply(width(rowRanges(airway)), sum, numeric(1))
meta <- as.data.frame(colData(airway))[, c("cell", "dex")]
saveRDS(list(counts = counts, length = lengths, meta = meta),
        "data/airway.rds")
cat("airway:", dim(counts), "\n")

# ---- label-free proteomics: NormalyzerDE peptide dilution series -----------
# 100 peptides x 24 runs, nine concentrations in triplicate, two batches.
p <- system.file("extdata", package = "NormalyzerDE")
pep <- read.delim(file.path(p, "data.tsv"), check.names = FALSE)
design <- read.delim(file.path(p, "design.tsv"))
cols <- match(design$sample, colnames(pep))
stopifnot(!anyNA(cols))
lfq <- as.matrix(pep[, cols])
rownames(lfq) <- pep[["Peptide Sequence"]]
saveRDS(list(x = lfq, design = design), "data/lfq.rds")
cat("LFQ:", dim(lfq), "\n")

# ---- single cell: 10x PBMC 3k ----------------------------------------------
base <- "data/filtered_gene_bc_matrices/hg19"
if (dir.exists(base)) {
  mtx <- readMM(file.path(base, "matrix.mtx"))
  genes <- read.delim(file.path(base, "genes.tsv"), header = FALSE)
  barcodes <- read.delim(file.path(base, "barcodes.tsv"), header = FALSE)
  rownames(mtx) <- genes$V2
  colnames(mtx) <- barcodes$V1
  # keep genes detected in at least 10 cells
  mtx <- mtx[rowSums(mtx > 0) >= 10, ]
  # keep 600 cells with at least 500 counts, sampled for a small CPU footprint
  lib <- colSums(mtx)
  keep <- which(lib >= 500)
  set.seed(1)
  keep <- sample(keep, min(600, length(keep)))
  mtx <- mtx[, keep]
  saveRDS(mtx, "data/pbmc.rds")
  cat("pbmc:", dim(mtx), "\n")
}

# ---- microarray: CLL batch --------------------------------------------------
data(CLLbatch, package = "CLL")
saveRDS(list(expr = exprs(CLLbatch), meta = pData(CLLbatch)),
        "data/cll.rds")
cat("CLL:", dim(exprs(CLLbatch)), "\n")

# ---- metabolomics: sacurine (183 urine samples x 109 metabolites) ----------
suppressPackageStartupMessages(library(ropls))
data(sacurine, package = "ropls")
metab <- list(x = t(sacurine$dataMatrix), meta = sacurine$sampleMetadata)
saveRDS(metab, "data/metab.rds")
cat("metabolomics:", dim(metab$x), "\n")
