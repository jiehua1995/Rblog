# E00 -- download and cache the real CPTAC spike-in MaxQuant output,
# attach theoretical peptide counts for every protein group.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})
source("R/io.R")
source("R/chem.R")

dir_data <- "data"
ensure_dir(dir_data)

files <- c(
  peptides = "peptides.txt",
  proteins = "proteinGroups.txt",
  annotation = "experimental_annotation.txt"
)
base_url <- "https://raw.githubusercontent.com/statOmics/msqrob2data/main/dda/cptacAvsB_lab3/"

for (f in names(files)) {
  path <- file.path(dir_data, files[[f]])
  if (!file.exists(path)) {
    message("downloading ", files[[f]])
    utils::download.file(paste0(base_url, files[[f]]), path, quiet = TRUE)
  }
}

pg <- read.delim(file.path(dir_data, files[["proteins"]]),
                 check.names = FALSE, stringsAsFactors = FALSE, quote = "")
pep <- read.delim(file.path(dir_data, files[["peptides"]]),
                  check.names = FALSE, stringsAsFactors = FALSE, quote = "")
ann <- read.delim(file.path(dir_data, files[["annotation"]]),
                  check.names = FALSE, stringsAsFactors = FALSE)

runs <- sub("^Intensity[.]", "", ann$quantCols)
sample_ids <- ann$sampleId
group <- ifelse(ann$treatment == "A", 1L, 2L)
int_cols <- paste0("Intensity ", runs)
lfq_cols <- paste0("LFQ intensity ", runs)
ms_cols <- paste0("MS/MS Count ", runs)

# ---- protein table -------------------------------------------------------
acc_from_header <- function(h) {
  h <- gsub('^"|"$', "", h)
  first <- strsplit(h, ";", fixed = TRUE)[[1]][1]
  tok <- sub("^>*\\s*", "", first)
  tok <- sub("\\s.*$", "", tok)
  if (grepl("\\|", tok)) {
    parts <- strsplit(tok, "\\|", fixed = FALSE)[[1]]
    tok <- if (parts[1] %in% c("sp", "tr")) parts[2] else parts[1]
  }
  tok <- sub("^SWISS-PROT:", "", tok)
  sub("ups$", "", sub("-[0-9]+$", "", tok))
}
first_acc <- vapply(strsplit(pg$`Fasta headers`, ";", fixed = TRUE),
                    function(x) acc_from_header(x[1]), character(1))

prot <- data.frame(
  id = pg$`Majority protein IDs`,
  acc = first_acc,
  length = pg$`Sequence length`,
  mw_kda = pg$`Mol. weight [kDa]`,
  n_pep = pg$`Razor + unique peptides`,
  msms = pg$`MS/MS Count`,
  ups = grepl("UPS", pg$`Fasta headers`, ignore.case = TRUE),
  reverse = pg$Reverse == "+",
  contaminant = pg$`Potential contaminant` == "+",
  stringsAsFactors = FALSE
)
prot$intensity <- as.matrix(pg[, int_cols])
prot$lfq <- as.matrix(pg[, lfq_cols])
prot$msms_run <- as.matrix(pg[, ms_cols])
colnames(prot$intensity) <- colnames(prot$lfq) <- colnames(prot$msms_run) <- runs
prot$intensity[prot$intensity <= 0] <- NA
prot$lfq[prot$lfq <= 0] <- NA

# ---- peptide table -------------------------------------------------------
pep2 <- data.frame(
  sequence = pep$Sequence,
  proteins = pep$Proteins,
  leading = pep$`Leading razor protein`,
  length = pep$Length,
  missed = pep$`Missed cleavages`,
  msms = pep$`MS/MS Count`,
  stringsAsFactors = FALSE
)
pep2$intensity <- as.matrix(pep[, int_cols])
colnames(pep2$intensity) <- runs
pep2$intensity[pep2$intensity <= 0] <- NA
pep2 <- pep2[pep$Reverse != "+" & pep$`Potential contaminant` != "+", ]

# ---- sequences and theoretical peptide counts ---------------------------
fasta_path <- file.path(dir_data, "uniprot.fasta")
accs <- unique(prot$acc[!prot$reverse & prot$acc != ""])
accs <- accs[grepl("^[A-Z0-9]{6,10}$", accs)]

if (!file.exists(fasta_path)) {
  message("fetching ", length(accs), " UniProt sequences in batches")
  chunks <- split(accs, ceiling(seq_along(accs) / 40))
  lines <- character()
  for (i in seq_along(chunks)) {
    url <- paste0("https://rest.uniprot.org/uniprotkb/accessions?accessions=",
                  paste(chunks[[i]], collapse = ","), "&format=fasta")
    txt <- tryCatch(readLines(url, warn = FALSE), error = function(e) character())
    lines <- c(lines, txt)
    Sys.sleep(0.3)
  }
  writeLines(lines, fasta_path)
}
fasta <- readLines(fasta_path)
hdr <- grep("^>", fasta)
seqs <- vapply(seq_along(hdr), function(i) {
  end <- if (i < length(hdr)) hdr[i + 1] - 1 else length(fasta)
  paste(fasta[(hdr[i] + 1):end], collapse = "")
}, character(1))
names(seqs) <- sub("^>([^ ]+).*$", "\\1", fasta[hdr])
nm <- names(seqs)
plain <- ifelse(grepl("\\|", nm),
                sub("^[a-z]+\\|([^|]+)\\|.*$", "\\1", nm),
                sub("\\s.*$", "", nm))
names(seqs) <- sub("-[0-9]+$", "", plain)

seq_lookup <- setNames(seqs, names(seqs))
n_obs <- vapply(prot$acc, function(a) {
  if (is.na(a) || !a %in% names(seq_lookup)) return(NA_real_)
  n_observable(seq_lookup[[a]])
}, numeric(1))
prot$n_obs <- n_obs

# ---- descriptives --------------------------------------------------------
clean <- !prot$reverse & !prot$contaminant
pep_mat <- log2(pep2$intensity)
prot_mat <- log2(prot$intensity[clean, ])

saveRDS(list(prot = prot, pep = pep2, runs = runs, group = group,
             seqs = seqs, annotation = ann),
        file.path(ensure_dir("outputs"), "cptac.rds"))

save_metrics("e00", list(
  n_protein_groups = nrow(prot),
  n_protein_clean = sum(clean),
  n_peptides = nrow(pep2),
  n_runs = length(runs),
  n_ups = sum(prot$ups & clean),
  expected_lfc = log2(0.74 / 0.25),
  pep_log2_range = as.numeric(diff(range(pep_mat, na.rm = TRUE))),
  pep_missing = mean(is.na(pep2$intensity)),
  prot_missing_raw = mean(is.na(prot$intensity[clean, ])),
  prot_missing_lfq = mean(is.na(prot$lfq[clean, ])),
  pep_per_protein_median = median(prot$n_pep[clean]),
  seq_found = sum(!is.na(prot$n_obs)),
  n_obs_median = median(prot$n_obs, na.rm = TRUE),
  n_obs_range = range(prot$n_obs, na.rm = TRUE)
))

message("E00 done: ", sum(clean), " proteins, ", nrow(pep2), " peptides")
