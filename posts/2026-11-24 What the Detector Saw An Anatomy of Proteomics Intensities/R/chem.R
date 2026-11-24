# Peptide chemistry: elemental composition, isotope envelope, GRAVY,
# and the in-silico tryptic digest that defines the iBAQ denominator.

# Residue formulas (free amino acid minus H2O): C, H, N, O, S
aa_formula <- list(
  A = c(C = 3, H = 5, N = 1, O = 1, S = 0),
  R = c(C = 6, H = 12, N = 4, O = 1, S = 0),
  N = c(C = 4, H = 6, N = 2, O = 2, S = 0),
  D = c(C = 4, H = 5, N = 1, O = 3, S = 0),
  C = c(C = 3, H = 5, N = 1, O = 1, S = 1),
  E = c(C = 5, H = 7, N = 1, O = 3, S = 0),
  Q = c(C = 5, H = 8, N = 2, O = 2, S = 0),
  G = c(C = 2, H = 3, N = 1, O = 1, S = 0),
  H = c(C = 6, H = 7, N = 3, O = 1, S = 0),
  I = c(C = 6, H = 11, N = 1, O = 1, S = 0),
  L = c(C = 6, H = 11, N = 1, O = 1, S = 0),
  K = c(C = 6, H = 12, N = 2, O = 1, S = 0),
  M = c(C = 5, H = 9, N = 1, O = 1, S = 1),
  F = c(C = 9, H = 9, N = 1, O = 1, S = 0),
  P = c(C = 5, H = 7, N = 1, O = 1, S = 0),
  S = c(C = 3, H = 5, N = 1, O = 2, S = 0),
  T = c(C = 4, H = 7, N = 1, O = 2, S = 0),
  W = c(C = 11, H = 10, N = 2, O = 1, S = 0),
  Y = c(C = 9, H = 9, N = 1, O = 2, S = 0),
  V = c(C = 5, H = 9, N = 1, O = 1, S = 0)
)

mono_mass <- c(C = 12, H = 1.007825032, N = 14.003074005,
               O = 15.994914620, S = 31.972071174)

# Kyte-Doolittle hydropathy
kd_scale <- c(
  A = 1.8, R = -4.5, N = -3.5, D = -3.5, C = 2.5, Q = -3.5, E = -3.5,
  G = -0.4, H = -3.2, I = 4.5, L = 3.8, K = -3.9, M = 1.9, F = 2.8,
  P = -1.6, S = -0.8, T = -0.7, W = -0.9, Y = -1.3, V = 4.2
)

peptide_formula <- function(seq) {
  aa <- strsplit(toupper(gsub("[^A-Z]", "", seq)), "")[[1]]
  aa <- aa[aa %in% names(aa_formula)]
  comp <- colSums(do.call(rbind, aa_formula[aa]))
  comp["H"] <- comp["H"] + 2   # two termini
  comp["O"] <- comp["O"] + 1   # C-terminal OH
  comp
}

peptide_mass <- function(seq) {
  comp <- peptide_formula(seq)
  sum(comp * mono_mass[names(comp)])
}

# Convolve two distributions given as data frames of (offset, probability).
.conv_sparse <- function(a, b) {
  out <- numeric(max(a$offset) + max(b$offset) + 1)
  for (i in seq_along(a$offset)) {
    idx <- a$offset[i] + b$offset + 1
    out[idx] <- out[idx] + a$p[i] * b$p
  }
  data.frame(offset = seq_along(out) - 1, p = out)
}

.binom_sparse <- function(n, p, shift) {
  k <- 0:n
  data.frame(offset = k * shift, p = dbinom(k, n, p))
}

# Isotope envelope up to `n_peaks` nominal mass units.
isotope_envelope <- function(seq, n_peaks = 8) {
  comp <- peptide_formula(seq)
  dist <- data.frame(offset = 0, p = 1)
  for (el in names(comp)) {
    n <- comp[[el]]
    if (n <= 0) next
    sub <- switch(
      el,
      C = .binom_sparse(n, 0.0107, 1),
      H = .binom_sparse(n, 0.000115, 1),
      N = .binom_sparse(n, 0.00364, 1),
      O = data.frame(
        offset = c(0, 1, 2),
        p = c(1 - 0.00038 - 0.00205, 0.00038, 0.00205)
      ),
      S = data.frame(
        offset = c(0, 1, 2),
        p = c(1 - 0.0075 - 0.0425, 0.0075, 0.0425)
      ),
      data.frame(offset = 0, p = 1)
    )
    if (el %in% c("O", "S")) {
      for (i in seq_len(n)) dist <- .conv_sparse(dist, sub)
    } else {
      dist <- .conv_sparse(dist, sub)
    }
  }
  dist <- dist[dist$offset < n_peaks, ]
  dist$p <- dist$p / sum(dist$p)
  dist
}

gravy <- function(seq) {
  aa <- strsplit(toupper(seq), "")[[1]]
  mean(kd_scale[aa], na.rm = TRUE)
}

# Trypsin: cut C-terminal to K/R unless the next residue is P.
tryptic_peptides <- function(seq, min_len = 6, max_len = 30, missed = 0) {
  s <- toupper(gsub("[^A-Z]", "", seq))
  chars <- strsplit(s, "")[[1]]
  n <- length(chars)
  cuts <- integer(0)
  if (n > 1) {
    for (i in seq_len(n - 1)) {
      if (chars[i] %in% c("K", "R") && chars[i + 1] != "P") cuts <- c(cuts, i)
    }
  }
  cuts <- sort(unique(c(0, cuts, n)))
  starts <- cuts[-length(cuts)] + 1
  ends <- cuts[-1]
  peps <- substring(s, starts, ends)
  peps <- peps[nchar(peps) >= min_len & nchar(peps) <= max_len]
  if (missed > 0 && length(cuts) > 2) {
    for (k in seq_len(missed)) {
      if (length(starts) > k) {
        extra <- substring(s, starts[seq_len(length(starts) - k)], ends[(k + 1):length(ends)])
        peps <- c(peps, extra[nchar(extra) >= min_len & nchar(extra) <= max_len])
      }
    }
  }
  unique(peps)
}

n_observable <- function(seq, min_len = 6, max_len = 30) {
  length(tryptic_peptides(seq, min_len, max_len))
}

# "sp|P02768|ALBU_HUMAN" / "P02768ups|ALBU_HUMAN_UPS" -> "P02768"
parse_accession <- function(x) {
  vapply(strsplit(x, "\\|"), function(p) {
    p <- gsub("^CON__", "", p[!p %in% c("sp", "tr", "CON__")])
    if (!length(p)) return(NA_character_)
    sub("ups$", "", sub("-[0-9]+$", "", p[1]))
  }, character(1))
}
