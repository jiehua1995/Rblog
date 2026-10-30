# =============================================================================
# 01_ca_models.R — cellular automata on a torus.
#
# Two minimal examples:
#   * Conway's Game of Life (2-D, 2 states, Q1...Q8 neighbourhood)
#   * elementary cellular automata (1-D, 2 states, radius-1 neighbourhood)
#
# Everything is written with integer matrix shifts so a 120 x 120 lattice runs
# fast enough to do in a few seconds inside a blog post.
# =============================================================================

# ---- Game of Life ----------------------------------------------------------
# Shift a matrix by (dr, dc) on a torus.  Positive dr moves rows downwards.
shift_torus <- function(m, dr, dc) {
  n <- nrow(m); p <- ncol(m)
  idx_r <- ((seq_len(n) - 1 + dr) %% n) + 1
  idx_c <- ((seq_len(p) - 1 + dc) %% p) + 1
  m[idx_r, idx_c, drop = FALSE]
}

# Discrete 5-point Laplacian with periodic boundaries, used by the
# reaction-diffusion models in 03_cell_communication.R.
laplacian_periodic <- function(f) {
  shift_torus(f, 1, 0) + shift_torus(f, -1, 0) +
    shift_torus(f, 0, 1) + shift_torus(f, 0, -1) - 4 * f
}

count_neighbours <- function(m) {
  nb <- matrix(0, nrow(m), ncol(m))
  for (dr in c(-1, 0, 1)) {
    for (dc in c(-1, 0, 1)) {
      if (dr == 0 && dc == 0) next
      nb <- nb + shift_torus(m, dr, dc)
    }
  }
  nb
}

# The whole rule is three lines: birth on 3, survival on 2 or 3, death otherwise.
gol_step <- function(m) {
  nb <- count_neighbours(m)
  ((m == 1) & (nb == 2 | nb == 3)) | ((m == 0) & (nb == 3))
}

gol_simulate <- function(m0, steps = 100, keep_every = 1) {
  m <- m0 * 1
  out <- vector("list", steps %/% keep_every + 1)
  out[[1]] <- m
  j <- 1
  for (t in seq_len(steps)) {
    m <- gol_step(m)
    if (t %% keep_every == 0) {
      j <- j + 1
      out[[j]] <- m
    }
  }
  out[seq_len(j)]
}

gol_random <- function(n = 120, p = 0.28, seed = 1) {
  set.seed(seed)
  matrix(runif(n * n) < p, n, n) * 1
}

# Classic seeds, embedded as text so the patterns are readable in the source.
gol_pattern <- function(name, n = 120, at = c(60, 60)) {
  pats <- list(
    glider = c("010", "001", "111"),
    blinker = c("111"),
    block = c("11", "11"),
    r_pentomino = c("011", "110", "010")
  )
  p <- pats[[name]]
  m <- matrix(0, n, n)
  r <- at[1]; c0 <- at[2]
  for (i in seq_along(p)) {
    bits <- as.integer(strsplit(p[i], "")[[1]])
    m[r + i - 1, c0 + seq_along(bits) - 1] <- bits
  }
  m
}

# ---- elementary cellular automata -----------------------------------------
# rule: integer 0..255; the 8-bit binary expansion is the lookup table for the
# neighbourhoods 111, 110, ..., 000 (Wolfram encoding).
eca_table <- function(rule) {
  bits <- as.integer(intToBits(rule))[1:8]
  # bits[k+1] is the output for neighbourhood value k (0..7)
  bits
}

eca_step <- function(row, tab) {
  n <- length(row)
  left  <- c(row[n], row[-n])
  right <- c(row[-1], row[1])
  idx <- left * 4 + row * 2 + right + 1     # 1-based index into tab
  tab[idx]
}

# Returns a steps x n matrix; row 1 is the initial condition.
eca_simulate <- function(rule, n = 201, steps = 200, init = "single") {
  tab <- eca_table(rule)
  row <- switch(init,
                single = { r <- integer(n); r[(n + 1) %/% 2] <- 1L; r },
                random = { set.seed(rule); as.integer(runif(n) < 0.5) })
  out <- matrix(0L, steps, n)
  out[1, ] <- row
  for (t in 2:steps) {
    row <- eca_step(row, tab)
    out[t, ] <- row
  }
  out
}
