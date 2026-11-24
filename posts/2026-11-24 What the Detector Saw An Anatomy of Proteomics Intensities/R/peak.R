# What the detector sees: elution profiles, isotope envelopes on an m/z axis,
# XIC integration, AGC-limited ion accumulation and ESI competition.

elution_profile <- function(rt, apex, sigma, tail = 1.8) {
  s <- ifelse(rt <= apex, sigma, sigma * tail)
  exp(-0.5 * ((rt - apex) / s)^2)
}

# Intensity of one peptide (amount in arbitrary ion units) on an RT x m/z grid.
peak_surface <- function(amount, seq, charge = 2, apex = 20, sigma = 0.25,
                         tail = 1.8, rt = seq(15, 27, length.out = 240),
                         mz_window = 0.9, resolution = 70000, n_mz = 320) {
  env <- isotope_envelope(seq, n_peaks = 6)
  m0 <- (peptide_mass(seq) + 1.007276 * charge) / charge
  mz <- seq(m0 - 0.35, m0 + mz_window, length.out = n_mz)
  fwhm <- m0 / resolution
  s_mz <- fwhm / 2.355
  el <- elution_profile(rt, apex, sigma, tail)
  grid <- expand.grid(rt = rt, mz = mz)
  inten <- numeric(nrow(grid))
  for (k in seq_len(nrow(env))) {
    mzk <- (peptide_mass(seq) + (k - 1) * 1.003355 + 1.007276 * charge) / charge
    inten <- inten + amount * env$p[k] * el[match(grid$rt, rt)] *
      dnorm(grid$mz, mzk, s_mz)
  }
  grid$intensity <- inten
  grid
}

# Extract-ion chromatogram: sum the surface inside an m/z window.
xic_from_surface <- function(surface, mz_lo, mz_hi) {
  sub <- surface[surface$mz >= mz_lo & surface$mz <= mz_hi, ]
  stats::aggregate(intensity ~ rt, data = sub, FUN = sum)
}

# Area under a chromatogram inside an RT window, relative to the full trace.
xic_area <- function(xic, rt_lo, rt_hi) {
  sub <- xic[xic$rt >= rt_lo & xic$rt <= rt_hi, ]
  if (nrow(sub) < 2) return(0)
  sum(diff(sub$rt) * (head(sub$intensity, -1) + tail(sub$intensity, -1)) / 2)
}

# Orbitrap-style AGC: ions are accumulated until the target is reached or the
# maximum injection time is exhausted.
agc_signal <- function(ion_rate, agc_target, max_injection = 0.1) {
  pmin(ion_rate * max_injection, agc_target) / max_injection
}

# ESI competition + detector capacity: the mixture shares a finite ion budget,
# and each species saturates as it approaches that budget.
saturating_signal <- function(x, capacity = Inf, load = 0, alpha = 0.5) {
  cap_eff <- capacity / (1 + alpha * load / capacity)
  x / (1 + x / cap_eff)
}
