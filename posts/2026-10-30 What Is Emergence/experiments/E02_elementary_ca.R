# =============================================================================
# E02_elementary_ca.R — one line of code, four different worlds.
#
# Elementary cellular automata: 8 bits decide the fate of every cell, so there
# are exactly 256 possible rules.  A few of them look "complex" while being
# almost structureless, which is the cleanest available counter-example to
# "pretty picture = emergence".
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")

n <- 201; steps <- 200
rules <- c(30, 90, 110, 184, 250)

ims <- list(); rows <- list()
for (r in rules) {
  # a single cell does not create a traffic jam, so rule 184 starts from a
  # random row instead; every other rule starts from one live cell
  init <- if (r == 184) "random" else "single"
  img <- eca_simulate(r, n = n, steps = steps, init = init)
  ims[[as.character(r)]] <- img
  last50 <- img[(steps - 49):steps, ]
  rows[[length(rows) + 1]] <- data.frame(
    system = paste0("rule ", r),
    density = mean(img),
    entropy = block_entropy(last50),
    compress = compressibility(img),
    persistence = mean(last50[-nrow(last50), ] == last50[-1, ]),
    ac_len = corr_length(spatial_autocorr(last50, 60))
  )
}

# ---- reference systems -----------------------------------------------------
gol <- readRDS("results/e01_gol.rds")

# iid noise: maximum entropy, zero structure, no memory
set.seed(2)
noise <- matrix(runif(n * steps) < 0.5, steps, n) * 1L
rows[[length(rows) + 1]] <- data.frame(
  system = "iid noise", density = mean(noise),
  entropy = block_entropy(noise),
  compress = compressibility(noise),
  persistence = 0.5,
  ac_len = corr_length(spatial_autocorr(noise, 60))
)

# Game of Life, random soup and glider, as 2-D reference points
for (nm in c("random", "glider")) {
  m <- if (nm == "random") gol$random_final else gol$glider_final
  rows[[length(rows) + 1]] <- data.frame(
    system = paste0("life: ", nm), density = live_fraction(m),
    entropy = block_entropy(m),
    compress = compressibility(m),
    persistence = NA_real_,
    ac_len = corr_length(spatial_autocorr(m, 60))
  )
}

mets <- do.call(rbind, rows)
saveRDS(list(metrics = mets, images = ims, n = n, steps = steps, rules = rules),
        "results/e02_eca.rds")

print(mets, row.names = FALSE, digits = 3)
