# =============================================================================
# E10_rule_space.R — where do the 256 elementary rules sit in metric space?
#
# E02 compares five rules by hand.  Here every rule runs from the same random
# row and is placed in the (block entropy, compressibility) plane.  The point
# is that "complex" is not a single axis: the rules occupy several distinct
# regions, and the two most-cited "chaotic" rules do not sit where intuition
# would put them.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")

t0 <- Sys.time()
n <- 201; steps <- 200

rules <- do.call(rbind, lapply(0:255, function(r) {
  img <- eca_simulate(r, n = n, steps = steps, init = "random")
  last50 <- img[(steps - 49):steps, ]
  data.frame(rule = r,
             density = mean(img),
             entropy = block_entropy(last50),
             compress = compressibility(img),
             persistence = mean(last50[-nrow(last50), ] == last50[-1, ]))
}))

# 2-D reference points already computed in E02 (iid noise and two GoL states)
refs <- readRDS("results/e02_eca.rds")$metrics
refs <- refs[grepl("iid|life", refs$system), c("system", "entropy", "compress")]

saveRDS(list(rules = rules, refs = refs, n = n, steps = steps),
        "results/e10_rule_space.rds")

cat("run time:", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
cat("entropy range:", round(range(rules$entropy), 3),
    " compress range:", round(range(rules$compress), 3), "\n")
cat("rules with entropy > 0.8:", sum(rules$entropy > 0.8), "\n")
cat("correlation(entropy, compress):", round(cor(rules$entropy, rules$compress), 3), "\n")
print(rules[rules$rule %in% c(30, 90, 110, 184, 250), ], row.names = FALSE, digits = 3)
