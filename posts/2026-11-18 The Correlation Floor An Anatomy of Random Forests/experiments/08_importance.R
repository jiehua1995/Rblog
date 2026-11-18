source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n <- 800
X <- matrix(rnorm(n * 10), n, 10)
colnames(X) <- paste0("x", 1:10)
X[, 2] <- X[, 1] + 0.3 * rnorm(n)
y <- 3 * X[, 1] + 2 * X[, 4] + rnorm(n)

Xt <- matrix(rnorm(2000 * 10), 2000, 10)
colnames(Xt) <- paste0("x", 1:10)
Xt[, 2] <- Xt[, 1] + 0.3 * rnorm(2000)
yt <- 3 * Xt[, 1] + 2 * Xt[, 4] + rnorm(2000)

set.seed(SEED + 1)
rf_mdi <- ranger(x = X, y = y, num.trees = 500, mtry = 3,
                 min.node.size = 1, importance = "impurity", seed = SEED + 1)
set.seed(SEED + 1)
rf_perm <- ranger(x = X, y = y, num.trees = 500, mtry = 3,
                  min.node.size = 1, seed = SEED + 2)

base <- mean((predict(rf_perm, data = Xt)$predictions - yt)^2)

permute_importance <- function(v, conditional = NULL, reps = 10) {
  vals <- replicate(reps, {
    Xp <- Xt
    if (is.null(conditional)) {
      Xp[, v] <- sample(Xp[, v])
    } else {
      br <- cut(conditional, breaks = quantile(conditional, 0:10 / 10),
                include.lowest = TRUE)
      for (b in levels(br)) {
        idx <- which(br == b)
        Xp[idx, v] <- sample(Xp[idx, v])
      }
    }
    mean((predict(rf_perm, data = Xp)$predictions - yt)^2) - base
  })
  mean(vals)
}

vars <- paste0("x", 1:6)
mdi <- rf_mdi$variable.importance[vars]
perm <- vapply(vars, permute_importance, numeric(1))
cond <- c(
  x1 = permute_importance("x1"),
  x2 = permute_importance("x2", conditional = Xt[, "x1"]),
  x3 = permute_importance("x3"),
  x4 = permute_importance("x4"),
  x5 = permute_importance("x5"),
  x6 = permute_importance("x6")
)

raw <- data.frame(variable = vars, MDI = as.numeric(mdi),
                  Permutation = as.numeric(perm),
                  `Conditional` = as.numeric(cond))
norm <- raw |>
  mutate(across(c(MDI, Permutation, Conditional), ~ .x / max(.x))) |>
  pivot_longer(-variable, names_to = "method", values_to = "value")

norm$method <- factor(norm$method,
                      levels = c("MDI", "Permutation", "Conditional"))
norm$variable <- factor(norm$variable, levels = vars)

p <- ggplot(norm, aes(variable, value, fill = method)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.7) +
  scale_fill_manual(values = c(MDI = blog_pal[["orange"]],
                               Permutation = blog_pal[["vermillion"]],
                               Conditional = blog_pal[["blue"]])) +
  labs(x = NULL, y = "Importance, scaled to its maximum", fill = NULL) +
  theme(legend.position = "bottom")

save_fig(p, "fig08_importance", width = 8.4, height = 4.2)

write_metrics(list(
  raw = lapply(seq_len(nrow(raw)), function(i) as.list(raw[i, ])),
  mdi_x2_over_x1 = raw$MDI[raw$variable == "x2"] / raw$MDI[raw$variable == "x1"],
  perm_x2_over_x1 = raw$Permutation[raw$variable == "x2"] / raw$Permutation[raw$variable == "x1"],
  cond_x2_over_x1 = raw$Conditional[raw$variable == "x2"] / raw$Conditional[raw$variable == "x1"],
  base_mse = base
), "e08_importance")
