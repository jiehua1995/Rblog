source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
n_train <- 600
n_test <- 2000
k <- 5

make_iid <- function(n, noise = 1) {
  d <- friedman_data(n, 10, noise)
  data.frame(X = d$X, y = d$y)
}

make_grouped <- function(n, n_groups = 60, sd_group = 3) {
  d <- friedman_data(n, 10, 1)
  g <- rep(seq_len(n_groups), length.out = n)
  b <- rnorm(n_groups, 0, sd_group)
  data.frame(X = d$X, y = d$y + b[g], group = g)
}

make_time <- function(n, phi = 0.9) {
  d <- friedman_data(n, 10, 0)
  e <- numeric(n)
  e[1] <- rnorm(1, 0, 1 / sqrt(1 - phi^2))
  for (i in 2:n) e[i] <- phi * e[i - 1] + rnorm(1, 0, 1)
  data.frame(X = d$X, y = d$y + e, time = seq_len(n))
}

fold_random <- function(n, k) sample(rep(seq_len(k), length.out = n))

fold_group <- function(g, k) {
  ug <- unique(g)
  assign <- setNames(sample(rep(seq_len(k), length.out = length(ug))), ug)
  unname(assign[as.character(g)])
}

fold_block <- function(n, k) {
  rep(seq_len(k), each = ceiling(n / k))[seq_len(n)]
}

cv_mse <- function(d, folds, mtry = 3, B = 300) {
  X <- as.matrix(d[, grep("^X", names(d))])
  y <- d$y
  pred <- rep(NA_real_, nrow(d))
  for (f in seq_len(max(folds))) {
    tr <- folds != f
    rf <- ranger(x = X[tr, , drop = FALSE], y = y[tr], num.trees = B,
                 mtry = mtry, min.node.size = 1, seed = SEED + f)
    pred[!tr] <- predict(rf, data = X[!tr, , drop = FALSE])$predictions
  }
  mean((pred - y)^2)
}

fit_test_mse <- function(d, test, mtry = 3, B = 300) {
  X <- as.matrix(d[, grep("^X", names(d))])
  Xt <- as.matrix(test[, grep("^X", names(test))])
  rf <- ranger(x = X, y = d$y, num.trees = B, mtry = mtry,
               min.node.size = 1, seed = SEED)
  mean((predict(rf, data = Xt)$predictions - test$y)^2)
}

oob_mse <- function(d, mtry = 3, B = 300) {
  X <- as.matrix(d[, grep("^X", names(d))])
  rf <- ranger(x = X, y = d$y, num.trees = B, mtry = mtry,
               min.node.size = 1, seed = SEED, keep.inbag = TRUE)
  mean((rf$predictions - d$y)^2)
}

scenarios <- list(
  iid = list(train = make_iid(n_train), test = make_iid(n_test)),
  grouped = list(train = make_grouped(n_train), test = make_grouped(n_test)),
  time = list(train = make_time(n_train), test = make_time(n_test))
)

res <- list()
for (s in names(scenarios)) {
  tr <- scenarios[[s]]$train
  te <- scenarios[[s]]$test
  folds_random <- fold_random(nrow(tr), k)
  folds_honest <- switch(s,
    iid = folds_random,
    grouped = fold_group(tr$group, k),
    time = fold_block(nrow(tr), k)
  )
  res[[s]] <- data.frame(
    scenario = s,
    oob = oob_mse(tr),
    cv_random = cv_mse(tr, folds_random),
    cv_honest = cv_mse(tr, folds_honest),
    test = fit_test_mse(tr, te)
  )
}
tab <- bind_rows(res)

long <- tab |>
  select(scenario, OOB = oob, `Random CV` = cv_random,
         `Honest CV` = cv_honest, `Independent test` = test) |>
  pivot_longer(-scenario, names_to = "estimator", values_to = "mse") |>
  group_by(scenario) |>
  mutate(ratio = mse / mse[estimator == "Independent test"]) |>
  ungroup()
long$scenario <- factor(long$scenario, levels = c("iid", "grouped", "time"),
                        labels = c("iid", "grouped", "serial"))
long$estimator <- factor(long$estimator,
                         levels = c("OOB", "Random CV", "Honest CV",
                                    "Independent test"))

p <- ggplot(long, aes(scenario, ratio, colour = estimator)) +
  geom_hline(yintercept = 1, linetype = 2, colour = blog_pal[["grey"]]) +
  geom_linerange(aes(ymin = pmin(ratio, 1), ymax = pmax(ratio, 1)),
                 position = position_dodge(width = 0.5), linewidth = 0.7) +
  geom_point(position = position_dodge(width = 0.5), size = 2.6) +
  coord_cartesian(ylim = c(0.85, 1.15)) +
  scale_colour_manual(values = c(OOB = blog_pal[["orange"]],
                                 `Random CV` = blog_pal[["vermillion"]],
                                 `Honest CV` = blog_pal[["blue"]],
                                 `Independent test` = "black")) +
  labs(x = NULL, y = "MSE relative to an independent test set", colour = NULL) +
  theme(legend.position = "bottom")

save_fig(p, "fig07_dependence", width = 8.4, height = 4.4)

write_metrics(list(
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  oob_over_test_iid = tab$oob[tab$scenario == "iid"] / tab$test[tab$scenario == "iid"],
  oob_over_test_grouped = tab$oob[tab$scenario == "grouped"] / tab$test[tab$scenario == "grouped"],
  oob_over_test_time = tab$oob[tab$scenario == "time"] / tab$test[tab$scenario == "time"],
  cv_random_over_test_grouped = tab$cv_random[tab$scenario == "grouped"] / tab$test[tab$scenario == "grouped"],
  cv_random_over_test_time = tab$cv_random[tab$scenario == "time"] / tab$test[tab$scenario == "time"]
), "e07_dependence")
