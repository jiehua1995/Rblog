source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))
suppressPackageStartupMessages(library(xgboost))

prep <- function(X) {
  X <- as.data.frame(X)
  for (j in names(X)) {
    if (is.character(X[[j]])) X[[j]] <- factor(X[[j]])
    if (is.factor(X[[j]])) X[[j]] <- as.numeric(X[[j]])
  }
  as.matrix(X)
}

knn_pred <- function(Xtr, ytr, Xte, k = 5, task = "reg") {
  mu <- colMeans(Xtr)
  sdv <- apply(Xtr, 2, sd)
  sdv[sdv == 0] <- 1
  Xtr <- scale(Xtr, center = mu, scale = sdv)
  Xte <- scale(Xte, center = mu, scale = sdv)
  D <- as.matrix(dist(rbind(Xte, Xtr)))
  D <- D[seq_len(nrow(Xte)), nrow(Xte) + seq_len(nrow(Xtr)), drop = FALSE]
  idx <- apply(D, 1, function(v) order(v)[seq_len(k)])
  if (task == "reg") {
    colMeans(matrix(ytr[idx], nrow = k))
  } else {
    as.integer(colMeans(matrix(ytr[idx], nrow = k)) > 0.5)
  }
}

fit_predict <- function(model, Xtr, ytr, Xte, task) {
  if (model %in% c("RF", "RF mtry=1")) {
    m <- if (model == "RF") max(1, floor(sqrt(ncol(Xtr)))) else 1
    rf <- ranger(x = Xtr, y = if (task == "reg") ytr else factor(ytr),
                 num.trees = 500, mtry = m, min.node.size = 1,
                 probability = task == "class", seed = SEED)
    pr <- predict(rf, data = Xte)
    if (task == "reg") pr$predictions else as.integer(pr$predictions[, "1"] > 0.5)
  } else if (model == "XGBoost") {
    obj <- if (task == "reg") "reg:squarederror" else "binary:logistic"
    dtr <- xgb.DMatrix(Xtr, label = ytr)
    dte <- xgb.DMatrix(Xte)
    bst <- xgb.train(list(objective = obj, max_depth = 3, eta = 0.05,
                          nthread = 1),
                     data = dtr, nrounds = 200, verbose = 0)
    pr <- predict(bst, dte)
    if (task == "reg") pr else as.integer(pr > 0.5)
  } else if (model == "Linear") {
    df <- data.frame(Xtr, y = ytr)
    if (task == "reg") {
      fit <- lm(y ~ ., data = df)
      as.numeric(predict(fit, newdata = data.frame(Xte)))
    } else {
      fit <- suppressWarnings(glm(y ~ ., data = df, family = binomial))
      as.integer(predict(fit, newdata = data.frame(Xte), type = "response") > 0.5)
    }
  } else {
    knn_pred(Xtr, ytr, Xte, k = 5, task = task)
  }
}

cv_eval <- function(X, y, task, model, k = 5) {
  set.seed(SEED)
  folds <- sample(rep(seq_len(k), length.out = nrow(X)))
  pred <- numeric(nrow(X))
  t0 <- proc.time()[["elapsed"]]
  for (f in seq_len(k)) {
    tr <- folds != f
    pred[!tr] <- fit_predict(model, X[tr, , drop = FALSE], y[tr],
                             X[!tr, , drop = FALSE], task)
  }
  elapsed <- proc.time()[["elapsed"]] - t0
  metric <- if (task == "reg") sqrt(mean((pred - y)^2)) else mean(pred != y)
  data.frame(model = model, metric = metric, time = elapsed)
}

set.seed(SEED)
datasets <- list()
d <- friedman_data(1000, 10, 1)
datasets$Friedman <- list(X = prep(d$X), y = d$y, task = "reg")

b <- MASS::Boston
datasets$Boston <- list(X = prep(b[, setdiff(names(b), "medv")]),
                        y = b$medv, task = "reg")

pima <- rbind(MASS::Pima.tr, MASS::Pima.te)
datasets$Pima <- list(X = prep(pima[, setdiff(names(pima), "type")]),
                      y = as.integer(pima$type == "Yes"), task = "class")

data("Sonar", package = "mlbench")
s <- Sonar
datasets$Sonar <- list(X = prep(s[, setdiff(names(s), "Class")]),
                       y = as.integer(s$Class == "R"), task = "class")

data("BreastCancer", package = "mlbench")
bc <- BreastCancer
bc <- bc[complete.cases(bc), ]
datasets$BreastCancer <- list(X = prep(bc[, 2:10]),
                              y = as.integer(bc$Class == "malignant"),
                              task = "class")

models <- c("RF", "RF mtry=1", "XGBoost", "Linear", "KNN")
res <- list()
for (nm in names(datasets)) {
  ds <- datasets[[nm]]
  for (m in models) {
    r <- cv_eval(ds$X, ds$y, ds$task, m)
    r$dataset <- nm
    r$task <- ds$task
    res[[paste(nm, m)]] <- r
  }
}
tab <- bind_rows(res)
tab <- tab |>
  group_by(dataset) |>
  mutate(relative = metric / min(metric),
         time = pmax(time, 0.001)) |>
  ungroup()

tab$task <- factor(tab$task, levels = c("reg", "class"),
                   labels = c("Regression (RMSE)", "Classification (error rate)"))
tab$model <- factor(tab$model, levels = models)

p <- ggplot(tab, aes(time, relative, colour = model, shape = dataset)) +
  geom_hline(yintercept = 1, linetype = 2, colour = blog_pal[["grey"]]) +
  geom_point(size = 2.6) +
  facet_wrap(~task, scales = "free_x") +
  scale_x_log10() +
  scale_y_log10() +
  scale_colour_manual(values = c(blog_pal[["vermillion"]], blog_pal[["orange"]],
                                 blog_pal[["green"]], blog_pal[["blue"]],
                                 blog_pal[["purple"]])) +
  labs(x = "Total time for 5-fold CV (s, log scale)",
       y = "Error relative to the best model on that dataset (log scale)",
       colour = NULL, shape = NULL) +
  theme(legend.position = "bottom", legend.box = "vertical")

save_fig(p, "fig12_benchmark", width = 9.4, height = 5.4)

write_metrics(list(
  table = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
  rf_relative = tab$relative[tab$model == "RF"],
  rf_time = tab$time[tab$model == "RF"],
  xgb_relative = tab$relative[tab$model == "XGBoost"],
  xgb_time = tab$time[tab$model == "XGBoost"]
), "e12_benchmark")
