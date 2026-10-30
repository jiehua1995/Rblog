# =============================================================================
# E04_vae.R — training cell autoencoders and VAEs in R/torch
#
# Same split, same preprocessing, same trunk.  Compared with PCA at the same
# latent width.  The count-likelihood variant (vae_nb) is trained on the same
# encoder input but scores raw counts.
# =============================================================================
source("R/00_setup.R")
source("R/01_simulate_cells.R")
source("R/02_preprocessing.R")
source("R/04_autoencoders.R")

sim <- readRDS("results/sim.rds")
tr  <- sim$truth
split <- make_split(nrow(tr), 0.7, seed = 7)

X <- build_input(sim, depth = TRUE, batch = TRUE, hvg = 1000, scale = TRUE)
counts <- as.matrix(sim$counts)
lib <- tr$lib
Xtr <- X[split$train, ]; Xte <- X[split$test, ]

Ztrue <- scale(as.matrix(tr[, c("Z1", "Z2", "Z3")]))

res <- list()
t_start <- Sys.time()

# ---- PCA baseline across latent widths -------------------------------------
pca_rows <- lapply(c(2, 4, 8, 16), function(d) {
  fit <- pca_svd(Xtr, d = d)
  rec <- (sweep(Xte, 2, colMeans(Xtr), "-") %*% fit$rotation) %*% t(fit$rotation)
  mse <- mean((sweep(Xte, 2, colMeans(Xtr), "-") - rec)^2)
  emb <- sweep(X, 2, colMeans(Xtr), "-") %*% fit$rotation
  data.frame(model = "PCA", d = d, test_mse = mse,
             r_t = embedding_metrics(emb, tr)$r_t,
             r_a = embedding_metrics(emb, tr)$r_a,
             knn = embedding_metrics(emb, tr)$knn,
             dist_cor = embedding_metrics(emb, tr)$dist_cor,
             seconds = NA_real_)
})
res$pca <- do.call(rbind, pca_rows)

# ---- autoencoder and Gaussian VAE -----------------------------------------
grid <- expand.grid(model = c("AE", "VAE"), d = c(2, 4, 8, 16),
                    stringsAsFactors = FALSE)
grid <- grid[order(grid$model, grid$d), ]
out <- vector("list", nrow(grid))
hist_list <- list()

for (i in seq_len(nrow(grid))) {
  d <- grid$d[i]; is_ae <- grid$model[i] == "AE"
  cat(sprintf("\n[%s d=%d]\n", grid$model[i], d))
  t0 <- Sys.time()
  fit <- train_model(if (is_ae) "ae" else "vae", Xtr, d = d, epochs = 120,
                     batch = 128, lr = 1e-3, beta = 1, warmup = 20, seed = 2026)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  emb <- encode_matrix(fit$model, X, d)
  # reconstruction error from the posterior mean, so VAE and AE are comparable
  rec_te <- if (is_ae) {
    fit$model$eval(); torch_no_grad({ as.matrix(fit$model(torch_tensor(Xte, dtype = torch_float()))$to(device = "cpu")) })
  } else {
    fit$model$eval(); torch_no_grad({
      z <- fit$model$encode(torch_tensor(Xte, dtype = torch_float()))$mu
      as.matrix(fit$model$decode(z)$to(device = "cpu"))
    })
  }
  m <- embedding_metrics(emb, tr)
  out[[i]] <- data.frame(model = grid$model[i], d = d,
                         test_mse = mean((rec_te - Xte)^2),
                         r_t = m$r_t, r_a = m$r_a, knn = m$knn,
                         dist_cor = m$dist_cor, seconds = secs)
  hist_list[[paste(grid$model[i], d)]] <- transform(fit$history, model = grid$model[i], d = d)
  cat(sprintf("  test MSE %.4f  r_t %.2f  r_a %.2f  (%.1fs)\n",
              out[[i]]$test_mse, m$r_t, m$r_a, secs))
  if (d == 4) torch_save(fit$model, sprintf("results/%s_d4.pt", tolower(grid$model[i])))
}

res$nn <- do.call(rbind, out)
res$history <- do.call(rbind, hist_list)

# ---- count-likelihood VAE --------------------------------------------------
cat("\n[vae_nb d=4]\n")
t0 <- Sys.time()
fit_nb <- train_model("vae_nb", Xtr, counts = counts[split$train, ],
                      lib = lib[split$train], d = 4, epochs = 320,
                      batch = 128, lr = 8e-4, beta = 1, warmup = 25,
                      patience = 40, seed = 2026)
secs_nb <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

emb_nb <- encode_matrix(fit_nb$model, X, 4)
fit_nb$model$eval()
torch_no_grad({
  o <- fit_nb$model(torch_tensor(Xte, dtype = torch_float()))
  mu_hat <- as.matrix((torch_exp(o$logmean) * (torch_tensor(lib[split$test]) / 1e4)$view(c(-1, 1)))$to(device = "cpu"))
  theta_hat <- as.numeric(o$theta$to(device = "cpu"))
})
m_nb <- embedding_metrics(emb_nb, tr)
res$nb <- data.frame(
  model = "VAE-NB", d = 4,
  test_nll_per_gene = {
    x <- counts[split$test, ]
    mu <- pmax(mu_hat, 1e-6)
    -mean(dnbinom(x, mu = mu, size = matrix(theta_hat, nrow(x), ncol(x), byrow = TRUE), log = TRUE))
  },
  spearman_obs_pred = cor(as.vector(mu_hat), as.vector(counts[split$test, ]), method = "spearman"),
  r_t = m_nb$r_t, r_a = m_nb$r_a, knn = m_nb$knn, dist_cor = m_nb$dist_cor,
  seconds = secs_nb
)
res$nb_history <- fit_nb$history
res$nb_embedding <- emb_nb
torch_save(fit_nb$model, "results/vae_nb_d4.pt")

print(res$pca, digits = 3)
print(res$nn, digits = 3)
print(res$nb, digits = 3)

res$total_seconds <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
res$X_meta <- list(mu = colMeans(Xtr), sd = apply(Xtr, 2, sd),
                   genes = colnames(X), split = split,
                   lib_median = median(lib), theta = theta_hat)
saveRDS(res, "results/e04_vae.rds")
cat(sprintf("total training time: %.0fs\n", res$total_seconds))
