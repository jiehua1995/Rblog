# =============================================================================
# 04_autoencoders.R — R/torch modules and training loops
#
# Three models share the same trunk:
#   ae      : deterministic bottleneck, squared-error reconstruction
#   vae     : Gaussian bottleneck, ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
#   vae_nb  : same bottleneck, but the decoder emits a negative-binomial mean
#             with a fixed library-size offset (a count likelihood instead of
#             a Gaussian one)
# =============================================================================
suppressPackageStartupMessages(library(torch))
torch_set_num_threads(max(1, min(4, parallel::detectCores() - 1)))

# Evaluation helper: keeps the call sites readable.  Autograd is still active,
# which costs a little memory but no correctness for the small tensors here.
torch_no_grad <- function(code) eval(substitute(code), envir = parent.frame())

# ---- modules ---------------------------------------------------------------

ae_module <- nn_module(
  "ae",
  initialize = function(p, h1 = 256, h2 = 64, d = 8) {
    self$enc <- nn_sequential(nn_linear(p, h1), nn_relu(),
                              nn_linear(h1, h2), nn_relu(),
                              nn_linear(h2, d))
    self$dec <- nn_sequential(nn_linear(d, h2), nn_relu(),
                              nn_linear(h2, h1), nn_relu(),
                              nn_linear(h1, p))
  },
  encode = function(x) self$enc(x),
  decode = function(z) self$dec(z),
  forward = function(x) self$dec(self$enc(x))
)

vae_module <- nn_module(
  "vae",
  initialize = function(p, h1 = 256, h2 = 64, d = 8) {
    self$enc <- nn_sequential(nn_linear(p, h1), nn_relu(),
                              nn_linear(h1, h2), nn_relu())
    self$mu_head <- nn_linear(h2, d)
    self$logvar_head <- nn_linear(h2, d)
    self$dec <- nn_sequential(nn_linear(d, h2), nn_relu(),
                              nn_linear(h2, h1), nn_relu(),
                              nn_linear(h1, p))
  },
  encode = function(x) {
    h <- self$enc(x)
    list(mu = self$mu_head(h), logvar = self$logvar_head(h))
  },
  decode = function(z) self$dec(z),
  forward = function(x) {
    e <- self$encode(x)
    z <- e$mu + torch_exp(0.5 * e$logvar) * torch_randn_like(e$mu)
    list(rec = self$decode(z), mu = e$mu, logvar = e$logvar, z = z)
  }
)

vae_nb_module <- nn_module(
  "vae_nb",
  initialize = function(p, h1 = 256, h2 = 64, d = 8) {
    self$enc <- nn_sequential(nn_linear(p, h1), nn_relu(),
                              nn_linear(h1, h2), nn_relu())
    self$mu_head <- nn_linear(h2, d)
    self$logvar_head <- nn_linear(h2, d)
    self$dec <- nn_sequential(nn_linear(d, h2), nn_relu(),
                              nn_linear(h2, h1), nn_relu(),
                              nn_linear(h1, p))
    self$theta_raw <- nn_parameter(torch_zeros(p))   # softplus -> dispersion
  },
  encode = function(x) {
    h <- self$enc(x)
    list(mu = self$mu_head(h), logvar = self$logvar_head(h))
  },
  decode = function(z) self$dec(z),
  forward = function(x) {
    e <- self$encode(x)
    z <- e$mu + torch_exp(0.5 * e$logvar) * torch_randn_like(e$mu)
    list(logmean = torch_clamp(self$decode(z), -8, 8),
         mu = e$mu, logvar = e$logvar, z = z,
         theta = nnf_softplus(self$theta_raw) + 1e-3)
  }
)

#' Negative-binomial negative log-likelihood for a batch.
#' mu is expressed in the same units as the counts (library-size offset applied).
nb_nll <- function(x, logmean, theta, lib) {
  logmean <- torch_clamp(logmean, -8, 8)
  mu <- torch_exp(logmean) * (lib / 1e4)$view(c(-1, 1)) + 1e-6
  lg <- torch_lgamma
  -(lg(x + theta) - lg(theta) - lg(x + 1) +
      theta * torch_log(theta) - (theta + x) * torch_log(theta + mu) +
      x * torch_log(mu))$sum(dim = 2)$mean()
}

# ---- training --------------------------------------------------------------

#' Train one model.  Returns a list with the fitted module and the loss history.
#'
#' Losses are summed over genes and averaged over the batch (the usual
#' variational convention), so that a beta of 1 is a meaningful default:
#'   ELBO term    = sum_g (x_ig - mu_ig)^2           (Gaussian, unit variance)
#'   NB term      = -sum_g log NB(x_ig | mu_ig, theta_g)
#'   KL term      = sum_j KL(q(z_j|x) || N(0, 1))
#' Early stopping keeps the weights with the lowest validation loss; without it
#' a 256-unit MLP memorises the 8% signal plus noise within ~40 epochs.
train_model <- function(type = c("ae", "vae", "vae_nb"), X, counts = NULL,
                        lib = NULL, d = 8, epochs = 120, batch = 128,
                        lr = 1e-3, beta = 1, warmup = 20, seed = 2026,
                        val_frac = 0.15, patience = 20, weight_decay = 1e-4,
                        verbose = TRUE) {
  type <- match.arg(type)
  torch_manual_seed(seed)
  p <- ncol(X)
  n <- nrow(X)
  set.seed(seed)
  val <- sample(n, floor(val_frac * n))
  trn <- setdiff(seq_len(n), val)

  Xt <- torch_tensor(X, dtype = torch_float())
  Xtr <- Xt[trn, ]; Xva <- Xt[val, ]
  Ctr <- if (!is.null(counts)) torch_tensor(counts, dtype = torch_float())[trn, ]
  Cva <- if (!is.null(counts)) torch_tensor(counts, dtype = torch_float())[val, ]
  Ltr <- if (!is.null(lib)) torch_tensor(lib, dtype = torch_float())[trn]
  Lva <- if (!is.null(lib)) torch_tensor(lib, dtype = torch_float())[val]

  model <- switch(type,
    ae     = ae_module(p, h1 = 256, h2 = 64, d = d),
    vae    = vae_module(p, h1 = 256, h2 = 64, d = d),
    vae_nb = vae_nb_module(p, h1 = 256, h2 = 64, d = d))
  opt <- optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)

  hist <- data.frame(epoch = seq_len(epochs), train = NA_real_, val = NA_real_)
  best <- Inf; best_state <- NULL; bad <- 0
  n_tr <- length(trn)
  for (ep in seq_len(epochs)) {
    model$train()
    perm <- sample(n_tr)   # R-level permutation; torch indices are 1-based
    tot <- 0; nb <- 0
    b <- beta * min(1, ep / warmup)
    for (s in seq(1, n_tr, by = batch)) {
      idx <- perm[s:min(s + batch - 1, n_tr)]
      xb <- Xtr[idx, ]
      if (type == "ae") {
        loss <- torch_sum((model(xb) - xb)^2, dim = 2)$mean()
      } else if (type == "vae") {
        out <- model(xb)
        kl <- -0.5 * torch_sum(1 + out$logvar - out$mu^2 - torch_exp(out$logvar),
                               dim = 2)$mean()
        loss <- torch_sum((out$rec - xb)^2, dim = 2)$mean() + b * kl
      } else {
        out <- model(xb)
        kl <- -0.5 * torch_sum(1 + out$logvar - out$mu^2 - torch_exp(out$logvar),
                               dim = 2)$mean()
        loss <- nb_nll(Ctr[idx, ], out$logmean, out$theta, Ltr[idx]) / p + b * kl / p
      }
      opt$zero_grad()
      loss$backward()
      try(nn_utils_clip_grad_norm_(model$parameters, 10), silent = TRUE)
      opt$step()
      tot <- tot + loss$item(); nb <- nb + 1
    }
    model$eval()
    vl <- torch_no_grad({
      if (type == "ae") {
        torch_sum((model(Xva) - Xva)^2, dim = 2)$mean()$item()
      } else if (type == "vae") {
        o <- model(Xva)
        kl <- -0.5 * torch_sum(1 + o$logvar - o$mu^2 - torch_exp(o$logvar), dim = 2)$mean()
        rec <- model$decode(o$mu)          # deterministic check, no sampling noise
        (torch_sum((rec - Xva)^2, dim = 2)$mean() + beta * kl)$item()
      } else {
        o <- model(Xva)
        kl <- -0.5 * torch_sum(1 + o$logvar - o$mu^2 - torch_exp(o$logvar), dim = 2)$mean()
        (nb_nll(Cva, model$decode(o$mu), o$theta, Lva) / p + beta * kl / p)$item()
      }
    })
    hist$train[ep] <- tot / nb
    hist$val[ep] <- vl
    if (!is.finite(vl)) {
      if (verbose) cat(sprintf("  non-finite validation loss at epoch %d, stopping\n", ep))
      hist <- hist[seq_len(ep), ]
      break
    }
    if (vl < best - 1e-3) {
      best <- vl; best_state <- lapply(model$state_dict(), function(t) t$clone())
      bad <- 0
    } else {
      bad <- bad + 1
      if (bad >= patience) {
        hist <- hist[seq_len(ep), ]
        if (verbose) cat(sprintf("  early stop at epoch %d (best val %.4f)\n", ep, best))
        break
      }
    }
    if (verbose && ep %% 20 == 0) cat(sprintf("  epoch %3d  train %.4f  val %.4f\n", ep, tot / nb, vl))
  }
  if (!is.null(best_state)) model$load_state_dict(best_state)
  list(model = model, history = hist, best_val = best)
}

#' Encode a matrix with a trained model (posterior mean for VAEs).
encode_matrix <- function(model, X, d) {
  model$eval()
  torch_no_grad({
    out <- model$encode(torch_tensor(X, dtype = torch_float()))
    z <- as.matrix((if (is.list(out)) out$mu else out)$to(device = "cpu"))
  })
  colnames(z) <- paste0("Z", seq_len(ncol(z)))
  z
}

#' Decode latent coordinates (deterministic mean profile).
decode_matrix <- function(model, Z) {
  model$eval()
  torch_no_grad({
    as.matrix(model$decode(torch_tensor(Z, dtype = torch_float()))$to(device = "cpu"))
  })
}

#' Test-set mean squared reconstruction error in the standardised space.
test_mse <- function(model, X, type = "vae") {
  model$eval()
  torch_no_grad({
    x <- torch_tensor(X, dtype = torch_float())
    rec <- if (type == "ae") model(x) else model(x)$rec
    torch_mean((rec - x)^2)$item()
  })
}
