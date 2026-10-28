#!/usr/bin/env Rscript
# Train the same two-layer MLP as the Python side, using torch for R.
# Usage: Rscript train-r-torch.R <moons.csv> <results.rds>

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[[1]]
rds_path <- args[[2]]

# torch's Windows build loads DLLs from its own lib/ directory; without that
# directory on PATH, dyn.load() fails with "The specified procedure could not be found"
torch_lib <- file.path(find.package("torch"), "lib")
Sys.setenv(PATH = paste(torch_lib, Sys.getenv("PATH"), sep = .Platform$path.sep))

t0 <- Sys.time()
suppressPackageStartupMessages(library(torch))
load_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

d <- read.csv(csv_path)
X <- as.matrix(d[, c("x", "y")])
y <- as.integer(d$label)

set.seed(7)
torch_manual_seed(7)

Xtr <- torch_tensor(X[1:320, ], dtype = torch_float())
ytr <- torch_tensor(y[1:320] + 1L, dtype = torch_long())
Xte <- torch_tensor(X[321:400, ], dtype = torch_float())
yte <- torch_tensor(y[321:400] + 1L, dtype = torch_long())

model <- nn_sequential(nn_linear(2, 16), nn_relu(), nn_linear(16, 2))
opt <- optim_adam(model$parameters, lr = 0.05)

t0 <- Sys.time()
losses <- numeric(200)
for (epoch in seq_len(200)) {
  opt$zero_grad()
  loss <- nnf_cross_entropy(model(Xtr), ytr)
  loss$backward()
  opt$step()
  losses[epoch] <- as.numeric(loss$item())
}
seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

with_no_grad({
  acc <- as.numeric(
    (model(Xte)$argmax(dim = 2) == yte)$to(dtype = torch_float())$mean()$item()
  )
})

# same 80 x 80 grid as the Python side: x varies fastest, y slowest
gx <- seq(min(X[, 1]) - 0.3, max(X[, 1]) + 0.3, length.out = 80)
gy <- seq(min(X[, 2]) - 0.3, max(X[, 2]) + 0.3, length.out = 80)
grid_flat <- as.matrix(expand.grid(x = gx, y = gy))
grid_tensor <- torch_tensor(grid_flat, dtype = torch_float())

t0 <- Sys.time()
with_no_grad({
  # argmax is 1-based in R, so shift back to 0/1 to match the Python side
  classes <- as.integer(model(grid_tensor)$argmax(dim = 2)) - 1L
})
infer_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

check <- as.numeric(torch_tensor(c(1, 2, 3))$exp()$sum())

cat(sprintf("torch %s | final loss %.4f | test acc %.3f | load %.2f s | train %.2f s | infer %.1f ms\n",
            packageVersion("torch"), losses[200], acc, load_seconds, seconds, infer_seconds * 1e3))

saveRDS(list(losses = losses, acc = acc, seconds = seconds,
             load = load_seconds, infer = infer_seconds,
             classes = classes, check = check,
             version = as.character(packageVersion("torch"))),
        rds_path)
