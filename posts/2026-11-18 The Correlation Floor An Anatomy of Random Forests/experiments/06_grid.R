source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

set.seed(SEED)
train <- friedman_data(400, p = 10, noise = 1)
test <- friedman_data(3000, p = 10, noise = 1)
Bs <- c(1, 2, 5, 10, 20, 50, 100, 200)
mtrys <- 1:10

grid <- expand.grid(B = Bs, mtry = mtrys)
grid$mse <- NA_real_
for (i in seq_len(nrow(grid))) {
  set.seed(SEED + i)
  rf <- ranger(x = train$X, y = train$y, num.trees = grid$B[i],
               mtry = grid$mtry[i], min.node.size = 1, replace = TRUE,
               sample.fraction = 1, seed = SEED + i)
  pr <- predict(rf, data = test$X)$predictions
  grid$mse[i] <- mean((pr - test$y)^2)
}

grid <- grid |>
  group_by(B) |>
  mutate(excess = mse / min(mse)) |>
  ungroup()

p <- ggplot(grid, aes(factor(mtry), factor(B), fill = excess)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  scale_fill_gradientn(colours = c(blog_pal[["sky"]], blog_pal[["blue"]],
                                   blog_pal[["purple"]], blog_pal[["vermillion"]]),
                       trans = "log10", name = "MSE relative\nto best for that B") +
  labs(x = "mtry", y = "Number of trees B") +
  theme(panel.border = element_blank(),
        axis.ticks = element_blank())

save_fig(p, "fig06_grid", width = 8.2, height = 4.6)

best <- grid[which.min(grid$mse), ]
write_metrics(list(
  best_mtry = best$mtry, best_B = best$B, best_mse = best$mse,
  mse_at_sqrt = grid$mse[grid$mtry == 3 & grid$B == 200],
  mse_mtry1 = grid$mse[grid$mtry == 1 & grid$B == 200],
  mse_mtry10 = grid$mse[grid$mtry == 10 & grid$B == 200],
  table = lapply(seq_len(nrow(grid)), function(i) as.list(grid[i, ]))
), "e06_grid")
