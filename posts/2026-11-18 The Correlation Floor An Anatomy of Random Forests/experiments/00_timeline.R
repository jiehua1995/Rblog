source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 "common", "setup.R"))

events <- data.frame(
  year = c(1963, 1979, 1984, 1995, 1996, 1997, 2001, 2006,
           2006, 2008, 2015, 2017, 2018, 2019, 2022, 2025),
  label = c("AID: automatic interaction detection",
            "Bootstrap (Efron)",
            "CART (Breiman et al.)",
            "Random subspace (Ho)",
            "Bagging (Breiman)",
            "Amit & Geman: randomized trees",
            "Random Forests (Breiman)",
            "Extremely randomized trees",
            "Quantile regression forests",
            "Random survival forests",
            "Consistency proof (Scornet et al.)",
            "ranger (Wright & Ziegler)",
            "Causal forests (Wager & Athey)",
            "Generalized random forests",
            "Tree-based models vs deep learning",
            "TabPFN v2 (Nature)"),
  family = c("Tree", "Resampling", "Tree", "Randomisation", "Resampling",
             "Randomisation", "Forest", "Randomisation", "Forest", "Forest",
             "Theory", "Software", "Inference", "Inference", "Modern", "Modern")
)

events$family <- factor(events$family,
                        levels = c("Tree", "Resampling", "Randomisation",
                                   "Forest", "Theory", "Inference",
                                   "Software", "Modern"))
events$side <- ifelse(seq_len(nrow(events)) %% 2 == 1, 1, -1)

p <- ggplot(events, aes(y = year, x = 0)) +
  geom_vline(xintercept = 0, colour = blog_pal[["grey"]], linewidth = 0.5) +
  geom_segment(aes(xend = side * 0.28, yend = year),
               colour = blog_pal[["grey"]], linewidth = 0.3) +
  geom_point(aes(colour = family), size = 2.4) +
  geom_text(aes(x = side * 0.34, label = label, colour = family),
            hjust = ifelse(events$side == 1, 0, 1), size = 2.8,
            show.legend = FALSE) +
  scale_colour_manual(values = c(
    Tree = blog_pal[["blue"]],
    Resampling = blog_pal[["green"]],
    Randomisation = blog_pal[["orange"]],
    Forest = blog_pal[["vermillion"]],
    Theory = blog_pal[["purple"]],
    Inference = blog_pal[["sky"]],
    Software = blog_pal[["grey"]],
    Modern = "black"
  )) +
  scale_x_continuous(limits = c(-1.25, 1.25)) +
  scale_y_continuous(breaks = seq(1960, 2020, 10), limits = c(1958, 2028)) +
  labs(x = NULL, y = NULL, colour = NULL) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        panel.border = element_blank(),
        legend.position = "top",
        legend.text = element_text(size = 7))

save_fig(p, "fig00_timeline", width = 7.4, height = 7.2)

write_metrics(list(n_events = nrow(events),
                   first_year = min(events$year),
                   last_year = max(events$year)),
              "e00_timeline")
