# =============================================================================
# E07_animations.R — the two point-cloud / field GIFs.
#
# The lattice GIFs (Game of Life, elementary CA) are rendered in Python
# (python/render_animations.py): a raster wants pixels, not ggplot objects.
# Here ggplot is the right tool, because the moving objects are particles and
# continuous fields.
#
# gganimate/gifski are not installed on this machine, so every frame is a plain
# ggplot saved with ragg and the frames are stitched with magick.  Frames are
# drawn on a light background instead of transparent: GIF supports only 1-bit
# transparency and would fringe badly.
# =============================================================================
source("R/00_theme.R")
source("R/01_ca_models.R")
source("R/04_metrics.R")
source("R/02_vicsek_model.R")
source("R/03_cell_communication.R")

dir.create("animations", showWarnings = FALSE)

# ---- 1. Vicsek: ordering in real time --------------------------------------
v <- readRDS("results/e03_vicsek2d.rds")
mean_dir <- function(d) atan2(mean(sin(d$theta)), mean(cos(d$theta)))
vic_frames <- lapply(v$anim, function(d) {
  m0 <- mean_dir(d)
  d$align <- cos(d$theta - m0)
  ggplot(d, aes(x, y)) +
    geom_segment(aes(xend = x + 0.9 * cos(theta), yend = y + 0.9 * sin(theta),
                     colour = align),
                 arrow = arrow(length = unit(1.6, "mm")), linewidth = 0.35) +
    scale_colour_gradientn(colours = em_div, limits = c(-1, 1)) +
    coord_fixed(xlim = c(0, v$anim_L), ylim = c(0, v$anim_L), expand = FALSE) +
    labs(title = sprintf("Vicsek model, t = %d", d$t[1]), colour = "alignment") +
    theme(axis.text = element_blank(), axis.title = element_blank(),
          axis.ticks = element_blank(), panel.border = element_blank(),
          legend.position = "none",
          plot.background = element_rect(fill = em_pal[["bg"]], colour = NA))
})
files <- vapply(vic_frames, frame_png, character(1), width = 5, height = 5)
save_gif(files, "animations/vicsek.gif", fps = 12)

# ---- 2. Cell communication: spots growing out of one seed ------------------
p <- readRDS("results/e05_patterns.rds")$spots
pat_frames <- lapply(seq_along(p$snaps), function(i) {
  d <- as.data.frame(as.table(p$snaps[[i]]))
  names(d) <- c("y", "x", "v")
  d$x <- as.integer(d$x); d$y <- as.integer(d$y)
  ggplot(d, aes(x, y, fill = v)) +
    geom_raster() +
    scale_fill_gradientn(colours = c("#FFFFFF", em_pal[["pale_blue"]],
                                     em_pal[["blue"]], em_pal[["deep_blue"]]),
                         limits = c(0, 1), guide = "none") +
    coord_fixed(expand = FALSE) +
    labs(title = sprintf("cell signalling, t = %d", p$metrics$t[i])) +
    theme_void() +
    theme(plot.background = element_rect(fill = em_pal[["bg"]], colour = NA),
          plot.title = element_text(size = 12, hjust = 0.5, colour = em_pal[["grey"]]))
})
files <- vapply(pat_frames, frame_png, character(1), width = 5, height = 5)
save_gif(files, "animations/pattern.gif", fps = 8)

cat("gifs written:\n")
print(file.info(c("animations/vicsek.gif", "animations/pattern.gif"))[, "size", drop = FALSE])
