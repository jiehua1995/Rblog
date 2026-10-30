# =============================================================================
# 00_theme.R — palette and small helpers shared by all emergence experiments.
#
# The site theme lives in ../../theme-blog.R (Okabe-Ito).  This post uses a
# deliberately narrower red/blue language: blue = one state / disordered phase,
# red = the other state / ordered phase, grey = neutral scaffolding.
# Colours are defined once here and never written inline in a plot.
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

# The site theme lives at the repository root.  Experiments are run from the
# post directory, while knitr runs from the project root, so resolve the path
# instead of hard-coding it.
.find_site_root <- function() {
  if (file.exists("theme-blog.R")) "." else
    if (file.exists(file.path("..", "..", "theme-blog.R"))) file.path("..", "..") else "."
}
source(file.path(.find_site_root(), "theme-blog.R"))

em_pal <- c(
  deep_blue = "#1F4E79",
  blue      = "#4F81BD",
  pale_blue = "#B7CDE4",
  deep_red  = "#C0504D",
  red       = "#E46C6C",
  pale_red  = "#F2C4C2",
  grey      = "#666666",
  light     = "#D9D9D9",
  bg        = "#FAFAFA"
)

# diverging scale used for every "ordered vs disordered" continuous map
em_div <- c(em_pal[["deep_blue"]], em_pal[["blue"]], "#FFFFFF",
            em_pal[["red"]], em_pal[["deep_red"]])

# sequential blue used for densities that have no natural zero on a red-blue axis
em_seq <- c("#FFFFFF", em_pal[["pale_blue"]], em_pal[["blue"]], em_pal[["deep_blue"]])

scale_fill_em <- function(mid = 0.5, ...) {
  scale_fill_gradientn(colours = em_div, limits = c(0, 1), oob = squish, ...)
}

scale_colour_em <- function(...) {
  scale_colour_manual(values = c(disordered = em_pal[["blue"]],
                                 ordered    = em_pal[["deep_red"]]), ...)
}

# ---- GIF assembly ----------------------------------------------------------
# gganimate/gifski are not installed, so frames are rendered with ragg and
# stitched with magick.  Frames are drawn on the page background colour rather
# than transparent: GIF has only 1-bit transparency and would fringe badly.
save_gif <- function(frames, out, fps = 10, width = 640, height = 640) {
  stopifnot(length(frames) > 1)
  # magick only accepts frame rates that divide 100
  fps <- c(1, 2, 4, 5, 10, 20, 25, 50, 100)[which.min(abs(c(1, 2, 4, 5, 10, 20, 25, 50, 100) - fps))]
  img <- magick::image_join(lapply(frames, magick::image_read))
  anim <- magick::image_animate(img, fps = fps, dispose = "previous",
                                optimize = TRUE)
  magick::image_write(anim, out)
  invisible(out)
}

# render one frame of a raster/point plot to a temporary png
frame_png <- function(plot, width = 6, height = 6, dpi = 110, bg = em_pal[["bg"]]) {
  f <- tempfile(fileext = ".png")
  ggsave(f, plot, width = width, height = height, dpi = dpi, bg = bg)
  f
}

# ---- small numeric helpers -------------------------------------------------
sigmoid <- function(x) 1 / (1 + exp(-x))

# Shannon entropy of a discrete vector, in bits
entropy_bits <- function(p) {
  p <- p[p > 0]
  -sum(p * log2(p))
}
