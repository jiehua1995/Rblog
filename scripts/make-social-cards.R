# Generate 1200x630 social preview cards for every post, and make sure each
# post's front matter points at its card via `image:` (used by og:image and
# twitter:image). Run from the repository root:
#
#   Rscript scripts/make-social-cards.R
#
# Re-running is safe: existing `image:` entries are left untouched.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ragg)
})

dir.create("images/social", showWarnings = FALSE, recursive = TRUE)

slugify <- function(title, date) {
  s <- tolower(title)
  s <- gsub("[^a-z0-9]+", "-", s)
  s <- gsub("(^-+|-+$)", "", s)
  s <- substr(s, 1, 58)
  s <- gsub("-+$", "", s)
  paste0(date, "-", s)
}

accent <- "#0f766e"
ink    <- "#121820"
muted  <- "#5b6772"
faint  <- "#8a949e"
paper  <- "#fbfcfd"

# The default "sans" family has no CJK coverage on Windows, so Chinese subtitles
# need a family that does. Fall back to "sans" elsewhere.
cjk_family <- if (.Platform$OS.type == "windows") "Microsoft YaHei" else "sans"

wrap_cjk <- function(x, n = 30) {
  x <- trimws(x)
  if (is.na(x) || !nzchar(x)) return("")
  if (nchar(x, type = "width") <= n) return(x)
  chars <- strsplit(x, "")[[1]]
  idx <- seq_along(chars)
  cuts <- ceiling(idx / n)
  paste(tapply(chars, cuts, paste, collapse = ""), collapse = "\n")
}

wrap_latin <- function(x, n = 24) {
  x <- trimws(x)
  if (is.na(x) || !nzchar(x)) return("")
  paste(strwrap(x, width = n), collapse = "\n")
}

card <- function(title, subtitle, date, out) {
  ttl <- wrap_latin(title, 26)
  subtitle <- if (is.na(subtitle)) "" else wrap_cjk(subtitle, 34)
  n_lines <- length(strsplit(ttl, "\n")[[1]])
  tsize <- if (n_lines >= 3) 11.2 else 13
  line_h <- tsize * 4.5

  sub_h <- if (nzchar(subtitle)) {
    24 + length(strsplit(subtitle, "\n")[[1]]) * 29
  } else 0
  title_y <- 330 + (n_lines * line_h + sub_h) / 2
  sub_y   <- title_y - n_lines * line_h - 24

  agg_png(out, width = 1200, height = 630, units = "px", res = 100,
          background = paper)
  on.exit(dev.off(), add = TRUE)

  p <- ggplot() +
    annotate("rect", xmin = 0, xmax = 1200, ymin = 0, ymax = 7, fill = accent) +
    annotate("rect", xmin = 0, xmax = 1200, ymin = 623, ymax = 630, fill = accent) +
    annotate("segment", x = 72, xend = 124, y = 552, yend = 552,
             colour = accent, linewidth = 1.5) +
    annotate("text", x = 136, y = 552, label = "BIOINFORMATICS NOTES",
             hjust = 0, vjust = 0.5, size = 3.0, colour = accent) +
    annotate("text", x = 72, y = title_y, label = ttl, hjust = 0, vjust = 1,
             size = tsize, lineheight = 1.05, colour = ink, family = "sans") +
    annotate("text", x = 72, y = 52, label = date, hjust = 0, vjust = 1,
             size = 3.1, colour = faint) +
    annotate("text", x = 1128, y = 52, label = "bioinfor.jiehua1995.xyz",
             hjust = 1, vjust = 1, size = 3.1, colour = faint) +
    coord_cartesian(xlim = c(0, 1200), ylim = c(0, 630), expand = FALSE) +
    theme_void() +
    theme(plot.background = element_rect(fill = paper, colour = NA),
          plot.margin = margin(0, 0, 0, 0))

  if (nzchar(subtitle)) {
    p <- p + annotate("text", x = 72, y = sub_y, label = subtitle,
                      hjust = 0, vjust = 1, size = 5.2, lineheight = 1.45,
                      colour = muted, family = cjk_family)
  }
  print(p)
}

get_field <- function(fm, key) {
  hit <- grep(paste0("^", key, ":[[:space:]]*"), fm)
  if (!length(hit)) return(NA_character_)
  v <- sub(paste0("^", key, ":[[:space:]]*"), "", fm[hit[1]])
  v <- sub('^"', "", v); v <- sub('"$', "", v)
  v <- sub("^'", "", v); v <- sub("'$", "", v)
  trimws(v)
}

posts <- sort(list.dirs("posts", recursive = FALSE))
changed <- character(0)

for (dir in posts) {
  qmd <- file.path(dir, "index.qmd")
  if (!file.exists(qmd)) next

  lines <- readLines(qmd, warn = FALSE, encoding = "UTF-8")
  if (!identical(trimws(lines[1]), "---")) next
  close_at <- which(trimws(lines) == "---")
  if (length(close_at) < 2) next
  fm <- lines[2:(close_at[2] - 1)]

  title    <- get_field(fm, "title")
  subtitle <- get_field(fm, "subtitle")
  date     <- get_field(fm, "date")
  if (is.na(title)) next

  # cards live in a spaces-free directory so every crawler can fetch them;
  # post folders contain spaces, which breaks some share previews
  slug <- slugify(title, if (is.na(date)) "post" else date)
  out  <- file.path("images", "social", paste0(slug, ".png"))
  image_value <- paste0("/images/social/", slug, ".png")

  card(title,
       if (is.na(subtitle)) "" else subtitle,
       if (is.na(date)) "" else date,
       out)

  current <- grep("^image:[[:space:]]*", fm, value = TRUE)
  current <- if (length(current)) trimws(sub("^image:[[:space:]]*", "", current[1])) else NA

  if (is.na(current) || identical(current, "social-card.png")) {
    if (is.na(current)) {
      at <- grep("^date:[[:space:]]*", fm)
      at <- if (length(at)) at[1] else length(fm)
      fm <- append(fm, paste0("image: ", image_value), after = at)
    } else {
      fm[grep("^image:[[:space:]]*", fm)[1]] <- paste0("image: ", image_value)
    }
    writeLines(c("---", fm, lines[close_at[2]:length(lines)]), qmd,
               useBytes = TRUE)
    changed <- c(changed, dir)
  }
}

# Site-wide fallback used by the home page and other non-post pages.
card("Jie Hua's Bioinformatics Notes",
     "可复现的生信笔记：基因组、结构与数据可视化",
     "", "images/social-card.png")

cat("cards written:", length(posts), "\n")
cat("front matter updated:", length(changed), "\n")
if (length(changed)) cat(paste0("  ", changed, collapse = "\n"), "\n")
