# =============================================================================
# theme-blog.R
# 全站统一的 R 绘图风格：
#   - 透明背景（panel / plot / legend / strip）
#   - 色盲友好的 Okabe-Ito 配色
#   - 统一的边框粗细、留白与图例位置
# 在文章的 setup chunk 中通过 source() 调用，不要在每张图里重复设置。
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

# ---- 色盲友好配色（Okabe-Ito） ----
blog_pal <- c(
  blue       = "#0072B2",
  vermillion = "#D55E00",
  green      = "#009E73",
  purple     = "#CC79A7",
  orange     = "#E69F00",
  sky        = "#56B4E9",
  yellow     = "#F0E442",
  grey       = "#7F7F7F"
)

# 顺序配色：离散变量直接按顺序取用
pal <- unname(blog_pal[c("blue", "vermillion", "green", "purple", "orange", "sky", "yellow")])

# 连续配色：低 → 高
blog_gradient <- c("#56B4E9", "#0072B2", "#CC79A7", "#D55E00")

# ---- 统一主题 ----
theme_blog <- function(base_size = 11, base_family = "sans", border = "black") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # 透明背景
      panel.background  = element_rect(fill = "transparent", color = NA),
      plot.background   = element_rect(fill = "transparent", color = NA),
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      strip.background  = element_rect(fill = "transparent", color = NA),
      legend.key        = element_rect(fill = "transparent", color = NA),
      # 网格与边框
      panel.grid        = element_blank(),
      panel.spacing.x   = unit(0.6, "lines"),
      panel.spacing.y   = unit(0.5, "lines"),
      panel.border      = element_rect(color = border, fill = NA, linewidth = 0.3),
      axis.ticks        = element_line(colour = border, linewidth = 0.3),
      # 文字
      plot.title        = element_text(face = "bold"),
      plot.subtitle     = element_text(colour = "grey35"),
      legend.position   = "top",
      legend.title      = element_text(size = rel(0.9)),
      legend.text       = element_text(size = rel(0.85)),
      strip.text        = element_text(face = "bold")
    )
}

theme_set(theme_blog())

# ---- 给自带主题的绘图包追加统一样式 ----
# 适用于 plotthis、ggtree 等返回 ggplot 对象的包：plot + theme_blog_overlay
theme_blog_overlay <- theme(
  panel.background  = element_rect(fill = "transparent", color = NA),
  plot.background   = element_rect(fill = "transparent", color = NA),
  legend.background = element_rect(fill = "transparent", color = NA),
  legend.box.background = element_rect(fill = "transparent", color = NA),
  legend.key        = element_rect(fill = "transparent", color = NA),
  strip.background  = element_rect(fill = "transparent", color = NA),
  panel.grid        = element_blank(),
  panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.3),
  axis.ticks        = element_line(colour = "black", linewidth = 0.3)
)

# tidyplots 对象不是 ggplot，需要用 adjust_theme_details() 追加
blog_tidyplots_theme <- function(plot) {
  tidyplots::adjust_theme_details(
    plot,
    panel.background  = element_rect(fill = "transparent", color = NA),
    plot.background   = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key        = element_rect(fill = "transparent", color = NA),
    strip.background  = element_rect(fill = "transparent", color = NA),
    panel.grid        = element_blank(),
    panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.3),
    axis.ticks        = element_line(colour = "black", linewidth = 0.3)
  )
}

# ---- 全局 chunk 选项：透明背景设备 ----
if (requireNamespace("knitr", quietly = TRUE)) {
  knitr::opts_chunk$set(
    dev = "ragg_png",
    dev.args = list(bg = "transparent"),
    fig.width = 8, fig.height = 4.4, dpi = 200, out.width = "100%",
    warning = FALSE, message = FALSE, comment = ""
  )
}
