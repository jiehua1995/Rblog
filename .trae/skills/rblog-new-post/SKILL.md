---
name: rblog-new-post
description: Write a new bilingual post for the Rblog Quarto site, including AI disclosure, a condensed prompt callout, paired Chinese and English paragraphs, folded code, transparent-background figures and DOI references. Use when the user asks to add or rewrite a post under posts/. Do not use for site-wide CSS, layout or deployment work.
---

# Rblog 新文章

仓库是一个 Quarto 站点（Netlify，发布到 `https://bioinfor.jiehua1995.xyz/`）。仓库可能被克隆到不同机器的不同路径，**本文所有路径都相对于仓库根目录**，任何地方都不要写死用户主目录或盘符。下面是从已发布文章里提炼出的固定写法，写新文章时按顺序执行，不要自行发明版式。

起稿可以直接复制 `assets/post-template.qmd`。

## 1. 建目录与 front matter

目录名：`posts/YYYY-MM-DD English Title in Title Case/index.qmd`，日期与 `date` 字段必须一致。日期可以早于今天（补写旧文是允许的），排序与归档都由这个字段决定。

```yaml
---
title: "English Title Only"
subtitle: "中文副标题"
description: "一句话中文摘要，会出现在列表页和 meta 标签里。"
author: "Jie Hua"
date: "YYYY-MM-DD"
categories: [Tooling]
engine: knitr
---
```

约定：

- `title` 必须是英文，`subtitle` 必须是中文（全站统一）。
- `categories` 用仓库里已有的单词分类：`Annotation`、`Assembly`、`Bash`、`Origins of Life`、`Plot`、`Proteomics`、`Single-cell`、`Structure`、`Tooling`。不要造新分类。
- 不要在 front matter 写 `code-fold`、`toc`、`freeze`、`author`，这些已在 `posts/_metadata.yml` 里统一设置。

## 2. 开头两个 callout

第一块声明 AI 参与，第二块放生成本文的提示词（默认折叠）。位置在 front matter 之后、第一个 `##` 之前。

```markdown
::: callout-important
## 写作说明 / Note

本文由 **AI（<模型名>）** 检索、整理并撰写，**尚未经过人工逐条校对**。<一句关于事实边界的说明，例如数字来源、版本可能变化、路径是本机路径。>

This article was researched and written by an **AI system (<model>)** and has **not been proofread line by line by a human**. <English mirror of the caveat.>
:::

::: {.callout-note collapse="true"}
## 生成本文所用的提示词（压缩后） / Prompt used to generate this article (condensed)

> <中文提示词：目标、结构、字数、必须覆盖的点、禁止的做法。>

> <English prompt, same content, can be slightly tighter.>
:::
```

禁止写"示意""示例提示词"这类措辞；标题固定用"压缩后 / condensed"。

## 3. setup chunk 与环境

每篇都要有，且必须 `include: false`。`theme-blog.R` 在仓库根目录，用相对路径引用：

```{r}
#| label: setup
#| include: false
source(file.path(dirname(knitr::current_input()), "..", "..", "theme-blog.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(knitr)
})
```

需要 Python 时再加 `library(reticulate)`。**不要让文章依赖某台机器的 Python 绝对路径**：

- 优先靠 `.Renviron`（已被 gitignore）里的 `RETICULATE_PYTHON`，或用户 shell 里已导出的同名变量；`.Renviron` 不随仓库同步，每台机器各自配置。
- 需要显式指定时，写成 `Sys.setenv(RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", unset = ""))` 这类"有就用、没有就交给 reticulate 自动发现"的形式，不要写死 `C:/...`。
- 文章里确实要展示路径（例如讲环境问题的段落）时，在正文说明"这是作者本机的路径"，不要让代码依赖它。
- 已知例外：`posts/2025-02-15 Testing R torch and PyTorch in Quarto/` 的 setup chunk 里写死了 Python 绝对路径，那是当时为了复现报错留下的；新文章不要照抄。

`library(torch)`（R 版）与 Python 的 PyTorch **不能同进程**：两边各带一份 libtorch，先加载的一方会让另一方报 `WinError 127`。两者都要用时，把 R 的那份放进独立 `.R` 脚本，用 `Rscript` 子进程跑，结果用 RDS/CSV 传回。

## 4. 双语正文（逐段对照，不是整篇翻译）

- 每个中文段落后面隔一个空行，紧跟对应英文段落；不是先写完整中文再写完整英文。
- 标题写成 `## 1. 中文标题 / English title`，编号连续。
- 列表：中文条目行尾加 `\` 强制换行，下一行缩进四个空格写 `*English.*`
- 表格：短表用双语表头（`引擎 / Engine`）；长文中的关键表格拆成两张独立表——先中文表，空行写一行 `*English*`，再英文表，表内文字不混排。
- 图注：`中文说明。 | English caption.`，在同一行用 `|` 隔开。
- 代码注释用英文；正文里的技术术语保留英文原词，不硬译。
- 中英段落数量必须一一对应；删段时成对删。

## 5. 图：透明背景 + 统一主题

`theme-blog.R` 已通过 `knitr::opts_chunk$set()` 全局设定 `dev = "ragg_png"`、`dev.args = list(bg = "transparent")`、`fig.width = 8`、`fig.height = 4.4`、`dpi = 200`、`out.width = "100%"`，并 `theme_set(theme_blog())`。因此：

- 直接写 `ggplot(...)` 即可，不要再叠加 `theme_minimal()`、不要设 `bg`。
- 颜色只用 `blog_pal[["blue"]]`、`blog_pal[["vermillion"]]` 等 Okabe-Ito 名称，或 `pal`、`blog_gradient`，不要手写十六进制（除浅色填充块）。
- 图里不放网格线、不加 `theme(legend.position = "bottom")`，这些由主题决定。
- 每张图必须有 `fig-cap`，中英对照。
- 多面板用 `facet_wrap()`；宽图把 `fig-width` 提到 9 左右，窄图 7 左右。
- 生成后用 Read 工具直接看 PNG，确认没有被裁成细条、没有空面板、分类编码正确（例如 R 的 `argmax` 是 1 起始，别把 0/1 画成 NA 灰块）。

## 6. 参考文献

放最后一节 `## 参考文献 / References`，编号列表，格式：

```markdown
1.  Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature* 618, 616–624. doi:10.1038/s41586-023-06139-9
```

- 已发表论文给作者、年份、标题、期刊、卷页、DOI；软件或数据库给标题加官方仓库 URL。
- 不要凭记忆写 DOI；不确定就只给仓库链接或标注"待核"。
- 结尾可加一段引用说明（哪些有 DOI、哪些是仓库），中英各一句。

## 7. 渲染与校验

先在新机器上跑一次环境探测，确认缺什么（只读，不会安装任何东西）：

```bash
Rscript --no-init-file .trae/skills/rblog-new-post/scripts/probe-env.R
```

`--no-init-file` 让它跳过 `.Rprofile`，否则新克隆的机器会因为缺 `renv/activate.R` 直接报错。输出包含 R 与 Rscript 路径、Quarto 版本、git、bash、`RETICULATE_PYTHON`、关键 R 包版本，以及当前 Python 能否 `import torch`。缺 Quarto 先装 Quarto CLI；缺 R 包用 `install.packages()`；纯 R 的文章不需要 Python。

渲染单篇（macOS/Linux 的 bash/zsh 版本）：

```bash
# 仓库里的 .Rprofile 引用 renv/activate.R，在没初始化 renv 的机器上会报错
[ -f .Rprofile ] && mv .Rprofile .Rprofile.bak
quarto render "posts/YYYY-MM-DD Title/index.qmd"
[ -f .Rprofile.bak ] && mv .Rprofile.bak .Rprofile
```

Windows PowerShell 版本：

```powershell
if (Test-Path .Rprofile) { Rename-Item .Rprofile .Rprofile.bak -Force }
try { quarto render "posts/YYYY-MM-DD Title/index.qmd" 2>&1 | Out-String }
finally { if (Test-Path .Rprofile.bak) { Rename-Item .Rprofile.bak .Rprofile -Force } }
```

- 路径含空格，必须整体加引号；不要一次传多个文件（pandoc 会报错）。
- `freeze: true` 是最大的坑：改正文会重渲染，但**改了外部脚本、数据或 `theme-blog.R` 后必须删该文章的缓存目录**再渲染，否则页面还是旧图。缓存目录是 `_freeze/posts/<文章目录名>/`。
- 渲染完成看三处：`_site/posts/.../index.html` 里的数字与正文是否一致、`index_files/figure-html/*.png` 是否正常、`_site/articles.html` 是否收录（首页只显示最新 7 篇，补写的旧文只在归档页出现）。
- 需要看整页版式时，用 `python -m http.server --directory _site`（或 `quarto preview`）起本地服务，再在浏览器里检查窄窗口下的表格与目录。

## 8. 发布

```bash
quarto publish netlify --no-prompt     # 先整站渲染，再上传
```

发布前确认工作区只剩本次改动；发布后再提交，提交信息写"为什么"而不是"改了什么"。`_site/` 在 `.gitignore` 里但部分文件仍被跟踪，改动这些文件要用 `git add -f`。每台机器都需要各自的 Netlify 授权（`netlify login` 或 `NETLIFY_AUTH_TOKEN`），授权状态不随仓库同步。

## 9. 跨机器同步

仓库本身就是同步机制，没有第二份副本需要维护：

| 内容 | 位置 | 是否随 GitHub 同步 |
|---|---|---|
| 本 skill 与模板 | `.trae/skills/rblog-new-post/` | 是（项目级 skill，提交即可） |
| 文章与图 | `posts/**` | 是 |
| 计算缓存 | `_freeze/posts/**` | 是（所以别的机器不必重跑） |
| 构建产物 | `_site/**` | 部分被跟踪，其余忽略 |
| `.Rprofile`、`.Renviron` | 仓库根目录 | 否（每台机器各自一份） |
| Netlify 授权、Python 路径 | 机器本地 | 否 |

因此：

- 在另一台机器上 `git clone` 之后，Trae 打开该仓库就会把 `.trae/skills/` 识别为项目级 skill，不需要重新安装；要更新就改仓库里的文件再提交。
- 只在一台机器上写文章、另一台只负责构建或发布时，**不要删别人的 `_freeze`**，否则会在那台机器上重跑并覆盖数字。
- 计时、版本号、绝对路径这类"机器指纹"会随机器变化。已经在 `_freeze` 里的旧文章不会被重跑，数字保持不变；重新渲染后数字会变，属于预期。
- 换行符：Windows 上 `git status` 会因为 CRLF 反复提示，提交前用 `git diff --stat` 确认只改了该改的文件。
- 新增机器相关配置一律写进 `.Renviron` 或 `.Rprofile`（均不提交），不要写进 `index.qmd`。

## 10. 交稿前自检

1. `title` 英文、`subtitle` 中文、`date` 与目录名一致、`categories` 属于既有集合。
2. 两个 callout 齐全，提示词那块默认折叠。
3. 每一段中文都有对应英文段落，标题是 `中文 / English` 格式。
4. 代码默认折叠，没有裸露的大段代码；伪代码用 `#| eval: false`，但**不要用 `{julia}`**——没有 JuliaCall 时 `eval: false` 也会中断渲染。
5. 所有图透明背景、配色来自 `blog_pal`、图注中英对照、缩略图没被裁坏。
6. 参考文献有 DOI 或仓库链接，没有编造条目。
7. 所有数字来自实际运行；正文里的数字用 inline R 生成（`` `r sprintf("%.2f", x)` ``），不要手抄。
8. 全文没有写死本机路径（`C:/`、`/Users/`、`/home/`）；必须在正文展示时已说明是作者机器。
9. 提交前 `git status` 只含本次改动，且 `.Renviron`、`.Rprofile` 没有被加进去。
