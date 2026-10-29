#!/usr/bin/env bash
# Render one post, then print the files worth inspecting.
#
#   .trae/skills/rblog-new-post/scripts/render-post.sh "posts/YYYY-MM-DD Title"
#   .trae/skills/rblog-new-post/scripts/render-post.sh "posts/YYYY-MM-DD Title" --fresh
#
# --fresh drops that post's _freeze cache first. Use it whenever the post reads
# an external file (data, another script, theme-blog.R) that changed, because
# freeze would otherwise keep serving the previous output.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$root"

if [ $# -lt 1 ]; then
  echo "usage: $0 \"posts/YYYY-MM-DD Title\" [--fresh]" >&2
  echo "available posts:" >&2
  ls -1 posts | grep -v '^_' | sed 's/^/  /' >&2
  exit 2
fi

post="${1%/}"
fresh="${2:-}"
src="$post/index.qmd"

if [ ! -f "$src" ]; then
  echo "no such file: $src" >&2
  exit 1
fi

if [ "$fresh" = "--fresh" ]; then
  rm -rf "_freeze/$post"
  echo "removed _freeze/$post"
fi

quarto render "$src"

echo
echo "inspect:"
echo "  _site/$post/index.html"
echo "  _site/$post/index_files/figure-html/"
