# Render one post, then print the files worth inspecting.
#
#   powershell -File .trae\skills\rblog-new-post\scripts\render-post.ps1 -Post "posts\YYYY-MM-DD Title"
#   powershell -File .trae\skills\rblog-new-post\scripts\render-post.ps1 -Post "posts\YYYY-MM-DD Title" -Fresh
#
# -Fresh drops that post's _freeze cache first. Use it whenever the post reads
# an external file (data, another script, theme-blog.R) that changed, because
# freeze would otherwise keep serving the previous output.

param(
  [Parameter(Mandatory = $true)][string]$Post,
  [switch]$Fresh
)

$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
Set-Location $root

$Post = $Post.TrimEnd('\', '/')
$src = Join-Path $Post 'index.qmd'
if (-not (Test-Path $src)) { throw "no such file: $src" }

if ($Fresh) {
  $cache = Join-Path '_freeze' $Post
  if (Test-Path $cache) {
    Remove-Item -Recurse -Force $cache
    Write-Host "removed $cache"
  }
}

quarto render $src

Write-Host ""
Write-Host "inspect:"
Write-Host ("  _site/{0}/index.html" -f $Post)
Write-Host ("  _site/{0}/index_files/figure-html/" -f $Post)
