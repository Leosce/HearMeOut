#!/usr/bin/env pwsh
# Convert MIRACL-VC1 PNG/JPG sequences (native ~15 fps, 640x480) into
# 25 fps mp4 clips so they match AV-HuBERT's expected frame rate.
#
# MIRACL-VC1 folder structure:
#   $ROOT/<speaker>/<words|phrases>/<id>/<instance>/color_*.jpg
#                                                   depth_*.jpg  (ignored)
#
# Output: one <instance>.mp4 next to each instance folder.
#
# Usage:  pwsh scripts/01_miracl_png_to_mp4.sh <MIRACL_ROOT>
param(
    [Parameter(Mandatory = $true)][string]$Root
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg not found on PATH"
}

$instances = Get-ChildItem -Path $Root -Directory -Recurse -Depth 4 |
    Where-Object {
        # Path depth: <ROOT>/<spk>/<words|phrases>/<id>/<instance>
        $rel = $_.FullName.Substring($Root.Length).TrimStart('\', '/')
        ($rel -split '[\\/]').Length -eq 4 -and
        (Test-Path (Join-Path $_.FullName "color_001.jpg"))
    }

Write-Host "Found $($instances.Count) instances to convert"

$i = 0
foreach ($inst in $instances) {
    $i++
    $outMp4 = "$($inst.FullName).mp4"
    if (Test-Path $outMp4) { continue }

    # MIRACL-VC1 frames are named color_001.jpg, color_002.jpg, ...
    # Use -start_number 1 and let ffmpeg stop at first missing index.
    & ffmpeg -hide_banner -loglevel error -y `
        -framerate 15 -start_number 1 `
        -i (Join-Path $inst.FullName "color_%03d.jpg") `
        -vf "fps=25,format=yuv420p" `
        -c:v libx264 -crf 18 -pix_fmt yuv420p `
        $outMp4

    if ($i % 100 -eq 0) { Write-Host "  $i / $($instances.Count)" }
}

Write-Host "Done. mp4 clips written next to instance folders."
