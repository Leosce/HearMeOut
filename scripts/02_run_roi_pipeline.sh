#!/usr/bin/env pwsh
# Run AV-HuBERT's stock mouth-ROI pipeline on the MIRACL-VC1 mp4 clips
# produced by 01_miracl_png_to_mp4.sh.
#
# Outputs (reused by later manifest scripts):
#   $MIRACL_ROOT/file.list         (relative ids, one per line, no ext)
#   $MIRACL_ROOT/landmark/...      (dlib landmarks)
#   $MIRACL_ROOT/video/<id>.mp4    (88x88 grayscale, mean-face aligned, 25 fps)
#   $MIRACL_ROOT/nframes.video     (int per line, same order as file.list)
#
# Assumes the following dlib / AV-HuBERT assets are on disk:
#   mmod_human_face_detector.dat
#   shape_predictor_68_face_landmarks.dat
#   20words_mean_face.npy
#
# Usage:
#   pwsh scripts/02_run_roi_pipeline.sh <MIRACL_ROOT> <AVHUBERT_ROOT> <ASSETS_DIR>
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$AvhubertRoot,
    [Parameter(Mandatory = $true)][string]$AssetsDir
)

$ErrorActionPreference = "Stop"
$prep = Join-Path $AvhubertRoot "avhubert/preparation"

# 1. Enumerate mp4s into file.list (id = relative path without .mp4).
$fileList = Join-Path $Root "file.list"
Get-ChildItem -Path $Root -Filter *.mp4 -Recurse |
    ForEach-Object {
        $rel = $_.FullName.Substring($Root.Length).TrimStart('\', '/')
        $rel -replace '\\', '/' -replace '\.mp4$', ''
    } | Set-Content -Encoding ascii $fileList

Write-Host "Wrote $fileList ($((Get-Content $fileList).Count) entries)"

# 2. Landmark detection.
python (Join-Path $prep "detect_landmark.py") `
    --root $Root `
    --landmark (Join-Path $Root "landmark") `
    --manifest $fileList `
    --cnn_detector (Join-Path $AssetsDir "mmod_human_face_detector.dat") `
    --face_detector (Join-Path $AssetsDir "shape_predictor_68_face_landmarks.dat") `
    --ffmpeg (Get-Command ffmpeg).Source `
    --rank 0 --nshard 1

# 3. Mean-face alignment + 88x88 grayscale crop.
python (Join-Path $prep "align_mouth.py") `
    --video-direc $Root `
    --landmark (Join-Path $Root "landmark") `
    --filename-path $fileList `
    --save-direc (Join-Path $Root "video") `
    --mean-face (Join-Path $AssetsDir "20words_mean_face.npy") `
    --ffmpeg (Get-Command ffmpeg).Source `
    --rank 0 --nshard 1

# 4. Frame counting.
python (Join-Path $prep "count_frames.py") `
    --root $Root --manifest $fileList --nshard 1 --rank 0

# count_frames.py writes nframes.video.0 / nframes.audio.0 shards; merge.
Get-Content (Join-Path $Root "nframes.video.0") |
    Set-Content -Encoding ascii (Join-Path $Root "nframes.video")

Write-Host "ROI pipeline complete: $(Join-Path $Root 'video') populated."
