#!/usr/bin/env pwsh
# End-to-end smoke test on a single MIRACL-VC1 speaker (F01 by default).
#
# Goal: validate the entire pipeline (preprocess -> manifest -> train -> eval)
# with a ~300 MB data download and ~10 minutes of training on a 16 GB GPU.
#
# This DOES NOT exercise anti-forgetting (LRS3 is not mixed in). For that,
# use the full train_stage_a.sh path. The smoke test is for plumbing only.
#
# Expected env vars:
#   AVHUBERT_ROOT   cloned av_hubert
#   MIRACL_ROOT     extracted MIRACL-VC1 containing just F01/ (or more)
#   ASSETS_DIR      dlib .dat files + 20words_mean_face.npy
#   CKPT_PATH       base_vsr_trainval.pt
#   BPE_MODEL       spm_unigram1000.model
#
# Run from repo root:  pwsh scripts/smoke_test.sh
param(
    [string]$AvhubertRoot = $env:AVHUBERT_ROOT,
    [string]$MiraclRoot   = $env:MIRACL_ROOT,
    [string]$AssetsDir    = $env:ASSETS_DIR,
    [string]$CkptPath     = $env:CKPT_PATH,
    [string]$BpeModel     = $env:BPE_MODEL
)

$ErrorActionPreference = "Stop"
foreach ($v in @("AvhubertRoot","MiraclRoot","AssetsDir","CkptPath","BpeModel")) {
    if (-not (Get-Variable $v).Value) { throw "Missing env var: $v" }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ---- 1. PNG sequences -> 25 fps mp4 ----------------------------------------
Write-Host "`n== Step 1: PNG -> mp4 =="
pwsh scripts/01_miracl_png_to_mp4.sh -Root $MiraclRoot

# ---- 2. ROI extraction ------------------------------------------------------
Write-Host "`n== Step 2: mouth ROI pipeline =="
pwsh scripts/02_run_roi_pipeline.sh -Root $MiraclRoot -AvhubertRoot $AvhubertRoot -AssetsDir $AssetsDir

# ---- 3. Manifest (12/1/2 default split still works if F01 in train list) ----
Write-Host "`n== Step 3: manifests =="
# For a 1-speaker smoke test, put F01 into all three splits by overriding.
# Loss numbers are meaningless — we only care that training runs end to end.
python scripts/03_build_miracl_manifest.py `
    --root $MiraclRoot --out data/miracl `
    --train-spk F01 --valid-spk F01 --test-spk F01

# Build a "joint" dir that is in fact just MIRACL (no LRS3, no GRID).
# We can reuse the join script with only one source, or just copy.
New-Item -ItemType Directory -Force -Path data/joint_smoke | Out-Null
Copy-Item data/miracl/train.tsv data/joint_smoke/train.tsv -Force
Copy-Item data/miracl/train.wrd data/joint_smoke/train.wrd -Force
Copy-Item data/miracl/valid.tsv data/joint_smoke/valid.tsv -Force
Copy-Item data/miracl/valid.wrd data/joint_smoke/valid.wrd -Force

# ---- 4. Tiny training run (50 updates) -------------------------------------
Write-Host "`n== Step 4: 50-update training =="
$expDir = Join-Path $RepoRoot "runs/smoke_stage_a"
New-Item -ItemType Directory -Force -Path $expDir | Out-Null

$env:PYTHONPATH = "$AvhubertRoot/avhubert;$env:PYTHONPATH"
fairseq-hydra-train `
    --config-dir (Join-Path $RepoRoot "conf/finetune") `
    --config-name miracl_joint_base `
    task.data=(Resolve-Path data/joint_smoke).Path `
    task.label_dir=(Resolve-Path data/joint_smoke).Path `
    task.tokenizer_bpe_model=$BpeModel `
    model.w2v_path=$CkptPath `
    common.user_dir=(Join-Path $RepoRoot "avhubert_ext") `
    optimization.max_update=50 `
    lr_scheduler.warmup_steps=10 lr_scheduler.decay_steps=40 `
    dataset.max_tokens=400 dataset.validate_interval_updates=25 `
    checkpoint.save_interval_updates=50 `
    hydra.run.dir=$expDir

# ---- 5. Decode: baseline checkpoint AND smoke-trained checkpoint -----------
Write-Host "`n== Step 5: decode MIRACL test split =="
$env:LRS3_TEST_DIR   = (Resolve-Path data/miracl).Path   # not really LRS3; reused slot
$env:MIRACL_TEST_DIR = (Resolve-Path data/miracl).Path
$env:RESULTS_DIR     = Join-Path $RepoRoot "runs/smoke_eval_baseline"
$env:CKPT_EVAL       = $CkptPath
Write-Host "--- baseline checkpoint ---"
pwsh scripts/eval_retention.sh

$env:RESULTS_DIR = Join-Path $RepoRoot "runs/smoke_eval_trained"
$env:CKPT_EVAL   = Join-Path $expDir "checkpoints/checkpoint_last.pt"
Write-Host "--- smoke-trained checkpoint ---"
pwsh scripts/eval_retention.sh

Write-Host "`nSmoke test complete. Compare hypo-*.txt under:"
Write-Host "  runs/smoke_eval_baseline/"
Write-Host "  runs/smoke_eval_trained/"
Write-Host "If you see reasonable English lip-reading on the baseline and"
Write-Host "the trained run did not crash, the pipeline is wired correctly."
