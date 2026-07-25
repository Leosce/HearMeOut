#!/usr/bin/env pwsh
# Decode on (a) LRS3 test (retention gate: <= baseline + 1.0 WER) and
# (b) MIRACL-VC1 test (target: >= 20% relative WER drop vs. baseline).
#
# Env vars:
#   AVHUBERT_ROOT   cloned av_hubert
#   CKPT_EVAL       checkpoint to evaluate (baseline or stage A/B best)
#   LRS3_TEST_DIR   dir containing test.tsv/test.wrd for LRS3
#   MIRACL_TEST_DIR dir containing test.tsv/test.wrd for MIRACL-VC1
#   RESULTS_DIR     output directory
param(
    [string]$AvhubertRoot = $env:AVHUBERT_ROOT,
    [string]$CkptEval    = $env:CKPT_EVAL,
    [string]$Lrs3TestDir = $env:LRS3_TEST_DIR,
    [string]$MiraclTestDir = $env:MIRACL_TEST_DIR,
    [string]$ResultsDir  = $env:RESULTS_DIR
)

$ErrorActionPreference = "Stop"
foreach ($v in @("AvhubertRoot","CkptEval","Lrs3TestDir","MiraclTestDir","ResultsDir")) {
    if (-not (Get-Variable $v).Value) { throw "Missing: $v" }
}

$infer = Join-Path $AvhubertRoot "avhubert/infer_s2s.py"
$conf  = Join-Path $AvhubertRoot "avhubert/conf"

foreach ($pair in @(
    @{ name = "lrs3_test"; dir = $Lrs3TestDir },
    @{ name = "miracl_test"; dir = $MiraclTestDir }
)) {
    $out = Join-Path $ResultsDir $pair.name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "=== Decoding $($pair.name) ==="
    python -B $infer `
        --config-dir $conf `
        --config-name s2s_decode `
        dataset.gen_subset=test `
        common_eval.path=$CkptEval `
        common_eval.results_path=$out `
        "+override.modalities=[video]" `
        "+override.data=$($pair.dir)" `
        "+override.label_dir=$($pair.dir)" `
        common.user_dir=(Join-Path $AvhubertRoot "avhubert")
}

Write-Host "Done. WER printed above and per-clip outputs under $ResultsDir."
