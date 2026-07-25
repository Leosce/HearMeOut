#!/usr/bin/env pwsh
# Stage A: head-only fine-tune (encoder frozen) on joint manifest.
# Effective batch ~ max_tokens * update_freq = 800 * 4 = 3200 tokens.
#
# Env vars expected:
#   AVHUBERT_ROOT   path to cloned facebookresearch/av_hubert
#   CKPT_PATH       path to base_vsr_trainval.pt
#   BPE_MODEL       path to spm_unigram1000.model shipped with the ckpt
#   DATA_DIR        dir containing joint {train,valid}.{tsv,wrd}
#   EXP_DIR         run output dir
param(
    [string]$AvhubertRoot = $env:AVHUBERT_ROOT,
    [string]$CkptPath    = $env:CKPT_PATH,
    [string]$BpeModel    = $env:BPE_MODEL,
    [string]$DataDir     = $env:DATA_DIR,
    [string]$ExpDir      = $env:EXP_DIR
)

$ErrorActionPreference = "Stop"

foreach ($v in @("AvhubertRoot", "CkptPath", "BpeModel", "DataDir", "ExpDir")) {
    if (-not (Get-Variable $v).Value) { throw "Missing: $v" }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfDir  = Join-Path $RepoRoot "conf/finetune"
$UserDir  = Join-Path $RepoRoot "avhubert_ext"

$env:PYTHONPATH = "$AvhubertRoot/avhubert;$env:PYTHONPATH"

fairseq-hydra-train `
    --config-dir $ConfDir `
    --config-name miracl_joint_base `
    task.data=$DataDir `
    task.label_dir=$DataDir `
    task.tokenizer_bpe_model=$BpeModel `
    model.w2v_path=$CkptPath `
    common.user_dir=$UserDir `
    hydra.run.dir=$ExpDir
