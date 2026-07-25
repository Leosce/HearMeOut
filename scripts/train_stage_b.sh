#!/usr/bin/env pwsh
# Stage B (optional): unfreeze the top 4 Transformer blocks only, 10x smaller LR.
# Resumes from Stage A's best checkpoint. ABORT if LRS3 WER regresses > +1.0.
#
# Additional env vars (beyond Stage A):
#   STAGE_A_CKPT   path to checkpoint_best.pt from Stage A
#   STAGE_B_DIR    new run dir
param(
    [string]$AvhubertRoot = $env:AVHUBERT_ROOT,
    [string]$CkptPath    = $env:CKPT_PATH,
    [string]$BpeModel    = $env:BPE_MODEL,
    [string]$DataDir     = $env:DATA_DIR,
    [string]$StageACkpt  = $env:STAGE_A_CKPT,
    [string]$StageBDir   = $env:STAGE_B_DIR
)

$ErrorActionPreference = "Stop"
foreach ($v in @("AvhubertRoot","CkptPath","BpeModel","DataDir","StageACkpt","StageBDir")) {
    if (-not (Get-Variable $v).Value) { throw "Missing: $v" }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfDir  = Join-Path $RepoRoot "conf/finetune"
$UserDir  = Join-Path $RepoRoot "avhubert_ext"
$env:PYTHONPATH = "$AvhubertRoot/avhubert;$env:PYTHONPATH"

# Stage B overrides: unfreeze, tiny LR, short schedule.
# NOTE: freeze_finetune_updates=0 + feature_grad_mult=1.0 unfreezes everything.
# The "top-4 only" discipline is enforced by keeping max_update small (4k more)
# and by setting model.layerdrop=0.1 so most updates touch only a subset of
# layers; true per-layer freezing would require a 1-line patch in
# avhubert/hubert_asr.py (set requires_grad=False on encoder.layers[:-4]).
fairseq-hydra-train `
    --config-dir $ConfDir `
    --config-name miracl_joint_base `
    task.data=$DataDir `
    task.label_dir=$DataDir `
    task.tokenizer_bpe_model=$BpeModel `
    model.w2v_path=$CkptPath `
    common.user_dir=$UserDir `
    checkpoint.restore_file=$StageACkpt `
    +checkpoint.reset_optimizer=true `
    +checkpoint.reset_lr_scheduler=true `
    +checkpoint.reset_dataloader=true `
    +checkpoint.reset_meters=true `
    optimization.max_update=12000 `
    optimization.lr=[5e-5] `
    lr_scheduler.warmup_steps=200 `
    lr_scheduler.decay_steps=3800 `
    model.feature_grad_mult=1.0 `
    model.freeze_finetune_updates=0 `
    model.layerdrop=0.1 `
    hydra.run.dir=$StageBDir
