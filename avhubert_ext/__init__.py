"""Fairseq user_dir package.

Registers a ConcatWeightedDataset-based alternative to the stock
av_hubert_pretraining task, under task name `joint_finetune`.

The default training path uses the *line-replicated* joint manifest built by
scripts/05_build_joint_manifest.py and does NOT need this module. Activate
this path only if you prefer sampling-weight control over manifest surgery:

  task._name=joint_finetune \
  +task.sub_manifests="[/path/to/lrs3,/path/to/miracl,/path/to/grid]" \
  +task.sub_weights="[0.70,0.20,0.10]"
"""
from . import joint_finetune_task  # noqa: F401
