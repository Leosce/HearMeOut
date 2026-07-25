"""Fairseq task `joint_finetune` = av_hubert_pretraining + weighted concat.

Extends the stock AVHubertPretrainingTask so that when sub_manifests is set
for a given split, load_dataset builds one AVHubertDataset per sub-manifest
and wraps them in ConcatWeightedDataset. Otherwise behaviour is unchanged
(falls through to the parent implementation).

This is an OPT-IN path. The default training command in this repo uses the
line-replicated joint manifest (scripts/05_build_joint_manifest.py) and the
stock `av_hubert_pretraining` task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from fairseq.tasks import register_task

# Imported from the upstream avhubert package (added to PYTHONPATH by the
# caller via common.user_dir of both avhubert/ and this package).
from hubert_pretraining import (  # type: ignore
    AVHubertPretrainingConfig,
    AVHubertPretrainingTask,
)

from .concat_weighted_dataset import ConcatWeightedDataset


@dataclass
class JointFinetuneConfig(AVHubertPretrainingConfig):
    sub_manifests: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of sub-manifest dirs; each must contain "
                          "{split}.tsv/{split}.wrd. Applied to train split only."},
    )
    sub_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Sampling weights, one per sub_manifests entry."},
    )


@register_task("joint_finetune", dataclass=JointFinetuneConfig)
class JointFinetuneTask(AVHubertPretrainingTask):
    cfg: JointFinetuneConfig

    def load_dataset(self, split: str, **kwargs) -> None:  # type: ignore[override]
        if split != self.cfg.train_subset or not self.cfg.sub_manifests:
            super().load_dataset(split, **kwargs)
            return
        if not self.cfg.sub_weights or \
                len(self.cfg.sub_weights) != len(self.cfg.sub_manifests):
            raise ValueError(
                "sub_weights must match sub_manifests in length"
            )

        sub_datasets = []
        original_data = self.cfg.data
        original_label_dir = self.cfg.label_dir
        try:
            for sub_dir in self.cfg.sub_manifests:
                # Re-point the task cfg to the sub-manifest, delegate to parent
                # so all the tokenizer / normalize / image_aug logic is reused.
                self.cfg.data = sub_dir
                self.cfg.label_dir = sub_dir
                super().load_dataset(split, **kwargs)
                sub_datasets.append(self.datasets.pop(split))
        finally:
            self.cfg.data = original_data
            self.cfg.label_dir = original_label_dir

        self.datasets[split] = ConcatWeightedDataset(
            sub_datasets, self.cfg.sub_weights, seed=self.cfg.random_crop or 1337,
        )
