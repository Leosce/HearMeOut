"""Weighted concatenation wrapper around several AVHubertDataset instances.

Each epoch the wrapper draws a virtual-length list of (dataset_idx, sample_idx)
pairs using the provided weights. All sub-datasets must share the same
tokenizer/label-processor so that collation is compatible.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from fairseq.data import FairseqDataset


class ConcatWeightedDataset(FairseqDataset):
    def __init__(
        self,
        datasets: Sequence[FairseqDataset],
        weights: Sequence[float],
        seed: int = 1337,
        epoch: int = 1,
    ) -> None:
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have equal length")
        if not datasets:
            raise ValueError("ConcatWeightedDataset needs >=1 sub-dataset")
        self.datasets: List[FairseqDataset] = list(datasets)
        w = np.asarray(weights, dtype=np.float64)
        if (w <= 0).any():
            raise ValueError("weights must be strictly positive")
        self.weights = w / w.sum()
        self.seed = seed
        self._epoch = -1
        self.set_epoch(epoch)

    # ---- epoch / indexing ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        if epoch == self._epoch:
            return
        self._epoch = epoch
        rng = np.random.RandomState(self.seed + epoch)
        virtual_len = int(
            max(len(d) / w for d, w in zip(self.datasets, self.weights))
        )
        ds_idx = rng.choice(len(self.datasets), size=virtual_len, p=self.weights)
        sample_idx = np.empty(virtual_len, dtype=np.int64)
        for i, d in enumerate(self.datasets):
            mask = ds_idx == i
            if mask.any():
                sample_idx[mask] = rng.randint(0, len(d), size=int(mask.sum()))
        self._ds_idx = ds_idx
        self._sample_idx = sample_idx

    def __len__(self) -> int:
        return int(self._ds_idx.shape[0])

    def __getitem__(self, index: int):
        d_i = int(self._ds_idx[index])
        s_i = int(self._sample_idx[index])
        item = self.datasets[d_i][s_i]
        if isinstance(item, dict) and "id" in item:
            item = dict(item)
            item["id"] = index
        return item

    # ---- Fairseq dataset protocol ------------------------------------------

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    def num_tokens(self, index: int) -> int:
        d_i = int(self._ds_idx[index])
        s_i = int(self._sample_idx[index])
        return self.datasets[d_i].num_tokens(s_i)

    def size(self, index: int):
        d_i = int(self._ds_idx[index])
        s_i = int(self._sample_idx[index])
        return self.datasets[d_i].size(s_i)

    def ordered_indices(self):
        sizes = np.fromiter(
            (self.num_tokens(i) for i in range(len(self))),
            count=len(self), dtype=np.int64,
        )
        return np.argsort(sizes, kind="mergesort")

    @property
    def sizes(self):
        return np.fromiter(
            (self.size(i) for i in range(len(self))),
            count=len(self), dtype=np.int64,
        )

    def can_reuse_epoch_itr_across_epochs(self) -> bool:
        return False
