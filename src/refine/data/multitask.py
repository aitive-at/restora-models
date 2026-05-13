"""MultiTaskWrapper: per-sample task picker on top of a clean-image dataset."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations.registry import Degradation


class MultiTaskWrapper(Dataset):
    def __init__(self, clean_ds: Dataset, degradations: list[Degradation],
                 weights: list[float], *, seed: int = 0) -> None:
        if len(degradations) == 0:
            raise ValueError("at least one degradation required")
        if len(degradations) != len(weights):
            raise ValueError("degradations/weights length mismatch")
        self.clean = clean_ds
        self.degs = degradations
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights sum must be > 0")
        self.cdf = np.cumsum(np.asarray([w / total for w in weights], dtype=np.float64))
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> dict:
        clean = self.clean[idx]
        rng = random.Random((self.seed * 1_000_003) ^ idx)
        task_idx = int(np.searchsorted(self.cdf, rng.random()))
        if task_idx >= len(self.degs):
            task_idx = len(self.degs) - 1
        deg = self.degs[task_idx]
        rgb_np = clean.permute(1, 2, 0).numpy()
        degraded_np = deg.degrade(rgb_np, rng)
        return {
            "clean": clean,
            "degraded": torch.from_numpy(degraded_np.transpose(2, 0, 1)).contiguous(),
            "task_id": torch.tensor(deg.task_id, dtype=torch.long),
            "task_name": deg.name,
        }


def collate_multitask(batch: list[dict]) -> dict:
    return {
        "clean": torch.stack([b["clean"] for b in batch]),
        "degraded": torch.stack([b["degraded"] for b in batch]),
        "task_id": torch.stack([b["task_id"] for b in batch]),
        "task_name": [b["task_name"] for b in batch],
    }
