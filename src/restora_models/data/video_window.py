"""Composite video dataset.

Pulls 7-frame clips from any number of sub-datasets, each of which
implements the VideoSubDataset protocol. The facade exposes a single
flat indexable Dataset over the union; sample_random() supports
weighted random sampling across sources.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from torch.utils.data import Dataset


@runtime_checkable
class VideoSubDataset(Protocol):
    """Sub-dataset protocol.

    Required:
    - __len__() -> int: number of available 7-frame clips
    - __getitem__(idx) -> dict with keys {frames: (7,3,H,W) torch.float32, source: str, key: str}
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...


class VideoWindowDataset(Dataset):
    """Concatenates N VideoSubDatasets and supports weighted random sampling."""

    def __init__(
        self,
        sub_datasets: Sequence[VideoSubDataset],
        weights: Sequence[float] | None = None,
    ):
        if not sub_datasets:
            raise ValueError("VideoWindowDataset requires >=1 sub-dataset")
        self.subs = list(sub_datasets)
        n = len(self.subs)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError(f"weights len {len(weights)} != sub_datasets len {n}")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        self.weights = np.array([w / total for w in weights], dtype=np.float64)
        self._cumlens = np.cumsum([len(s) for s in self.subs])

    def __len__(self) -> int:
        return int(self._cumlens[-1])

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        bucket = int(np.searchsorted(self._cumlens, idx, side="right"))
        local = idx - (self._cumlens[bucket - 1] if bucket > 0 else 0)
        return self.subs[bucket][int(local)]

    def sample_random(self, rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        bucket = int(rng.choice(len(self.subs), p=self.weights))
        sub = self.subs[bucket]
        local = int(rng.integers(0, len(sub)))
        return sub[local]
