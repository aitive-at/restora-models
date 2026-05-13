"""CompoundDegradationWrapper: per-sample random subset of restoration axes,
applied to a clean image in real-world causal order."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import colorization as _colorization  # noqa: F401
from .degradations import deblur as _deblur  # noqa: F401
from .degradations import denoise as _denoise  # noqa: F401
from .degradations import jpeg as _jpeg  # noqa: F401
from .degradations import superres as _superres  # noqa: F401
from .degradations.registry import build_degradation

_AXES = ("colorize", "denoise", "sharpen", "dejpeg", "deblur")
# Real-world causal order: blur first (camera), then noise (sensor),
# then downsample (resolution), then jpeg (compression), then gray (color loss).
_DEGRADE_ORDER = ("deblur", "denoise", "sharpen", "dejpeg", "colorize")

# Map axis name -> registry name (for axes whose registry name differs)
_AXIS_TO_REG: dict[str, str] = {
    "colorize": "colorize",
    "denoise": "denoise",
    "sharpen": "sharpen",
    "dejpeg": "jpeg",
    "deblur": "deblur",
}


class CompoundDegradationWrapper(Dataset):
    def __init__(
        self,
        clean_ds: Dataset,
        *,
        axis_probs: dict[str, float],
        identity_prob: float = 0.05,
        degradation_params: dict[str, dict] | None = None,
        seed: int = 0,
    ) -> None:
        self.clean = clean_ds
        self.axis_probs = {a: float(axis_probs.get(a, 0.5)) for a in _AXES}
        self.identity_prob = float(identity_prob)
        params = degradation_params or {}
        self.degs = {
            a: build_degradation(_AXIS_TO_REG[a], params.get(a, {}))
            for a in _AXES
        }
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> dict:
        clean = self.clean[idx]  # (3, H, W) float [0,1]
        rng = random.Random((self.seed * 1_000_003) ^ idx)
        if rng.random() < self.identity_prob:
            flags = {a: 0 for a in _AXES}
        else:
            flags = {a: int(rng.random() < self.axis_probs[a]) for a in _AXES}

        rgb_np = clean.permute(1, 2, 0).numpy().copy()
        for axis in _DEGRADE_ORDER:
            if flags[axis]:
                rgb_np = self.degs[axis].degrade(rgb_np, rng)

        config = torch.tensor([flags[a] for a in _AXES], dtype=torch.float32)
        active = [a for a in _AXES if flags[a]]
        axes_label = "+".join(active) if active else "identity"
        return {
            "clean": clean,
            "degraded": torch.from_numpy(rgb_np.transpose(2, 0, 1)).contiguous(),
            "config": config,
            "axes": axes_label,
        }


def collate_compound(batch: list[dict]) -> dict:
    return {
        "clean":    torch.stack([b["clean"]    for b in batch]),
        "degraded": torch.stack([b["degraded"] for b in batch]),
        "config":   torch.stack([b["config"]   for b in batch]),
        "axes":     [b["axes"] for b in batch],
    }


AXES = _AXES  # public re-export
