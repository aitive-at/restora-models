"""Video-pair compound degradation wrapper.

Wraps a `VideoPairDataset` (which yields {clean_t, clean_tk, flow_t_tk})
and applies the SAME randomly-sampled compound degradation to both frames
using a SHARED RNG seed. The shared RNG matters because:

- Sharpen (SR) picks a downsample factor: must match across the pair so
  the model sees the same task scale on both frames.
- JPEG picks a quality factor: should match across the pair (real
  video compression uses constant QF within a clip).
- Deblur/denoise stochastic parameters likewise match for clean signal.

Same RNG means the noise pattern is identical across frames too — a
slight simplification of the real-world case (per-frame independent
sensor noise). For temporal-consistency training this is fine: even with
matching noise, the underlying clean content differs between frames
because of motion, so temporal_pair loss measures whether the model
*commutes* with optical-flow warping. A perfect model produces
flow-consistent outputs; an imperfect one produces measurable error.

Returned dict (per item):
    clean_t       : (3, H, W) clean RGB at time t
    degraded_t    : (3, H, W) degraded input at time t
    clean_tk      : (3, H, W) clean RGB at time t+k
    degraded_tk   : (3, H, W) degraded input at time t+k
    flow_t_tk     : (2, H, W) pixel-displacement flow t -> t+k
    config        : (5,)      same per-axis flag vector for both frames
    axes          : str       "+".join of active axes (e.g. "colorize+denoise")
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .compound import AXES, AXIS_TO_REG, DEGRADE_ORDER
from .degradations.registry import build_degradation
from .video import VideoPairDataset


def _degrade(clean_chw: torch.Tensor, flags: dict[str, int],
             degs: dict[str, Any], rng: random.Random) -> torch.Tensor:
    """Apply compound degradation in causal order; return (3,H,W) float tensor."""
    rgb_np = clean_chw.permute(1, 2, 0).numpy().copy()
    for axis in DEGRADE_ORDER:
        if flags[axis]:
            rgb_np = degs[axis].degrade(rgb_np, rng)
    return torch.from_numpy(rgb_np.transpose(2, 0, 1)).contiguous()


class VideoCompoundDegradationWrapper(Dataset):
    """Pairs degraded frames for temporal-pair training.

    Parameters mirror CompoundDegradationWrapper but operate on a
    VideoPairDataset rather than a flat image dataset.
    """

    def __init__(
        self,
        video_ds: VideoPairDataset,
        *,
        axis_probs: dict[str, float],
        identity_prob: float = 0.05,
        degradation_params: dict[str, dict] | None = None,
        seed: int = 0,
    ) -> None:
        self.video = video_ds
        self.axis_probs = {a: float(axis_probs.get(a, 0.5)) for a in AXES}
        self.identity_prob = float(identity_prob)
        params = degradation_params or {}
        self.degs = {
            a: build_degradation(AXIS_TO_REG[a], params.get(a, {}))
            for a in AXES
        }
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.video)

    def __getitem__(self, idx: int) -> dict:
        pair = self.video[idx]
        clean_t = pair["clean_t"]
        clean_tk = pair["clean_tk"]
        flow = pair["flow_t_tk"]

        # Decide axis flags from a sample-specific RNG seed.
        flag_rng = random.Random((self.seed * 7919) ^ idx)
        if flag_rng.random() < self.identity_prob:
            flags = {a: 0 for a in AXES}
        else:
            flags = {a: int(flag_rng.random() < self.axis_probs[a]) for a in AXES}

        # SAME degradation parameters for both frames — fresh Random(seed)
        # per frame produces an identical pull sequence so SR factor /
        # JPEG QF / sigma / noise pattern all match between t and t+k.
        deg_seed = (self.seed * 1_000_003) ^ (idx * 31 + 17)
        rng_t = random.Random(deg_seed)
        rng_tk = random.Random(deg_seed)

        degraded_t = _degrade(clean_t, flags, self.degs, rng_t)
        degraded_tk = _degrade(clean_tk, flags, self.degs, rng_tk)

        config = torch.tensor([flags[a] for a in AXES], dtype=torch.float32)
        active = [a for a in AXES if flags[a]]
        axes_label = "+".join(active) if active else "identity"
        return {
            "clean_t":     clean_t,
            "degraded_t":  degraded_t,
            "clean_tk":    clean_tk,
            "degraded_tk": degraded_tk,
            "flow_t_tk":   flow,
            "config":      config,
            "axes":        axes_label,
        }


def collate_video_compound(batch: list[dict]) -> dict:
    return {
        "clean_t":     torch.stack([b["clean_t"]     for b in batch]),
        "degraded_t":  torch.stack([b["degraded_t"]  for b in batch]),
        "clean_tk":    torch.stack([b["clean_tk"]    for b in batch]),
        "degraded_tk": torch.stack([b["degraded_tk"] for b in batch]),
        "flow_t_tk":   torch.stack([b["flow_t_tk"]   for b in batch]),
        "config":      torch.stack([b["config"]      for b in batch]),
        "axes":        [b["axes"] for b in batch],
    }
