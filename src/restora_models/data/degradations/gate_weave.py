"""Gate-weave degradation: per-frame sub-pixel translation jitter.

Models the optical-printer "gate weave" of physical film. Operates on a
full (T,3,H,W) clip. Not part of the registry-based per-frame pipeline
because the jitter pattern must be temporally smooth across frames.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GateWeaveDegradation:
    def __init__(self, max_shift_px: float = 2.0):
        self.max_shift_px = max_shift_px

    def apply_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 4:
            raise ValueError(f"expected (T,3,H,W), got {tuple(clip.shape)}")
        if self.max_shift_px <= 0.0:
            return clip.clone()
        t, _, h, w = clip.shape
        raw = torch.randn(t, 2) * self.max_shift_px
        smooth = F.avg_pool1d(raw.T.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).T
        out = torch.empty_like(clip)
        for k in range(t):
            dy, dx = smooth[k].tolist()
            theta = torch.tensor([
                [1.0, 0.0, 2.0 * dx / max(w - 1, 1)],
                [0.0, 1.0, 2.0 * dy / max(h - 1, 1)],
            ], dtype=clip.dtype, device=clip.device).unsqueeze(0)
            grid = F.affine_grid(theta, [1, 3, h, w], align_corners=True)
            out[k] = F.grid_sample(clip[k:k + 1], grid, mode="bilinear",
                                    padding_mode="border", align_corners=True).squeeze(0)
        return out
