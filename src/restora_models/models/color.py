"""nn.Module wrappers around the pure color conversions.

These are frozen (no learned parameters) and always run in fp32. The
bf16 autocast is explicitly disabled inside the forward — pow(x, 1/2.4)
and the lab inverse functions overflow bf16 dynamic range, which was
the dominant training-instability failure mode in coliraz v1.

The modules also include a fixed normalization so the LAB output has
roughly N(0, 1) statistics, which lets downstream NAFBlocks process
under bf16 cleanly.
"""
from __future__ import annotations

import torch
from torch import nn

from restora_models.utils.color import lab_to_rgb, rgb_to_lab


class RgbToLab(nn.Module):
    """(B, 3, H, W) RGB in [0, 1]  →  (B, 3, H, W) normalized LAB fp32."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(rgb.device.type, enabled=False):
            lab = rgb_to_lab(rgb.float())
            L = (lab[:, 0:1] - 50.0) / 50.0
            a = lab[:, 1:2] / 110.0
            b = lab[:, 2:3] / 110.0
            return torch.cat([L, a, b], dim=1)


class LabToRgb(nn.Module):
    """(B, 3, H, W) normalized LAB fp32  →  (B, 3, H, W) RGB clamped [0, 1]."""

    def forward(self, lab_n: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(lab_n.device.type, enabled=False):
            x = lab_n.float()
            L = x[:, 0:1] * 50.0 + 50.0
            a = x[:, 1:2] * 110.0
            b = x[:, 2:3] * 110.0
            lab = torch.cat([L, a, b], dim=1)
            return lab_to_rgb(lab).clamp(0, 1)
