"""Shared output heads for refine restoration models.

The single class here, `DualOutputHead`, is the structural fix for the
colorize-quality gap surfaced in the 2026-05-14 design discussion.
Mirrors v1 DDColor's contract: a dedicated `ab`-prediction head whose
contribution is gated by the `config[0]` (colorize) axis, with the
luminance channel always carried by the parallel RGB-delta head.
"""
from __future__ import annotations

import torch
from torch import nn

from .color import LabToRgb, RgbToLab


class DualOutputHead(nn.Module):
    """RGB-delta head + Lab-ab head, composed by a linear gate on config[0].

    forward(features, rgb_input, config) -> rgb_output

    - rgb_intermediate = rgb_input + head_rgb(features)
    - ab_pred          = head_ab(features)
    - new_ab = config[0] * ab_pred + (1 - config[0]) * lab(rgb_intermediate).ab
    - output = lab_to_rgb(L=lab(rgb_intermediate).L, ab=new_ab)

    Properties:
      - config[0] = 0 -> output == rgb_intermediate (modulo lab round-trip).
                         head_ab receives zero gradient on this sample.
      - config[0] = 1 -> output's ab channels equal head_ab(features) exactly.
                         L is carried by head_rgb's contribution.

    Init:
      head_rgb: zero - initial delta = 0 so initial output == input
                (preserves identity-config behavior, matches the NAFNet
                output-head convention, and keeps rgb_intermediate inside
                [0, 1] at step 0 so the Lab round-trip is stable).
      head_ab : zero - initial ab = 0, so colorize=1 at step 0 yields gray.

    Note: rgb_intermediate is clamped to [0, 1] before the Lab round-trip
    to (a) keep the model inside the valid sRGB domain the Lab conversion
    is defined on, and (b) avoid NaN gradients from the
    `torch.where(c <= thr, c/12.92, ((c+0.055)/1.055)**2.4)` branch in
    `_srgb_to_linear` (autograd flows through both branches even when the
    selected one is finite — the unselected ``**2.4`` on negative bases
    produces NaN gradients).
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.rgb_to_lab = RgbToLab()
        self.lab_to_rgb = LabToRgb()
        self.head_rgb = nn.Conv2d(in_dim, 3, kernel_size=3, padding=1)
        self.head_ab  = nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_rgb.weight)
        nn.init.zeros_(self.head_rgb.bias)
        nn.init.zeros_(self.head_ab.weight)
        nn.init.zeros_(self.head_ab.bias)

    def forward(self, features: torch.Tensor,
                rgb_input: torch.Tensor,
                config: torch.Tensor) -> torch.Tensor:
        rgb_intermediate = (rgb_input + self.head_rgb(features)).clamp(0.0, 1.0)
        ab_pred          = self.head_ab(features)

        lab = self.rgb_to_lab(rgb_intermediate)
        w   = config[:, 0:1].view(-1, 1, 1, 1)
        new_ab = w * ab_pred + (1.0 - w) * lab[:, 1:3]
        lab_out = torch.cat([lab[:, 0:1], new_ab], dim=1)
        return self.lab_to_rgb(lab_out)
