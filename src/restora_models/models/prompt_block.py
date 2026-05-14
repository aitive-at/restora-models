"""Config-driven PromptBlock.

Replaces PromptIR's paper-original blind self-attention prompt-selection
with a router driven by the 5-axis config embedding. The same `cond`
that drives every AdaLN in the network also picks which learned prompts
to mix here.

Property: identical config -> identical mix -> identical output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PromptBlock(nn.Module):
    def __init__(self, *, feat_c: int, prompt_n: int = 5,
                 prompt_dim: int, prompt_hw: int, cond_dim: int) -> None:
        super().__init__()
        # Learnable prompt bank, shape (N, prompt_dim, P_h, P_w).
        # Small Gaussian init so different prompts start at different signals.
        self.prompts = nn.Parameter(
            torch.randn(prompt_n, prompt_dim, prompt_hw, prompt_hw) * 0.02
        )
        self.router = nn.Linear(cond_dim, prompt_n)
        self.fuse = nn.Conv2d(feat_c + prompt_dim, feat_c, kernel_size=1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        alpha = F.softmax(self.router(cond), dim=-1)            # (B, N)
        mix = (alpha[:, :, None, None, None]
               * self.prompts.unsqueeze(0)).sum(dim=1)          # (B, P_c, P_h, P_w)
        mix = F.interpolate(mix, size=(h, w), mode="bilinear",
                            align_corners=False)
        return self.fuse(torch.cat([feat, mix], dim=1))
