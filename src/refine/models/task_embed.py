"""Conditioning embedding: turns a (B, num_axes) float vector into a
(B, dim) task vector for FiLM/AdaLN modulation throughout the model."""
from __future__ import annotations

import torch
from torch import nn


class ConfigEmbed(nn.Module):
    def __init__(self, *, num_axes: int = 5, dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(num_axes, dim)
        self.mlp = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, config: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.proj(config))
