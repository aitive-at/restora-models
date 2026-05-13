"""NAFBlock from 'Simple Baselines for Image Restoration' (Chen et al. ECCV'22),
with FiLM conditioning on a task vector."""
from __future__ import annotations

import torch
from torch import nn


class _ChannelLayerNorm(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class _SimpleChannelAttention(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c: int, *, task_dim: int, expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        self.film = nn.Linear(task_dim, 2 * c)
        self.norm1 = _ChannelLayerNorm(c)
        self.conv1 = nn.Conv2d(c, c * expand * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(c * expand * 2, c * expand * 2, kernel_size=3, padding=1,
                                groups=c * expand * 2)
        self.gate1 = _SimpleGate()
        self.sca = _SimpleChannelAttention(c * expand)
        self.conv2 = nn.Conv2d(c * expand, c, kernel_size=1)
        self.norm2 = _ChannelLayerNorm(c)
        self.conv3 = nn.Conv2d(c, c * ffn_expand * 2, kernel_size=1)
        self.gate2 = _SimpleGate()
        self.conv4 = nn.Conv2d(c * ffn_expand, c, kernel_size=1)

    def forward(self, x: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(task_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = self.norm1(x)
        h = (1.0 + gamma) * h + beta
        h = self.conv1(h)
        h = self.dwconv(h)
        h = self.gate1(h)
        h = self.sca(h)
        h = self.conv2(h)
        x = x + h
        h = self.norm2(x)
        h = self.conv3(h)
        h = self.gate2(h)
        h = self.conv4(h)
        return x + h
