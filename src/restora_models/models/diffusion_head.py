"""LatentDiffusionRefineHead — single-step diffusion refinement in SD 1.5
VAE latent space.

See docs/superpowers/specs/2026-05-16-latent-diffusion-refine-head-design.md
for the design rationale.

This file contains:
  - sinusoidal_timestep_embedding: positional encoding for the diffusion t
  - AdaLNResBlock:                 conditional residual block (AdaLN + conv)
  - LatentDiffusionRefineHead:     the full head (added in a later task)
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for a continuous timestep t in [0, 1].
    Maps (B,) -> (B, dim) with values in [-1, 1]."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 10000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _AdaLN(nn.Module):
    """Group-norm + per-channel scale/shift conditioned on cond (B, cond_dim)."""

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, c), num_channels=c, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1.0 + gamma.view(-1, x.shape[1], 1, 1)) + beta.view(-1, x.shape[1], 1, 1)


class AdaLNResBlock(nn.Module):
    """Residual block with two AdaLN-conditioned conv layers.

      h = adaLN1(x, cond) -> SiLU -> conv1
      h = adaLN2(h, cond) -> SiLU -> conv2
      out = x + h
    """

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.adaln1 = _AdaLN(c, cond_dim)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.adaln2 = _AdaLN(c, cond_dim)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.adaln2(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h
