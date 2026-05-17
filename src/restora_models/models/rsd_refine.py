"""RSD: one-step residual-shift diffusion in RGB space.

No external VAE. Operates directly on the backbone's coarse RGB output.
Conditioned on the 5-axis task vector + a per-axis t_inf scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from restora_models.models.config_embed import ConfigEmbed


TASK_DIM = 128


def _t_inf_for(config: torch.Tensor) -> torch.Tensor:
    """Per-axis t_inf table from the spec section 3.4. Returns (B,) in [0,1].

    For samples that have multiple axes active, take the max -- the hardest
    axis controls the noise level.
    """
    per_axis = torch.tensor([0.3, 0.05, 0.3, 0.05, 0.05],
                             device=config.device, dtype=config.dtype)
    weighted = config * per_axis.unsqueeze(0)
    return weighted.max(dim=1).values


class _FiLMBlock(nn.Module):
    def __init__(self, c: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, c), num_channels=c)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + g.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + h


class RSDRefineHead(nn.Module):
    """Single-step RGB-space residual refinement."""

    def __init__(self, width: int = 64, num_axes: int = 5, depth: int = 4):
        super().__init__()
        # CORRECTED: ConfigEmbed is keyword-only and has no .out_dim
        self.cfg_embed = ConfigEmbed(num_axes=num_axes, dim=TASK_DIM)
        cond_dim = TASK_DIM + 1  # +1 for t_inf scalar
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList([_FiLMBlock(width, cond_dim) for _ in range(depth)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coarse_rgb: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        b = coarse_rgb.shape[0]
        task = self.cfg_embed(config)
        t_inf = _t_inf_for(config).view(b, 1)
        cond = torch.cat([task, t_inf], dim=-1)
        h = self.stem(coarse_rgb)
        for blk in self.blocks:
            h = blk(h, cond)
        residual = self.head(h)
        return (coarse_rgb + residual).clamp(0.0, 1.0)
