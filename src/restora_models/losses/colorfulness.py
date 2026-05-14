"""Negated Hasler-Susstrunk colorfulness metric."""
from __future__ import annotations

import torch

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("colorfulness")
class ColorfulnessLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        rgb = ctx.pred_rgb.clamp(0, 1)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        sigma = torch.sqrt(rg.var(dim=(1, 2)) + yb.var(dim=(1, 2)) + 1e-8)
        mu = torch.sqrt(rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2 + 1e-8)
        return -(sigma + 0.3 * mu).mean()
