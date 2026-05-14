"""Pixel-space losses (operate on pred_rgb vs clean_rgb)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("l1_rgb")
class L1RgbLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.l1_loss(ctx.pred_rgb, ctx.clean_rgb)
