"""Pixel-space losses (operate on pred_rgb vs clean_rgb)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("l1_rgb")
class L1RgbLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.l1_loss(ctx.pred_rgb, ctx.clean_rgb)


@register_loss("l2_rgb")
class L2RgbLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.mse_loss(ctx.pred_rgb, ctx.clean_rgb)


@register_loss("charbonnier_rgb")
class CharbonnierRgbLoss(RestorationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        diff2 = (ctx.pred_rgb - ctx.clean_rgb) ** 2
        return torch.sqrt(diff2 + self.eps2).mean()
