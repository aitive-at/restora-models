"""Pixel-space losses (operate on the predicted AB channels)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import ColorizationLoss, LossContext, register_loss


@register_loss("l1_ab")
class L1AbLoss(ColorizationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.l1_loss(ctx.pred_ab, ctx.gt_ab)


@register_loss("l2_ab")
class L2AbLoss(ColorizationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        return F.mse_loss(ctx.pred_ab, ctx.gt_ab)


@register_loss("charbonnier_ab")
class CharbonnierAbLoss(ColorizationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        diff2 = (ctx.pred_ab - ctx.gt_ab) ** 2
        return torch.sqrt(diff2 + self.eps2).mean()
