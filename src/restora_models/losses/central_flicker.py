"""Central flicker loss: L1 between model outputs for two overlapping
7-frame windows on the same physical frame."""
from __future__ import annotations

import torch

from restora_models.losses.registry import LossContext, RestorationLoss, register_loss


@register_loss("central_flicker")
class CentralFlickerLoss(RestorationLoss):
    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.secondary_pred_rgb is None:
            return ctx.pred_rgb.new_zeros(())
        if ctx.pred_rgb.shape != ctx.secondary_pred_rgb.shape:
            raise ValueError("pred_rgb and secondary_pred_rgb shapes must match")
        return (ctx.pred_rgb - ctx.secondary_pred_rgb).abs().mean()
