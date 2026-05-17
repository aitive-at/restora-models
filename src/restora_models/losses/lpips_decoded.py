"""LPIPS perceptual loss on decoded RGB. Used in temporal_v1 preset and SLKD distillation."""
from __future__ import annotations

import torch

from restora_models.losses.registry import LossContext, RestorationLoss, register_loss


@register_loss("lpips_decoded")
class LpipsDecodedLoss(RestorationLoss):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1]
        pred_n = ctx.pred_rgb * 2.0 - 1.0
        target_n = ctx.clean_rgb * 2.0 - 1.0
        return self.model(pred_n, target_n).mean()
