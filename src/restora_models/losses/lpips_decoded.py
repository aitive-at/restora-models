"""LPIPS perceptual loss on decoded RGB. Used in temporal_v1 preset and SLKD distillation."""
from __future__ import annotations

import warnings

import torch

from restora_models.losses.registry import LossContext, RestorationLoss, register_loss


@register_loss("lpips_decoded")
class LpipsDecodedLoss(RestorationLoss):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        import lpips
        # lpips 0.1.x calls torchvision.models.vgg16(pretrained=True), which
        # warns under torchvision >=0.13. Suppress at the construction site
        # so trainer stdout stays readable; remove once lpips switches to
        # the `weights=` API upstream.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"torchvision\..*")
            self.model = lpips.LPIPS(net=net, verbose=False)
        self.model.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1]
        pred_n = ctx.pred_rgb * 2.0 - 1.0
        target_n = ctx.clean_rgb * 2.0 - 1.0
        return self.model(pred_n, target_n).mean()
