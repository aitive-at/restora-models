"""L1 between FFT log-magnitude spectra of pred and clean."""
from __future__ import annotations

import torch

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("freq_l1")
class FreqL1Loss(RestorationLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, ctx: LossContext) -> torch.Tensor:
        with torch.amp.autocast(ctx.pred_rgb.device.type, enabled=False):
            pred = torch.fft.rfft2(ctx.pred_rgb.float(), norm="ortho")
            clean = torch.fft.rfft2(ctx.clean_rgb.float(), norm="ortho")
            pred_mag = torch.log1p(pred.abs() + self.eps)
            clean_mag = torch.log1p(clean.abs() + self.eps)
            return (pred_mag - clean_mag).abs().mean()
