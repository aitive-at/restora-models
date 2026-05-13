"""Chroma loss — L1 on Lab ab-channels.

Anchors hue+saturation to ground truth, independent of luminance.
Used to counter the colorfulness-loss-driven "infrared map" failure
mode where the model maximizes opponent-color variance without
respecting the true hue.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from refine.utils.color import rgb_to_lab

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("chroma_lab")
class ChromaLabLoss(RestorationLoss):
    """L1 between predicted and ground-truth `ab` channels in CIELab.

    Inputs are RGB [0, 1] sRGB; converted on the fly. The `ab` channels
    have approximate range [-128, 127], so the raw loss is in the same
    units — at a weight of 1.0 it's commensurate with l1_rgb at weight
    ~0.01. Callers should weight accordingly (the loss preset does).
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        pred_lab  = rgb_to_lab(ctx.pred_rgb.clamp(0, 1))
        clean_lab = rgb_to_lab(ctx.clean_rgb.clamp(0, 1))
        return F.l1_loss(pred_lab[:, 1:3], clean_lab[:, 1:3]) * self.scale
