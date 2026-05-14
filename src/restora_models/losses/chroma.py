"""Chroma loss — L1 on Lab ab-channels.

Anchors hue+saturation to ground truth, independent of luminance.
Used to counter the colorfulness-loss-driven "infrared map" failure
mode where the model maximizes opponent-color variance without
respecting the true hue.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from restora_models.utils.color import rgb_to_lab

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("chroma_lab")
class ChromaLabLoss(RestorationLoss):
    """L1 between predicted and ground-truth `ab` channels in normalized CIELab.

    Inputs are RGB [0, 1] sRGB. Converted to CIELab on the fly, then
    `ab` channels are divided by 110 (matching the `RgbToLab` nn.Module
    normalization elsewhere in the codebase) so the L1 output is in
    [0, ~2] — directly commensurate with `l1_rgb` at the same weight.

    Without this normalization the raw Lab `ab` channels span ~[-128, 127]
    so the loss is in [0, ~250] — at weight=1.0 it would dominate every
    other term in the loss stack by ~180×, which was the cause of the
    "training stalls / losses move unevenly" pathology observed on
    2026-05-14 dual-head experiments.

    The `_AB_NORM` divisor matches `RgbToLab` in `restora_models.models.color`.
    """

    _AB_NORM: float = 110.0

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, ctx: LossContext) -> torch.Tensor:
        pred_lab  = rgb_to_lab(ctx.pred_rgb.clamp(0, 1))
        clean_lab = rgb_to_lab(ctx.clean_rgb.clamp(0, 1))
        pred_ab_n  = pred_lab[:, 1:3]  / self._AB_NORM
        clean_ab_n = clean_lab[:, 1:3] / self._AB_NORM
        return F.l1_loss(pred_ab_n, clean_ab_n) * self.scale
