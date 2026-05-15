"""Diffusion training losses for the LatentDiffusionRefineHead."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


@register_loss("l1_latent")
class L1LatentLoss(RestorationLoss):
    """L1 between predicted and target VAE latents. Returns 0 if either
    field is missing (no-op on non-diffusion batches)."""

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.pred_latent is None or ctx.target_latent is None:
            return torch.zeros((), device=ctx.pred_rgb.device,
                                dtype=ctx.pred_rgb.dtype)
        return F.l1_loss(ctx.pred_latent, ctx.target_latent)
