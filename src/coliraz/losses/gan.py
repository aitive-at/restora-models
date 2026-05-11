"""GAN losses (generator-side via registry; discriminator step as a helper)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .registry import ColorizationLoss, LossContext, register_loss


def _g_loss(logits: torch.Tensor, gan_type: str) -> torch.Tensor:
    if gan_type == "vanilla":
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    if gan_type == "lsgan":
        return F.mse_loss(logits, torch.ones_like(logits))
    if gan_type == "hinge":
        return -logits.mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


def _d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, gan_type: str) -> torch.Tensor:
    if gan_type == "vanilla":
        r = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        f = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return r + f
    if gan_type == "lsgan":
        return F.mse_loss(real_logits, torch.ones_like(real_logits)) + F.mse_loss(
            fake_logits, torch.zeros_like(fake_logits)
        )
    if gan_type == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


@register_loss("gan")
class GeneratorGANLoss(ColorizationLoss):
    def __init__(self, gan_type: str = "hinge", discriminator: dict | None = None) -> None:
        super().__init__()
        self.gan_type = gan_type
        self._disc_cfg = discriminator or {"type": "unet", "nf": 64}

    @property
    def disc_config(self) -> dict:
        return self._disc_cfg

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.discriminator is None:
            raise RuntimeError("GeneratorGANLoss requires LossContext.discriminator")
        fake_logits = ctx.discriminator(ctx.pred_rgb)
        return _g_loss(fake_logits, self.gan_type)


def discriminator_loss(
    disc: nn.Module, real_rgb: torch.Tensor, fake_rgb: torch.Tensor, gan_type: str
) -> torch.Tensor:
    real_logits = disc(real_rgb)
    fake_logits = disc(fake_rgb.detach())
    return _d_loss(real_logits, fake_logits, gan_type)
