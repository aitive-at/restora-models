"""Generator/discriminator GAN losses (vanilla, lsgan, hinge)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .registry import LossContext, RestorationLoss, register_loss


def _g_loss(logits, gan_type):
    if gan_type == "vanilla":
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    if gan_type == "lsgan":
        return F.mse_loss(logits, torch.ones_like(logits))
    if gan_type == "hinge":
        return -logits.mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


def _d_loss(real, fake, gan_type):
    if gan_type == "vanilla":
        return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) +
                F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
    if gan_type == "lsgan":
        return F.mse_loss(real, torch.ones_like(real)) + F.mse_loss(fake, torch.zeros_like(fake))
    if gan_type == "hinge":
        return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    raise ValueError(f"unknown gan_type: {gan_type}")


@register_loss("gan")
class GeneratorGANLoss(RestorationLoss):
    def __init__(self, gan_type: str = "hinge", discriminator: dict | None = None) -> None:
        super().__init__()
        self.gan_type = gan_type
        self._disc_cfg = discriminator or {"type": "unet", "nf": 64}

    @property
    def disc_config(self) -> dict:
        return self._disc_cfg

    def forward(self, ctx: LossContext) -> torch.Tensor:
        if ctx.discriminator is None:
            raise RuntimeError("GeneratorGANLoss requires ctx.discriminator")
        return _g_loss(ctx.discriminator(ctx.pred_rgb), self.gan_type)


def discriminator_loss(disc, real_rgb, fake_rgb, gan_type):
    return _d_loss(disc(real_rgb), disc(fake_rgb.detach()), gan_type)
