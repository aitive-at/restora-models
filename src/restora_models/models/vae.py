"""Frozen wrapper around Stability AI's SD 1.5 VAE (sd-vae-ft-ema).

Used by the latent diffusion refine head — encode the deterministic
coarse output to a 4-channel latent, run the diffusion UNet there, then
decode back. The VAE weights are frozen; never train them.
"""
from __future__ import annotations

import torch
from torch import nn

# Canonical SD 1.5 latent scale.
_SCALE = 0.18215


class FrozenSD15VAE(nn.Module):
    """Wraps `diffusers.AutoencoderKL` from `stabilityai/sd-vae-ft-ema`,
    freezes all weights, exposes encode()/decode() in the RGB [0,1] domain.

    Inputs/outputs:
      encode(rgb_01: (B,3,H,W) in [0,1]) -> z: (B,4,H/8,W/8)
      decode(z: (B,4,H/8,W/8)) -> rgb_01: (B,3,H,W) in [0,1]

    H and W must be multiples of 8 (VAE downsample factor).
    """

    def __init__(self) -> None:
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.train(False)

    @torch.no_grad()
    def encode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Stochastic encode (samples from the latent distribution).
        Use for training where noise in the target is standard SD practice."""
        x = rgb_01 * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.sample() * _SCALE
        return z

    @torch.no_grad()
    def encode_mode(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """Deterministic encode (returns the latent distribution's mean).
        Use for inference where we want reproducible outputs."""
        x = rgb_01 * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.mode() * _SCALE
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        rgb_m11 = self.vae.decode(z / _SCALE).sample
        return ((rgb_m11 + 1.0) / 2.0).clamp(0.0, 1.0)
