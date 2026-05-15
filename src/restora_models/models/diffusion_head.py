"""LatentDiffusionRefineHead — single-step diffusion refinement in SD 1.5
VAE latent space.

See docs/superpowers/specs/2026-05-16-latent-diffusion-refine-head-design.md
for the design rationale.

This file contains:
  - sinusoidal_timestep_embedding: positional encoding for the diffusion t
  - AdaLNResBlock:                 conditional residual block (AdaLN + conv)
  - LatentDiffusionRefineHead:     the full head (added in a later task)
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding for a continuous timestep t in [0, 1].
    Maps (B,) -> (B, dim) with values in [-1, 1]."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 10000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _AdaLN(nn.Module):
    """Group-norm + per-channel scale/shift conditioned on cond (B, cond_dim)."""

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(8, c), num_channels=c, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * c)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1.0 + gamma.view(-1, x.shape[1], 1, 1)) + beta.view(-1, x.shape[1], 1, 1)


class AdaLNResBlock(nn.Module):
    """Residual block with two AdaLN-conditioned conv layers.

      h = adaLN1(x, cond) -> SiLU -> conv1
      h = adaLN2(h, cond) -> SiLU -> conv2
      out = x + h
    """

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.adaln1 = _AdaLN(c, cond_dim)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.adaln2 = _AdaLN(c, cond_dim)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.adaln2(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


from .vae import FrozenSD15VAE


def _down(in_c: int, out_c: int) -> nn.Module:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)


def _up(in_c: int, out_c: int) -> nn.Module:
    return nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)


class LatentDiffusionRefineHead(nn.Module):
    """Single-step diffusion refine head in SD 1.5 VAE latent space.

    Inference forward:
        z_coarse        = VAE.encode_mode(coarse_rgb)
        eps             = noise_for(z_coarse)
        z_t             = (1 - t_inf) * z_coarse + t_inf * eps
        feat_at_latent  = feat_proj(backbone_features)
        pred_z_clean    = unet(z_t, z_coarse, feat_at_latent, cond)
        refined_rgb     = VAE.decode(pred_z_clean)

    Training forward (forward_with_targets):
        Same as inference but:
          - t is sampled from Uniform(0, 1) per-sample
          - clean_rgb is provided so we can also compute z_clean
          - returns (pred_z_clean, z_clean, decoded_rgb)
    """

    def __init__(
        self,
        *,
        feat_dim: int = 64,
        num_axes: int = 5,
        base_c: int = 96,
        num_blocks_per_stage: int = 2,
        timestep_emb_dim: int = 128,
        t_inference: float = 0.2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_axes = num_axes
        self.t_inference = float(t_inference)
        self._noise_mode = "random"
        self._fixed_noise: torch.Tensor | None = None

        self.vae = FrozenSD15VAE()

        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, 32, kernel_size=8, stride=8),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=1),
        )

        self._cond_dim = base_c * 4
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_axes + timestep_emb_dim, self._cond_dim),
            nn.SiLU(),
            nn.Linear(self._cond_dim, self._cond_dim),
        )
        self._timestep_emb_dim = timestep_emb_dim

        c = base_c
        self.stem = nn.Conv2d(4 + 4 + 16, c, kernel_size=1)

        self.enc1 = nn.ModuleList([AdaLNResBlock(c, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])
        self.down1 = _down(c, c * 2)

        self.enc2 = nn.ModuleList([AdaLNResBlock(c * 2, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])
        self.down2 = _down(c * 2, c * 4)

        self.bottle = nn.ModuleList([AdaLNResBlock(c * 4, self._cond_dim)
                                      for _ in range(num_blocks_per_stage * 2)])

        self.up2 = _up(c * 4, c * 2)
        self.dec2 = nn.ModuleList([AdaLNResBlock(c * 2, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])

        self.up1 = _up(c * 2, c)
        self.dec1 = nn.ModuleList([AdaLNResBlock(c, self._cond_dim)
                                    for _ in range(num_blocks_per_stage)])

        self.head_out = nn.Conv2d(c, 4, kernel_size=3, padding=1)
        nn.init.normal_(self.head_out.weight, std=0.01)
        nn.init.zeros_(self.head_out.bias)

    def set_inference_noise_mode(self, mode: str,
                                  fixed: torch.Tensor | None = None) -> None:
        """Choose how noise is sampled at inference time.

        Modes:
          'random': torch.randn_like at each call.
          'zero':   noise = 0 (degenerate; VAE-decodes z_coarse).
          'fixed':  use a pre-supplied noise tensor.
        """
        if mode not in ("random", "zero", "fixed"):
            raise ValueError(f"unknown noise mode {mode!r}")
        if mode == "fixed" and fixed is None:
            raise ValueError("noise_mode='fixed' requires a fixed= tensor")
        self._noise_mode = mode
        self._fixed_noise = fixed

    def _noise_for(self, z: torch.Tensor) -> torch.Tensor:
        if self._noise_mode == "zero":
            return torch.zeros_like(z)
        if self._noise_mode == "fixed":
            return self._fixed_noise.to(z.device, z.dtype).expand_as(z)
        return torch.randn_like(z)

    def _condition(self, config: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_timestep_embedding(t, self._timestep_emb_dim)
        return self.cond_mlp(torch.cat([config, t_emb], dim=-1))

    def _unet(self, z_t: torch.Tensor, z_coarse: torch.Tensor,
              feat_at_latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(torch.cat([z_t, z_coarse, feat_at_latent], dim=1))
        for b in self.enc1:
            x = b(x, cond)
        skip1 = x
        x = self.down1(x)
        for b in self.enc2:
            x = b(x, cond)
        skip2 = x
        x = self.down2(x)
        for b in self.bottle:
            x = b(x, cond)
        x = self.up2(x)
        x = x + skip2
        for b in self.dec2:
            x = b(x, cond)
        x = self.up1(x)
        x = x + skip1
        for b in self.dec1:
            x = b(x, cond)
        return self.head_out(x)

    def forward(self, backbone_features: torch.Tensor,
                coarse_rgb: torch.Tensor,
                config: torch.Tensor) -> torch.Tensor:
        z_coarse = self.vae.encode_mode(coarse_rgb)
        eps = self._noise_for(z_coarse)
        t = torch.full((coarse_rgb.shape[0],),
                       self.t_inference, device=coarse_rgb.device)
        z_t = (1.0 - self.t_inference) * z_coarse + self.t_inference * eps
        feat = self.feat_proj(backbone_features)
        cond = self._condition(config, t)
        pred_z = self._unet(z_t, z_coarse, feat, cond)
        return self.vae.decode(pred_z).clamp(0.0, 1.0)

    def forward_with_targets(
        self,
        backbone_features: torch.Tensor,
        coarse_rgb: torch.Tensor,
        clean_rgb: torch.Tensor,
        config: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-time forward. Returns (pred_z_clean, target_z_clean, decoded_rgb)."""
        B = coarse_rgb.shape[0]
        z_coarse = self.vae.encode_mode(coarse_rgb)
        target_z_clean = self.vae.encode(clean_rgb)
        eps = torch.randn_like(z_coarse)
        t = torch.rand(B, device=coarse_rgb.device)
        t_b = t.view(B, 1, 1, 1)
        z_t = (1.0 - t_b) * z_coarse + t_b * eps
        feat = self.feat_proj(backbone_features)
        cond = self._condition(config, t)
        pred_z_clean = self._unet(z_t, z_coarse, feat, cond)
        decoded_rgb = self.vae.decode(pred_z_clean).clamp(0.0, 1.0)
        return pred_z_clean, target_z_clean, decoded_rgb
