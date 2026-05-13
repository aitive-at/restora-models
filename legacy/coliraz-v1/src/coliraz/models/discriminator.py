"""UNet-style image discriminator with per-pixel logits."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _down(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        spectral_norm(nn.Conv2d(in_c, out_c, 3, padding=1)),
        nn.LeakyReLU(0.2, inplace=True),
    )


class UNetDiscriminator(nn.Module):
    def __init__(self, *, in_ch: int = 3, nf: int = 64) -> None:
        super().__init__()
        self.d1 = _down(in_ch, nf)
        self.d2 = _down(nf, nf * 2)
        self.d3 = _down(nf * 2, nf * 4)
        self.d4 = _down(nf * 4, nf * 8)
        self.u3 = _up(nf * 8, nf * 4)
        self.u2 = _up(nf * 4 + nf * 4, nf * 2)
        self.u1 = _up(nf * 2 + nf * 2, nf)
        self.out = nn.Conv2d(nf + nf, 1, kernel_size=3, padding=1)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u3 = self.u3(d4)
        u2 = self.u2(torch.cat([u3, d3], dim=1))
        u1 = self.u1(torch.cat([u2, d2], dim=1))
        y = self.out(torch.cat([u1, d1], dim=1))
        return self.up_final(y)
