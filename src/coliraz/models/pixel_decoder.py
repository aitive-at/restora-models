"""Pixel decoder: UNet upsample path producing 3 mid-scale memory features + 1 hi-res map."""
from __future__ import annotations

import torch
from torch import nn

from .unet_blocks import PixelShuffleICNR, UnetBlockWide


class PixelDecoder(nn.Module):
    def __init__(self, *, feature_channels: list[int], nf: int = 512) -> None:
        super().__init__()
        c0, c1, c2, c3 = feature_channels
        out_c = nf
        self.u1 = UnetBlockWide(in_c=c3, skip_c=c2, out_c=out_c)
        self.u2 = UnetBlockWide(in_c=out_c, skip_c=c1, out_c=out_c)
        self.u3 = UnetBlockWide(in_c=out_c, skip_c=c0, out_c=out_c // 2)
        self.last_shuf = PixelShuffleICNR(out_c // 2, out_c // 2, scale=4)

    def forward(self, feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        f0, f1, f2, f3 = feats
        m0 = self.u1(f3, f2)
        m1 = self.u2(m0, f1)
        m2 = self.u3(m1, f0)
        hi = self.last_shuf(m2)
        return [m0, m1, m2], hi
