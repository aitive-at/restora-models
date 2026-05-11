"""Pixel-shuffle ICNR upsampler and UNet wide block (modernized)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _icnr_init(tensor: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Initialize a sub-pixel conv weight to mimic nearest-neighbor upsampling at start."""
    out_c, in_c, kh, kw = tensor.shape
    sub_out = out_c // (scale * scale)
    sub = torch.empty(sub_out, in_c, kh, kw)
    nn.init.kaiming_normal_(sub)
    sub = sub.repeat_interleave(scale * scale, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)
    return tensor


class PixelShuffleICNR(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, scale: int = 2, blur: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel_size=1)
        _icnr_init(self.conv.weight, scale=scale)
        self.shuf = nn.PixelShuffle(scale)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.blur = (
            nn.Sequential(nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1))
            if blur else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuf(x)
        x = self.blur(x)
        x = self.norm(x)
        return self.act(x)


class UnetBlockWide(nn.Module):
    """Upsample deep feature, concat with skip feature from encoder, project to out_c."""

    def __init__(self, in_c: int, skip_c: int, out_c: int, *, use_spectral: bool = True) -> None:
        super().__init__()
        self.up = PixelShuffleICNR(in_c, in_c // 2, scale=2)
        conv = nn.Conv2d(in_c // 2 + skip_c, out_c, kernel_size=3, padding=1)
        self.proj = spectral_norm(conv) if use_spectral else conv
        self.norm = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(deep)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)
