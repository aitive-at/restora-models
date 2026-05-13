"""Final 1x1 refinement conv that mixes the einsum coarse map with the input RGB."""
from __future__ import annotations

from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def build_refine(in_ch: int, out_ch: int, *, norm: str = "spectral") -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
    if norm == "spectral":
        return spectral_norm(conv)
    if norm == "batch":
        return nn.Sequential(conv, nn.BatchNorm2d(out_ch))
    if norm == "none":
        return conv
    raise ValueError(f"unknown refine norm: {norm!r}")
