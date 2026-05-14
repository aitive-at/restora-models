"""Vectorized RGB <-> LAB conversion on torch tensors.

Conventions match cv2's `COLOR_RGB2LAB` for float32 inputs in [0, 1]:
- L in [0, 100]
- a, b in approximately [-128, 127]
All ops are pure-tensor so they autograd through and run on GPU.
"""
from __future__ import annotations

import torch

_RGB2XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ2RGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])  # D65


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    threshold = 0.04045
    return torch.where(c <= threshold, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    threshold = 0.0031308
    return torch.where(
        c <= threshold, c * 12.92, 1.055 * c.clamp(min=1e-8).pow(1 / 2.4) - 0.055
    )


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(
        t > delta**3, t.clamp(min=1e-8).pow(1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0
    )


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta, t.pow(3), 3 * delta**2 * (t - 4.0 / 29.0))


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B, 3, H, W) in [0, 1] sRGB → (B, 3, H, W) LAB."""
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(rgb.shape)}")
    m = _RGB2XYZ.to(rgb.device, dtype=rgb.dtype)
    w = _WHITE.to(rgb.device, dtype=rgb.dtype)

    lin = _srgb_to_linear(rgb)
    xyz = torch.einsum("ij,bjhw->bihw", m, lin) / w.view(1, 3, 1, 1)
    f = _f_lab(xyz)
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """lab: (B, 3, H, W) → (B, 3, H, W) sRGB in [0, 1] (may exceed range; clamp at call site)."""
    if lab.dim() != 4 or lab.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(lab.shape)}")
    m = _XYZ2RGB.to(lab.device, dtype=lab.dtype)
    w = _WHITE.to(lab.device, dtype=lab.dtype)

    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = torch.cat([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], dim=1) * w.view(1, 3, 1, 1)
    lin = torch.einsum("ij,bjhw->bihw", m, xyz)
    return _linear_to_srgb(lin)


def derive_gray_rgb_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B, 3, H, W) → 3-channel grayscale-as-RGB via LAB-L (model input contract)."""
    lab = rgb_to_lab(rgb)
    L = lab[:, 0:1]
    gray_lab = torch.cat([L, torch.zeros_like(L), torch.zeros_like(L)], dim=1)
    return lab_to_rgb(gray_lab).clamp(0, 1)


def color_enhance_blend(rgb: torch.Tensor, factor: float = 1.2) -> torch.Tensor:
    """Saturation-boost by linear extrapolation between Rec.601 luma and color.

    out = luma * (1 - factor) + rgb * factor, clamped to [0, 1].

    factor > 1 extrapolates *past* the original color away from gray, boosting
    saturation. factor = 1 is a no-op; factor < 1 desaturates. The default 1.2
    matches the original DDColor training recipe.

    Used by the trainer (when train.color_enhance is true) to push the model
    toward more vivid output: applied to gt_rgb before the perceptual loss,
    while L1 still sees the original gt_ab — productive tension between
    "match exactly" and "look more saturated".
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got {tuple(rgb.shape)}")
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    gray = luma.expand_as(rgb)
    return (gray * (1.0 - factor) + rgb * factor).clamp(0, 1)
