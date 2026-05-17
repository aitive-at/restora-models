"""Per-sample PSNR / SSIM / LPIPS metrics (no grad)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = (pred.float() - clean.float()).pow(2).flatten(1).mean(dim=1)
    eps = 1e-10
    return 10.0 * torch.log10(max_val**2 / (mse + eps))


@torch.no_grad()
def lpips_per_sample(model: nn.Module, pred: torch.Tensor,
                     clean: torch.Tensor) -> torch.Tensor:
    """Per-sample LPIPS distance from a pre-instantiated lpips.LPIPS model.

    Inputs are RGB in [0, 1]; LPIPS internally expects [-1, 1]. Lower is
    better. Returns a 1-D tensor of length B.

    Pass `model` from the existing LpipsDecodedLoss instance in the loss
    set so we don't load the ~500 MB VGG weights twice — see
    ``find_lpips_model`` for the lookup helper.
    """
    pred_n = pred * 2.0 - 1.0
    target_n = clean * 2.0 - 1.0
    return model(pred_n, target_n).view(-1)


def find_lpips_model(loss_set) -> Optional[nn.Module]:
    """Return the inner lpips.LPIPS model from a LossSet, or None.

    Used by the trainer to compute a per-axis LPIPS metric without
    duplicating the VGG-based feature extractor that's already in memory
    for the lpips_decoded loss.
    """
    for _, loss, _ in getattr(loss_set, "entries", []):
        if getattr(loss, "name", None) == "lpips_decoded":
            return loss.model
    return None


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


@torch.no_grad()
def ssim(pred: torch.Tensor, clean: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    kernel = _gaussian_kernel().to(pred.device, pred.dtype)
    kernel = kernel.expand(pred.shape[1], 1, *kernel.shape).contiguous()
    pad = kernel.shape[-1] // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

    mu_x = conv(pred); mu_y = conv(clean)
    mu_x2 = mu_x.pow(2); mu_y2 = mu_y.pow(2); mu_xy = mu_x * mu_y
    sigma_x2 = conv(pred * pred) - mu_x2
    sigma_y2 = conv(clean * clean) - mu_y2
    sigma_xy = conv(pred * clean) - mu_xy
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den).mean(dim=(1, 2, 3))
