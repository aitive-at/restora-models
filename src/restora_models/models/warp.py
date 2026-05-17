"""Flow-based warping + cycle-consistency occlusion mask.

Pure ops; no learnable params; ONNX-safe via grid_sample (opset 16+).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _identity_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).expand(b, h, w, 2).contiguous()


def flow_warp(rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp rgb (B,C,H,W) by flow (B,2,H,W). Output pixel (y,x) is sampled from
    (y + flow_y, x + flow_x) in the input."""
    if rgb.dim() != 4 or flow.dim() != 4:
        raise ValueError(f"expected 4D tensors, got rgb={rgb.shape} flow={flow.shape}")
    b, _, h, w = rgb.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError(f"flow shape mismatch: rgb={tuple(rgb.shape)} flow={tuple(flow.shape)}")
    base = _identity_grid(b, h, w, rgb.device, rgb.dtype)
    scale_x = 2.0 / max(w - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    offset = torch.stack([flow[:, 0] * scale_x, flow[:, 1] * scale_y], dim=-1)
    sample_grid = base + offset
    return F.grid_sample(rgb, sample_grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def visibility_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor,
                    threshold: float = 0.5) -> torch.Tensor:
    """Soft visibility mask from cycle consistency.

    A pixel p is visible if flow_fwd(p) + flow_bwd(p + flow_fwd(p)) is near zero.
    """
    if flow_fwd.shape != flow_bwd.shape:
        raise ValueError("flow shapes must match")
    warped_bwd = flow_warp(flow_bwd, flow_fwd)
    cycle = flow_fwd + warped_bwd
    err = torch.linalg.vector_norm(cycle, ord=2, dim=1, keepdim=True)
    return torch.sigmoid(-(err - threshold) * 12.0)
