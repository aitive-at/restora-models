"""Temporal-pair consistency loss for video restoration training.

Trains the model to produce naturally temporally-consistent outputs even
under per-frame inference (no recurrence, no flicker). The mechanism:

    out_t   = model(degrade(frame_t),   config)
    out_t+k = model(degrade(frame_t+k), config)
    warped  = flow_warp(out_t, flow)
    L_temp  = L1(warped, out_t+k)

Flow convention: `flow_warp(image, flow)` samples image at `(p +
flow[p])`. For `warped ≈ out_t+k` we therefore need `flow` to be the
backward optical flow from t+k to t — for each pixel p in t+k's grid,
where in t lives the same physical content. RAFT precomputes this as
`RAFT(frame_t+k, frame_t)` (note the argument order). The
LossContext.flow_t_to_secondary field carries this backward flow
despite its name; the field name is kept for backward compatibility.

The LossContext is extended with optional `secondary_pred_rgb` and
`flow_t_to_secondary` fields. If both are present, this loss fires;
otherwise it returns 0 (so the loss can sit in the loss stack always,
contributing only on video batches).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import LossContext, RestorationLoss, register_loss


def flow_warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp `image` from time t to time t+k using `flow` (t → t+k).

    image: (B, C, H, W)
    flow:  (B, 2, H, W) — float pixel displacement; flow[:, 0] is dx, flow[:, 1] is dy
    Returns: (B, C, H, W) — image sampled at (x + dx, y + dy).

    Uses grid_sample with bilinear interpolation and zero-padding for
    out-of-bounds. Standard backward-warping per Baker et al.
    """
    B, _, H, W = image.shape
    # Build a normalized identity grid in [-1, 1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, W, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    # Convert flow from pixels to normalized coords
    flow_norm = flow.permute(0, 2, 3, 1).clone()       # (B, H, W, 2)
    flow_norm[..., 0] = flow_norm[..., 0] * 2.0 / max(W - 1, 1)
    flow_norm[..., 1] = flow_norm[..., 1] * 2.0 / max(H - 1, 1)
    sample_grid = grid + flow_norm
    return F.grid_sample(image, sample_grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


@register_loss("temporal_pair")
class TemporalPairLoss(RestorationLoss):
    """L1 consistency between warped(out_t) and out_t+k.

    Reads `secondary_pred_rgb` and `flow_t_to_secondary` from the
    LossContext. Returns 0 if either is absent (image-only batches).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, ctx: LossContext) -> torch.Tensor:
        sec = getattr(ctx, "secondary_pred_rgb", None)
        flow = getattr(ctx, "flow_t_to_secondary", None)
        if sec is None or flow is None:
            # Image-only batch; return a connected-to-graph zero so backward works.
            return torch.zeros((), device=ctx.pred_rgb.device, dtype=ctx.pred_rgb.dtype)
        warped = flow_warp(ctx.pred_rgb, flow)
        return F.l1_loss(warped, sec)
