"""Flow-based warping + cycle-consistency occlusion mask.

Pure ops; no learnable params; ONNX-safe via grid_sample (opset 16+).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _identity_grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    # Build the identity sampling grid in fp32 unconditionally — see the
    # `flow_warp` docstring for why. Caller is responsible for casting to
    # the desired dtype at the boundary.
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).expand(b, h, w, 2).contiguous()


def flow_warp(rgb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp rgb (B,C,H,W) by flow (B,2,H,W). Output pixel (y,x) is sampled from
    (y + flow_y, x + flow_x) in the input.

    fp16-safe ONNX export contract: TensorRT's `IGridSampleLayer` (and ORT's
    TRT EP) reject mixed-dtype GridSample — `input` and `grid` must be the
    same dtype. PyTorch's eager dispatch hides this via aten's internal
    promotion, but ONNX export of `torch.linspace` lowers to a sequence of
    `Range/Mul/Add` ops that ignore the `dtype` kwarg and produce fp32
    intermediates regardless of the model's dtype. So an fp16 model that
    naively built the grid in `rgb.dtype` still produced an fp32 sampling
    grid in the ONNX graph, with predictable TRT-compile failures.

    Fix (defensive, applies to fp16 and any future precision):
    1. Build the identity grid + flow offsets explicitly in fp32 — matches
       what ONNX would produce anyway, and avoids relying on linspace
       respecting a dtype kwarg.
    2. Cast the final `sample_grid` to `rgb.dtype` *unconditionally*. This
       cast is a real op at trace time (fp32 → target dtype), so ONNX
       captures an actual `Cast` node feeding into `GridSample` — which
       TRT fuses away during engine compile. In eager fp32 mode the cast
       is a no-op; in eager fp16 mode the perf cost is a single
       `meshgrid → cast` per warp, negligible vs the conv stack.

    Without this construction, fp16 ONNX export hits
    "ITensor::getDimensions: Error Code 4 (input and grid must be of
    same type)" at TRT compile on every GridSample node. See
    docs/integration/training-side-fp16-export.md for the full handoff.
    """
    if rgb.dim() != 4 or flow.dim() != 4:
        raise ValueError(f"expected 4D tensors, got rgb={rgb.shape} flow={flow.shape}")
    b, _, h, w = rgb.shape
    if flow.shape != (b, 2, h, w):
        raise ValueError(f"flow shape mismatch: rgb={tuple(rgb.shape)} flow={tuple(flow.shape)}")
    base = _identity_grid(b, h, w, rgb.device)                  # fp32
    flow_f32 = flow if flow.dtype == torch.float32 else flow.float()
    scale_x = 2.0 / max(w - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    offset = torch.stack(
        [flow_f32[:, 0] * scale_x, flow_f32[:, 1] * scale_y], dim=-1)
    sample_grid = (base + offset).to(rgb.dtype)                 # unconditional Cast
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
