"""Helpers to build a 7-frame window from variable-length input."""
from __future__ import annotations

import torch


def replicate_to_window(
    frames: torch.Tensor, *, num_frames: int = 7, center_index: int = 3,
) -> torch.Tensor:
    """Pad/replicate to exactly num_frames frames.

    - (3, H, W): single image -> all num_frames are copies.
    - (T, 3, H, W) with T < num_frames: center input at center_index, replicate edges.
    - (T, 3, H, W) with T >= num_frames: center-crop num_frames.
    """
    if frames.dim() == 3:
        return frames.unsqueeze(0).expand(num_frames, *frames.shape).contiguous()
    if frames.dim() != 4:
        raise ValueError(f"expected (T,3,H,W) or (3,H,W), got {tuple(frames.shape)}")
    t = frames.shape[0]
    if t >= num_frames:
        start = (t - num_frames) // 2
        return frames[start:start + num_frames].contiguous()
    center_in = t // 2
    out = []
    for k in range(num_frames):
        idx = k - center_index + center_in
        idx = max(0, min(t - 1, idx))
        out.append(frames[idx])
    return torch.stack(out, dim=0).contiguous()
