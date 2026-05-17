"""Static-unroll RAFT-style flow estimator for the temporal stem.

Designed for ONNX-clean export: no while-loop, no dynamic shape inside
the graph. Trained via distillation from torchvision raft_large in a
separate one-shot script (`restora train-flow-distill`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _residual_block(c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        nn.Conv2d(c, c, 3, padding=1),
    )


class _FeatureExtractor(nn.Module):
    def __init__(self, dims=(32, 64, 96, 128)):
        super().__init__()
        self.stem = nn.Conv2d(3, dims[0], 7, stride=2, padding=3)
        self.act = nn.GELU()
        self.b1 = _residual_block(dims[0])
        self.d1 = nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1)
        self.b2 = _residual_block(dims[1])
        self.d2 = nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1)
        self.b3 = _residual_block(dims[2])
        self.proj = nn.Conv2d(dims[2], dims[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.stem(x))
        x = x + self.b1(x)
        x = self.act(self.d1(x))
        x = x + self.b2(x)
        x = self.act(self.d2(x))
        x = x + self.b3(x)
        return self.proj(x)


class _UpdateBlock(nn.Module):
    def __init__(self, c: int = 128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2 + 2, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        )
        self.delta = nn.Conv2d(c, 2, 3, padding=1)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        h = torch.cat([fa, fb, flow], dim=1)
        return self.delta(self.fuse(h))


class FlowDistill(nn.Module):
    """Static-unroll RAFT student.

    Forward input:  frames (B, 2, 3, H, W)  -- frame_a, frame_b
    Forward output: flow   (B, 2, H, W)     -- backward flow b -> a
    """

    def __init__(self, iters: int = 4):
        super().__init__()
        if iters < 1:
            raise ValueError(f"iters must be >=1, got {iters}")
        self.iters = iters
        self.feat = _FeatureExtractor()
        self.updates = nn.ModuleList([_UpdateBlock() for _ in range(iters)])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.shape[1] != 2:
            raise ValueError(f"expected (B,2,3,H,W), got {tuple(frames.shape)}")
        b, _, _, h, w = frames.shape
        fa = self.feat(frames[:, 0])
        fb = self.feat(frames[:, 1])
        flow = torch.zeros(b, 2, h // 8, w // 8, device=frames.device, dtype=frames.dtype)
        for blk in self.updates:
            flow = flow + blk(fa, fb, flow)
        return F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * 8.0
