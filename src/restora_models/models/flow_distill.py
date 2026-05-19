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

    fp16 deployment contract: this module is an **fp32 island** inside an
    otherwise-fp16 graph. The iterative flow refinement (the
    `for blk in self.updates` loop) accumulates intermediate residuals
    whose magnitudes reach ±65000 after just two iterations — the model
    was trained in bf16 (range ±3.4e38), so the learned weights aren't
    constrained to stay representable in fp16 (range ±65504). Native-fp16
    GPU silicon overflows on the third iteration; CPU EP survives only
    because it emulates fp16 with fp32 accumulators internally.

    Protection mechanism: `_apply` is overridden to refuse fp16 dtype
    conversions, so `model.half()` on the parent leaves this submodule's
    weights in fp32. `forward` then casts at the I/O boundary — fp16 in,
    fp32 inside, fp16 out — preserving the model-level fp16 contract.
    bf16 conversions are allowed through unchanged because bf16 has the
    same dynamic range as fp32; only fp16 is blocked.
    """

    def __init__(self, iters: int = 4):
        super().__init__()
        if iters < 1:
            raise ValueError(f"iters must be >=1, got {iters}")
        self.iters = iters
        self.feat = _FeatureExtractor()
        self.updates = nn.ModuleList([_UpdateBlock() for _ in range(iters)])

    def _apply(self, fn, recurse: bool = True):
        # Filter out fp16 dtype conversions for this subtree. Device moves
        # (.cuda()/.cpu()) and other dtypes (bf16, fp32) pass through
        # unchanged. See class docstring for why.
        def _filtered(t: torch.Tensor) -> torch.Tensor:
            new_t = fn(t)
            if new_t.dtype == torch.float16 and t.dtype != torch.float16:
                # Caller asked for fp16; refuse, return the original tensor
                # but apply any non-dtype transforms (device, etc.) by
                # re-running with dtype preserved.
                if new_t.device != t.device:
                    return new_t.to(t.dtype)
                return t
            return new_t
        return super()._apply(_filtered, recurse=recurse)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.shape[1] != 2:
            raise ValueError(f"expected (B,2,3,H,W), got {tuple(frames.shape)}")
        in_dtype = frames.dtype
        # Cast the input to fp32 if it arrived as fp16; the weights are
        # already fp32 thanks to the _apply override, so the convs need
        # matching fp32 input. bf16 inputs pass through unchanged
        # (conv2d natively handles bf16-vs-fp32-weights mixing).
        if in_dtype == torch.float16:
            frames = frames.float()
        b, _, _, h, w = frames.shape
        fa = self.feat(frames[:, 0])
        fb = self.feat(frames[:, 1])
        flow = torch.zeros(b, 2, h // 8, w // 8, device=frames.device, dtype=frames.dtype)
        for blk in self.updates:
            flow = flow + blk(fa, fb, flow)
        out = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * 8.0
        # Clamp the final flow output to a fp16-safe range before any cast.
        # Real video flow magnitudes are bounded by image dimensions
        # (typically <100 px, extreme motion <1024 px), so ±1024 is well
        # above any plausible real input but safely inside fp16's ±65504.
        # On synthetic / adversarial inputs (uniform noise, static frames)
        # the unrolled refinement can produce huge spurious values that
        # would overflow on cast to fp16 — clamping yields a deterministic
        # valid result rather than NaN/Inf propagating through the model.
        # The clamp is a no-op on any real video input.
        out = torch.clamp(out, -1024.0, 1024.0)
        # Restore the caller's dtype so downstream modules see what they
        # expect. fp32 in → fp32 out (no-op cast); fp16 in → fp16 out
        # (explicit Cast which TRT can fuse away during engine compile).
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        return out
