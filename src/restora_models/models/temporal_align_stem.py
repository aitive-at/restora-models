"""Temporal alignment stem: flow + warp + visibility into a 28-channel
tensor consumed by TemporalNAFNet.

Input:  frames (B, 7, 3, H, W)
Output: (B, 28, H, W) = 7*3 RGB (center + 6 warped neighbors) + 7 visibility masks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from restora_models.models.flow_distill import FlowDistill
from restora_models.models.warp import flow_warp, visibility_mask


class TemporalAlignStem(nn.Module):
    CENTER_INDEX = 3
    NUM_FRAMES = 7

    def __init__(self, flow_iters: int = 4):
        super().__init__()
        self.flow = FlowDistill(iters=flow_iters)

    @staticmethod
    def _pair(frames: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        return torch.stack([frames[:, src], frames[:, dst]], dim=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5 or frames.shape[1] != self.NUM_FRAMES:
            raise ValueError(f"expected (B,{self.NUM_FRAMES},3,H,W), got {tuple(frames.shape)}")
        b, _, _, h, w = frames.shape
        center = frames[:, self.CENTER_INDEX]
        ones = torch.ones(b, 1, h, w, device=frames.device, dtype=frames.dtype)
        warped, masks = [], []
        for k in range(self.NUM_FRAMES):
            if k == self.CENTER_INDEX:
                warped.append(center)
                masks.append(ones)
                continue
            flow_n_c = self.flow(self._pair(frames, k, self.CENTER_INDEX))
            flow_c_n = self.flow(self._pair(frames, self.CENTER_INDEX, k))
            warped.append(flow_warp(frames[:, k], flow_n_c))
            mask_k = visibility_mask(flow_n_c, flow_c_n).squeeze(1).unsqueeze(1)
            masks.append(mask_k)
        rgb_cat = torch.cat(warped, dim=1)
        mask_cat = torch.cat(masks, dim=1)
        return torch.cat([rgb_cat, mask_cat], dim=1)
