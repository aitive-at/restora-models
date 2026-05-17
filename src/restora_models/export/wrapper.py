"""Stable ONNX export wrappers for the temporal restoration model.

Two wrappers pin the exported graph's I/O contract:

- `ONNXExportWrapper` — generic 2-input. `forward(frames, config) -> rgb`.
- `ONNXExportWrapperBaked` — task-specific 1-input. Bakes the 5-axis
  config as a constant buffer, so the exported ONNX has ONLY one input:
  `forward(frames) -> rgb`.
"""
from __future__ import annotations

import torch
from torch import nn


class ONNXExportWrapper(nn.Module):
    """Generic 2-input wrapper.

    Inputs:
        frames: (B, 7, 3, H, W) float in [0, 1] — 7-frame symmetric window
        config: (B, 5) float in [0, 1] — task vector (colorize, denoise,
                sharpen, dejpeg, deblur)

    Output:
        rgb:    (B, 3, H, W) float in [0, 1] — restored central frame
    """

    def __init__(self, model: nn.Module, *, clamp_output: bool = True) -> None:
        super().__init__()
        self.model = model
        self.clamp_output = bool(clamp_output)

    def forward(self, frames: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        out = self.model(frames, config)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out


class ONNXExportWrapperBaked(nn.Module):
    """1-input wrapper with the 5-axis config tensor baked in.

    Use one exported file per task: `colorize.onnx`, `denoise.onnx`, etc.
    """

    def __init__(self, model: nn.Module, config: torch.Tensor,
                 *, clamp_output: bool = True) -> None:
        super().__init__()
        self.model = model
        # Persist the config as a non-trainable buffer; will appear as a constant
        # in the ONNX graph.
        self.register_buffer("baked_config", config.detach().clone(), persistent=True)
        self.clamp_output = bool(clamp_output)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b = frames.shape[0]
        cfg = self.baked_config.unsqueeze(0).expand(b, -1)
        out = self.model(frames, cfg)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out
