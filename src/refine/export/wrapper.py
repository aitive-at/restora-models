"""Stable ONNX export entry point for refine models.

Two wrappers, both with the same purpose — pin the exported graph's I/O
signature behind a thin nn.Module so future backbone changes can't drift
the ONNX contract:

- `ONNXExportWrapper` — generic. Exports a 2-input ONNX:
  `forward(input, config) -> output`
- `ONNXExportWrapperBaked` — task-specific. Bakes the 5-axis config as
  a constant buffer, so the exported ONNX has ONLY one input:
  `forward(input) -> output`
  This is the "RGB in, RGB out" wrapper the deployment side asked for.
  Use one exported file per task (colorize.onnx, denoise.onnx, etc.)
  and consumers don't need to know about the config tensor.
"""
from __future__ import annotations

import torch
from torch import nn


class ONNXExportWrapper(nn.Module):
    """Generic 2-input wrapper. forward(input, config) -> output.

    Args:
        model: any nn.Module with `forward(rgb, config) -> rgb` where
               rgb is (B, 3, H, W) float in [0, 1] and config is
               (B, num_axes) float in [0, 1].
        clamp_output: if True, the wrapped output is clamped to [0, 1].
                      Defaults to False to keep training-time behavior
                      where loss/metric code does its own range handling.
    """

    def __init__(self, model: nn.Module, *, clamp_output: bool = False) -> None:
        super().__init__()
        self.model = model
        self.clamp_output = bool(clamp_output)

    def forward(self, input: torch.Tensor,            # noqa: A002 (intentional shadow)
                config: torch.Tensor) -> torch.Tensor:
        out = self.model(input, config)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out


class ONNXExportWrapperBaked(nn.Module):
    """Single-input wrapper with the config tensor baked in as a buffer.

    Exports an ONNX with ONLY `input` (RGB) as input — no config tensor.
    The consumer just does `sess.run(None, {"input": rgb})[0]` and gets
    RGB back; the task is encoded inside the model file. One exported
    file per task is the intended use:

        refine export --model ckpt.pt --output colorize.onnx --task colorize
        refine export --model ckpt.pt --output denoise.onnx  --task denoise
        ...

    The fixed_config buffer is broadcast to the input batch size at
    forward time, so the resulting ONNX still works for any batch dim.

    Args:
        model: backbone with `forward(rgb, config) -> rgb`.
        fixed_config: list of `num_axes` floats — the config to bake in.
                      e.g. [1, 0, 0, 0, 0] for colorize-only.
        clamp_output: if True, output is clamped to [0, 1] before return.
                      Recommended True for inference-time deployment.
    """

    def __init__(self, model: nn.Module, *,
                 fixed_config: list[float], clamp_output: bool = True) -> None:
        super().__init__()
        self.model = model
        self.clamp_output = bool(clamp_output)
        # Buffer (not parameter) — serialized with the module, no grad.
        cfg = torch.tensor(fixed_config, dtype=torch.float32).view(1, -1)
        self.register_buffer("_fixed_config", cfg)

    def forward(self, input: torch.Tensor) -> torch.Tensor:    # noqa: A002
        # Broadcast the baked config to the current batch size.
        b = input.shape[0]
        config = self._fixed_config.expand(b, -1).contiguous()
        out = self.model(input, config)
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out
