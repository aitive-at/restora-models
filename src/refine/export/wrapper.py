"""Stable ONNX export entry point for refine models.

Pins the exported graph signature `forward(input, config) -> output`
regardless of future changes to a backbone's Python forward. Today the
wrapper is a pure pass-through; in the future it can host stable
preprocessing (input normalization, dtype coercion, output clamping)
without touching the backbones themselves.
"""
from __future__ import annotations

import torch
from torch import nn


class ONNXExportWrapper(nn.Module):
    """Wraps a backbone whose Python forward is `(rgb, config) -> rgb`.

    The parameter names `input` and `config` are deliberately chosen
    because torch.onnx.export uses parameter names for the exported
    graph's input names by default. Output name is "output".

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
