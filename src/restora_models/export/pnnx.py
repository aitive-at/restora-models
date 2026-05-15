"""PNNX export entry point for restora-models.

PNNX (PyTorch Neural Network eXchange) is the format used by ncnn for
mobile/edge deployment. https://github.com/pnnx/pnnx

PNNX is fundamentally a TorchScript-based exporter. The export procedure:
  1. Wrap the model in an `ONNXExportWrapper` (or `ONNXExportWrapperBaked`
     for per-task baked exports) so the I/O signature is stable — same
     pattern as the ONNX export so both share the wrapper.
  2. torch.jit.trace the wrapper with an example input.
  3. Call pnnx.convert on the traced .pt file. PNNX then emits several
     companion files (.pnnx.bin / .pnnx.param / .ncnn.bin / .ncnn.param /
     .pnnx.onnx / a Python recreate script).

Dynamic axes: PNNX infers which spatial dims are dynamic by comparing two
example inputs at different shapes. We pass shape A (input_size x input_size)
and shape B (1.5x input_size, square but different) when `dynamic_hw=True`.

Output files:
  Given export_path = "model.pt":
    model.pt              - traced TorchScript (small, mostly for debug)
    model.pnnx.bin        - PNNX intermediate weights
    model.pnnx.param      - PNNX intermediate graph
    model.ncnn.bin        - ncnn weights (target for mobile deployment)
    model.ncnn.param      - ncnn graph
    model.pnnx.onnx       - bonus ONNX export from PNNX (different graph
                            than torch.onnx.export — usually simpler)
    model_pnnx.py         - Python recreate script (for debugging)
    model_ncnn.py         - ncnn Python recreate script
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .wrapper import ONNXExportWrapper, ONNXExportWrapperBaked


def export_pnnx_from_model(
    model: nn.Module,
    *,
    num_axes: int,
    input_size: int,
    export_path: str | Path,
    dynamic_hw: bool = False,
    fp16: bool = True,
    fixed_config: list[float] | None = None,
    task_map: dict | None = None,
    check_trace: bool = False,
) -> None:
    """Export a restora model to PNNX format (and ncnn as a byproduct).

    Args:
        model: a backbone with `forward(rgb, config) -> rgb` (B,3,H,W in/out
            float in [0,1]; config (B, num_axes) float).
        num_axes: number of restoration axes (5 in the current codebase).
        input_size: spatial resolution for the first example input (square).
            When `dynamic_hw=True` we also pass an alternate resolution
            (1.5x) so PNNX can infer which spatial dims are dynamic.
        export_path: target path; PNNX writes companion files alongside it
            (.pnnx.bin/.param, .ncnn.bin/.param, .pnnx.onnx, etc.).
        dynamic_hw: if True, pass two example inputs at different spatial
            sizes so PNNX marks H/W as dynamic dimensions.
        fp16: emit fp16-quantized ncnn weights (default True). Internal
            kernels are quantized; I/O layout in the consumer is dictated
            by ncnn runtime (typically fp32 input/output by default).
        fixed_config: if provided, the config tensor is baked into the
            traced graph and the exported model takes only a single
            image input. Use for per-task deployment artifacts.
        task_map: optional metadata written to `<export_path>.task_map.json`
            (same convention as the ONNX exporter).
        check_trace: pass through to torch.jit.trace's `check_trace` flag.
            Default False because the model has small Lab-conversion
            numerical paths that produce sub-1e-6 differences on retrace,
            which would otherwise fail the check.

    The function blocks until PNNX finishes (~10-60s on a small model,
    minutes on a NAFNet-large). The PNNX binary runs as a subprocess
    inside the `pnnx` Python wrapper.
    """
    import pnnx   # imported lazily — heavy package + only needed at export

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the export wrapper (same pattern as the ONNX exporter so the
    # I/O contract stays in lockstep between the two formats).
    model = model.train(False)
    if fixed_config is not None:
        if len(fixed_config) != num_axes:
            raise ValueError(
                f"fixed_config has {len(fixed_config)} entries; expected {num_axes}"
            )
        wrapped: nn.Module = ONNXExportWrapperBaked(
            model, fixed_config=fixed_config, clamp_output=True,
        )
        ex_inputs = (torch.rand(1, 3, input_size, input_size),)
        ex_inputs2: tuple[torch.Tensor, ...] | None = (
            (torch.rand(1, 3, int(input_size * 1.5), int(input_size * 1.5)),)
            if dynamic_hw else None
        )
    else:
        wrapped = ONNXExportWrapper(model, clamp_output=True)
        # Use a colorize-enabled config so all heads see non-trivial activation.
        cfg_vec = [1.0] + [0.0] * (num_axes - 1)
        ex_inputs = (
            torch.rand(1, 3, input_size, input_size),
            torch.tensor([cfg_vec], dtype=torch.float32),
        )
        ex_inputs2 = (
            (
                torch.rand(1, 3, int(input_size * 1.5), int(input_size * 1.5)),
                torch.tensor([cfg_vec], dtype=torch.float32),
            )
            if dynamic_hw else None
        )

    # Map our explicit naming to pnnx's expected output paths. PNNX likes
    # to derive filenames from the .pt path; we override every sidecar
    # explicitly so the user's --output choice controls everything.
    base = str(export_path.with_suffix(""))  # strip .pt if present
    ncnn_param = f"{base}.ncnn.param"
    ncnn_bin = f"{base}.ncnn.bin"

    # PNNX's export() runs the binary AND then imports its auto-generated
    # `_pnnx.py` recreate script to validate. The recreate script has a
    # known limitation: for some ops (e.g. GroupNorm in our refine head)
    # it omits constructor args, which raises a TypeError on import. The
    # actual deployment files (.ncnn.bin/.param) are written BEFORE that
    # import, so we tolerate the validation failure if those files landed.
    try:
        pnnx.export(
            wrapped,
            ptpath=str(export_path) if export_path.suffix == ".pt" else f"{base}.pt",
            inputs=ex_inputs,
            inputs2=ex_inputs2,
            pnnxparam=f"{base}.pnnx.param",
            pnnxbin=f"{base}.pnnx.bin",
            pnnxpy=f"{base}_pnnx.py",
            pnnxonnx=f"{base}.pnnx.onnx",
            ncnnparam=ncnn_param,
            ncnnbin=ncnn_bin,
            ncnnpy=f"{base}_ncnn.py",
            check_trace=check_trace,
            fp16=fp16,
        )
    except TypeError as exc:
        # Diagnostic-only — if the deployment files are present, surface a
        # warning rather than failing the whole export.
        if Path(ncnn_bin).exists() and Path(ncnn_param).exists():
            import warnings
            warnings.warn(
                f"PNNX recreate-script validation raised TypeError "
                f"({exc.__class__.__name__}: {exc}), but deployment files "
                f"({ncnn_param}, {ncnn_bin}) were written successfully. "
                "This is a known pnnx limitation for some ops (e.g. GroupNorm) "
                "and does NOT affect the .ncnn.bin/.param artifacts that "
                "consumers actually load.",
                stacklevel=2,
            )
        else:
            raise

    if task_map is not None:
        sidecar = Path(f"{base}.task_map.json")
        sidecar.write_text(json.dumps(task_map, indent=2))
