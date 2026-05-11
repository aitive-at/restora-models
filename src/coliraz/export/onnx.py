"""Export DDColor to ONNX with shape inference, simplification, and parity check."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx_from_model(
    model: nn.Module,
    *,
    input_size: int,
    export_path: str | Path,
    opset: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    parity_atol: float = 1e-3,
    dynamic_hw: bool = False,
) -> None:
    """Export DDColor to ONNX.

    Args:
        input_size: spatial size of the dummy tracing input. Baked into the
            graph as the example shape; ignored at runtime when dynamic_hw=True.
        dynamic_hw: if True, export height and width as dynamic axes. The
            exported model then accepts any (B, 3, H, W) fp32 input — useful
            for variable-resolution inference at the cost of some ORT kernel
            specialization. Transformer self-attention is O((HW)^2) so memory
            scales with image size.
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dynamic_axes: dict[str, dict[int, str]] = {
        "input": {0: "batch"},
        "output": {0: "batch"},
    }
    if dynamic_hw:
        dynamic_axes["input"][2] = "height"
        dynamic_axes["input"][3] = "width"
        dynamic_axes["output"][2] = "height"
        dynamic_axes["output"][3] = "width"
    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        try:
            import onnx
            import onnxsim

            m = onnx.load(str(export_path))
            m, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m, str(export_path))
        except Exception:
            pass

    if verify_parity:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        # Parity at the export size.
        x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        ort_out = sess.run(None, {"input": x})[0]
        with torch.no_grad():
            t_out = model(torch.from_numpy(x)).numpy()
        diff = float(np.abs(ort_out - t_out).max())
        if diff > parity_atol:
            raise RuntimeError(
                f"ONNX parity failed: max_abs_diff={diff:.3e} > atol={parity_atol}"
            )
        # When dynamic_hw is enabled, also verify a non-square non-export size
        # so we catch silent shape-baking in the exporter or simplifier.
        if dynamic_hw:
            alt_h, alt_w = max(64, input_size // 2), max(64, input_size // 2 + 64)
            x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
            try:
                ort_out = sess.run(None, {"input": x})[0]
            except Exception as e:
                raise RuntimeError(
                    f"dynamic_hw ONNX rejected ({alt_h}x{alt_w}) input: {e}"
                ) from e
            with torch.no_grad():
                t_out = model(torch.from_numpy(x)).numpy()
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(
                    f"ONNX dynamic-hw parity failed at {alt_h}x{alt_w}: "
                    f"max_abs_diff={diff:.3e} > atol={parity_atol}"
                )
