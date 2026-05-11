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
) -> None:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(export_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
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
        x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        ort_out = sess.run(None, {"input": x})[0]
        with torch.no_grad():
            t_out = model(torch.from_numpy(x)).numpy()
        diff = float(np.abs(ort_out - t_out).max())
        if diff > parity_atol:
            raise RuntimeError(
                f"ONNX parity failed: max_abs_diff={diff:.3e} > atol={parity_atol}"
            )
