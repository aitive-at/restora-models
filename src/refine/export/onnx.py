"""Export multi-task refine model to ONNX with per-task parity verification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx_from_model(
    model: nn.Module, *,
    num_tasks: int,
    input_size: int,
    export_path: str | Path,
    opset: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    parity_atol: float = 1e-3,
    dynamic_hw: bool = False,
    task_map: dict | None = None,
) -> None:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy_rgb = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_task = torch.tensor([0], dtype=torch.long)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input":  {0: "batch"},
        "task":   {0: "batch"},
        "output": {0: "batch"},
    }
    if dynamic_hw:
        dynamic_axes["input"][2] = "height"; dynamic_axes["input"][3] = "width"
        dynamic_axes["output"][2] = "height"; dynamic_axes["output"][3] = "width"

    torch.onnx.export(
        model, (dummy_rgb, dummy_task), str(export_path),
        opset_version=opset,
        input_names=["input", "task"], output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        try:
            import onnx
            import onnxsim
            m_onnx = onnx.load(str(export_path))
            m_onnx, ok = onnxsim.simplify(m_onnx)
            if ok:
                onnx.save(m_onnx, str(export_path))
        except Exception:
            pass

    if verify_parity:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        for tid in range(num_tasks):
            x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            t = np.array([tid], dtype=np.int64)
            ort_out = sess.run(None, {"input": x, "task": t})[0]
            with torch.no_grad():
                t_out = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(
                    f"ONNX parity failed for task {tid}: max_abs_diff={diff:.3e}")
        if dynamic_hw:
            alt_h = max(48, input_size // 2); alt_w = max(48, input_size // 2 + 32)
            for tid in range(num_tasks):
                x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
                t = np.array([tid], dtype=np.int64)
                try:
                    ort_out = sess.run(None, {"input": x, "task": t})[0]
                except Exception as e:
                    raise RuntimeError(f"dynamic_hw ONNX rejected {alt_h}x{alt_w}: {e}") from e
                with torch.no_grad():
                    t_out = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
                diff = float(np.abs(ort_out - t_out).max())
                if diff > parity_atol:
                    raise RuntimeError(
                        f"dynamic-hw parity failed for task {tid} at {alt_h}x{alt_w}: "
                        f"max_abs_diff={diff:.3e}")

    if task_map is not None:
        sidecar = export_path.with_suffix(".task_map.json")
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map, indent=2))
        os.replace(sidecar_tmp, sidecar)
