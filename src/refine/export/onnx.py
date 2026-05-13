"""Export refine compound model to ONNX with per-config parity verification."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _reference_configs(num_axes: int) -> list[tuple[str, list[float]]]:
    """Identity, all-on, and each single-axis-on."""
    configs = [("identity", [0.0] * num_axes), ("all-on", [1.0] * num_axes)]
    for i in range(num_axes):
        v = [0.0] * num_axes
        v[i] = 1.0
        configs.append((f"axis{i}-only", v))
    return configs


def export_onnx_from_model(
    model: nn.Module, *,
    num_axes: int,
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
    dummy_cfg = torch.zeros(1, num_axes, dtype=torch.float32)

    dynamic_axes: dict[str, dict[int, str]] = {
        "input":  {0: "batch"},
        "config": {0: "batch"},
        "output": {0: "batch"},
    }
    if dynamic_hw:
        dynamic_axes["input"][2] = "height"; dynamic_axes["input"][3] = "width"
        dynamic_axes["output"][2] = "height"; dynamic_axes["output"][3] = "width"

    torch.onnx.export(
        model, (dummy_rgb, dummy_cfg), str(export_path),
        opset_version=opset,
        input_names=["input", "config"], output_names=["output"],
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
        for label, vec in _reference_configs(num_axes):
            x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            c = np.array([vec], dtype=np.float32)
            ort_out = sess.run(None, {"input": x, "config": c})[0]
            with torch.no_grad():
                t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(f"ONNX parity failed for {label}: max_abs_diff={diff:.3e}")
        if dynamic_hw:
            alt_h = max(48, input_size // 2); alt_w = max(48, input_size // 2 + 32)
            for label, vec in _reference_configs(num_axes):
                x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
                c = np.array([vec], dtype=np.float32)
                try:
                    ort_out = sess.run(None, {"input": x, "config": c})[0]
                except Exception as e:
                    raise RuntimeError(f"dynamic_hw ONNX rejected {alt_h}x{alt_w}: {e}") from e
                with torch.no_grad():
                    t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
                diff = float(np.abs(ort_out - t_out).max())
                if diff > parity_atol:
                    raise RuntimeError(
                        f"dynamic-hw parity failed for {label} at {alt_h}x{alt_w}: "
                        f"max_abs_diff={diff:.3e}")

    if task_map is not None:
        sidecar = export_path.with_suffix(".task_map.json")
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map, indent=2))
        os.replace(sidecar_tmp, sidecar)
