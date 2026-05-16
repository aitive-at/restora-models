"""Export refine compound model to ONNX with per-config parity verification.

Supports precision={"fp32" (default), "fp16", "fp8", "fp4"}. fp16 is a post-export
conversion via onnxconverter-common. fp8 attempts post-training quantization via
onnxruntime.quantization (requires opset 19+ and ORT 1.17+); on unsupported
runtimes it raises with a clear message. fp4 raises NotImplementedError pointing
at TensorRT 10+ / NVIDIA modelopt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from restora_models.utils.color import graph_friendly_color


Precision = Literal["fp32", "fp16", "fp8", "fp4"]
_VALID_PRECISION = ("fp32", "fp16", "fp8", "fp4")


def _reference_configs(num_axes: int) -> list[tuple[str, list[float]]]:
    configs = [("identity", [0.0] * num_axes), ("all-on", [1.0] * num_axes)]
    for i in range(num_axes):
        v = [0.0] * num_axes
        v[i] = 1.0
        configs.append((f"axis{i}-only", v))
    return configs


def _convert_to_fp16(path: Path) -> None:
    """Convert an ONNX model's internal weights + activations to fp16,
    preserving the input/output dtypes as float32.

    keep_io_types=True is deliberate: it lets the C# / Python downstream
    consumer feed normal float32 tensors without per-call dtype casts.
    Internal kernels still run in fp16 for ~2× speedup and ~half memory
    on modern GPUs. The fp32 I/O cost is negligible (a few MB per frame).
    """
    try:
        from onnxconverter_common import float16
    except ImportError as e:
        raise RuntimeError(
            "fp16 export requires `onnxconverter-common`; install with "
            "`uv pip install onnxconverter-common`"
        ) from e
    import onnx
    m = onnx.load(str(path))
    m_fp16 = float16.convert_float_to_float16(
        m, keep_io_types=True, disable_shape_infer=False,
    )
    # Save with weights inline. After fp16 conversion the model is half the
    # size and easily fits inline (~200 MB for a 100M-param model); no need
    # for external data. If torch.onnx.export wrote a `.data` sidecar
    # earlier, it's now orphaned fp32 data — remove it so the artifact is
    # a single self-contained .onnx file the C# / Python downstream can ship.
    onnx.save(m_fp16, str(path), save_as_external_data=False)
    orphan = Path(str(path) + ".data")
    if orphan.exists():
        orphan.unlink()


def _quantize_to_fp8(path: Path, opset: int) -> None:
    if opset < 19:
        raise RuntimeError(f"fp8 requires ONNX opset >= 19; got opset={opset}")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "fp8 export requires onnxruntime>=1.17 with quantization support"
        ) from e
    if not hasattr(QuantType, "QFloat8E4M3FN"):
        raise RuntimeError(
            "Local onnxruntime build lacks fp8 (E4M3) QuantType — "
            "upgrade to onnxruntime>=1.17 with cuda12-fp8 support"
        )
    tmp = path.with_suffix(path.suffix + ".pre-fp8")
    os.replace(path, tmp)
    quantize_dynamic(
        model_input=str(tmp), model_output=str(path),
        weight_type=QuantType.QFloat8E4M3FN,
    )
    os.remove(tmp)


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
    precision: Precision = "fp32",
    fixed_config: list[float] | None = None,
) -> None:
    """Export a refine model to ONNX.

    If ``fixed_config`` is None, the exported ONNX has two inputs
    (``input``, ``config``). If a list of `num_axes` floats is supplied,
    the config tensor is BAKED into the ONNX as a constant buffer and
    the exported graph has only the single ``input`` tensor — that's
    the "RGB in, RGB out" variant for per-task deployment.
    """
    if precision not in _VALID_PRECISION:
        raise ValueError(f"unknown precision {precision!r}; must be one of {_VALID_PRECISION}")
    # Loosen the parity check tolerance to match the precision's natural
    # quantization error. fp16 has ~1 ULP ≈ 1e-3 near unit values, so the
    # round-trip diff lands in [1e-3, 1e-2] in practice. Default atol=1e-3
    # is set for fp32 numerics and would falsely flag every fp16 export.
    _precision_floor = {"fp32": 1e-3, "fp16": 1.5e-2, "fp8": 5e-2, "fp4": 1e-1}
    parity_atol = max(parity_atol, _precision_floor[precision])
    if fixed_config is not None:
        if len(fixed_config) != num_axes:
            raise ValueError(
                f"fixed_config has {len(fixed_config)} entries; expected {num_axes}"
            )
    if precision == "fp4":
        raise NotImplementedError(
            "fp4 / NVFP4 export not yet supported by stable tooling. "
            "Requires TensorRT 10+ on a Blackwell-class GPU (B100/B200/GB200); "
            "see NVIDIA modelopt (https://github.com/NVIDIA/TensorRT-Model-Optimizer) "
            "for the current path. This stub will be replaced once onnxruntime "
            "gains stable fp4 support."
        )

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.train(False)

    dummy_rgb = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    dummy_cfg = torch.zeros(1, num_axes, dtype=torch.float32)

    # Pin the ONNX contract behind a stable wrapper module so future
    # backbone changes can't drift the exported graph's I/O signature.
    # Two flavors:
    #   - generic 2-input (input, config) -> output
    #   - per-task 1-input (input) -> output, with config baked as a buffer
    from .wrapper import ONNXExportWrapper, ONNXExportWrapperBaked

    if fixed_config is not None:
        export_model = ONNXExportWrapperBaked(model, fixed_config=fixed_config,
                                              clamp_output=True)
        export_model.train(False)
        dynamic_axes_baked: dict[str, dict[int, str]] = {
            "input":  {0: "batch"},
            "output": {0: "batch"},
        }
        if dynamic_hw:
            dynamic_axes_baked["input"][2] = "height"
            dynamic_axes_baked["input"][3] = "width"
            dynamic_axes_baked["output"][2] = "height"
            dynamic_axes_baked["output"][3] = "width"
        # graph_friendly_color() replaces torch.where(condition, low, high)
        # in the four piecewise color functions (sRGB<->linear and Lab f /
        # f_inv) with a smooth-blend formulation using only Mul/Sub/Add/
        # Clip ops. This eliminates Where + LessOrEqual + Greater + Cast
        # nodes from the exported graph. Why it matters for ONNX:
        #   - ORT CUDA EP can run Where on GPU, but the boolean mask from
        #     LessOrEqual often forces a CPU stay (the EP can't easily
        #     prove the mask consumer is GPU-resident), triggering
        #     host<->device memcpy on every inference call.
        #   - Mixed CPU/GPU ops break CUDA graph capture (a 1.5-2x win on
        #     repeated-shape inference like video frame loops).
        # The smooth blend is numerically equivalent to <1 LSB on the
        # exact piecewise (120 dB PSNR on real model outputs).
        with graph_friendly_color():
            torch.onnx.export(
                export_model, (dummy_rgb,), str(export_path),
                opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes=dynamic_axes_baked,
            )
    else:
        dynamic_axes: dict[str, dict[int, str]] = {
            "input":  {0: "batch"},
            "config": {0: "batch"},
            "output": {0: "batch"},
        }
        if dynamic_hw:
            dynamic_axes["input"][2] = "height"; dynamic_axes["input"][3] = "width"
            dynamic_axes["output"][2] = "height"; dynamic_axes["output"][3] = "width"
        export_model = ONNXExportWrapper(model)
        export_model.train(False)
        with graph_friendly_color():
            torch.onnx.export(
                export_model, (dummy_rgb, dummy_cfg), str(export_path),
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

    if precision == "fp16":
        _convert_to_fp16(export_path)
    elif precision == "fp8":
        _quantize_to_fp8(export_path, opset=opset)

    if verify_parity:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        _ort_dtype_map = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        in_dtypes = {i.name: _ort_dtype_map.get(i.type, np.float32) for i in sess.get_inputs()}

        def _ort_inputs(x_f32: np.ndarray, c_f32: np.ndarray | None) -> dict:
            d = {"input": x_f32.astype(in_dtypes.get("input", np.float32), copy=False)}
            if c_f32 is not None:
                d["config"] = c_f32.astype(in_dtypes.get("config", np.float32), copy=False)
            return d

        # Reference configs to check against PyTorch. For the baked variant
        # there's only ONE meaningful config (the one baked in), so we skip
        # the loop over reference configs and use the fixed one directly.
        if fixed_config is not None:
            refs = [("baked", list(fixed_config))]
        else:
            refs = _reference_configs(num_axes)

        for label, vec in refs:
            x = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            c = np.array([vec], dtype=np.float32)
            if fixed_config is not None:
                ort_out = sess.run(None, _ort_inputs(x, None))[0].astype(np.float32)
            else:
                ort_out = sess.run(None, _ort_inputs(x, c))[0].astype(np.float32)
            with torch.no_grad():
                t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
                if fixed_config is not None:
                    # Baked wrapper clamps output by default; mirror here for parity.
                    t_out = np.clip(t_out, 0.0, 1.0)
            diff = float(np.abs(ort_out - t_out).max())
            if diff > parity_atol:
                raise RuntimeError(
                    f"ONNX parity failed for {label} ({precision}): max_abs_diff={diff:.3e}"
                )
        if dynamic_hw:
            alt_h = max(48, input_size // 2); alt_w = max(48, input_size // 2 + 32)
            for label, vec in refs:
                x = np.random.rand(1, 3, alt_h, alt_w).astype(np.float32)
                c = np.array([vec], dtype=np.float32)
                try:
                    if fixed_config is not None:
                        ort_out = sess.run(None, _ort_inputs(x, None))[0].astype(np.float32)
                    else:
                        ort_out = sess.run(None, _ort_inputs(x, c))[0].astype(np.float32)
                except Exception as e:
                    raise RuntimeError(f"dynamic_hw ONNX rejected {alt_h}x{alt_w}: {e}") from e
                with torch.no_grad():
                    t_out = model(torch.from_numpy(x), torch.from_numpy(c)).numpy()
                    if fixed_config is not None:
                        t_out = np.clip(t_out, 0.0, 1.0)
                diff = float(np.abs(ort_out - t_out).max())
                if diff > parity_atol:
                    raise RuntimeError(
                        f"dynamic-hw parity failed for {label} at {alt_h}x{alt_w} ({precision}): "
                        f"max_abs_diff={diff:.3e}")

    if task_map is not None:
        sidecar = export_path.with_suffix(".task_map.json")
        task_map_with_prec = dict(task_map)
        task_map_with_prec["precision"] = precision
        if fixed_config is not None:
            task_map_with_prec["baked_config"] = list(fixed_config)
            task_map_with_prec["onnx_inputs"] = ["input"]    # signal: single-input ONNX
        else:
            task_map_with_prec["onnx_inputs"] = ["input", "config"]
        sidecar_tmp = sidecar.with_suffix(".json.tmp")
        sidecar_tmp.write_text(json.dumps(task_map_with_prec, indent=2))
        os.replace(sidecar_tmp, sidecar)
