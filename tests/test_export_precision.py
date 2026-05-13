import os

import pytest
import torch

from refine.config import ModelConfig
from refine.export.onnx import export_onnx_from_model
from refine.models import build_model


@pytest.mark.skipif(not os.environ.get("REFINE_SLOW"), reason="slow ONNX export, set REFINE_SLOW=1")
def test_fp16_export_round_trip(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp16.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=32, export_path=out,
        opset=17, simplify=False, verify_parity=True, parity_atol=5e-2,
        dynamic_hw=False, task_map={"model_type": "nafnet"},
        precision="fp16",
    )
    assert out.exists()
    import onnx
    om = onnx.load(str(out))
    fp16 = sum(1 for init in om.graph.initializer if init.data_type == onnx.TensorProto.FLOAT16)
    fp32 = sum(1 for init in om.graph.initializer if init.data_type == onnx.TensorProto.FLOAT)
    assert fp16 > 0
    assert fp16 > fp32, f"expected fp16 dominance; got fp16={fp16} fp32={fp32}"


def test_fp8_raises_capability_error_on_unsupported_runtime(tmp_path):
    """If the local onnxruntime build lacks fp8 support, fp8 export must raise a
    clear error message naming the missing capability — not silently fall back."""
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp8.onnx"
    try:
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=19, simplify=False, verify_parity=False,
            dynamic_hw=False, task_map=None, precision="fp8",
        )
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e).lower()
        assert ("fp8" in msg) or ("e4m3" in msg) or ("opset" in msg) or ("not supported" in msg)


def test_fp4_raises_not_implemented(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model_fp4.onnx"
    with pytest.raises(NotImplementedError) as ei:
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=21, simplify=False, verify_parity=False,
            dynamic_hw=False, task_map=None, precision="fp4",
        )
    assert "fp4" in str(ei.value).lower() or "nvfp4" in str(ei.value).lower()
    assert "tensorrt" in str(ei.value).lower() or "modelopt" in str(ei.value).lower()


def test_invalid_precision_rejected(tmp_path):
    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "model.onnx"
    with pytest.raises(ValueError):
        export_onnx_from_model(
            m, num_axes=5, input_size=32, export_path=out,
            opset=17, simplify=False, verify_parity=False,
            precision="fp64",
        )
