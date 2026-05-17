"""End-to-end ONNX export test for the temporal model.

Builds the tiniest temporal model (nano), exports to ONNX, runs the
graph through ORT CPU and checks shape + numerical sanity.
"""
from pathlib import Path

import numpy as np
import pytest

from restora_models.config import ModelConfig
from restora_models.export.onnx import export_onnx_from_model
from restora_models.models.registry import build_model


def _load_ort_or_skip():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        pytest.skip("onnxruntime not installed")


def test_export_temporal_generic_onnx(tmp_path: Path):
    """Generic 2-input export, dynamic spatial."""
    ort = _load_ort_or_skip()
    cfg = ModelConfig(type="temporal_restora_nano")
    m = build_model(cfg, num_axes=5)
    m.train(False)
    out_path = tmp_path / "tiny.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=64, export_path=out_path,
        opset=17, simplify=True, dynamic_hw=True,
        task_map=None, precision="fp32", fixed_config=None,
    )
    assert out_path.exists()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    assert inames == ["frames", "config"]

    # Smoke at a *different* resolution from export (96x96) to confirm
    # dynamic axes work.
    frames = np.random.rand(1, 7, 3, 96, 96).astype(np.float32)
    config = np.zeros((1, 5), dtype=np.float32)
    config[0, 0] = 1.0
    out = sess.run(None, {"frames": frames, "config": config})[0]
    assert out.shape == (1, 3, 96, 96)
    assert (out >= 0).all() and (out <= 1).all()


def test_export_temporal_baked_onnx(tmp_path: Path):
    """Per-task baked export, 1-input."""
    ort = _load_ort_or_skip()
    cfg = ModelConfig(type="temporal_restora_nano")
    m = build_model(cfg, num_axes=5)
    m.train(False)
    out_path = tmp_path / "tiny_colorize.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=64, export_path=out_path,
        opset=17, simplify=True, dynamic_hw=True,
        task_map={"task": "colorize"}, precision="fp32",
        fixed_config=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert out_path.exists()
    sidecar = out_path.with_suffix(".task_map.json")
    assert sidecar.exists()
    assert '"colorize"' in sidecar.read_text()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    assert inames == ["frames"]
    frames = np.random.rand(1, 7, 3, 64, 64).astype(np.float32)
    out = sess.run(None, {"frames": frames})[0]
    assert out.shape == (1, 3, 64, 64)
