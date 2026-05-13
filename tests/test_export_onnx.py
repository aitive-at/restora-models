import json
import os

import pytest


@pytest.mark.skipif(os.environ.get("REFINE_SLOW") != "1",
                    reason="onnx export is slow; set REFINE_SLOW=1 to run")
def test_onnx_export_parity_all_configs(tmp_path):
    import numpy as np
    import torch

    from refine.config import ModelConfig
    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    num_axes = 5
    m = build_model(cfg, num_axes=num_axes)
    m.train(False)
    path = tmp_path / "m.onnx"
    task_map = {
        "model_type": "nafnet",
        "model_size": "tiny",
        "input_size": 32,
        "config_axes": ["colorize", "denoise", "sharpen", "dejpeg", "deblur"],
        "version": "0.2.0",
    }
    export_onnx_from_model(m, num_axes=num_axes, input_size=32, export_path=path,
                            opset=17, simplify=False, task_map=task_map)
    assert path.exists()
    sidecar = path.with_suffix(".task_map.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["config_axes"] == ["colorize", "denoise", "sharpen", "dejpeg", "deblur"]
    assert data["version"] == "0.2.0"

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Test 7 reference configs: identity, all-on, each single-axis-on
    reference_configs = [
        ("identity", [0.0] * num_axes),
        ("all-on", [1.0] * num_axes),
    ]
    for i in range(num_axes):
        v = [0.0] * num_axes
        v[i] = 1.0
        reference_configs.append((f"axis{i}-only", v))

    for label, vec in reference_configs:
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        c = np.array([vec], dtype=np.float32)
        onnx_out = sess.run(None, {"input": x, "config": c})[0]
        with torch.no_grad():
            torch_out = m(torch.from_numpy(x), torch.from_numpy(c)).numpy()
        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-2,
                                   err_msg=f"parity failed for config {label}")
