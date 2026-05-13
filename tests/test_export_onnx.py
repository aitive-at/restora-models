import json
import os

import pytest


@pytest.mark.skipif(os.environ.get("REFINE_SLOW") != "1",
                    reason="onnx export is slow; set REFINE_SLOW=1 to run")
def test_onnx_export_parity_all_tasks(tmp_path):
    import numpy as np
    import torch

    from refine.config import ModelConfig
    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    num_tasks = 3
    m = build_model(cfg, num_tasks=num_tasks)
    m.train(False)
    path = tmp_path / "m.onnx"
    task_map = {"tasks": {"colorize": 0, "denoise": 1, "sr_x4": 2}}
    export_onnx_from_model(m, num_tasks=num_tasks, input_size=32, export_path=path,
                            opset=17, simplify=False, task_map=task_map)
    assert path.exists()
    sidecar = path.with_suffix(".task_map.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["tasks"]["sr_x4"] == 2

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for tid in range(num_tasks):
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        t = np.array([tid], dtype=np.int64)
        onnx_out = sess.run(None, {"input": x, "task": t})[0]
        with torch.no_grad():
            torch_out = m(torch.from_numpy(x), torch.from_numpy(t)).numpy()
        np.testing.assert_allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-2)
