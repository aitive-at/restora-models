import os

import pytest

from coliraz.config import ModelConfig
from coliraz.export.onnx import export_onnx_from_model
from coliraz.models import build_ddcolor


@pytest.mark.skipif(
    os.environ.get("COLIRAZ_SLOW") != "1",
    reason="onnx export is slow; set COLIRAZ_SLOW=1 to run",
)
def test_onnx_export_parity(tmp_path):
    import numpy as np
    import torch

    cfg = ModelConfig(
        size="tiny", input_size=32, dec_layers=1, num_queries=2, nf=64, hidden_dim=32
    )
    model = build_ddcolor(cfg, pretrained=False)
    model.train(False)
    path = tmp_path / "m.onnx"
    export_onnx_from_model(
        model, input_size=32, export_path=path, opset=17, simplify=False
    )
    assert path.exists()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    onnx_out = sess.run(None, {"input": x})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-3, rtol=1e-2)


@pytest.mark.skipif(
    os.environ.get("COLIRAZ_SLOW") != "1",
    reason="onnx dynamic-hw export is slow; set COLIRAZ_SLOW=1 to run",
)
def test_onnx_export_dynamic_hw_accepts_different_sizes(tmp_path):
    import numpy as np
    import torch

    cfg = ModelConfig(
        size="tiny", input_size=64, dec_layers=1, num_queries=2, nf=64, hidden_dim=32
    )
    model = build_ddcolor(cfg, pretrained=False)
    model.train(False)
    path = tmp_path / "m.onnx"
    # The parity check inside export already exercises a non-square non-export
    # size, so a successful call here is the assertion we need.
    export_onnx_from_model(
        model,
        input_size=64,
        export_path=path,
        opset=17,
        simplify=False,
        dynamic_hw=True,
    )
    assert path.exists()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Also verify a wildly different (square, smaller) size works.
    x = np.random.rand(1, 3, 96, 96).astype(np.float32)
    onnx_out = sess.run(None, {"input": x})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    assert onnx_out.shape == torch_out.shape == (1, 2, 96, 96)
    np.testing.assert_allclose(onnx_out, torch_out, atol=2e-3, rtol=1e-2)
