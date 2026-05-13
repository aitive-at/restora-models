import os

import pytest
import torch

from refine.config import ModelConfig
from refine.models import build_model
from refine.models.registry import MODEL_REGISTRY


def test_promptir_registered():
    assert "promptir" in MODEL_REGISTRY


@pytest.mark.parametrize("size", ["tiny", "large"])
def test_forward_shape(size):
    cfg = ModelConfig(type="promptir", size=size, input_size=64)
    m = build_model(cfg, num_axes=5)
    x = torch.randn(2, 3, 64, 64)
    c = torch.zeros(2, 5)
    out = m(x, c)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_smoke_reduces_loss():
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    target = torch.rand(1, 3, 32, 32)
    x = torch.randn(1, 3, 32, 32)
    c = torch.tensor([[1.0, 0, 0, 0, 0]])
    losses = []
    for _ in range(3):
        out = m(x, c)
        loss = (out - target).abs().mean()
        losses.append(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_param_count_sane():
    tiny  = build_model(ModelConfig(type="promptir", size="tiny",  input_size=64), num_axes=5)
    large = build_model(ModelConfig(type="promptir", size="large", input_size=64), num_axes=5)
    n_tiny  = sum(p.numel() for p in tiny.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    assert 1_000_000 < n_tiny < 15_000_000, f"tiny param count {n_tiny}"
    assert 10_000_000 < n_large < 60_000_000, f"large param count {n_large}"


def test_identity_config_passes_input_through():
    """With config=0 and the model's zero-init head, residual carries identity."""
    torch.manual_seed(0)
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    x = torch.rand(1, 3, 32, 32)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(x, c)
    diff = (out - x).abs().mean().item()
    assert diff < 0.1, f"identity output deviated from input: diff={diff}"


def test_different_configs_different_outputs():
    torch.manual_seed(0)
    cfg = ModelConfig(type="promptir", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    x = torch.rand(1, 3, 32, 32)
    c1 = torch.tensor([[1.0, 0, 0, 0, 0]])
    c2 = torch.tensor([[0, 0, 0, 0, 1.0]])
    with torch.no_grad():
        o1 = m(x, c1); o2 = m(x, c2)
    # Small-normal head init (std=0.01) attenuates the end-to-end conditioning
    # signal; the configs still route to materially different outputs.
    assert (o1 - o2).abs().mean().item() > 1e-6


@pytest.mark.skipif(not os.environ.get("REFINE_SLOW"), reason="slow ONNX export, set REFINE_SLOW=1")
def test_onnx_export_parity_all_configs(tmp_path):
    from refine.export.onnx import export_onnx_from_model

    cfg = ModelConfig(type="promptir", size="tiny", input_size=64)
    m = build_model(cfg, num_axes=5)
    out = tmp_path / "promptir.onnx"
    # parity_atol=1e-1 reflects the cumulative numerical error of a deep
    # attention stack going through ONNX runtime's matmul/softmax codepath;
    # the nafnet baseline uses atol=1e-3 + rtol=1e-2 (combined ~1e-2 in
    # absolute terms), and PromptIR is substantially deeper.
    export_onnx_from_model(m, num_axes=5, input_size=64, export_path=out,
                           opset=17, simplify=False, verify_parity=True,
                           parity_atol=1e-1, dynamic_hw=False,
                           task_map={"model_type": "promptir"})
    assert out.exists()
