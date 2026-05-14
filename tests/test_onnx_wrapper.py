import torch
from torch import nn

from refine.export.wrapper import ONNXExportWrapper


class _Toy(nn.Module):
    """Minimal model satisfying the refine forward contract."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.proj.weight, 0.1)

    def forward(self, rgb, config):
        out = self.proj(rgb)
        return out + config.sum(dim=-1).view(-1, 1, 1, 1) * 0.0


def test_wrapper_is_pure_passthrough():
    m = _Toy()
    w = ONNXExportWrapper(m, clamp_output=False)
    x = torch.rand(2, 3, 16, 16)
    c = torch.tensor([[1.0, 0, 1, 0, 0], [0, 1, 0, 1, 0]])
    direct = m(x, c)
    via_wrapper = w(x, c)
    assert torch.equal(direct, via_wrapper)


def test_clamp_applied():
    class _Overshoot(nn.Module):
        def forward(self, rgb, config):
            return rgb * 5.0 - 1.0
    w = ONNXExportWrapper(_Overshoot(), clamp_output=True)
    x = torch.rand(1, 3, 4, 4)
    out = w(x, torch.zeros(1, 5))
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_clamp_off_by_default():
    class _Overshoot(nn.Module):
        def forward(self, rgb, config):
            return rgb * 5.0 - 1.0
    w = ONNXExportWrapper(_Overshoot())
    out = w(torch.rand(1, 3, 4, 4), torch.zeros(1, 5))
    assert (out < 0.0).any() or (out > 1.0).any()


def test_signature_names():
    import inspect
    sig = inspect.signature(ONNXExportWrapper.forward)
    params = list(sig.parameters.keys())
    assert params[1] == "input", f"first arg must be 'input', got {params[1]!r}"
    assert params[2] == "config", f"second arg must be 'config', got {params[2]!r}"


def test_exporter_uses_wrapper(tmp_path):
    """Smoke: full export through export_onnx_from_model produces an ONNX
    whose input names are exactly ['input', 'config'] and output is 'output'."""
    import os
    if not os.environ.get("REFINE_SLOW"):
        import pytest
        pytest.skip("slow ONNX export, set REFINE_SLOW=1 to run")

    from refine.config import ModelConfig
    from refine.models import build_model
    from refine.export.onnx import export_onnx_from_model

    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "wrapped.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=32, export_path=out,
        opset=17, simplify=False, verify_parity=True, parity_atol=1e-3,
        dynamic_hw=False, task_map=None, precision="fp32",
    )
    import onnx
    om = onnx.load(str(out))
    input_names = sorted([i.name for i in om.graph.input])
    output_names = sorted([o.name for o in om.graph.output])
    assert input_names == ["config", "input"]
    assert output_names == ["output"]
