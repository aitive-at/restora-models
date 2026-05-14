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


def test_baked_wrapper_single_input():
    """ONNXExportWrapperBaked exposes forward(input) -> output; no config arg."""
    from refine.export.wrapper import ONNXExportWrapperBaked

    class _Toy2(nn.Module):
        def forward(self, rgb, config):
            scale = (config.sum(dim=-1).view(-1, 1, 1, 1) + 1.0) * 0.1
            return rgb * scale

    w = ONNXExportWrapperBaked(
        _Toy2(), fixed_config=[1.0, 0.0, 1.0, 0.0, 0.0], clamp_output=False
    )
    x = torch.rand(2, 3, 8, 8)
    out = w(x)
    # config = [1,0,1,0,0] -> sum=2 -> scale=0.3 -> expected = x * 0.3
    expected = x * 0.3
    assert torch.allclose(out, expected, atol=1e-6)


def test_baked_wrapper_config_broadcasts_to_batch():
    """The baked (1, num_axes) config must broadcast cleanly to any batch."""
    from refine.export.wrapper import ONNXExportWrapperBaked

    class _Toy3(nn.Module):
        def forward(self, rgb, config):
            return rgb + config[:, 0:1].view(-1, 1, 1, 1)

    w = ONNXExportWrapperBaked(_Toy3(), fixed_config=[0.5, 0, 0, 0, 0], clamp_output=False)
    for batch in (1, 4, 7):
        x = torch.rand(batch, 3, 4, 4)
        out = w(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x + 0.5, atol=1e-6)


def test_baked_wrapper_clamp_default_true():
    """Default clamp_output=True for baked wrapper — deployment-friendly."""
    from refine.export.wrapper import ONNXExportWrapperBaked

    class _Over(nn.Module):
        def forward(self, rgb, config):
            return rgb * 5.0 - 1.0

    w = ONNXExportWrapperBaked(_Over(), fixed_config=[1, 0, 0, 0, 0])
    out = w(torch.rand(1, 3, 4, 4))
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()


def test_baked_wrapper_signature_is_single_input():
    """forward parameter is just 'input' — exported ONNX has 1 input."""
    import inspect
    from refine.export.wrapper import ONNXExportWrapperBaked
    sig = inspect.signature(ONNXExportWrapperBaked.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input"], f"expected (self, input); got {params}"


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


def test_exporter_baked_emits_single_input_onnx(tmp_path):
    """With fixed_config, the exported ONNX must have ONLY 'input' (no 'config')."""
    import os
    if not os.environ.get("REFINE_SLOW"):
        import pytest
        pytest.skip("slow ONNX export, set REFINE_SLOW=1 to run")

    from refine.config import ModelConfig
    from refine.models import build_model
    from refine.export.onnx import export_onnx_from_model

    m = build_model(ModelConfig(type="nafnet", size="tiny", input_size=32), num_axes=5)
    out = tmp_path / "colorize.onnx"
    export_onnx_from_model(
        m, num_axes=5, input_size=32, export_path=out,
        opset=17, simplify=False, verify_parity=True, parity_atol=1e-3,
        dynamic_hw=False, task_map={"model_type": "nafnet"}, precision="fp32",
        fixed_config=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    import onnx
    om = onnx.load(str(out))
    input_names = sorted([i.name for i in om.graph.input])
    output_names = sorted([o.name for o in om.graph.output])
    assert input_names == ["input"], \
        f"baked ONNX must have only 'input'; got {input_names}"
    assert output_names == ["output"]
    # Sidecar JSON should reflect the bake
    import json
    sidecar = out.with_suffix(".task_map.json")
    tm = json.loads(sidecar.read_text())
    assert tm["baked_config"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert tm["onnx_inputs"] == ["input"]
