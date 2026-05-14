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
