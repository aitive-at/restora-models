import torch

from refine.models.heads import DualOutputHead


def _head(in_dim: int = 8) -> DualOutputHead:
    return DualOutputHead(in_dim=in_dim)


def test_forward_shape():
    h = _head()
    features = torch.randn(2, 8, 16, 16)
    rgb = torch.rand(2, 3, 16, 16)
    config = torch.zeros(2, 5)
    out = h(features=features, rgb_input=rgb, config=config)
    assert out.shape == rgb.shape
    assert torch.isfinite(out).all()


def test_passthrough_when_colorize_zero():
    """With config[0]=0, output must equal rgb + head_rgb(features) exactly
    (modulo the Lab round-trip). The ab override path contributes nothing."""
    h = _head()
    features = torch.randn(1, 8, 16, 16)
    rgb = torch.rand(1, 3, 16, 16)
    config = torch.zeros(1, 5)
    with torch.no_grad():
        rgb_delta = h.head_rgb(features)
        expected = rgb + rgb_delta
        out = h(features=features, rgb_input=rgb, config=config)
    assert (out - expected).abs().mean().item() < 1e-3


def test_ab_head_zero_init_means_gray_for_colorize_one():
    """With head_ab zero-initialized and config[0]=1, the prediction's ab
    channels should be 0 (the Lab-gray axis), so the output is grayscale
    derived from input's L."""
    h = _head()
    rgb = torch.rand(1, 3, 16, 16)
    features = torch.randn(1, 8, 16, 16)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    with torch.no_grad():
        out = h(features=features, rgb_input=rgb, config=config)
    rgb_chan_var = out.var(dim=1).mean().item()
    assert rgb_chan_var < 1e-2, f"output is not gray; per-pixel RGB variance = {rgb_chan_var}"


def test_linear_gate_distinct_endpoints():
    """The gate is linear in Lab-ab space; verify the mid-point output is
    distinct from both endpoints when head_ab predicts a nonzero value."""
    h = _head()
    with torch.no_grad():
        h.head_ab.weight.fill_(0.01)
        # Bias must stay inside the normalized ab range (~[-1, 1]) so that
        # the final LabToRgb clamp doesn't collapse the half/full endpoints
        # onto the same sRGB-gamut boundary.
        h.head_ab.bias.fill_(0.3)
    features = torch.randn(1, 8, 16, 16)
    rgb = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out_0    = h(features, rgb, torch.tensor([[0.0, 0, 0, 0, 0]]))
        out_1    = h(features, rgb, torch.tensor([[1.0, 0, 0, 0, 0]]))
        out_half = h(features, rgb, torch.tensor([[0.5, 0, 0, 0, 0]]))
    delta_to_0 = (out_half - out_0).abs().mean().item()
    delta_to_1 = (out_half - out_1).abs().mean().item()
    assert delta_to_0 > 1e-3 and delta_to_1 > 1e-3, \
        "out_half is not distinct from both endpoints"


def test_gradient_routing():
    """When config[0]=0 across the whole batch, head_ab must receive zero grad
    (it's gated out). head_rgb receives grad on every sample."""
    h = _head()
    features = torch.randn(2, 8, 16, 16, requires_grad=False)
    rgb = torch.rand(2, 3, 16, 16, requires_grad=False)
    config = torch.zeros(2, 5)
    out = h(features=features, rgb_input=rgb, config=config)
    out.sum().backward()
    assert h.head_rgb.weight.grad is not None
    assert h.head_rgb.weight.grad.abs().sum().item() > 0
    assert h.head_ab.weight.grad is not None
    assert h.head_ab.weight.grad.abs().sum().item() < 1e-6


def test_param_count_small():
    """The dual head should be a thin output module, not a chunky decoder."""
    h = DualOutputHead(in_dim=64)
    n = sum(p.numel() for p in h.parameters())
    assert n < 5000, f"DualOutputHead too large: {n} params"
