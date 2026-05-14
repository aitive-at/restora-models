import torch

from refine.config import ModelConfig
from refine.models import build_model


def test_nafnet_tiny_forward_shape():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_axes=5)
    rgb = torch.rand(2, 3, 32, 32)
    config = torch.rand(2, 5).clamp(0, 1)
    out = m(rgb, config)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_nafnet_at_init_is_near_identity():
    """Residual + zero-init head: untrained model passes input through."""
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    rgb = torch.rand(1, 3, 32, 32)
    config = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(rgb, config)
    assert (out - rgb).abs().mean() < 0.05


def test_nafnet_backward_flows():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 32, 32)
    config = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    out = m(rgb, config)
    out.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_nafnet_has_dual_heads():
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    assert hasattr(m, "head_lab_delta"), "head_lab_delta missing"
    assert hasattr(m, "head_ab_abs"), "head_ab_abs missing"
    assert not hasattr(m, "head"), \
        "bare self.head must be removed; replaced by head_lab_delta + head_ab_abs"
    assert m.head_lab_delta.out_channels == 3
    assert m.head_ab_abs.out_channels == 2


def test_nafnet_colorize_off_preserves_input():
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        out = m(x, c)
    diff = (out - x).abs().mean().item()
    assert diff < 0.05, f"identity-config output drifted: diff={diff}"


def test_nafnet_colorize_on_predicts_gray_at_init():
    """With config[0]=1 and untrained heads (both zero), output should be
    grayscale (head_ab_abs produces 0 -> Lab ab = 0 -> gray)."""
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    m.train(False)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    c = torch.tensor([[1.0, 0, 0, 0, 0]])
    with torch.no_grad():
        out = m(x, c)
    chan_var = out.var(dim=1).mean().item()
    assert chan_var < 5e-2, f"colorize=1 with zero-init head_ab is not gray: var={chan_var}"
