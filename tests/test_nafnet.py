import torch

from refine.config import ModelConfig
from refine.models import build_model


def test_nafnet_tiny_forward_shape():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    rgb = torch.rand(2, 3, 32, 32)
    task = torch.tensor([0, 2], dtype=torch.long)
    out = m(rgb, task)
    assert out.shape == rgb.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_nafnet_at_init_is_near_identity():
    """Residual + zero-init head: untrained model passes input through."""
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    m.train(False)
    rgb = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = m(rgb, torch.tensor([0]))
    assert (out - rgb).abs().mean() < 0.05


def test_nafnet_backward_flows():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_tasks=3)
    rgb = torch.rand(1, 3, 32, 32)
    out = m(rgb, torch.tensor([1]))
    out.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
