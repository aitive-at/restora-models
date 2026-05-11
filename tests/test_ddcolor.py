import torch

from coliraz.config import ModelConfig
from coliraz.models.ddcolor import build_ddcolor


def test_ddcolor_tiny_forward_shape():
    cfg = ModelConfig(size="tiny", input_size=64, dec_layers=2, num_queries=8)
    model = build_ddcolor(cfg, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 2, 64, 64)


def test_ddcolor_backward_flows():
    cfg = ModelConfig(size="tiny", input_size=64, dec_layers=2, num_queries=8)
    model = build_ddcolor(cfg, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
