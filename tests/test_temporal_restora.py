"""Tests for TemporalRestora composite (backbone + RSD refine)."""
import torch

from restora_models.config import ModelConfig
from restora_models.models.registry import build_model


def test_composite_contract():
    cfg = ModelConfig(type="temporal_restora_small")
    m = build_model(cfg, num_axes=5).eval()
    frames = torch.rand(1, 7, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0]])
    out = m(frames, config)
    assert out.shape == (1, 3, 64, 64)


def test_composite_has_components():
    cfg = ModelConfig(type="temporal_restora_small")
    m = build_model(cfg, num_axes=5)
    assert hasattr(m, "refine"), "composite must expose .refine"
    assert hasattr(m, "backbone"), "composite must expose .backbone"
