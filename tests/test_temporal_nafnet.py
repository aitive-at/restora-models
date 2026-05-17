"""Tests for TemporalNAFNet backbone."""
import torch

from restora_models.config import ModelConfig
from restora_models.models.registry import build_model


def test_temporal_nafnet_contract():
    cfg = ModelConfig(type="temporal_nafnet_small")
    m = build_model(cfg, num_axes=5).eval()
    frames = torch.rand(2, 7, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 1.0, 0, 0]])
    out = m(frames, config)
    assert out.shape == (2, 3, 64, 64), f"got {tuple(out.shape)}"
    assert out.dtype == frames.dtype


def test_temporal_nafnet_any_resolution():
    cfg = ModelConfig(type="temporal_nafnet_small")
    m = build_model(cfg, num_axes=5).eval()
    for hw in [(96, 96), (128, 192), (256, 144), (96, 256)]:
        frames = torch.rand(1, 7, 3, *hw)
        cfgvec = torch.zeros(1, 5)
        out = m(frames, cfgvec)
        assert out.shape == (1, 3, *hw)


def test_temporal_nafnet_param_count_by_size():
    bands = {
        "temporal_nafnet_nano": (4_000_000, 12_000_000),
        "temporal_nafnet_small": (15_000_000, 30_000_000),
    }
    for name, (lo, hi) in bands.items():
        m = build_model(ModelConfig(type=name), num_axes=5)
        n = sum(p.numel() for p in m.parameters())
        assert lo <= n <= hi, f"{name}: {n} not in [{lo}, {hi}]"
