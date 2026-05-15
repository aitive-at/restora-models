"""Tests for NAFNetMultiTask using the diffusion refine head."""
import torch

from restora_models.config import ModelConfig
from restora_models.models import build_model


def test_nafnet_with_refine_type_none_returns_coarse_only():
    cfg = ModelConfig(type="nafnet", size="tiny", refine_type="none", input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)


def test_nafnet_with_diffusion_refine_returns_correct_shape():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       refine_type="diffusion", input_size=64)
    model = build_model(cfg, num_axes=5)
    model.train(False)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)


def test_nafnet_with_diffusion_exposes_latents_via_forward_with_extras():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       refine_type="diffusion", input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    clean = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    pred_rgb, extras = model.forward_with_extras(rgb, clean, config)
    assert pred_rgb.shape == (1, 3, 64, 64)
    assert "pred_latent" in extras and "target_latent" in extras
    assert extras["pred_latent"].shape == (1, 4, 8, 8)
    assert extras["target_latent"].shape == (1, 4, 8, 8)


def test_nafnet_legacy_adversarial_refine_still_works():
    cfg = ModelConfig(type="nafnet", size="tiny",
                       adversarial_refine=True, input_size=64)
    model = build_model(cfg, num_axes=5)
    rgb = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = model(rgb, config)
    assert out.shape == (1, 3, 64, 64)
