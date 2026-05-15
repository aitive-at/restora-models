"""End-to-end tests for LatentDiffusionRefineHead."""
import pytest
import torch


@pytest.fixture
def head():
    from restora_models.models.diffusion_head import LatentDiffusionRefineHead
    return LatentDiffusionRefineHead(feat_dim=64, num_axes=5)


def test_forward_inference_returns_rgb_at_input_resolution(head):
    head.train(False)
    feat = torch.randn(2, 64, 256, 256)
    coarse = torch.rand(2, 3, 256, 256)
    config = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = head(feat, coarse, config)
    assert out.shape == (2, 3, 256, 256)
    assert 0.0 <= out.min().item() and out.max().item() <= 1.001


def test_forward_training_returns_pred_and_target_latent(head):
    head.train(True)
    feat = torch.randn(2, 64, 256, 256)
    coarse = torch.rand(2, 3, 256, 256)
    clean = torch.rand(2, 3, 256, 256)
    config = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    pred_latent, target_latent, decoded_rgb = head.forward_with_targets(
        feat, coarse, clean, config)
    assert pred_latent.shape == (2, 4, 32, 32)
    assert target_latent.shape == (2, 4, 32, 32)
    assert decoded_rgb.shape == (2, 3, 256, 256)


def test_inference_is_deterministic_with_zero_noise(head):
    head.train(False)
    head.set_inference_noise_mode("zero")
    feat = torch.randn(1, 64, 64, 64)
    coarse = torch.rand(1, 3, 64, 64)
    config = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = head(feat, coarse, config)
        out2 = head(feat, coarse, config)
    assert torch.allclose(out1, out2)


def test_param_count_in_budget():
    from restora_models.models.diffusion_head import LatentDiffusionRefineHead
    head = LatentDiffusionRefineHead(feat_dim=64, num_axes=5)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    assert 15_000_000 < trainable < 35_000_000, \
        f"head has {trainable:,} trainable params (target ~25M)"
