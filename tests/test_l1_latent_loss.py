"""Tests for the l1_latent diffusion loss."""
import torch

from restora_models.losses.registry import LossContext, build_loss


def _ctx_with_latents(B=2, H=256, W=256):
    return LossContext(
        pred_rgb=torch.zeros(B, 3, H, W),
        clean_rgb=torch.zeros(B, 3, H, W),
        degraded_rgb=torch.zeros(B, 3, H, W),
        config=torch.zeros(B, 5),
        axes_active=["identity"] * B,
        pred_latent=torch.randn(B, 4, H // 8, W // 8),
        target_latent=torch.randn(B, 4, H // 8, W // 8),
    )


def test_l1_latent_returns_scalar():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    out = loss(ctx)
    assert out.dim() == 0


def test_l1_latent_zero_when_pred_equals_target():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    ctx.pred_latent = ctx.target_latent.clone()
    out = loss(ctx)
    assert out.item() == 0.0


def test_l1_latent_returns_zero_when_latents_absent():
    loss = build_loss("l1_latent")
    ctx = LossContext(
        pred_rgb=torch.zeros(1, 3, 64, 64),
        clean_rgb=torch.zeros(1, 3, 64, 64),
        degraded_rgb=torch.zeros(1, 3, 64, 64),
        config=torch.zeros(1, 5),
        axes_active=["identity"],
    )
    out = loss(ctx)
    assert out.item() == 0.0


def test_l1_latent_positive_when_latents_differ():
    loss = build_loss("l1_latent")
    ctx = _ctx_with_latents()
    out = loss(ctx)
    assert out.item() > 0
