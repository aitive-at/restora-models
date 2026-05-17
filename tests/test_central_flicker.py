"""Tests for central_flicker temporal-consistency loss."""
import torch

from restora_models.losses.registry import LossContext
from restora_models.losses.central_flicker import CentralFlickerLoss


def _ctx(pred_a, pred_b):
    return LossContext(
        pred_rgb=pred_a,
        clean_rgb=pred_a,
        degraded_rgb=pred_a,
        config=torch.zeros(pred_a.shape[0], 5),
        axes_active=["identity"] * pred_a.shape[0],
        secondary_pred_rgb=pred_b,
    )


def test_central_flicker_zero_when_identical():
    loss = CentralFlickerLoss()
    pred = torch.rand(2, 3, 32, 32)
    val = loss(_ctx(pred, pred))
    assert val.item() < 1e-6


def test_central_flicker_positive_when_different():
    loss = CentralFlickerLoss()
    a = torch.rand(2, 3, 32, 32)
    b = a + 0.1 * torch.rand_like(a)
    val = loss(_ctx(a, b))
    assert val.item() > 0.01


def test_central_flicker_zero_when_no_secondary():
    """When secondary_pred_rgb is None (image batch), loss returns 0."""
    loss = CentralFlickerLoss()
    pred = torch.rand(2, 3, 32, 32)
    ctx = LossContext(pred_rgb=pred, clean_rgb=pred, degraded_rgb=pred,
                      config=torch.zeros(2, 5), axes_active=["identity"] * 2)
    val = loss(ctx)
    assert val.item() == 0.0
