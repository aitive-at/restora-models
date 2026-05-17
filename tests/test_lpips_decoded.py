"""Tests for lpips_decoded loss."""
import torch

from restora_models.losses.registry import LossContext
from restora_models.losses.lpips_decoded import LpipsDecodedLoss


def _ctx(pred, target):
    return LossContext(
        pred_rgb=pred,
        clean_rgb=target,
        degraded_rgb=target,
        config=torch.zeros(pred.shape[0], 5),
        axes_active=["identity"] * pred.shape[0],
    )


def test_lpips_zero_for_identical():
    loss = LpipsDecodedLoss()
    img = torch.rand(2, 3, 64, 64)
    val = loss(_ctx(img, img))
    assert val.item() < 0.05


def test_lpips_positive_for_different():
    loss = LpipsDecodedLoss()
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    val = loss(_ctx(a, b))
    assert val.item() > 0.1
