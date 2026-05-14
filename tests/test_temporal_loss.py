"""Tests for the temporal-pair consistency loss + flow_warp helper."""
import torch

from restora_models.losses.registry import LossContext, build_loss
from restora_models.losses.temporal import flow_warp


def _ctx(pred, secondary=None, flow=None):
    return LossContext(
        pred_rgb=pred, clean_rgb=torch.zeros_like(pred),
        degraded_rgb=torch.zeros_like(pred),
        config=torch.zeros(pred.shape[0], 5),
        axes_active=["denoise"] * pred.shape[0],
        secondary_pred_rgb=secondary,
        flow_t_to_secondary=flow,
    )


def test_flow_warp_zero_flow_is_identity():
    """Zero displacement field should produce the same image."""
    img = torch.rand(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16)
    warped = flow_warp(img, flow)
    assert torch.allclose(warped, img, atol=1e-5)


def test_flow_warp_shifts_correctly():
    """A flow of (dx=1, dy=0) on a step image should shift it by 1 pixel."""
    img = torch.zeros(1, 1, 8, 8)
    img[..., :4] = 1.0    # left half = 1, right half = 0
    # Flow says: for each pixel, sample from (x+1, y) — so output is input shifted LEFT by 1
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 1.0
    warped = flow_warp(img, flow)
    # Output column 0 should now be 1 (sampled from input col 1, which is 1)
    # Output column 3 should be 1 (sampled from input col 4, which is 0... wait)
    # Actually: warped[0, 0, 0] = img[0, 0, 0, 1] = 1 (col 1 still in left half)
    # warped[0, 0, 0, 3] = img[0, 0, 0, 4] = 0 (col 4 in right half)
    assert warped[0, 0, 0, 0].item() == 1.0
    assert warped[0, 0, 0, 3].item() == 0.0


def test_temporal_loss_zero_when_no_secondary():
    """Image-only batch: returns 0 (so the loss is safe in the loss stack)."""
    loss = build_loss("temporal_pair")
    pred = torch.rand(1, 3, 16, 16)
    out = loss(_ctx(pred))
    assert out.item() == 0.0


def test_temporal_loss_zero_when_perfectly_consistent():
    """If sec = warp(pred, flow), the L1 should be ~0."""
    loss = build_loss("temporal_pair")
    pred = torch.rand(1, 3, 16, 16)
    flow = torch.randn(1, 2, 16, 16) * 0.5      # small random flow
    sec = flow_warp(pred, flow)
    out = loss(_ctx(pred, sec, flow))
    assert out.item() < 1e-5


def test_temporal_loss_positive_when_inconsistent():
    """If sec is unrelated to pred, the loss should be substantially > 0."""
    loss = build_loss("temporal_pair")
    pred = torch.rand(1, 3, 16, 16)
    sec = torch.rand(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16)
    out = loss(_ctx(pred, sec, flow))
    assert out.item() > 0.1


def test_temporal_loss_backprop():
    """Gradient must flow back to pred_rgb."""
    loss = build_loss("temporal_pair")
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    sec = torch.rand(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16)
    out = loss(_ctx(pred, sec, flow))
    out.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0
