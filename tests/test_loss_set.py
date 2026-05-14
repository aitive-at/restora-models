import torch

from restora_models.config import LossConfig
from restora_models.losses import LossSet
from restora_models.losses.registry import LossContext


def _ctx_two_samples():
    rgb = torch.ones(2, 3, 4, 4)
    z = torch.zeros(2, 3, 4, 4)
    # Sample 0: colorize active (config[0]=1), sample 1: denoise active (config[1]=1)
    config = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0]])
    return LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                       config=config, axes_active=["colorize", "denoise"])


def test_l1_aggregates_with_weight():
    ls = LossSet([LossConfig(name="l1_rgb", weight=2.0)])
    total, log = ls(_ctx_two_samples())
    assert total.item() == 2.0
    assert log["l1_rgb"] == 1.0


def test_apply_to_axes_filters_samples():
    ls = LossSet([LossConfig(name="colorfulness", weight=1.0, apply_to_axes=["colorize"])])
    total, log = ls(_ctx_two_samples())
    assert "colorfulness" in log
    assert torch.isfinite(total)


def test_apply_to_axes_empty_mask():
    ls = LossSet([LossConfig(name="l1_rgb", weight=1.0, apply_to_axes=["dejpeg"])])
    total, log = ls(_ctx_two_samples())
    assert total.item() == 0.0
    assert log["l1_rgb"] == 0.0


def test_apply_to_axes_any_semantics():
    """Both samples qualify if mask covers both of their active axes."""
    ls = LossSet([LossConfig(name="l1_rgb", weight=1.0, apply_to_axes=["colorize", "denoise"])])
    total, log = ls(_ctx_two_samples())
    # Both samples active; l1 of ones vs zeros = 1.0, weighted 1.0
    assert log["l1_rgb"] == 1.0
    assert torch.isfinite(total)


def test_has_gan_detected():
    ls_yes = LossSet([LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge"})])
    ls_no = LossSet([LossConfig(name="l1_rgb", weight=1.0)])
    assert ls_yes.has_gan and not ls_no.has_gan


def test_apply_to_axes_preserves_video_fields():
    """When apply_to_axes filters samples, the sub-context must keep
    secondary_pred_rgb / flow_t_to_secondary so temporal_pair can still
    fire on the filtered subset. Regression test for a silent drop bug."""
    rgb = torch.rand(2, 3, 8, 8)
    sec = torch.rand(2, 3, 8, 8)
    flow = torch.zeros(2, 2, 8, 8)
    config = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0]])
    ctx = LossContext(pred_rgb=rgb, clean_rgb=torch.zeros_like(rgb),
                      degraded_rgb=torch.zeros_like(rgb),
                      config=config, axes_active=["colorize", "denoise"],
                      secondary_pred_rgb=sec, flow_t_to_secondary=flow)
    ls = LossSet([LossConfig(name="temporal_pair", weight=1.0,
                              apply_to_axes=["colorize"])])
    total, log = ls(ctx)
    # If the fix is in place, temporal_pair sees the filtered sec+flow and
    # produces a positive value (uncorrelated pred vs sec). If the fix is
    # missing, sec is None inside sub_ctx and the loss returns 0.
    assert log["temporal_pair"] > 0.0
    assert torch.isfinite(total)
