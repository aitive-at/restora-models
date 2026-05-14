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
