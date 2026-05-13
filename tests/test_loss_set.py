import torch

from refine.config import LossConfig
from refine.losses import LossSet
from refine.losses.registry import LossContext


def _ctx_two_tasks():
    rgb = torch.ones(2, 3, 4, 4)
    z = torch.zeros(2, 3, 4, 4)
    return LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                       task_ids=torch.tensor([0, 1]), task_names=["colorize", "denoise"])


def test_l1_aggregates_with_weight():
    ls = LossSet([LossConfig(name="l1_rgb", weight=2.0)])
    total, log = ls(_ctx_two_tasks())
    assert total.item() == 2.0
    assert log["l1_rgb"] == 1.0


def test_apply_to_tasks_filters_samples():
    ls = LossSet([LossConfig(name="colorfulness", weight=1.0, apply_to_tasks=["colorize"])])
    total, log = ls(_ctx_two_tasks())
    assert "colorfulness" in log
    assert torch.isfinite(total)


def test_apply_to_tasks_empty_mask():
    ls = LossSet([LossConfig(name="l1_rgb", weight=1.0, apply_to_tasks=["jpeg"])])
    total, log = ls(_ctx_two_tasks())
    assert total.item() == 0.0
    assert log["l1_rgb"] == 0.0


def test_has_gan_detected():
    ls_yes = LossSet([LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge"})])
    ls_no = LossSet([LossConfig(name="l1_rgb", weight=1.0)])
    assert ls_yes.has_gan and not ls_no.has_gan
