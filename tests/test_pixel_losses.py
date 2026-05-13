import torch

from refine.losses.pixel import CharbonnierRgbLoss, L1RgbLoss
from refine.losses.registry import LossContext


def _ctx(pred=None, clean=None):
    z = torch.zeros(2, 3, 4, 4)
    return LossContext(
        pred_rgb=pred if pred is not None else z.clone(),
        clean_rgb=clean if clean is not None else z.clone(),
        degraded_rgb=z.clone(),
        task_ids=torch.tensor([0, 0]),
        task_names=["colorize", "colorize"],
    )


def test_l1_zero_when_equal():
    assert L1RgbLoss()(_ctx()).item() == 0.0


def test_l1_positive_when_unequal():
    pred = torch.ones(2, 3, 4, 4)
    assert L1RgbLoss()(_ctx(pred=pred)).item() == 1.0


def test_charbonnier_grad():
    pred = torch.randn(2, 3, 4, 4, requires_grad=True)
    CharbonnierRgbLoss()(_ctx(pred=pred)).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
