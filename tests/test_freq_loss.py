import torch

from refine.losses.freq import FreqL1Loss
from refine.losses.registry import LossContext


def test_freq_zero_when_equal():
    rgb = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=rgb, clean_rgb=rgb.clone(), degraded_rgb=z,
                      task_ids=torch.tensor([0]), task_names=["sr"])
    assert FreqL1Loss()(ctx).item() < 1e-5


def test_freq_grad():
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    clean = torch.rand(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=z,
                      task_ids=torch.tensor([0]), task_names=["sr"])
    FreqL1Loss()(ctx).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
