import torch
from torch import nn

from refine.losses.perceptual import VGG16BNPerceptualLoss
from refine.losses.registry import LossContext


def test_perceptual_loss_grad_flows():
    """Stubbed VGG to avoid downloading weights in CI."""
    loss = VGG16BNPerceptualLoss.__new__(VGG16BNPerceptualLoss)
    nn.Module.__init__(loss)
    stub = nn.ModuleDict({
        "conv1_1": nn.Conv2d(3, 4, 3, padding=1),
        "conv2_1": nn.Conv2d(4, 8, 3, padding=1),
        "conv3_1": nn.Conv2d(8, 16, 3, padding=1),
    })
    loss._stages = stub
    loss._weights = {"conv1_1": 1.0, "conv2_1": 1.0, "conv3_1": 1.0}
    loss._criterion = nn.L1Loss()
    loss.style_weight = 0.0
    loss._input_norm = True
    loss.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    loss.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    pred = torch.randn(1, 3, 16, 16, requires_grad=True)
    gt = torch.randn(1, 3, 16, 16)
    z = torch.zeros(1, 3, 16, 16)
    ctx = LossContext(pred_rgb=pred, clean_rgb=gt, degraded_rgb=z,
                      config=torch.zeros(1, 5),
                      axes_active=["denoise"])
    loss(ctx).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
