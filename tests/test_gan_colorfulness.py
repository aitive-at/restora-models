import torch
from torch import nn

from refine.losses.colorfulness import ColorfulnessLoss
from refine.losses.gan import GeneratorGANLoss, discriminator_loss
from refine.losses.registry import LossContext
from refine.models.discriminator import UNetDiscriminator


def _ctx(disc=None):
    rgb = torch.rand(2, 3, 16, 16, requires_grad=True)
    z = torch.zeros(2, 3, 16, 16)
    return LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                       task_ids=torch.tensor([0, 0]), task_names=["x", "x"],
                       discriminator=disc), rgb


def test_unet_discriminator_shape():
    d = UNetDiscriminator(nf=8)
    assert d(torch.randn(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_gen_gan_grad():
    disc = UNetDiscriminator(nf=8)
    loss = GeneratorGANLoss(gan_type="hinge")
    ctx, rgb = _ctx(disc=disc)
    loss(ctx).backward()
    assert rgb.grad is not None


def test_disc_loss_scalar():
    disc = UNetDiscriminator(nf=8)
    assert discriminator_loss(disc, torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16),
                              gan_type="hinge").dim() == 0


def test_colorfulness_grad():
    rgb = torch.rand(1, 3, 4, 4, requires_grad=True)
    z = torch.zeros(1, 3, 4, 4)
    ctx = LossContext(pred_rgb=rgb, clean_rgb=z, degraded_rgb=z,
                     task_ids=torch.tensor([0]), task_names=["colorize"])
    ColorfulnessLoss()(ctx).backward()
    assert rgb.grad is not None
