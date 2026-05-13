"""Tests for loss registry, individual losses, and the LossSet composer."""
import torch
from torch import nn

from coliraz.config import LossConfig
from coliraz.losses import LossContext, LossSet
from coliraz.losses.colorfulness import ColorfulnessLoss
from coliraz.losses.gan import GeneratorGANLoss, discriminator_loss
from coliraz.losses.pixel import CharbonnierAbLoss, L1AbLoss, L2AbLoss
from coliraz.losses.registry import LOSS_REGISTRY, build_loss, register_loss


def _ctx(pred_ab=None, gt_ab=None, pred_rgb=None, gt_rgb=None, gray_rgb=None, disc=None):
    z2 = torch.zeros(1, 2, 4, 4)
    z3 = torch.zeros(1, 3, 4, 4)
    return LossContext(
        pred_ab=pred_ab if pred_ab is not None else z2,
        gt_ab=gt_ab if gt_ab is not None else z2,
        pred_rgb=pred_rgb if pred_rgb is not None else z3,
        gt_rgb=gt_rgb if gt_rgb is not None else z3,
        gray_rgb=gray_rgb if gray_rgb is not None else z3,
        discriminator=disc,
    )


# ---------- registry ----------

def test_registry_collects_decorated_class():
    @register_loss("toy_test_loss")
    class _Toy(_ToyBase := type("_ToyBase", (nn.Module,), {"name": ""})):
        name = "toy_test_loss"

        def forward(self, ctx):
            return ctx.pred_ab.abs().mean()

    assert "toy_test_loss" in LOSS_REGISTRY
    LOSS_REGISTRY.pop("toy_test_loss")


# ---------- pixel ----------

def test_l1_zero_when_equal():
    assert L1AbLoss()(_ctx()).item() == 0.0


def test_l1_positive_when_unequal():
    pred = torch.zeros(1, 2, 4, 4) + 1.0
    assert L1AbLoss()(_ctx(pred_ab=pred)).item() == 1.0


def test_l2_positive_when_unequal():
    pred = torch.zeros(1, 2, 4, 4) + 2.0
    assert L2AbLoss()(_ctx(pred_ab=pred)).item() == 4.0


def test_charbonnier_grad():
    pred = torch.randn(1, 2, 4, 4, requires_grad=True)
    out = CharbonnierAbLoss()(_ctx(pred_ab=pred))
    out.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


# ---------- gan ----------

class _TinyDisc(nn.Module):
    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


def test_generator_gan_loss_hinge_grad():
    disc = _TinyDisc()
    loss = GeneratorGANLoss(gan_type="hinge")
    rgb = torch.randn(1, 3, 8, 8, requires_grad=True)
    z2 = torch.zeros(1, 2, 8, 8)
    ctx = LossContext(pred_ab=z2, gt_ab=z2, pred_rgb=rgb, gt_rgb=rgb, gray_rgb=rgb, discriminator=disc)
    out = loss(ctx)
    out.backward()
    assert rgb.grad is not None


def test_discriminator_loss_returns_scalar():
    disc = _TinyDisc()
    real = torch.randn(1, 3, 8, 8)
    fake = torch.randn(1, 3, 8, 8)
    d = discriminator_loss(disc, real, fake, gan_type="hinge")
    assert d.dim() == 0


# ---------- colorfulness ----------

def test_colorfulness_decreases_with_more_color():
    z2 = torch.zeros(1, 2, 4, 4)
    gray = torch.zeros(1, 3, 4, 4) + 0.5
    color = torch.zeros(1, 3, 4, 4); color[0, 0] = 1.0
    ctx_gray = LossContext(pred_ab=z2, gt_ab=z2, pred_rgb=gray, gt_rgb=gray, gray_rgb=gray)
    ctx_color = LossContext(pred_ab=z2, gt_ab=z2, pred_rgb=color, gt_rgb=color, gray_rgb=gray)
    loss = ColorfulnessLoss()
    assert loss(ctx_color) < loss(ctx_gray)


# ---------- LossSet ----------

def test_loss_set_aggregates_with_weights():
    ls = LossSet([LossConfig(name="l1_ab", weight=2.0)])
    pred = torch.zeros(1, 2, 4, 4) + 1.0
    z2 = torch.zeros(1, 2, 4, 4)
    rgb = torch.zeros(1, 3, 4, 4)
    ctx = LossContext(pred_ab=pred, gt_ab=z2, pred_rgb=rgb, gt_rgb=rgb, gray_rgb=rgb)
    total, log = ls(ctx)
    assert total.item() == 2.0
    assert log["l1_ab"] == 1.0


def test_loss_set_has_gan_detected():
    with_gan = LossSet([
        LossConfig(name="l1_ab", weight=1.0),
        LossConfig(name="gan", weight=1.0, config={"gan_type": "hinge"}),
    ])
    without_gan = LossSet([LossConfig(name="l1_ab", weight=1.0)])
    assert with_gan.has_gan is True
    assert without_gan.has_gan is False


def test_loss_set_disc_cfg_only_if_gan():
    ls = LossSet([
        LossConfig(
            name="gan", weight=1.0,
            config={"gan_type": "hinge", "discriminator": {"type": "unet", "nf": 32}},
        ),
    ])
    assert ls.discriminator_cfg == {"type": "unet", "nf": 32}


def test_build_loss_with_kwargs():
    # Test that build_loss accepts config kwargs correctly
    loss = build_loss("charbonnier_ab", {"eps": 1e-2})
    assert isinstance(loss, CharbonnierAbLoss)
    assert abs(loss.eps2 - 1e-4) < 1e-12
