import torch

from refine.losses.registry import LOSS_REGISTRY, LossContext, build_loss


def _ctx(pred, clean):
    return LossContext(
        pred_rgb=pred, clean_rgb=clean,
        degraded_rgb=torch.zeros_like(pred), config=torch.zeros(pred.shape[0], 5),
        axes_active=["colorize"] * pred.shape[0],
    )


def test_chroma_loss_registered():
    assert "chroma_lab" in LOSS_REGISTRY


def test_chroma_loss_zero_on_identical():
    rgb = torch.rand(2, 3, 32, 32)
    loss = build_loss("chroma_lab")
    out = loss(_ctx(rgb.clone(), rgb.clone()))
    assert out.item() == 0.0 or out.item() < 1e-5


def test_chroma_loss_positive_on_different_hue():
    red   = torch.zeros(1, 3, 16, 16); red[:, 0] = 1.0
    green = torch.zeros(1, 3, 16, 16); green[:, 1] = 1.0
    loss = build_loss("chroma_lab")
    out = loss(_ctx(red, green))
    assert out.item() > 1.0


def test_chroma_loss_ignores_luminance():
    bright = torch.full((1, 3, 16, 16), 0.8); bright[:, 0] = 1.0
    dark   = torch.full((1, 3, 16, 16), 0.2); dark[:, 0] = 0.4
    loss = build_loss("chroma_lab")
    bright_dark = loss(_ctx(bright, dark)).item()

    red    = torch.zeros(1, 3, 16, 16); red[:, 0]   = 0.8
    green  = torch.zeros(1, 3, 16, 16); green[:, 1] = 0.8
    hue_flip = loss(_ctx(red, green)).item()

    assert hue_flip > bright_dark * 2


def test_chroma_loss_backprop():
    pred  = torch.rand(1, 3, 16, 16, requires_grad=True)
    clean = torch.rand(1, 3, 16, 16)
    loss = build_loss("chroma_lab")
    out = loss(_ctx(pred, clean))
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
