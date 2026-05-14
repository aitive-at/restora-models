import torch

from restora_models.losses.registry import LOSS_REGISTRY, LossContext, build_loss


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
    """Pure red vs pure green at full saturation differs by ~0.8 in
    normalized-ab L1 (raw ab spans ~90 / 110 ≈ 0.82 between the hues).
    The exact threshold isn't critical — what matters is the loss is
    in the [0, 2] range (commensurate with l1_rgb), NOT in the [0, 250]
    range that un-normalized Lab produces."""
    red   = torch.zeros(1, 3, 16, 16); red[:, 0] = 1.0
    green = torch.zeros(1, 3, 16, 16); green[:, 1] = 1.0
    loss = build_loss("chroma_lab")
    out = loss(_ctx(red, green))
    assert 0.3 < out.item() < 2.0, f"out={out.item()}"


def test_chroma_loss_magnitude_commensurate_with_l1_rgb():
    """Regression: the loss MUST be in the [0, ~2] range so it doesn't
    180x-dominate l1_rgb in the loss stack. This was the cause of the
    2026-05-14 training-stalls bug."""
    pred  = torch.full((4, 3, 32, 32), 0.5)
    clean = torch.zeros(4, 3, 32, 32); clean[:, 0] = 0.8; clean[:, 1] = 0.4; clean[:, 2] = 0.2
    loss = build_loss("chroma_lab")
    out = loss(_ctx(pred, clean)).item()
    assert out < 2.0, \
        f"chroma_lab returned {out:.2f} on a typical training input — too large; "\
        f"will dominate other losses. Normalization regression?"


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
