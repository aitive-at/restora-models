import torch

from restora_models.models.color import LabToRgb, RgbToLab


def test_rgb_to_lab_module_shape():
    m = RgbToLab()
    rgb = torch.rand(2, 3, 16, 16)
    lab_n = m(rgb)
    assert lab_n.shape == rgb.shape


def test_round_trip_through_modules():
    rgb = torch.rand(2, 3, 16, 16)
    lab_n = RgbToLab()(rgb)
    rgb_back = LabToRgb()(lab_n)
    assert rgb_back.shape == rgb.shape
    assert (rgb_back - rgb).abs().mean() < 0.05


def test_fp32_dispatch_under_bf16_autocast():
    """Even when autocast(bf16) is active, conversion runs in fp32 and
    returns fp32 output (no NaN/Inf)."""
    m = RgbToLab()
    rgb = torch.rand(1, 3, 8, 8)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = m(rgb)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
