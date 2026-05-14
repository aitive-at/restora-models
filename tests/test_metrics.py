import torch

from restora_models.losses.metrics import psnr, ssim


def test_psnr_identical_is_inf():
    rgb = torch.rand(1, 3, 16, 16)
    out = psnr(rgb, rgb)
    assert out.item() > 60.0


def test_psnr_decreases_with_noise():
    rgb = torch.rand(1, 3, 16, 16)
    noisy = (rgb + torch.randn_like(rgb) * 0.1).clamp(0, 1)
    assert psnr(noisy, rgb).item() < psnr(rgb, rgb).item()


def test_ssim_shape():
    a = torch.rand(2, 3, 32, 32); b = torch.rand(2, 3, 32, 32)
    s = ssim(a, b)
    assert s.shape == (2,)
    assert (s >= -1.0).all() and (s <= 1.0).all()
