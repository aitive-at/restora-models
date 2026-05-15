"""Tests for the FrozenSD15VAE wrapper."""
import pytest
import torch


def test_vae_encode_returns_4ch_latent_at_8x_downsample():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    rgb = torch.rand(2, 3, 256, 256)
    z = vae.encode(rgb)
    assert z.shape == (2, 4, 32, 32), f"got {tuple(z.shape)}"
    assert -3 < z.mean().item() < 3
    assert 0.1 < z.std().item() < 3


def test_vae_decode_returns_3ch_rgb_at_8x_upsample():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    z = torch.randn(2, 4, 32, 32) * 0.5
    rgb = vae.decode(z)
    assert rgb.shape == (2, 3, 256, 256)
    assert 0.0 <= rgb.min().item() and rgb.max().item() <= 1.001


def test_vae_encode_decode_roundtrip_psnr():
    """A clean roundtrip on natural-looking content should have PSNR > 25 dB."""
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    h = w = 256
    xs = torch.linspace(0, 1, w).unsqueeze(0).unsqueeze(0).expand(3, h, w)
    ys = torch.linspace(0, 1, h).unsqueeze(0).unsqueeze(-1).expand(3, h, w)
    rgb_in = ((xs + ys) / 2.0).unsqueeze(0).clamp(0, 1)
    rgb_out = vae.decode(vae.encode(rgb_in)).clamp(0, 1)
    mse = ((rgb_in - rgb_out) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-9))).item()
    assert psnr > 25.0, f"VAE roundtrip too lossy: {psnr:.2f} dB"


def test_vae_params_are_frozen():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    assert all(not p.requires_grad for p in vae.parameters())


def test_vae_encode_mode_is_deterministic():
    from restora_models.models.vae import FrozenSD15VAE
    vae = FrozenSD15VAE()
    rgb = torch.rand(1, 3, 128, 128)
    z1 = vae.encode_mode(rgb)
    z2 = vae.encode_mode(rgb)
    assert torch.allclose(z1, z2)
