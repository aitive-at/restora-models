import torch

from coliraz.models.unet_blocks import PixelShuffleICNR, UnetBlockWide


def test_pixel_shuffle_icnr_doubles_resolution():
    blk = PixelShuffleICNR(in_ch=32, out_ch=16, scale=2)
    x = torch.randn(2, 32, 8, 8)
    y = blk(x)
    assert y.shape == (2, 16, 16, 16)


def test_pixel_shuffle_icnr_scale4():
    blk = PixelShuffleICNR(in_ch=64, out_ch=32, scale=4)
    x = torch.randn(1, 64, 4, 4)
    y = blk(x)
    assert y.shape == (1, 32, 16, 16)


def test_unet_block_wide_shapes():
    blk = UnetBlockWide(in_c=128, skip_c=64, out_c=96)
    deep = torch.randn(2, 128, 8, 8)
    skip = torch.randn(2, 64, 16, 16)
    y = blk(deep, skip)
    assert y.shape == (2, 96, 16, 16)
