import torch

from coliraz.models.pixel_decoder import PixelDecoder


def test_pixel_decoder_outputs():
    feature_channels = [96, 192, 384, 768]
    dec = PixelDecoder(feature_channels=feature_channels, nf=512)
    feats = [
        torch.randn(2, 96, 16, 16),
        torch.randn(2, 192, 8, 8),
        torch.randn(2, 384, 4, 4),
        torch.randn(2, 768, 2, 2),
    ]
    mem, hi = dec(feats)
    assert len(mem) == 3
    assert hi.shape[2] == 64 and hi.shape[3] == 64
    assert hi.shape[1] == 256
