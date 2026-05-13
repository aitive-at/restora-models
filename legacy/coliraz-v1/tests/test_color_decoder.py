import torch

from coliraz.models.color_decoder import MultiScaleColorDecoder


def test_color_decoder_einsum_output():
    in_chs = [512, 512, 256]
    dec = MultiScaleColorDecoder(
        in_channels=in_chs, num_queries=8, num_scales=3, dec_layers=2,
        hidden_dim=64, color_embed_dim=64,
    )
    memories = [
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 512, 8, 8),
        torch.randn(2, 256, 4, 4),
    ]
    hi = torch.randn(2, 64, 64, 64)
    out = dec(memories, hi)
    assert out.shape == (2, 8, 64, 64)
