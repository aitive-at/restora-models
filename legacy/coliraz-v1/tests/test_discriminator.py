import torch

from coliraz.models.discriminator import UNetDiscriminator


def test_discriminator_forward_shape():
    d = UNetDiscriminator(in_ch=3, nf=16)
    x = torch.randn(2, 3, 64, 64)
    y = d(x)
    assert y.shape == (2, 1, 64, 64)
