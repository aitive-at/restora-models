import torch

from refine.models.task_embed import ConfigEmbed


def test_config_embed_shape():
    m = ConfigEmbed(num_axes=5, dim=128)
    config = torch.rand(4, 5)
    out = m(config)
    assert out.shape == (4, 128)


def test_config_embed_distinguishes_configs():
    m = ConfigEmbed(num_axes=5, dim=64)
    a = m(torch.zeros(1, 5)).detach()
    b = m(torch.ones(1, 5)).detach()
    assert (a - b).abs().sum() > 0
