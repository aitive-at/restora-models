"""Unit tests for the LatentDiffusionRefineHead building blocks."""
import torch


def test_sinusoidal_timestep_embedding_shape_and_range():
    from restora_models.models.diffusion_head import sinusoidal_timestep_embedding
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_timestep_embedding(t, dim=64)
    assert emb.shape == (3, 64)
    assert (-1.001 <= emb).all() and (emb <= 1.001).all()


def test_sinusoidal_timestep_embedding_distinct_per_t():
    from restora_models.models.diffusion_head import sinusoidal_timestep_embedding
    t = torch.tensor([0.1, 0.2, 0.3])
    emb = sinusoidal_timestep_embedding(t, dim=128)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])


def test_adaln_resblock_preserves_shape():
    from restora_models.models.diffusion_head import AdaLNResBlock
    block = AdaLNResBlock(c=96, cond_dim=384)
    x = torch.randn(2, 96, 32, 32)
    cond = torch.randn(2, 384)
    out = block(x, cond)
    assert out.shape == x.shape


def test_adaln_resblock_has_residual_path():
    """At zero-init the residual path should be ~identity."""
    from restora_models.models.diffusion_head import AdaLNResBlock
    block = AdaLNResBlock(c=64, cond_dim=128)
    torch.nn.init.zeros_(block.conv2.weight)
    torch.nn.init.zeros_(block.conv2.bias)
    x = torch.randn(1, 64, 16, 16)
    cond = torch.randn(1, 128)
    out = block(x, cond)
    assert torch.allclose(out, x)
