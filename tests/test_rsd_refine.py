"""Tests for RSDRefineHead -- one-step residual-shift diffusion in RGB."""
import torch

from restora_models.models.rsd_refine import RSDRefineHead


def test_rsd_refine_output_shape():
    head = RSDRefineHead(width=64, num_axes=5).eval()
    coarse = torch.rand(2, 3, 64, 64)
    config = torch.tensor([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
    out = head(coarse, config)
    assert out.shape == coarse.shape


def test_rsd_refine_near_identity_at_init():
    head = RSDRefineHead(width=64, num_axes=5).eval()
    coarse = torch.rand(1, 3, 64, 64)
    config = torch.zeros(1, 5)
    out = head(coarse, config)
    assert torch.allclose(out, coarse, atol=0.05), f"max diff {(out - coarse).abs().max().item()}"
