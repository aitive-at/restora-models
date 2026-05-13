import torch

from refine.models.nafblock import NAFBlock


def test_nafblock_shape():
    blk = NAFBlock(c=16, task_dim=32)
    x = torch.randn(2, 16, 8, 8)
    t = torch.randn(2, 32)
    assert blk(x, t).shape == x.shape


def test_film_conditions_output():
    blk = NAFBlock(c=16, task_dim=32)
    x = torch.randn(1, 16, 8, 8)
    t1 = torch.randn(1, 32)
    t2 = torch.randn(1, 32) * 3.0
    assert (blk(x, t1) - blk(x, t2)).abs().sum() > 0


def test_residual_path():
    blk = NAFBlock(c=8, task_dim=16)
    x = torch.randn(1, 8, 6, 6, requires_grad=True)
    t = torch.randn(1, 16)
    blk(x, t).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
