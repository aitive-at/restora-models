import torch

from restora_models.models.restormer_block import RestormerBlock


def test_forward_shape():
    blk = RestormerBlock(c=16, num_heads=2, task_dim=8)
    x = torch.randn(2, 16, 32, 32)
    task = torch.randn(2, 8)
    out = blk(x, task)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward():
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    task = torch.randn(1, 4)
    out = blk(x, task)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for n, p in blk.named_parameters():
        assert p.grad is not None, f"{n} got no gradient"


def test_adaln_changes_output():
    """AdaLN init is zero so output equals x at init. Bump AdaLN weights
    manually, then verify different task vectors produce different outputs."""
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    with torch.no_grad():
        for adaln in (blk.adaln1, blk.adaln2):
            adaln.weight.normal_(0, 0.1)
            adaln.bias.normal_(0, 0.1)
    x = torch.randn(1, 8, 16, 16)
    t1 = torch.zeros(1, 4)
    t2 = torch.ones(1, 4) * 5.0
    out1 = blk(x, t1); out2 = blk(x, t2)
    assert (out1 - out2).abs().mean().item() > 1e-3


def test_handles_non_square():
    blk = RestormerBlock(c=8, num_heads=2, task_dim=4)
    x = torch.randn(1, 8, 24, 40)
    out = blk(x, torch.zeros(1, 4))
    assert out.shape == x.shape
