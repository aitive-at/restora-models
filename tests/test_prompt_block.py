import torch

from restora_models.models.prompt_block import PromptBlock


def test_forward_shape():
    blk = PromptBlock(feat_c=16, prompt_n=5, prompt_dim=16, prompt_hw=8, cond_dim=8)
    x = torch.randn(2, 16, 32, 32)
    cond = torch.randn(2, 8)
    out = blk(x, cond)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_config_determinism_same_config_same_output():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    blk.train(False)
    x = torch.randn(1, 8, 16, 16)
    c = torch.tensor([[1.0, 0, 0, 0]])
    o1 = blk(x, c); o2 = blk(x, c)
    assert torch.allclose(o1, o2)


def test_different_configs_different_outputs():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    with torch.no_grad():
        for i in range(blk.prompts.shape[0]):
            blk.prompts[i].fill_(float(i + 1) * 0.5)
    x = torch.randn(1, 8, 16, 16)
    c1 = torch.tensor([[5.0, -5, -5, -5]])
    c2 = torch.tensor([[-5, 5.0, -5, -5]])
    o1 = blk(x, c1); o2 = blk(x, c2)
    assert (o1 - o2).abs().mean().item() > 1e-4


def test_backward_grads_all_params():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    cond = torch.randn(1, 4)
    out = blk(x, cond)
    out.pow(2).mean().backward()
    for n, p in blk.named_parameters():
        assert p.grad is not None, f"{n} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{n} has non-finite grad"


def test_handles_non_square_feat():
    blk = PromptBlock(feat_c=8, prompt_n=5, prompt_dim=8, prompt_hw=4, cond_dim=4)
    x = torch.randn(1, 8, 24, 40)
    out = blk(x, torch.zeros(1, 4))
    assert out.shape == x.shape
