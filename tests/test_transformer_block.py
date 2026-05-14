import torch

from restora_models.models.transformer_block import TransformerBlock


def test_transformer_block_shape():
    blk = TransformerBlock(c=64, task_dim=32, num_heads=4, ffn_dim=128)
    x = torch.randn(2, 64, 8, 8)
    t = torch.randn(2, 32)
    assert blk(x, t).shape == x.shape


def test_adaln_conditions_output():
    blk = TransformerBlock(c=64, task_dim=32, num_heads=4, ffn_dim=128)
    x = torch.randn(1, 64, 8, 8)
    t1 = torch.zeros(1, 32)
    t2 = torch.ones(1, 32) * 2.0
    assert (blk(x, t1) - blk(x, t2)).abs().sum() > 0
