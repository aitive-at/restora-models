import torch

from refine.models.task_embed import TaskEmbed


def test_task_embed_shape():
    m = TaskEmbed(num_tasks=6, dim=128)
    task = torch.tensor([0, 1, 5, 2], dtype=torch.long)
    out = m(task)
    assert out.shape == (4, 128)


def test_task_embed_distinguishes_tasks():
    m = TaskEmbed(num_tasks=6, dim=64)
    a = m(torch.tensor([0])).detach()
    b = m(torch.tensor([1])).detach()
    assert (a - b).abs().sum() > 0
