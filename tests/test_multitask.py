import random

import numpy as np
import torch

from refine.data.degradations import colorization, denoise, superres  # noqa: F401
from refine.data.degradations.registry import build_degradation
from refine.data.multitask import MultiTaskWrapper, collate_multitask


class _DummyCleanDS:
    def __len__(self): return 8
    def __getitem__(self, idx):
        return torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(idx))


def test_wrapper_picks_tasks_via_weights():
    ds = _DummyCleanDS()
    degs = [build_degradation("colorize"), build_degradation("denoise")]
    degs[0].task_id = 0; degs[1].task_id = 1
    wrap = MultiTaskWrapper(ds, degs, weights=[0.5, 0.5], seed=0)
    counts = {0: 0, 1: 0}
    for i in range(200):
        s = wrap[i]
        counts[int(s["task_id"])] += 1
    assert counts[0] >= 60 and counts[1] >= 60


def test_wrapper_sample_shapes():
    ds = _DummyCleanDS()
    deg = build_degradation("sr_x2"); deg.task_id = 0
    wrap = MultiTaskWrapper(ds, [deg], weights=[1.0], seed=0)
    s = wrap[0]
    assert s["clean"].shape == (3, 32, 32)
    assert s["degraded"].shape == (3, 32, 32)
    assert s["task_id"].item() == 0
    assert s["task_name"] == "sr_x2"


def test_collate_stacks():
    ds = _DummyCleanDS()
    deg = build_degradation("denoise"); deg.task_id = 0
    wrap = MultiTaskWrapper(ds, [deg], weights=[1.0], seed=0)
    batch = collate_multitask([wrap[i] for i in range(4)])
    assert batch["clean"].shape == (4, 3, 32, 32)
    assert batch["degraded"].shape == (4, 3, 32, 32)
    assert batch["task_id"].shape == (4,)
    assert batch["task_name"] == ["denoise"] * 4
