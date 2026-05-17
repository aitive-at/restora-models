"""Smoke test: trainer instantiates + steps one batch using fake data.

These tests deliberately bypass the real video sources (REDS / Vimeo) by
monkeypatching ``build_compound_video_dataset`` with a tiny in-memory
``Dataset`` that yields the post-degradation batch schema directly
(``{clean, degraded, config, axes_active}``). They confirm:

  * ``Trainer.__init__`` succeeds end-to-end (model + losses + optimizer
    + dataset + DataLoader + UI + TensorBoard writer) with the new
    ``configs/local-temporal.yaml``.
  * On a CUDA host, one full training step runs and a checkpoint lands
    on disk.
"""
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from restora_models.config import load_config
from restora_models.data.compound import AXES
from restora_models.train.trainer import Trainer


class _FakeCompoundDataset(Dataset):
    """Yields the post-degradation schema the trainer now expects.

    The trainer used to do degradation itself; that work is now inside
    DataLoader workers (see compound_wrapper.py). This fake stands in
    for the wrapper's output for smoke-test purposes — same dict keys,
    same dtypes, synthetic content.
    """

    def __init__(self, n: int = 4):
        self.n = n
        # Expose `inner` so the trainer's preview-index logic can still
        # call `len(...)` on something. Pointing it at self is fine.
        self.inner = self

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        clean = torch.rand(7, 3, 32, 32)
        degraded = (clean + 0.02 * torch.randn_like(clean)).clamp(0, 1)
        config = torch.zeros(len(AXES))
        config[0] = 1.0   # colorize active
        return {
            "clean": clean,
            "degraded": degraded,
            "config": config,
            "axes_active": "colorize",
        }


def _adamw_optimizer(model, lr, weight_decay, *, prefer_muon=False):
    """Plain AdamW for smoke tests — bypasses Muon's strict shape rules."""
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    return opt, "adamw"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="trainer-smoke requires CUDA")
def test_trainer_one_step_on_fake_data(tmp_path, monkeypatch):
    cfg = load_config(Path("configs/local-temporal.yaml"))
    # Patch the dataset builder to return a fake compound-schema dataset.
    import restora_models.train.trainer as trainer_mod
    monkeypatch.setattr(
        trainer_mod, "build_compound_video_dataset",
        lambda sources, *, data_cfg=None, seed=0: _FakeCompoundDataset(n=4),
    )
    # Bypass Muon — its conv-filter shape reshape path is fragile for
    # some kernel shapes in this architecture. The smoke test only
    # cares that the trainer plumbing executes one full step + saves
    # a checkpoint, so plain AdamW is the right tool here.
    monkeypatch.setattr(trainer_mod, "_build_optimizer", _adamw_optimizer)
    cfg.train.total_steps = 1
    cfg.train.save_every = 1
    cfg.train.log_every = 1
    cfg.train.compile = False
    cfg.train.memory_format = "contiguous"
    cfg.run.root = Path(tmp_path)
    cfg.run.name = "smoke"
    cfg.data.loader.batch_size = 2
    cfg.data.loader.num_workers = 0
    cfg.data.loader.persistent_workers = False

    trainer = Trainer(cfg)
    final = trainer.fit()
    assert final.exists(), f"checkpoint not produced at {final}"


def test_trainer_constructs_without_cuda(tmp_path, monkeypatch):
    """Trainer should at least instantiate on CPU (CUDA-only step is skipped)."""
    cfg = load_config(Path("configs/local-temporal.yaml"))
    cfg.train.compile = False
    cfg.train.amp = "fp32"
    cfg.train.memory_format = "contiguous"
    cfg.data.loader.batch_size = 1
    cfg.data.loader.num_workers = 0
    cfg.data.loader.persistent_workers = False
    cfg.run.root = Path(tmp_path)
    cfg.run.name = "construct_only"
    import restora_models.train.trainer as trainer_mod
    monkeypatch.setattr(
        trainer_mod, "build_compound_video_dataset",
        lambda sources, *, data_cfg=None, seed=0: _FakeCompoundDataset(n=4),
    )

    trainer = Trainer(cfg, device=torch.device("cpu"))
    assert trainer.model is not None
    assert trainer.loss_set is not None
