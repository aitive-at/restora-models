"""Smoke test: trainer.run_one_step() with refine_type='diffusion'."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from restora_models.config import (
    AugmentConfig, CompoundConfig, Config, DataConfig, ExportConfig, LoaderConfig,
    LossConfig, ModelConfig, OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
    VideoConfig,
)


@pytest.mark.skipif(
    not __import__('os').environ.get('REFINE_SLOW'),
    reason="downloads SD VAE; set REFINE_SLOW=1 to run",
)
def test_diffusion_trainer_runs_one_step(tmp_path):
    data_root = tmp_path / "imgs"
    data_root.mkdir()
    for i in range(8):
        img = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(data_root / f"img_{i}.jpg"), img)

    cfg = Config(
        run=RunConfig(name="diff-smoke", output_dir=str(tmp_path / "run")),
        model=ModelConfig(type="nafnet", size="tiny", refine_type="diffusion",
                          input_size=64, nf=8, enc_depths=[1,1,1,1],
                          bottle_blocks=1, hidden_dim=32),
        data=DataConfig(root=str(data_root), val_fraction=0.0,
                        loader=LoaderConfig(batch_size=2, num_workers=0)),
        compound=CompoundConfig(),
        losses=[LossConfig(name="l1_rgb", weight=1.0),
                LossConfig(name="l1_latent", weight=1.0)],
        optim_g=OptimConfig(lr=1e-4, fused=False),
        optim_d=OptimConfig(lr=1e-4, fused=False, weight_decay=0.0),
        scheduler=SchedulerConfig(total_steps=10, warmup_steps=2),
        train=TrainConfig(total_steps=10, amp="fp32", compile=False,
                          ema_decay=0.0, preview_every_s=0.0,
                          ckpt_every_steps=0, log_every_steps=10),
        export=ExportConfig(on_finish=False),
        video=VideoConfig(enabled=False),
    )
    from restora_models.train import Trainer
    t = Trainer(cfg)
    log = t.run_one_step()
    assert "total_g" in log
    assert torch.isfinite(torch.tensor(log["total_g"]))
