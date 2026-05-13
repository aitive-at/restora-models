import torch

from refine.config import (
    Config, DataConfig, DegradationConfig, LoaderConfig, LossConfig, ModelConfig,
    OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from refine.train.trainer import Trainer


def _make_cfg(image_dir, out_dir):
    return Config(
        run=RunConfig(name="t", output_dir=str(out_dir), seed=0),
        model=ModelConfig(type="nafnet", size="tiny", nf=8,
                          enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                          task_embed_dim=16, input_size=64),
        data=DataConfig(
            root=str(image_dir),
            val_fraction=0.0,
            num_fixed_preview_samples=0,
            num_random_preview_samples=0,
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        degradations={
            "colorize": DegradationConfig(weight=1.0),
            "denoise":  DegradationConfig(weight=1.0, sigma_range=[0.02, 0.05]),
        },
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=10),
        train=TrainConfig(total_steps=10, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0,
                          ckpt_every_steps=10000, log_every_steps=1),
    )


def test_trainer_overfit_reduces_loss(tmp_image_dir, tmp_path):
    cfg = _make_cfg(tmp_image_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._iter)
    initial = trainer._train_step(batch)
    for _ in range(30):
        last = trainer._train_step(batch)
    assert last["total_g"] < initial["total_g"]
