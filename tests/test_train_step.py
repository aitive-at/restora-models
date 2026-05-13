import torch

from refine.config import (
    AxisProbs, CompoundConfig, CompoundDegradations,
    Config, DataConfig, LoaderConfig, LossConfig, ModelConfig,
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
        compound=CompoundConfig(
            identity_prob=0.0,
            # Force denoise always on so every sample has large degradation
            axis_probs=AxisProbs(colorize=0.0, denoise=1.0, sharpen=0.0,
                                  dejpeg=0.0, deblur=0.0),
            degradations=CompoundDegradations(
                # Use large sigma so degraded is far from clean
                denoise={"sigma_range": [0.2, 0.3]},
            ),
        ),
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(lr=3e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=200),
        train=TrainConfig(total_steps=200, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0,
                          ckpt_every_steps=10000, log_every_steps=1),
    )


def test_trainer_overfit_reduces_loss(tmp_image_dir, tmp_path):
    cfg = _make_cfg(tmp_image_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._iter)
    # Warm up to escape near-identity init plateau
    for _ in range(10):
        trainer._train_step(batch)
    # Record loss before and after overfitting
    initial = trainer._train_step(batch)["total_g"]
    for _ in range(200):
        last = trainer._train_step(batch)
    assert last["total_g"] < initial
