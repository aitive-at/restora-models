import torch

from coliraz.config import (
    Config,
    DataConfig,
    LoaderConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SchedulerConfig,
    TrainConfig,
)
from coliraz.train.trainer import Trainer


def _make_cfg(image_dir, out_dir):
    return Config(
        run=RunConfig(name="t", output_dir=str(out_dir), seed=0),
        model=ModelConfig(
            size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32
        ),
        data=DataConfig(
            root=str(image_dir),
            val_fraction=0.0,
            num_fixed_preview_samples=0,
            num_random_preview_samples=0,
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        losses=[LossConfig(name="l1_ab", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=10),
        train=TrainConfig(
            total_steps=10,
            amp="fp32",
            memory_format="contiguous",
            compile=False,
            ema_decay=0.0,
            preview_every_s=0,
            ckpt_every_steps=10000,
            log_every_steps=1,
        ),
    )


def test_trainer_reduces_loss_on_overfit(tmp_image_dir, tmp_path):
    """Repeatedly train on the SAME batch; loss should decrease (true overfit)."""
    cfg = _make_cfg(tmp_image_dir, tmp_path)
    trainer = Trainer(
        cfg,
        device=torch.device("cpu"),
        pretrained_encoder=False,
        headless=True,
    )
    # Grab one batch and re-train on it 20 times.
    batch = next(trainer._iter)
    initial = trainer._train_step(batch)
    for _ in range(30):
        last = trainer._train_step(batch)
    assert last["total_g"] < initial["total_g"], (
        f"loss did not decrease: initial={initial['total_g']:.4f} last={last['total_g']:.4f}"
    )
