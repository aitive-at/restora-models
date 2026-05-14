"""Preview must contain per-factor SR rows so the user sees sr2x/sr4x/sr8x."""
from __future__ import annotations

import torch

from restora_models.config import (
    Config, RunConfig, ModelConfig, DataConfig, LoaderConfig,
    AugmentConfig, CompoundConfig, LossConfig, OptimConfig,
    SchedulerConfig, TrainConfig, ExportConfig,
)


def _minimal_cfg(root: str) -> Config:
    return Config(
        run=RunConfig(name="test", output_dir=root, seed=0),
        model=ModelConfig(type="nafnet", size="tiny", input_size=64),
        data=DataConfig(
            root=root,
            val_fraction=0.5,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            augment=AugmentConfig(),
            loader=LoaderConfig(batch_size=2, num_workers=0,
                                pin_memory=False, persistent_workers=False),
        ),
        compound=CompoundConfig(),
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(), optim_d=OptimConfig(),
        scheduler=SchedulerConfig(total_steps=1),
        train=TrainConfig(total_steps=1),
        export=ExportConfig(on_finish=False),
    )


def test_preview_includes_per_factor_sr_rows(tmp_image_dir, monkeypatch):
    from restora_models.train.trainer import Trainer

    cfg = _minimal_cfg(str(tmp_image_dir))
    trainer = Trainer(cfg)
    samples = trainer._build_preview_samples()
    keys = list(samples.keys())
    assert "sharpen-2x" in keys
    assert "sharpen-4x" in keys
    assert "sharpen-8x" in keys
    assert "sharpen-only" not in keys


def test_preview_per_factor_uses_different_factors(tmp_image_dir):
    """The degraded image in sharpen-2x and sharpen-8x must differ — different
    downsample factors yield different blur amounts."""
    torch.manual_seed(20260514)   # isolation: other tests may have set seed
    from restora_models.train.trainer import Trainer

    cfg = _minimal_cfg(str(tmp_image_dir))
    trainer = Trainer(cfg)
    samples = trainer._build_preview_samples()
    deg2 = samples["sharpen-2x"][0]["degraded"]
    deg8 = samples["sharpen-8x"][0]["degraded"]
    diff = (deg2 - deg8).abs().mean().item()
    assert diff > 1e-3, f"factors produced identical degraded images: diff={diff}"
