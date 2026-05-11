import os

import cv2
import numpy as np
import pytest
import torch

from coliraz.config import (
    Config,
    DataConfig,
    ExportConfig,
    LoaderConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SchedulerConfig,
    TrainConfig,
)
from coliraz.infer.pipeline import load_pipeline
from coliraz.train import Trainer


@pytest.mark.skipif(
    os.environ.get("COLIRAZ_SLOW") != "1",
    reason="end-to-end smoke is slow; set COLIRAZ_SLOW=1 to run",
)
def test_train_then_infer_e2e(tmp_path):
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        img = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / f"img{i}.png"), img)

    out_dir = tmp_path / "run"
    cfg = Config(
        run=RunConfig(name="smoke", output_dir=str(out_dir), seed=0),
        model=ModelConfig(
            size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32
        ),
        data=DataConfig(
            root=str(data_dir),
            val_fraction=0.25,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            loader=LoaderConfig(batch_size=2, num_workers=0, persistent_workers=False),
        ),
        losses=[LossConfig(name="l1_ab", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=5),
        train=TrainConfig(
            total_steps=5,
            amp="fp32",
            memory_format="contiguous",
            compile=False,
            ema_decay=0.0,
            preview_every_s=0.001,
            preview_history_every=0,
            ckpt_every_steps=5,
            log_every_steps=1,
        ),
        export=ExportConfig(on_finish=False),
    )
    trainer = Trainer(
        cfg, device=torch.device("cpu"), pretrained_encoder=False, headless=True
    )
    trainer.fit()

    final_ckpt = out_dir / "ckpt" / "final.pt"
    assert final_ckpt.exists()
    assert (out_dir / "samples" / "latest.png").exists()

    pipe = load_pipeline(final_ckpt, input_size=64, device=torch.device("cpu"))
    img = cv2.imread(str(data_dir / "img0.png"))
    out = pipe.process(img)
    assert out.shape == img.shape and out.dtype == np.uint8
