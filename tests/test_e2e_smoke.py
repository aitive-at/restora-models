import os

import cv2
import numpy as np
import pytest
import torch

from refine.config import (
    AxisProbs, CompoundConfig, CompoundDegradations,
    Config, DataConfig, ExportConfig, LoaderConfig, LossConfig,
    ModelConfig, OptimConfig, RunConfig, SchedulerConfig, TrainConfig,
)
from refine.infer.pipeline import load_pipeline
from refine.train import Trainer


@pytest.mark.skipif(os.environ.get("REFINE_SLOW") != "1",
                    reason="e2e smoke is slow; set REFINE_SLOW=1 to run")
def test_train_then_infer_e2e(tmp_path):
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(str(data_dir / f"img{i}.png"),
                    rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8))

    out_dir = tmp_path / "run"
    cfg = Config(
        run=RunConfig(name="smoke", output_dir=str(out_dir), seed=0),
        model=ModelConfig(type="nafnet", size="tiny", nf=8,
                          enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                          task_embed_dim=16, input_size=64),
        data=DataConfig(root=str(data_dir), val_fraction=0.25,
                        num_fixed_preview_samples=1, num_random_preview_samples=0,
                        loader=LoaderConfig(batch_size=2, num_workers=0,
                                            persistent_workers=False)),
        compound=CompoundConfig(
            identity_prob=0.05,
            axis_probs=AxisProbs(colorize=0.5, denoise=0.5, sharpen=0.0,
                                  dejpeg=0.0, deblur=0.0),
            degradations=CompoundDegradations(
                denoise={"sigma_range": [0.02, 0.05]},
            ),
        ),
        losses=[LossConfig(name="l1_rgb", weight=1.0)],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=5),
        train=TrainConfig(total_steps=5, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0.001,
                          preview_history_every=0, ckpt_every_steps=5,
                          log_every_steps=1),
        export=ExportConfig(on_finish=False),
    )
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    trainer.fit()

    final_ckpt = out_dir / "ckpt" / "final.pt"
    assert final_ckpt.exists()
    assert (out_dir / "samples" / "latest.png").exists()
    sidecar = final_ckpt.with_suffix(".task_map.json")
    assert sidecar.exists()

    pipe = load_pipeline(final_ckpt, device=torch.device("cpu"))
    img = cv2.imread(str(data_dir / "img0.png"))
    out = pipe.process(img, config={"colorize": True, "denoise": False,
                                    "sharpen": False, "dejpeg": False, "deblur": False})
    assert out.shape == img.shape and out.dtype == np.uint8
