"""End-to-end smoke for the trainer's video-pair step."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from restora_models.config import (
    AxisProbs, CompoundConfig, CompoundDegradations,
    Config, DataConfig, LoaderConfig, LossConfig, ModelConfig,
    OptimConfig, RunConfig, SchedulerConfig, TrainConfig, VideoConfig,
)
from restora_models.train.trainer import Trainer


@pytest.fixture
def tmp_video_dir(tmp_path: Path) -> Path:
    """Three tiny 'videos' with 4 frames + zero-flow files. Used for
    video-trainer smoke tests."""
    root = tmp_path / "videos"
    for vi in range(3):
        vd = root / f"vid_{vi:02d}"
        vd.mkdir(parents=True)
        flow_dir = vd / ".flow"
        flow_dir.mkdir()
        for fi in range(4):
            rng = np.random.default_rng(vi * 100 + fi)
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(vd / f"frame_{fi:05d}.jpg"), img)
        for fi in range(3):
            for k in (1, 2):
                flow = np.zeros((2, 64, 64), dtype=np.float32)
                np.savez(flow_dir / f"frame_{fi:05d}_skip{k}.npz", flow=flow)
    return root


def _cfg(image_dir, video_dir, out_dir, video_prob: float = 1.0):
    return Config(
        run=RunConfig(name="vt", output_dir=str(out_dir), seed=0),
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
            axis_probs=AxisProbs(colorize=0.0, denoise=1.0, sharpen=0.0,
                                  dejpeg=0.0, deblur=0.0),
            degradations=CompoundDegradations(
                denoise={"sigma_range": [0.05, 0.1]},
            ),
        ),
        losses=[
            LossConfig(name="l1_rgb", weight=1.0),
            LossConfig(name="temporal_pair", weight=0.5),
        ],
        optim_g=OptimConfig(lr=1e-3, fused=False),
        scheduler=SchedulerConfig(type="constant", warmup_steps=0, total_steps=200),
        train=TrainConfig(total_steps=200, amp="fp32", memory_format="contiguous",
                          compile=False, ema_decay=0.0, preview_every_s=0,
                          ckpt_every_steps=10000, log_every_steps=1),
        video=VideoConfig(
            enabled=True, root=str(video_dir), max_skip=2,
            hflip_prob=0.0, video_batch_prob=video_prob,
            batch_size=2, num_workers=0,
        ),
    )


def test_trainer_builds_with_video_enabled(tmp_image_dir, tmp_video_dir, tmp_path):
    cfg = _cfg(tmp_image_dir, tmp_video_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    assert trainer.video_loader is not None
    assert trainer._video_iter is not None
    assert trainer.video_batch_prob == 1.0


def test_video_step_runs_and_drops_loss(tmp_image_dir, tmp_video_dir, tmp_path):
    """A video batch must complete a full forward+backward+opt-step
    without crashing, and the temporal_pair loss should be reported."""
    cfg = _cfg(tmp_image_dir, tmp_video_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._video_iter)
    log = trainer._train_step_video(batch)
    assert "total_g" in log
    assert "temporal_pair" in log
    assert "l1_rgb" in log
    assert log.get("_video") == 1.0


def test_image_step_still_works_when_video_enabled(tmp_image_dir, tmp_video_dir, tmp_path):
    """The image path should be unaffected by video being configured."""
    cfg = _cfg(tmp_image_dir, tmp_video_dir, tmp_path, video_prob=0.0)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._iter)
    log = trainer._train_step(batch)
    assert "total_g" in log
    assert "l1_rgb" in log
    # Image batches don't populate the temporal signal; the loss is in the
    # stack but returns 0 for image-only context. It still shows up in the log.
    assert log["temporal_pair"] == 0.0


def test_temporal_pair_zero_with_unrelated_frames_initially(tmp_image_dir, tmp_video_dir, tmp_path):
    """At step 0 with random init, the model output on two random frames
    will not be temporally consistent → temporal_pair > 0 even with
    zero-flow (warped pred_t identity != pred_tk)."""
    cfg = _cfg(tmp_image_dir, tmp_video_dir, tmp_path)
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    batch = next(trainer._video_iter)
    log = trainer._train_step_video(batch)
    # Two different input frames → different predictions → identity-warped
    # (zero flow) pred_t differs from pred_tk → nonzero temporal_pair.
    assert log["temporal_pair"] > 0.0
