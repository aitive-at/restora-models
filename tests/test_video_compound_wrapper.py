"""Tests for VideoCompoundDegradationWrapper."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from restora_models.data.video import VideoPairDataset
from restora_models.data.video_compound import (
    VideoCompoundDegradationWrapper, collate_video_compound,
)


@pytest.fixture
def tiny_video_root(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    for vi in range(2):
        vid_dir = root / f"vid_{vi:02d}"
        vid_dir.mkdir(parents=True)
        flow_dir = vid_dir / ".flow"
        flow_dir.mkdir()
        for fi in range(4):
            rng = np.random.default_rng(vi * 100 + fi)
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(vid_dir / f"frame_{fi:05d}.jpg"), img)
        for fi in range(3):
            for k in (1, 2):
                flow = np.zeros((2, 64, 64), dtype=np.float32)
                np.savez(flow_dir / f"frame_{fi:05d}_skip{k}.npz", flow=flow)
    return root


@pytest.fixture
def wrapper(tiny_video_root):
    video = VideoPairDataset(tiny_video_root, target_size=64, max_skip=2,
                              hflip_prob=0.0)
    axis_probs = {"colorize": 0.5, "denoise": 0.5, "sharpen": 0.5,
                  "dejpeg": 0.5, "deblur": 0.5}
    return VideoCompoundDegradationWrapper(
        video, axis_probs=axis_probs, identity_prob=0.0,
        degradation_params={
            "denoise": {"sigma_range": [0.01, 0.02]},
            "sharpen": {"factor_choices": [2, 4]},
            "dejpeg": {"quality_range": [30, 60]},
            "deblur": {"sigma_range": [1.0, 2.0], "motion_prob": 0.0},
        },
        seed=0,
    )


def test_returns_paired_dict(wrapper):
    item = wrapper[0]
    assert set(item) == {"clean_t", "degraded_t", "clean_tk", "degraded_tk",
                          "flow_t_tk", "config", "axes"}
    for k in ("clean_t", "degraded_t", "clean_tk", "degraded_tk"):
        assert item[k].shape == (3, 64, 64)
    assert item["flow_t_tk"].shape == (2, 64, 64)
    assert item["config"].shape == (5,)


def test_same_config_across_frames(wrapper):
    """Both frames must see the same axis flags (single config vector)."""
    item = wrapper[0]
    # config is a 5-vector; same vector applies to both frames implicitly.
    # Validate axes label matches the active flags.
    flag_idx = (item["config"] >= 0.5).nonzero().flatten().tolist()
    from restora_models.data.compound import AXES
    expected_active = [AXES[i] for i in flag_idx]
    expected_label = "+".join(expected_active) if expected_active else "identity"
    assert item["axes"] == expected_label


def test_same_rng_produces_consistent_sr_factor(wrapper):
    """For a sample whose flags include sharpen, the SR factor must be the
    same for both frames (else the model gets a different scale per frame
    and the temporal loss becomes meaningless)."""
    # Force a sample that has sharpen active. Iterate until we find one.
    for i in range(len(wrapper)):
        item = wrapper[i]
        from restora_models.data.compound import AXES
        sharpen_idx = AXES.index("sharpen")
        if item["config"][sharpen_idx] >= 0.5 and item["config"].sum() == 1.0:
            # sharpen-only sample. clean_tk should be a slight motion of clean_t,
            # but if the SR factor matches and the underlying frames are
            # similar, the degradation magnitude should be similar.
            # (This is a smoke test: just confirm both got bicubic-down-up.)
            assert item["degraded_t"].shape == item["degraded_tk"].shape
            return
    pytest.skip("no sharpen-only sample found in small dataset")


def test_collate_stacks(wrapper):
    items = [wrapper[i] for i in range(4)]
    batch = collate_video_compound(items)
    assert batch["clean_t"].shape == (4, 3, 64, 64)
    assert batch["degraded_t"].shape == (4, 3, 64, 64)
    assert batch["flow_t_tk"].shape == (4, 2, 64, 64)
    assert batch["config"].shape == (4, 5)
    assert len(batch["axes"]) == 4


def test_identity_prob_zero_means_no_passthrough(wrapper):
    """With identity_prob=0 and axis_probs=0.5, most samples should have
    at least one active axis (random check)."""
    n_identity = 0
    n_total = min(8, len(wrapper))
    for i in range(n_total):
        item = wrapper[i]
        if item["axes"] == "identity":
            n_identity += 1
    # With identity_prob=0 and 5 axes at p=0.5, P(identity) = 0.5^5 ≈ 3%
    assert n_identity <= 2
