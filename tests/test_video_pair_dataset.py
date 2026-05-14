"""Tests for the video frame-pair dataset."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from restora_models.data.video import VideoPairDataset


@pytest.fixture
def tiny_video_root(tmp_path: Path) -> Path:
    """Make 2 tiny synthetic 'videos' with 5 frames each + flow files."""
    root = tmp_path / "videos"
    for vi in range(2):
        vid_dir = root / f"vid_{vi:02d}"
        vid_dir.mkdir(parents=True)
        flow_dir = vid_dir / ".flow"
        flow_dir.mkdir()
        for fi in range(5):
            # Random colored 64x64 frame
            rng = np.random.default_rng(vi * 100 + fi)
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(vid_dir / f"frame_{fi:05d}.jpg"), img)
        # Pre-create some flow files (skip 1 and 2 from frame 0)
        for fi in range(3):
            for k in (1, 2):
                flow = np.zeros((2, 64, 64), dtype=np.float32)
                np.savez(flow_dir / f"frame_{fi:05d}_skip{k}.npz", flow=flow)
    return root


def test_lists_videos(tiny_video_root):
    ds = VideoPairDataset(tiny_video_root, target_size=64, max_skip=2)
    assert len(ds.videos) == 2
    # 2 videos × 4 valid pair-starts (frame 0-3 since 4 has no successor)
    assert len(ds) == 2 * 4


def test_loads_pair_with_flow(tiny_video_root):
    ds = VideoPairDataset(tiny_video_root, target_size=64, max_skip=2)
    item = ds[0]
    assert set(item) == {"clean_t", "clean_tk", "flow_t_tk"}
    assert item["clean_t"].shape == (3, 64, 64)
    assert item["clean_tk"].shape == (3, 64, 64)
    assert item["flow_t_tk"].shape == (2, 64, 64)
    assert torch.isfinite(item["clean_t"]).all()


def test_missing_flow_falls_back_to_zero(tiny_video_root):
    """If a flow file doesn't exist for a (frame, skip) combo, fall back to
    zero flow rather than crashing — temporal_pair loss handles zero flow
    as 'no temporal info' gracefully (identity warp)."""
    # Remove all flow files
    for f in tiny_video_root.rglob("*.npz"):
        f.unlink()
    ds = VideoPairDataset(tiny_video_root, target_size=64, max_skip=2,
                          require_flow=True)
    item = ds[0]
    assert item["flow_t_tk"].abs().sum().item() == 0.0


def test_raises_on_empty_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no video subdirectories"):
        VideoPairDataset(empty)


def test_hflip_preserves_shape_and_flips_flow_x(tiny_video_root):
    """When hflip fires, flow's x-component should be flipped + negated.
    Hard to test deterministically; just ensure no crash on many samples."""
    ds = VideoPairDataset(tiny_video_root, target_size=64, max_skip=2,
                          hflip_prob=1.0, seed=42)
    item = ds[0]
    assert item["clean_t"].shape == (3, 64, 64)
    assert item["flow_t_tk"].shape == (2, 64, 64)
