"""Tests for REDSDataset 7-frame window sampler."""
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.data.reds import REDSDataset


def _make_fake_reds_flat(tmp_path: Path, n_seqs: int = 2, n_frames: int = 30) -> Path:
    """REDS with flat layout: <root>/<seq>/<frame_id>.png."""
    root = tmp_path / "REDS"
    for s in range(n_seqs):
        seq_dir = root / f"{s:03d}"
        seq_dir.mkdir(parents=True)
        for f in range(n_frames):
            img = (np.random.rand(48, 48, 3) * 255).astype("uint8")
            cv2.imwrite(str(seq_dir / f"{f:08d}.png"), img)
    return root


def _make_fake_reds_split(tmp_path: Path, n_seqs: int = 2, n_frames: int = 30) -> Path:
    """REDS with traditional layout: <root>/train_sharp/<seq>/<frame_id>.png."""
    root = tmp_path / "REDS_split"
    for s in range(n_seqs):
        seq_dir = root / "train_sharp" / f"{s:03d}"
        seq_dir.mkdir(parents=True)
        for f in range(n_frames):
            img = (np.random.rand(48, 48, 3) * 255).astype("uint8")
            cv2.imwrite(str(seq_dir / f"{f:08d}.png"), img)
    return root


def test_reds_flat_layout_window_count(tmp_path):
    """Each 30-frame sequence yields (30 - 7 + 1) = 24 windows; 2 seqs = 48."""
    root = _make_fake_reds_flat(tmp_path, n_seqs=2, n_frames=30)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    assert len(ds) == 48


def test_reds_split_layout_window_count(tmp_path):
    """Traditional REDS layout with train_sharp/ subdir."""
    root = _make_fake_reds_split(tmp_path, n_seqs=2, n_frames=30)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    assert len(ds) == 48


def test_reds_dataset_canonical_sample(tmp_path):
    root = _make_fake_reds_flat(tmp_path, n_seqs=1, n_frames=10)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert sample["source"] == "reds"
    assert "key" in sample
    assert 0.0 <= sample["frames"].min().item()
    assert sample["frames"].max().item() <= 1.0


def test_reds_dataset_stride_2(tmp_path):
    root = _make_fake_reds_flat(tmp_path, n_seqs=1, n_frames=30)
    ds = REDSDataset(root, split="train_sharp", window=7, stride=2, crop=32)
    # (30 - 7) // 2 + 1 = 12 windows
    assert len(ds) == 12


def test_reds_ignores_zone_identifier_files(tmp_path):
    """Windows-downloaded REDS dirs have :Zone.Identifier sidecars. Loader must ignore them."""
    root = _make_fake_reds_flat(tmp_path, n_seqs=1, n_frames=10)
    seq_dir = root / "000"
    # Create some fake sidecars that pathlib glob might pick up
    (seq_dir / "00000000.png:Zone.Identifier").write_text("[ZoneTransfer]\n")
    ds = REDSDataset(root, split="train_sharp", window=7, stride=1, crop=32)
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
