from pathlib import Path

import cv2
import numpy as np
import torch

from coliraz.data.dataset import (
    MANIFEST_NAME,
    RecursiveImageDataset,
    build_manifest,
)


def test_build_manifest_finds_all_images(tmp_image_dir: Path):
    paths = build_manifest(tmp_image_dir)
    assert len(paths) == 6
    rels = sorted(str(p.relative_to(tmp_image_dir)) for p in paths)
    assert "img0.png" in rels
    assert "a/b/img4.jpeg" in rels


def test_manifest_is_cached(tmp_image_dir: Path):
    build_manifest(tmp_image_dir)
    assert (tmp_image_dir / MANIFEST_NAME).exists()


def test_dataset_returns_correct_shapes(tmp_image_dir: Path):
    ds = RecursiveImageDataset(tmp_image_dir, target_size=32, augment_hflip=False)
    sample = ds[0]
    assert sample["gray_rgb"].shape == (3, 32, 32)
    assert sample["gt_ab"].shape == (2, 32, 32)
    assert sample["L_full"].shape[0] == 1
    assert isinstance(sample["gray_rgb"], torch.Tensor)


def test_dataset_skips_too_small_images(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "tiny.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    ds = RecursiveImageDataset(tmp_path, target_size=32, min_side=32, augment_hflip=False)
    assert len(ds) == 1


def test_holdout_split_is_deterministic(tmp_image_dir: Path):
    a = RecursiveImageDataset(
        tmp_image_dir, target_size=32, val_fraction=0.34,
        split="val", augment_hflip=False,
    )
    b = RecursiveImageDataset(
        tmp_image_dir, target_size=32, val_fraction=0.34,
        split="val", augment_hflip=False,
    )
    assert [str(p) for p in a._paths] == [str(p) for p in b._paths]
