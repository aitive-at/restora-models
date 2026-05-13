from pathlib import Path

import cv2
import numpy as np
import torch

from refine.data.dataset import MANIFEST_NAME, RecursiveImageDataset, build_manifest


def test_build_manifest_finds_all_images(tmp_image_dir):
    assert len(build_manifest(tmp_image_dir)) == 6


def test_manifest_is_cached(tmp_image_dir):
    build_manifest(tmp_image_dir)
    assert (tmp_image_dir / MANIFEST_NAME).exists()


def test_dataset_returns_clean_rgb(tmp_image_dir):
    ds = RecursiveImageDataset(tmp_image_dir, target_size=32, augment_hflip=False)
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32)
    assert sample.dtype == torch.float32
    assert sample.min() >= 0.0 and sample.max() <= 1.0


def test_skip_too_small(tmp_path):
    cv2.imwrite(str(tmp_path / "ok.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "tiny.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    ds = RecursiveImageDataset(tmp_path, target_size=32, min_side=32, augment_hflip=False)
    assert len(ds) == 1


def test_deterministic_split(tmp_image_dir):
    a = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val",
                              augment_hflip=False)
    b = RecursiveImageDataset(tmp_image_dir, target_size=32, val_fraction=0.34, split="val",
                              augment_hflip=False)
    assert [str(p) for p in a._paths] == [str(p) for p in b._paths]
