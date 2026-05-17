"""Tests for VimeoSeptupletDataset sub-dataset."""
from pathlib import Path

import cv2
import numpy as np
import torch

from restora_models.data.vimeo_septuplet import VimeoSeptupletDataset


def _make_fake_vimeo(tmp_path: Path) -> Path:
    root = tmp_path / "vimeo"
    seqs = ["00001/0001", "00001/0002"]
    for s in seqs:
        d = root / "sequences" / s
        d.mkdir(parents=True)
        for i in range(1, 8):
            img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
            cv2.imwrite(str(d / f"im{i}.png"), img)
    (root / "sep_trainlist.txt").write_text("\n".join(seqs) + "\n")
    return root


def test_vimeo_loader_canonical_sample(tmp_path):
    root = _make_fake_vimeo(tmp_path)
    ds = VimeoSeptupletDataset(root, split="train", crop=32)
    sample = ds[0]
    assert sample["frames"].shape == (7, 3, 32, 32)
    assert sample["frames"].dtype == torch.float32
    assert sample["source"] == "vimeo_septuplet"
    assert "key" in sample
