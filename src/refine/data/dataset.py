"""Recursive image dataset with manifest cache and deterministic train/val split.

Returns *clean* (3, H, W) float32 RGB. Degradation lives outside the dataset
(see refine.data.multitask.MultiTaskWrapper).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import hflip, random_crop

MANIFEST_NAME = ".refine-manifest.txt"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def build_manifest(root: Path, *, force: bool = False) -> list[Path]:
    root = Path(root)
    mf = _manifest_path(root)
    if mf.exists() and not force:
        try:
            lines = mf.read_text().splitlines()
            mtime = float(lines[0])
            if abs(mtime - root.stat().st_mtime) < 1.0:
                return [root / line for line in lines[1:]]
        except Exception:
            pass
    out = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in _EXTS and p.is_file()]
    try:
        mf.write_text(f"{root.stat().st_mtime}\n" + "\n".join(str(p.relative_to(root)) for p in out))
    except OSError:
        pass
    return out


def _hash_to_unit(path: Path) -> float:
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class RecursiveImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        target_size: int,
        min_side: int | None = None,
        val_fraction: float = 0.0,
        split: Literal["train", "val", "all"] = "all",
        augment_hflip: bool = True,
        augment_rotate90: bool = False,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.target_size = target_size
        self.min_side = min_side if min_side is not None else target_size
        self.augment_hflip = augment_hflip
        self.augment_rotate90 = augment_rotate90
        self._seed = seed

        from PIL import Image

        all_paths = build_manifest(self.root)
        kept: list[Path] = []
        for p in all_paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                if h < self.min_side or w < self.min_side:
                    continue
                kept.append(p)
            except Exception:
                continue
        if val_fraction > 0 and split != "all":
            wanted = "val" if split == "val" else "train"
            kept = [p for p in kept if ((_hash_to_unit(p) < val_fraction) == (wanted == "val"))]
        self._paths = kept

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self._paths[idx]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        rng = random.Random((self._seed * 1_000_003) ^ idx)
        if self.augment_hflip and rng.random() < 0.5:
            rgb = hflip(rgb)
        if self.augment_rotate90 and rng.random() < 0.5:
            rgb = np.ascontiguousarray(np.rot90(rgb, k=rng.choice([1, 2, 3])))

        rgb = random_crop(rgb, self.target_size, rng)
        rgb_f32 = rgb.astype(np.float32) / 255.0
        return torch.from_numpy(rgb_f32.transpose(2, 0, 1)).contiguous()
