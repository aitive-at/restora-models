"""Recursive image dataset with manifest cache and deterministic train/val split."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .grayscale import derive_pair
from .transforms import hflip, random_crop

MANIFEST_NAME = ".coliraz-manifest.txt"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def build_manifest(root: Path, *, force: bool = False) -> list[Path]:
    root = Path(root)
    mf = _manifest_path(root)
    if mf.exists() and not force:
        try:
            mtime_recorded = float(mf.read_text().splitlines()[0])
            if abs(mtime_recorded - root.stat().st_mtime) < 1.0:
                return [root / line for line in mf.read_text().splitlines()[1:]]
        except Exception:
            pass

    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in _EXTS and p.is_file():
            out.append(p)
    try:
        mf.write_text(
            f"{root.stat().st_mtime}\n"
            + "\n".join(str(p.relative_to(root)) for p in out)
        )
    except OSError:
        pass
    return out


def _hash_to_unit(path: Path) -> float:
    h = hashlib.md5(str(path).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


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

        all_paths = build_manifest(self.root)
        kept: list[Path] = []
        # Reading the full image with cv2.imread just to check shape is
        # *very* slow on network/Windows-mount filesystems (30K x 30-100ms
        # = 15-50 min stall at trainer construction). PIL reads only the
        # file header by default and is 10-50x faster.
        from PIL import Image

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
            kept = [
                p for p in kept
                if ((_hash_to_unit(p) < val_fraction) == (wanted == "val"))
            ]
        self._paths = kept

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        pair = derive_pair(rgb, target_size=self.target_size)
        return {
            "gray_rgb": torch.from_numpy(pair["gray_rgb"]),
            "gt_ab": torch.from_numpy(pair["gt_ab"]),
            "L_full": torch.from_numpy(pair["L_full"]),
            "path": str(p),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "gray_rgb": torch.stack([b["gray_rgb"] for b in batch]),
        "gt_ab": torch.stack([b["gt_ab"] for b in batch]),
        "L_full": torch.stack([b["L_full"] for b in batch]),
        "path": [b["path"] for b in batch],
    }
