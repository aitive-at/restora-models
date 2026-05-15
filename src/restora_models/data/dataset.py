"""Recursive image dataset with manifest cache and deterministic train/val split.

Returns *clean* (3, H, W) float32 RGB. Degradation lives outside the dataset
(see restora_models.data.compound.CompoundDegradationWrapper).
"""
from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import hflip, random_crop

MANIFEST_NAME = ".restora-manifest.txt"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Progress cadence — print at most every N seconds so logs aren't flooded
# but the user still sees that the scan is alive.
_PROGRESS_EVERY_S = 5.0


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def build_manifest(root: Path, *, force: bool = False) -> list[Path]:
    """Discover all image files under `root`, recursively.

    Caches results to `<root>/.restora-manifest.txt` (one line per relative
    path) so subsequent training launches read it in milliseconds instead
    of walking the tree. Cache invalidates when `root.stat().st_mtime`
    changes (new images added at the top level).

    Prints discovery progress every 5s — important because on a 1M-file
    tree the walk can take minutes and was previously silent.
    """
    root = Path(root)
    mf = _manifest_path(root)
    if mf.exists() and not force:
        try:
            lines = mf.read_text().splitlines()
            mtime = float(lines[0])
            if abs(mtime - root.stat().st_mtime) < 1.0:
                print(f"[manifest] using cached: {len(lines)-1:,} images under {root}",
                      flush=True)
                return [root / line for line in lines[1:]]
        except Exception:
            pass

    print(f"[manifest] scanning {root} (no cache or stale) ...", flush=True)
    t0 = time.time()
    out: list[Path] = []
    last_print = t0
    for p in root.rglob("*"):
        if p.suffix.lower() in _EXTS and p.is_file():
            out.append(p)
            now = time.time()
            if now - last_print > _PROGRESS_EVERY_S:
                elapsed = now - t0
                rate = len(out) / max(elapsed, 1e-3)
                print(f"[manifest]   {len(out):,} files found  "
                      f"({rate:.0f}/s, {elapsed:.0f}s elapsed)", flush=True)
                last_print = now

    out.sort()
    elapsed = time.time() - t0
    print(f"[manifest] discovery done: {len(out):,} files in {elapsed:.1f}s",
          flush=True)

    try:
        mf.write_text(f"{root.stat().st_mtime}\n"
                       + "\n".join(str(p.relative_to(root)) for p in out))
        print(f"[manifest] cached to {mf}", flush=True)
    except OSError:
        pass
    return out


def _hash_to_unit(path: Path) -> float:
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _check_one(args: tuple[Path, int]) -> tuple[Path, bool]:
    """Open one image header, return (path, big_enough). Used by the
    parallel dimension-filter below."""
    p, min_side = args
    from PIL import Image
    try:
        with Image.open(p) as im:
            w, h = im.size
        return (p, w >= min_side and h >= min_side)
    except Exception:
        return (p, False)


def _filter_by_dimensions(paths: list[Path], min_side: int, *,
                           workers: int = 32) -> list[Path]:
    """Keep paths whose images are >= min_side on the shorter side.

    Parallelizes the per-file PIL header read across `workers` threads.
    PIL's Image.open just reads the header (not pixels), so this is I/O
    bound — ThreadPoolExecutor scales it linearly until disk-bound.

    Prints progress every 5s. On a 1M-file ImageNet tree with NVMe disk:
    serial ≈ 17 min, 32 threads ≈ 1-2 min.
    """
    if not paths:
        return []
    print(f"[dataset] checking dimensions >= {min_side}px on "
          f"{len(paths):,} files ({workers} parallel I/O threads) ...",
          flush=True)
    t0 = time.time()
    kept: list[Path] = []
    n_dropped = 0
    last_print = t0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # ex.map preserves input order, so kept[] remains sorted
        # (the manifest came in sorted from build_manifest). chunksize
        # batches reduce per-task dispatch overhead for ~1M items.
        for i, (p, ok) in enumerate(
            ex.map(_check_one, ((p, min_side) for p in paths), chunksize=64),
            start=1,
        ):
            if ok:
                kept.append(p)
            else:
                n_dropped += 1
            now = time.time()
            if now - last_print > _PROGRESS_EVERY_S:
                elapsed = now - t0
                rate = i / max(elapsed, 1e-3)
                eta = (len(paths) - i) / max(rate, 1e-3)
                print(f"[dataset]   {i:,}/{len(paths):,}  "
                      f"({rate:.0f}/s, ETA {eta:.0f}s, kept {len(kept):,})",
                      flush=True)
                last_print = now
    elapsed = time.time() - t0
    print(f"[dataset] dimension filter done in {elapsed:.1f}s: "
          f"kept {len(kept):,}, dropped {n_dropped:,} too-small/unreadable",
          flush=True)
    return kept


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
        kept = _filter_by_dimensions(all_paths, self.min_side)

        if val_fraction > 0 and split != "all":
            wanted = "val" if split == "val" else "train"
            kept = [p for p in kept if ((_hash_to_unit(p) < val_fraction) == (wanted == "val"))]
            print(f"[dataset] split={split} (val_fraction={val_fraction}): "
                  f"{len(kept):,} paths", flush=True)
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
