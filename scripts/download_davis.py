#!/usr/bin/env python
"""Download DAVIS-2017 (480p, train+val) and lay out frames for VideoPairDataset.

The DAVIS-2017 trainval-480p archive contains:

    DAVIS/JPEGImages/480p/<video_name>/00000.jpg
    DAVIS/JPEGImages/480p/<video_name>/00001.jpg
    ...
    DAVIS/Annotations/...   (we discard — segmentation masks we don't need)
    DAVIS/ImageSets/...     (we discard)

This script:
1. Downloads the archive if not present (skips if already there).
2. Unzips it to a staging directory.
3. Moves only `JPEGImages/480p/<video>/` subdirs into
   `<out_root>/<video_name>/frame_NNNNN.jpg`, renaming `00000.jpg` →
   `frame_00000.jpg` to match what `VideoPairDataset` expects.
4. Removes the staging directory.

Re-runs are idempotent — videos already present under `<out_root>` are
skipped.

Usage:
    uv run python scripts/download_davis.py --out ~/data/laion-videos
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[davis] archive present, skip download ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[davis] downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_pct = [-1]
    def hook(chunk_i, chunk_sz, total):
        if total <= 0:
            return
        downloaded = chunk_i * chunk_sz
        pct = int(100 * downloaded / total)
        if pct != last_pct[0]:
            print(f"\r[davis]   {pct:3d}%  ({downloaded / 1e6:.0f} / {total / 1e6:.0f} MB)",
                  end="", flush=True)
            last_pct[0] = pct
    urllib.request.urlretrieve(url, tmp, hook)
    print()
    tmp.rename(dest)


def _unzip(zip_path: Path, staging: Path) -> Path:
    """Returns the path to DAVIS/JPEGImages/480p inside staging."""
    print(f"[davis] unzipping into {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # Only extract JPEGImages/480p/* — saves time and disk.
        names = [n for n in zf.namelist() if "/JPEGImages/480p/" in n and n.endswith(".jpg")]
        for n in names:
            zf.extract(n, staging)
    return staging / "DAVIS" / "JPEGImages" / "480p"


def _layout(jpegs_root: Path, out_root: Path) -> int:
    """Move each video's frames into <out_root>/<video>/frame_NNNNN.jpg.
    Returns count of videos newly written."""
    out_root.mkdir(parents=True, exist_ok=True)
    new = 0
    for video_dir in sorted(jpegs_root.iterdir()):
        if not video_dir.is_dir():
            continue
        target = out_root / video_dir.name
        if target.exists() and any(target.glob("frame_*.jpg")):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for src in sorted(video_dir.iterdir()):
            if not src.suffix.lower() == ".jpg":
                continue
            stem = src.stem
            try:
                fi = int(stem)
            except ValueError:
                continue
            dst = target / f"frame_{fi:05d}.jpg"
            shutil.move(str(src), str(dst))
        new += 1
    return new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("~/data/laion-videos"),
                   help="Output root for videos. Default: ~/data/laion-videos")
    p.add_argument("--cache", type=Path, default=Path("~/.cache/davis"),
                   help="Where to put the downloaded zip. Default: ~/.cache/davis")
    p.add_argument("--keep-staging", action="store_true",
                   help="Don't delete the unzipped staging dir afterwards.")
    args = p.parse_args()
    out_root = args.out.expanduser()
    cache = args.cache.expanduser()

    zip_path = cache / "DAVIS-2017-trainval-480p.zip"
    _download(DAVIS_URL, zip_path)

    staging = cache / "extract"
    if staging.exists():
        shutil.rmtree(staging)
    jpegs_root = _unzip(zip_path, staging)

    new = _layout(jpegs_root, out_root)
    total = sum(1 for _ in out_root.iterdir() if _.is_dir() and not _.name.startswith("."))
    print(f"[davis] done. {new} new videos, {total} total under {out_root}")

    if not args.keep_staging and staging.exists():
        shutil.rmtree(staging)
        print(f"[davis] cleaned up staging {staging}")


if __name__ == "__main__":
    sys.exit(main())
