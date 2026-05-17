"""REDS (REalistic and Dynamic Scenes) sub-dataset.

Official site: https://seungjunnah.github.io/Datasets/reds.html
Supports two layouts:
    Traditional:    <root>/<split>/<seq_id>/<frame_id>.png    (e.g. train_sharp/000/00000000.png)
    Flat:           <root>/<seq_id>/<frame_id>.png            (user laid it out flat)

Each sample is a window of `window` contiguous frames. Random cropping inline.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_FRAME_PATTERN = re.compile(r"^\d+\.png$")


def _is_real_png(p: Path) -> bool:
    """Reject `:Zone.Identifier` sidecars and other non-frame files."""
    return _FRAME_PATTERN.match(p.name) is not None


class REDSDataset(Dataset):
    SOURCE_NAME = "reds"

    def __init__(
        self,
        root: Path | str,
        split: str = "train_sharp",
        window: int = 7,
        stride: int = 1,
        crop: int = 256,
    ):
        if window < 1:
            raise ValueError(f"window must be >=1, got {window}")
        if stride < 1:
            raise ValueError(f"stride must be >=1, got {stride}")
        self.root = Path(root)
        self.window = window
        self.stride = stride
        self.crop = crop

        # Try <root>/<split>/ first, fall back to <root>/ for flat layouts.
        split_dir = self.root / split
        if split_dir.is_dir():
            seq_root = split_dir
        elif self.root.is_dir():
            seq_root = self.root
        else:
            raise FileNotFoundError(f"REDS root not found: {self.root}")

        self.windows: list[tuple[Path, int]] = []
        for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            frames = sorted(p for p in seq_dir.glob("*.png") if _is_real_png(p))
            n = len(frames)
            if n < window:
                continue
            for off in range(0, n - window + 1, stride):
                self.windows.append((seq_dir, off))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        seq_dir, off = self.windows[idx]
        all_frames = sorted(p for p in seq_dir.glob("*.png") if _is_real_png(p))
        frames_files = all_frames[off:off + self.window]
        clip = []
        for p in frames_files:
            arr = cv2.imread(str(p))
            if arr is None:
                raise RuntimeError(f"failed to read {p}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            clip.append(arr)
        clip_np = np.stack(clip).astype(np.float32) / 255.0
        h, w = clip_np.shape[1:3]
        ch = min(self.crop, h)
        cw = min(self.crop, w)
        y0 = int(np.random.randint(0, h - ch + 1)) if h > ch else 0
        x0 = int(np.random.randint(0, w - cw + 1)) if w > cw else 0
        clip_np = clip_np[:, y0:y0 + ch, x0:x0 + cw, :]
        clip_t = torch.from_numpy(clip_np).permute(0, 3, 1, 2).contiguous()
        return {
            "frames": clip_t,
            "source": self.SOURCE_NAME,
            "key": f"{seq_dir.name}@{off}",
        }
