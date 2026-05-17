"""Vimeo Septuplet (Xue et al., http://toflow.csail.mit.edu/) sub-dataset.

Layout:
    <root>/sequences/<seqA>/<seqB>/im{1..7}.png
    <root>/sep_trainlist.txt   (lines: <seqA>/<seqB>)
    <root>/sep_testlist.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VimeoSeptupletDataset(Dataset):
    NUM_FRAMES = 7
    SOURCE_NAME = "vimeo_septuplet"

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "test"] = "train",
        crop: int = 256,
    ) -> None:
        self.root = Path(root)
        list_name = "sep_trainlist.txt" if split == "train" else "sep_testlist.txt"
        list_path = self.root / list_name
        if not list_path.exists():
            raise FileNotFoundError(f"Vimeo Septuplet list missing: {list_path}")
        self.entries = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        self.crop = crop

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        seq = self.entries[idx]
        d = self.root / "sequences" / seq
        frames = []
        for i in range(1, self.NUM_FRAMES + 1):
            arr = cv2.imread(str(d / f"im{i}.png"))
            if arr is None:
                raise RuntimeError(f"failed to read {d / f'im{i}.png'}")
            frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        clip = np.stack(frames).astype(np.float32) / 255.0
        h, w = clip.shape[1:3]
        ch = min(self.crop, h)
        cw = min(self.crop, w)
        y0 = int(np.random.randint(0, h - ch + 1)) if h > ch else 0
        x0 = int(np.random.randint(0, w - cw + 1)) if w > cw else 0
        clip = clip[:, y0:y0 + ch, x0:x0 + cw, :]
        clip_t = torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()
        return {"frames": clip_t, "source": self.SOURCE_NAME, "key": seq}
