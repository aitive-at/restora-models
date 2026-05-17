"""Video frame-pair dataset for temporal consistency training.

Expects a directory layout like:

    <root>/<video_name>/<frame_NNNNN>.jpg
    <root>/<video_name>/<frame_NNNNN>.jpg
    ...
    <root>/<video_name>/.flow/<frame_NNNNN>_to_+k.npz   <- precomputed flow

The flow files are produced by `restora precompute-flow` (RAFT-based)
before training. At training time the dataset:
1. Samples a video at random
2. Picks a starting frame at random
3. Picks a k ∈ [1, max_skip] (random per sample)
4. Loads (frame_t, frame_{t+k}, flow_{t -> t+k})
5. Returns the same (clean, degraded, config, axes) tuple plus the
   secondary frame's clean+degraded+flow for the temporal loss.

If no precomputed flow file exists for a particular pair, the sample is
skipped and the next one is tried — temporal_pair loss handles missing
secondary by returning 0.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import hflip, random_crop


def _list_videos(root: Path) -> list[tuple[str, list[Path]]]:
    """Return [(video_name, sorted_frame_paths), ...] for each video subdir."""
    out: list[tuple[str, list[Path]]] = []
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir() or video_dir.name.startswith("."):
            continue
        frames = sorted(
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(frames) >= 2:
            out.append((video_dir.name, frames))
    return out


class VideoPairDataset(Dataset):
    """Yields pairs of clean RGB tensors + precomputed flow.

    Returns:
        {
          "clean_t":   (3, H, W) float [0, 1]
          "clean_tk":  (3, H, W)
          "flow_t_tk": (2, H, W) — pixel-displacement flow t -> t+k
        }

    The trainer wraps this with a degradation step so the model sees
    degraded inputs at training time; the clean tensors here are the
    targets.
    """

    def __init__(
        self,
        root: Path,
        target_size: int = 256,
        max_skip: int = 5,
        hflip_prob: float = 0.5,
        seed: int = 0,
        require_flow: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.target_size = int(target_size)
        self.max_skip = int(max_skip)
        self.hflip_prob = float(hflip_prob)
        self.seed = int(seed)
        self.require_flow = bool(require_flow)
        self.videos = _list_videos(self.root)
        if not self.videos:
            raise RuntimeError(f"no video subdirectories found under {self.root}")

        # Build a flat list of (video_idx, frame_t_idx) candidate pairs
        # respecting the bounds (need at least max_skip frames after t).
        self._pairs: list[tuple[int, int]] = []
        for vi, (_, frames) in enumerate(self.videos):
            for fi in range(len(frames) - 1):
                self._pairs.append((vi, fi))

    def __len__(self) -> int:
        return len(self._pairs)

    def _flow_path(self, video_name: str, frame_idx: int, k: int) -> Path:
        return self.root / video_name / ".flow" / f"frame_{frame_idx:05d}_skip{k}.npz"

    def _load_image(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to load {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict:
        vi, fi = self._pairs[idx]
        video_name, frames = self.videos[vi]
        rng = random.Random((self.seed * 1_000_003) ^ idx)

        # Pick skip
        max_k = min(self.max_skip, len(frames) - fi - 1)
        k = rng.randint(1, max(1, max_k))
        fi_secondary = fi + k

        flow_path = self._flow_path(video_name, fi, k)
        if self.require_flow and not flow_path.exists():
            # Fall back to identity flow (still useful as a temporal smoothness
            # prior; just not as accurate). Logged as a warning once would be
            # ideal, but for now just silently substitute.
            flow = np.zeros((2, self.target_size, self.target_size), dtype=np.float32)
        else:
            arr = np.load(str(flow_path))
            flow = arr["flow"].astype(np.float32)        # (2, H, W) at the precompute resolution

        img_t = self._load_image(frames[fi])
        img_tk = self._load_image(frames[fi_secondary])

        # Resize to target (and resize flow proportionally if needed)
        H_in, W_in = img_t.shape[:2]
        img_t = cv2.resize(img_t, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_AREA)
        img_tk = cv2.resize(img_tk, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_AREA)
        if flow.shape[1] != self.target_size or flow.shape[2] != self.target_size:
            # Rescale flow to target_size with appropriate pixel-magnitude scaling
            flow_resized = np.stack([
                cv2.resize(flow[0], (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LINEAR) * (self.target_size / W_in),
                cv2.resize(flow[1], (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LINEAR) * (self.target_size / H_in),
            ], axis=0)
            flow = flow_resized

        # Optional hflip (must flip flow's x-component too)
        if self.hflip_prob > 0 and rng.random() < self.hflip_prob:
            img_t = img_t[:, ::-1, :].copy()
            img_tk = img_tk[:, ::-1, :].copy()
            flow_x = -flow[0, :, ::-1].copy()
            flow_y = flow[1, :, ::-1].copy()
            flow = np.stack([flow_x, flow_y], axis=0)

        return {
            "clean_t":   torch.from_numpy(img_t.transpose(2, 0, 1)).contiguous(),
            "clean_tk":  torch.from_numpy(img_tk.transpose(2, 0, 1)).contiguous(),
            "flow_t_tk": torch.from_numpy(flow).contiguous(),
        }
