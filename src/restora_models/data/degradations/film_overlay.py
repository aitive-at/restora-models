"""Film overlay degradation: composite real scratch/dust/grain textures.

Textures come from the DeepRemaster noise_data.zip pack (898 MB, 6152 PNGs
of fractal noise / grain / dust / scratches). Auto-download path lives
elsewhere; this class just consumes a list of loaded textures.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from restora_models.data.degradations.registry import Degradation, register_degradation


@register_degradation("film_overlay")
@dataclass
class FilmOverlayDegradation(Degradation):
    textures: Sequence[np.ndarray] | None = None
    alpha_range: tuple[float, float] = (0.1, 0.4)

    @classmethod
    def from_dir(cls, root: Path, max_textures: int = 2000) -> "FilmOverlayDegradation":
        import cv2
        paths = sorted(root.rglob("*.png"))[:max_textures]
        textures = []
        for p in paths:
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            textures.append(arr.astype(np.float32) / 255.0)
        return cls(textures=textures or None)

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        if self.textures is None or len(self.textures) == 0:
            return rgb
        h, w = rgb.shape[:2]
        tex_idx = rng.randrange(len(self.textures))
        tex = self.textures[tex_idx]
        # Tile if needed so we have at least (h, w) to crop from.
        th, tw = tex.shape
        if th < h or tw < w:
            reps_y = (h + th - 1) // th + 1
            reps_x = (w + tw - 1) // tw + 1
            tex = np.tile(tex, (reps_y, reps_x))
            th, tw = tex.shape
        y0 = rng.randrange(0, th - h + 1)
        x0 = rng.randrange(0, tw - w + 1)
        crop = tex[y0:y0 + h, x0:x0 + w]
        alpha = rng.uniform(*self.alpha_range)
        out = rgb + alpha * crop[..., None]
        return np.clip(out, 0.0, 1.0).astype(rgb.dtype)
