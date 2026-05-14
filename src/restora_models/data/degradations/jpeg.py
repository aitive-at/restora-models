"""JPEG-restore degradation: encode then decode at random quality."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .registry import Degradation, register_degradation


@register_degradation("jpeg")
class JpegDegradation(Degradation):
    def __init__(self, quality_range: tuple[int, int] = (20, 70)) -> None:
        super().__init__()
        self.qmin, self.qmax = int(quality_range[0]), int(quality_range[1])

    def degrade(self, rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        quality = rng.randint(self.qmin, self.qmax)
        bgr_uint8 = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * 255.0).round().clip(0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return rgb
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(decoded_rgb, 0.0, 1.0).astype(np.float32)
