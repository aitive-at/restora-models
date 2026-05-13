"""Grayscale pair derivation used by both training and inference.

Routes RGB through LAB-L exactly like the original ColorizationPipeline,
so training distribution == inference distribution.
"""
from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np


class GrayPair(TypedDict):
    gray_rgb: np.ndarray  # (3, H, W) float32
    gt_ab: np.ndarray  # (2, H, W) float32 — LAB ab channels in cv2 range
    L_full: np.ndarray  # (1, H_full, W_full) float32 — full-res L for inference re-merge


def derive_pair(rgb_uint8: np.ndarray, *, target_size: int) -> GrayPair:
    """rgb_uint8: (H, W, 3) RGB → resized (3, T, T) gray RGB + GT AB + full-res L."""
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(
            f"expected (H, W, 3) uint8 RGB, got {rgb_uint8.shape}/{rgb_uint8.dtype}"
        )

    # cv2 expects BGR
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    img_f32 = bgr.astype(np.float32) / 255.0

    # Full-res L for inference re-merge path consistency
    L_full = cv2.cvtColor(img_f32, cv2.COLOR_BGR2Lab)[:, :, :1]  # (H, W, 1)

    img_resized = cv2.resize(img_f32, (target_size, target_size))
    lab_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)  # (T, T, 3)
    L = lab_resized[:, :, :1]
    gt_ab = lab_resized[:, :, 1:].astype(np.float32)

    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    return GrayPair(
        gray_rgb=gray_rgb.transpose(2, 0, 1),
        gt_ab=gt_ab.transpose(2, 0, 1),
        L_full=L_full.transpose(2, 0, 1),
    )
