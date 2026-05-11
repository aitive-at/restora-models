import cv2
import numpy as np
import torch

from coliraz.utils.color import (
    derive_gray_rgb_from_rgb,
    lab_to_rgb,
    rgb_to_lab,
)


def test_rgb_to_lab_matches_cv2_within_tolerance(small_image_uint8):
    rgb_uint8 = small_image_uint8
    rgb_f32 = rgb_uint8.astype(np.float32) / 255.0

    expected = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)  # H,W,3
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    got = rgb_to_lab(t).squeeze(0).permute(1, 2, 0).numpy()

    np.testing.assert_allclose(got, expected, atol=1.0)


def test_lab_to_rgb_round_trip(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    lab = rgb_to_lab(t)
    back = lab_to_rgb(lab).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(back, rgb_f32, atol=0.02)


def test_derive_gray_rgb_matches_reference(small_image_uint8):
    rgb_f32 = small_image_uint8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)
    L = lab[:, :, :1]
    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    expected = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)

    t = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)
    got = derive_gray_rgb_from_rgb(t).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=0.01)
