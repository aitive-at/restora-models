import cv2
import numpy as np

from coliraz.data.grayscale import derive_pair


def test_derive_pair_shapes(small_image_uint8):
    out = derive_pair(small_image_uint8, target_size=16)
    assert out["gray_rgb"].shape == (3, 16, 16)
    assert out["gt_ab"].shape == (2, 16, 16)
    assert out["L_full"].shape == (1, 32, 32)
    assert out["gray_rgb"].dtype.name == "float32"


def test_gray_rgb_matches_reference_pipeline(small_image_uint8):
    bgr = cv2.cvtColor(small_image_uint8, cv2.COLOR_RGB2BGR)
    img_f32 = bgr.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_f32, (16, 16))
    L = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    gray_lab = np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
    expected_gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB).transpose(2, 0, 1)

    out = derive_pair(small_image_uint8, target_size=16)
    np.testing.assert_allclose(out["gray_rgb"], expected_gray_rgb, atol=1e-5)
