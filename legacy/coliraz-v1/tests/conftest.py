import pathlib

import numpy as np
import pytest


@pytest.fixture
def tmp_image_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tmp directory containing 6 small synthetic RGB images in a nested tree."""
    import cv2

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    rng = np.random.default_rng(0)
    paths = [
        tmp_path / "img0.png",
        tmp_path / "img1.jpg",
        tmp_path / "a" / "img2.png",
        tmp_path / "a" / "img3.webp",
        tmp_path / "a" / "b" / "img4.jpeg",
        tmp_path / "a" / "b" / "img5.bmp",
    ]
    for p in paths:
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)
    return tmp_path


@pytest.fixture
def small_image_uint8() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
