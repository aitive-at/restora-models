import numpy as np
import torch

from restora_models.train.preview import render_multitask_grid, write_png_atomic


def test_render_grid_shape():
    samples = {
        "colorize": [{"clean": torch.rand(3, 32, 32), "degraded": torch.rand(3, 32, 32),
                      "predicted": torch.rand(3, 32, 32)}],
        "denoise":  [{"clean": torch.rand(3, 32, 32), "degraded": torch.rand(3, 32, 32),
                      "predicted": torch.rand(3, 32, 32)}],
    }
    img = render_multitask_grid(samples, caption="step 100", cell_size=32)
    assert img.dtype == np.uint8
    assert img.shape[1] >= 32 * 4
    assert img.shape[0] >= 32 * 2


def test_atomic_write(tmp_path):
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    write_png_atomic(p, img)
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()
