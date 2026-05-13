import numpy as np
import torch

from refine.config import ModelConfig
from refine.infer.pipeline import CompoundRefinerPipeline, pad_to_multiple, unpad
from refine.models import build_model


def test_pad_unpad_round_trip():
    img = np.random.rand(45, 71, 3).astype(np.float32)
    padded, pads = pad_to_multiple(img, multiple=16, mode="reflect")
    assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0
    back = unpad(padded, *pads)
    np.testing.assert_array_equal(back, img)


def test_pipeline_rgb_to_rgb_shape():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_axes=5)
    pipe = CompoundRefinerPipeline(m, device=torch.device("cpu"))
    img = (np.random.rand(33, 55, 3) * 255).astype(np.uint8)
    out = pipe.process(img, config={"colorize": True, "denoise": False,
                                    "sharpen": False, "dejpeg": False, "deblur": False})
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_pipeline_config_as_list():
    cfg = ModelConfig(type="nafnet", size="tiny", nf=8,
                      enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
                      task_embed_dim=16)
    m = build_model(cfg, num_axes=5)
    pipe = CompoundRefinerPipeline(m, device=torch.device("cpu"))
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    out = pipe.process(img, config=[1.0, 0.0, 0.0, 0.0, 0.0])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
