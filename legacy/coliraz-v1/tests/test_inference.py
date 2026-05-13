import numpy as np
import torch

from coliraz.config import ModelConfig
from coliraz.infer.pipeline import ColorizationPipeline
from coliraz.models import build_ddcolor


def test_pipeline_returns_same_shape_bgr_uint8():
    cfg = ModelConfig(
        size="tiny", input_size=64, dec_layers=1, num_queries=4, nf=64, hidden_dim=32
    )
    model = build_ddcolor(cfg, pretrained=False)
    pipe = ColorizationPipeline(model, input_size=64, device=torch.device("cpu"))
    img = (np.random.rand(48, 72, 3) * 255).astype(np.uint8)
    out = pipe.process(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
