"""End-to-end smoke for PromptIR: train 10 steps -> ckpt -> preview -> infer -> ONNX fp16."""
from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="full smoke test (~60s on CPU); set REFINE_SLOW=1 to run",
)


def test_promptir_full_pipeline(tmp_path, tmp_image_dir):
    from refine.config import (
        AugmentConfig, CompoundConfig, Config, DataConfig, ExportConfig,
        LoaderConfig, ModelConfig, OptimConfig, RunConfig, SchedulerConfig,
        TrainConfig, expand_loss_preset,
    )
    from refine.train.trainer import Trainer

    cfg = Config(
        run=RunConfig(name="smoke", output_dir=str(tmp_path), seed=0),
        model=ModelConfig(type="promptir", size="tiny", input_size=64),
        data=DataConfig(
            root=str(tmp_image_dir),
            val_fraction=0.25,
            num_fixed_preview_samples=1,
            num_random_preview_samples=0,
            augment=AugmentConfig(),
            loader=LoaderConfig(batch_size=2, num_workers=0,
                                pin_memory=False, persistent_workers=False),
        ),
        compound=CompoundConfig(),
        losses=expand_loss_preset("standard"),
        optim_g=OptimConfig(fused=False),
        optim_d=OptimConfig(fused=False),
        scheduler=SchedulerConfig(total_steps=10),
        train=TrainConfig(total_steps=10, amp="fp32",
                          memory_format="contiguous", compile=False,
                          ema_decay=0.0, preview_every_s=999999,
                          preview_history_every=0, ckpt_every_steps=10,
                          log_every_steps=1),
        export=ExportConfig(on_finish=False),
    )
    trainer = Trainer(cfg, device=torch.device("cpu"), headless=True)
    trainer.fit()

    final = tmp_path / "ckpt" / "final.pt"
    assert final.exists(), f"no checkpoint at {final}"

    latest = tmp_path / "samples" / "latest.png"
    assert latest.exists(), "preview not written"

    import json
    sidecar = final.with_suffix(".task_map.json")
    assert sidecar.exists(), "task_map sidecar not written"
    tm = json.loads(sidecar.read_text())
    assert tm["model_type"] == "promptir", \
        f"sidecar should reflect actual model type; got {tm.get('model_type')!r}"

    # Dual-head verification: after training, head_ab.weight should have
    # changed from its zero init (it received gradient on colorize samples).
    # head_rgb.weight should have moved as well (gradient on every sample).
    payload = torch.load(str(final), map_location="cpu", weights_only=False)
    sd = payload["model"]
    head_ab_w = sd.get("dual_head.head_ab.weight")
    head_rgb_w = sd.get("dual_head.head_rgb.weight")
    assert head_ab_w is not None,  "dual_head.head_ab.weight not in checkpoint"
    assert head_rgb_w is not None, "dual_head.head_rgb.weight not in checkpoint"
    assert head_ab_w.abs().max().item() > 1e-6, \
        "head_ab.weight didn't get any gradient — dual-head wiring broken"
    assert head_rgb_w.abs().max().item() > 1e-6

    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    mcfg = ModelConfig(**(payload["extra"]["cfg"]["model"]))
    m = build_model(mcfg, num_axes=5)
    m.load_state_dict(payload["model"])
    onnx_path = tmp_path / "model.onnx"
    # parity_atol=2e-1: deep PromptIR + fp16 accumulates substantial numerical
    # drift through the attention stack's matmul/softmax + fp16 IO conversion.
    export_onnx_from_model(
        m, num_axes=5, input_size=64, export_path=onnx_path,
        opset=17, simplify=False, verify_parity=True, parity_atol=2e-1,
        dynamic_hw=False, task_map={"model_type": "promptir"},
        precision="fp16",
    )
    assert onnx_path.exists()

    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 64, 64).astype(np.float16)
    c = np.array([[1.0, 0, 0, 0, 0]], dtype=np.float16)
    y = sess.run(None, {"input": x, "config": c})[0]
    assert y.shape == (1, 3, 64, 64)
    assert np.isfinite(y).all()
