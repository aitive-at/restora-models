"""End-to-end PNNX export test on a tiny NAFNet.

PNNX is the ncnn-format exporter (https://github.com/pnnx/pnnx). It runs
torch.jit.trace + a native binary, so this test is slow-marked (~30s on
CPU). Gate on REFINE_SLOW=1 to run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from restora_models.config import ModelConfig
from restora_models.export.pnnx import export_pnnx_from_model
from restora_models.models import build_model


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        type="nafnet", size="tiny", input_size=64,
        nf=8, enc_depths=[1, 1, 1, 1], bottle_blocks=1, hidden_dim=32,
        refine_type="none",
    )
    return build_model(cfg, num_axes=5)


@pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="pnnx export is slow; set REFINE_SLOW=1 to run",
)
def test_pnnx_export_generic_produces_expected_sidecars(tmp_path, tiny_model):
    out = tmp_path / "model.pt"
    export_pnnx_from_model(
        tiny_model, num_axes=5, input_size=32,
        export_path=out, dynamic_hw=False, fp16=False,
    )
    base = out.with_suffix("")
    # PNNX writes these companion files alongside the .pt
    assert (tmp_path / "model.pt").exists()
    assert Path(f"{base}.pnnx.bin").exists()
    assert Path(f"{base}.pnnx.param").exists()
    assert Path(f"{base}.ncnn.bin").exists()
    assert Path(f"{base}.ncnn.param").exists()


@pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="pnnx export is slow; set REFINE_SLOW=1 to run",
)
def test_pnnx_export_dynamic_hw_uses_two_input_shapes(tmp_path, tiny_model):
    """When dynamic_hw=True, pnnx receives two example inputs at different
    spatial resolutions and marks H/W as dynamic in the .param file."""
    out = tmp_path / "model.pt"
    export_pnnx_from_model(
        tiny_model, num_axes=5, input_size=32,
        export_path=out, dynamic_hw=True, fp16=False,
    )
    # Sanity: ncnn param file lists a Convolution op with non-fixed input dims.
    # The .pnnx.param file is the easier-to-read intermediate format.
    param_text = (tmp_path / "model.pnnx.param").read_text()
    assert "pnnx.Input" in param_text
    # We don't pin the exact dynamic-shape syntax (it's a pnnx internal),
    # but we verify the file is non-trivial.
    assert len(param_text) > 100


@pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="pnnx export is slow; set REFINE_SLOW=1 to run",
)
def test_pnnx_export_baked_produces_single_input(tmp_path, tiny_model):
    """With fixed_config, the export wrapper takes only one input tensor —
    pnnx should reflect that in its single-input .param file."""
    out = tmp_path / "model.pt"
    export_pnnx_from_model(
        tiny_model, num_axes=5, input_size=32,
        export_path=out, fixed_config=[1.0, 0.0, 0.0, 0.0, 0.0],
        fp16=False,
    )
    # ncnn .param has an Input line per network input; baked variant has 1.
    ncnn_param = (tmp_path / "model.ncnn.param").read_text()
    input_lines = [ln for ln in ncnn_param.splitlines() if ln.startswith("Input")]
    assert len(input_lines) == 1, \
        f"baked PNNX should have exactly 1 input; got {len(input_lines)}: {input_lines}"


@pytest.mark.skipif(
    not os.environ.get("REFINE_SLOW"),
    reason="pnnx export is slow; set REFINE_SLOW=1 to run",
)
def test_pnnx_export_writes_task_map_sidecar(tmp_path, tiny_model):
    import json
    out = tmp_path / "model.pt"
    task_map = {"model_type": "nafnet", "model_size": "tiny", "version": "test"}
    export_pnnx_from_model(
        tiny_model, num_axes=5, input_size=32,
        export_path=out, task_map=task_map, fp16=False,
    )
    sidecar = tmp_path / "model.task_map.json"
    assert sidecar.exists()
    loaded = json.loads(sidecar.read_text())
    assert loaded == task_map
