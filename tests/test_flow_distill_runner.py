"""Smoke test for the flow-distill runner."""
from pathlib import Path

import pytest


@pytest.mark.skipif(True, reason="downloads RAFT weights; skip in CI")
def test_run_flow_distill_smoke(tmp_path: Path):
    from restora_models.train.flow_distill import run_flow_distill
    # 2 steps with batch 1, real teacher download
    final = run_flow_distill(
        out_dir=tmp_path, config_path=Path("configs/local-temporal.yaml"),
        steps=2, batch_size=1, log_every=1,
    )
    assert final.exists()


def test_flow_distill_module_importable():
    """At least verify the module + entry point exist."""
    from restora_models.train.flow_distill import run_flow_distill
    assert callable(run_flow_distill)
