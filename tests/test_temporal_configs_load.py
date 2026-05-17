"""Tests that the three temporal configs load cleanly."""
from pathlib import Path

from restora_models.config import load_config


def test_default_config_loads():
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.model.type.startswith("temporal_")
    assert isinstance(cfg.data.sources, list)


def test_local_temporal_config_loads():
    cfg = load_config(Path("configs/local-temporal.yaml"))
    assert cfg.run.name == "local_temporal"
    assert cfg.model.type == "temporal_restora_small"
    assert len(cfg.data.sources) >= 1
    assert cfg.data.sources[0]["type"] == "reds"


def test_local_temporal_production_targets():
    cfg = load_config(Path("configs/local-temporal.yaml"))
    assert cfg.train.total_steps == 100000
    assert cfg.train.compile is True
    # bs bumped to 32 after the data-pipeline refactor; ~55 GB on a 96 GB
    # Blackwell. See `configs/local-temporal.yaml` for the rationale.
    assert cfg.data.loader.batch_size == 32


def test_temporal_v1_preset_has_expected_losses():
    cfg = load_config(Path("configs/default.yaml"))
    names = [c.name for c in cfg.losses]
    # At least the temporal-specific losses should appear
    assert "lpips_decoded" in names
    assert "central_flicker" in names
