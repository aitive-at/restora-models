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
    # 60k @ bs=32 sees ~1.9 M samples in ~18 h on the Blackwell, vs the
    # old 100k @ bs=12 which saw ~1.2 M in ~16 h. See the config comment.
    assert cfg.train.total_steps == 60000
    assert cfg.train.compile is True
    assert cfg.data.loader.batch_size == 32


def test_temporal_v1_preset_has_expected_losses():
    cfg = load_config(Path("configs/default.yaml"))
    names = [c.name for c in cfg.losses]
    # At least the temporal-specific losses should appear
    assert "lpips_decoded" in names
    assert "central_flicker" in names
