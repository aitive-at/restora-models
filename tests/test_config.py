from pathlib import Path

import pytest

from restora_models.config import (
    Config, DataConfig, ModelConfig, deep_merge,
    expand_loss_preset, load_config,
)


def test_preset_minimal():
    losses = expand_loss_preset("minimal")
    assert [l.name for l in losses] == ["l1_rgb"]


def test_preset_standard_has_colorfulness_for_colorize_only():
    losses = expand_loss_preset("standard")
    cf = [l for l in losses if l.name == "colorfulness"]
    assert len(cf) == 1
    assert cf[0].apply_to_axes == ["colorize"]


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"b": 99}}
    assert deep_merge(base, over) == {"a": {"b": 99, "c": 2}, "d": 3}


def test_load_config_with_preset(tmp_path: Path):
    (tmp_path / "x.yaml").write_text(
        "data: { sources: [] }\nlosses: !preset minimal\n"
    )
    cfg = load_config(tmp_path / "x.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.sources == []
    assert [l.name for l in cfg.losses] == ["l1_rgb"]


def test_chained_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text(
        "data: { sources: [] }\nlosses: !preset minimal\n"
    )
    (tmp_path / "child.yaml").write_text(
        "defaults: base.yaml\ndata: { val_fraction: 0.05 }\n"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.sources == []
    assert cfg.data.val_fraction == 0.05


def test_required_fields_raise():
    with pytest.raises(Exception):
        Config.model_validate({})


def test_data_config_film_overlay_knobs_default_to_disabled():
    """film_overlay_root is None unless explicitly pointed at noise_data/."""
    d = DataConfig()
    assert d.film_overlay_root is None
    assert 0.0 <= d.film_overlay_prob <= 1.0
    assert 0.0 <= d.gate_weave_prob <= 1.0
    assert 0.0 <= d.mpeg_transcode_prob <= 1.0


def test_model_config_defaults_to_temporal_restora_small():
    """The new contract: type names the architecture+size in one string."""
    m = ModelConfig()
    assert m.type == "temporal_restora_small"
    assert m.input_size == 256
    assert m.task_embed_dim == 128


def test_legacy_model_config_fields_are_ignored():
    """Old configs with refine_type / adversarial_refine validate but the
    fields are silently dropped (pydantic v2 'ignore' extras default)."""
    m = ModelConfig(type="temporal_restora_small", refine_type="adversarial",
                    adversarial_refine=True, nf=32)
    assert m.type == "temporal_restora_small"
    assert not hasattr(m, "refine_type")
    assert not hasattr(m, "adversarial_refine")
