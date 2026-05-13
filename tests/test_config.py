from pathlib import Path

import pytest

from refine.config import Config, deep_merge, expand_loss_preset, load_config


def test_preset_minimal():
    losses = expand_loss_preset("minimal")
    assert [l.name for l in losses] == ["l1_rgb"]


def test_preset_standard_has_colorfulness_for_colorize_only():
    losses = expand_loss_preset("standard")
    cf = [l for l in losses if l.name == "colorfulness"]
    assert len(cf) == 1
    assert cf[0].apply_to_tasks == ["colorize"]


def test_deep_merge():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    over = {"a": {"b": 99}}
    assert deep_merge(base, over) == {"a": {"b": 99, "c": 2}, "d": 3}


def test_load_config_with_preset(tmp_path: Path):
    (tmp_path / "x.yaml").write_text(
        "data: { root: /tmp/x }\nlosses: !preset minimal\n"
        "degradations: { colorize: { weight: 1.0 } }\n"
    )
    cfg = load_config(tmp_path / "x.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.root == "/tmp/x"
    assert [l.name for l in cfg.losses] == ["l1_rgb"]


def test_chained_defaults(tmp_path: Path):
    (tmp_path / "base.yaml").write_text(
        "data: { root: /a }\nlosses: !preset minimal\n"
        "degradations: { colorize: { weight: 1.0 } }\n"
    )
    (tmp_path / "child.yaml").write_text("defaults: base.yaml\ndata: { val_fraction: 0.05 }\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.data.root == "/a"
    assert cfg.data.val_fraction == 0.05


def test_required_fields_raise():
    with pytest.raises(Exception):
        Config.model_validate({})
