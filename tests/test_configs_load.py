from pathlib import Path

from refine.config import load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_yaml_loads():
    cfg = load_config(ROOT / "tiny.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "tiny"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"


def test_laion_multitask_loads():
    cfg = load_config(ROOT / "laion-multitask.yaml", overrides={"data": {"root": "/tmp"}})
    assert "colorize" in cfg.degradations
    assert "sr_x4" in cfg.degradations
