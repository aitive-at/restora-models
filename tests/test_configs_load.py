from pathlib import Path

from refine.config import load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_tiny_yaml_loads():
    cfg = load_config(ROOT / "tiny.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "tiny"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"


def test_laion_compound_loads():
    cfg = load_config(ROOT / "laion-compound.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.compound.identity_prob == 0.05
    assert cfg.compound.axis_probs.colorize == 0.5
    assert cfg.compound.axis_probs.sharpen == 0.5


def test_promptir_laion_config_loads():
    from pathlib import Path
    from refine.config import load_config
    cfg = load_config(Path("configs/promptir-laion.yaml"))
    assert cfg.model.type == "promptir"
    assert cfg.model.size == "large"
    names = [l.name for l in cfg.losses]
    assert "chroma_lab" in names


def test_nafnet_tiny_vivid_config_loads():
    cfg = load_config(ROOT / "nafnet-tiny-vivid.yaml")
    assert cfg.model.type == "nafnet"
    assert cfg.model.size == "tiny"
    by_name = {l.name: l for l in cfg.losses}
    # The cheap-experiment recipe deltas
    assert by_name["chroma_lab"].weight == 0.25
    assert by_name["colorfulness"].weight == 0.10
    assert by_name["freq_l1"].weight == 0.40
    assert by_name["freq_l1"].apply_to_axes == ["sharpen"]   # deblur dropped
    assert by_name["gan"].weight == 0.05
    assert set(by_name["gan"].apply_to_axes) == {"colorize", "sharpen"}
