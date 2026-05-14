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
    # axis_probs uniform 0.5 (proven balanced; was 0.75/0.40 — too aggressive,
    # caused contention that crushed easy tasks)
    assert cfg.compound.axis_probs.colorize == 0.50
    assert cfg.compound.axis_probs.sharpen == 0.50


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
    # The vivid variant has slightly heavier color weights than balanced
    assert "chroma_lab" in by_name
    assert "gan" in by_name  # vivid keeps GAN (caller adds warmup if needed)


def test_nafnet_tiny_vivid_nogan_config_loads():
    """The canonical balanced recipe — chroma_lab 0.10, colorfulness 0.05,
    freq_l1 0.30 sharpen-only — proven on 2026-05-14 iter-6 (nafnet-large)."""
    cfg = load_config(ROOT / "nafnet-tiny-vivid-nogan.yaml")
    assert cfg.model.type == "nafnet"
    assert cfg.model.size == "tiny"
    names = [l.name for l in cfg.losses]
    assert "gan" not in names, "balanced recipe should NOT include GAN from cold"
    by_name = {l.name: l for l in cfg.losses}
    assert by_name["chroma_lab"].weight == 0.10
    assert by_name["freq_l1"].weight == 0.30
    assert by_name["colorfulness"].weight == 0.05
    assert by_name["freq_l1"].apply_to_axes == ["sharpen"]


def test_data_root_expands_tilde():
    """Regression: data.root in YAML can use ~ — it must expand to $HOME at
    load time so Path() / directory walks work."""
    import os
    from refine.config import DataConfig, LoaderConfig, AugmentConfig
    home = os.path.expanduser("~")
    cfg = DataConfig(root="~/data/laion-images",
                     loader=LoaderConfig(), augment=AugmentConfig())
    assert cfg.root == f"{home}/data/laion-images"
    assert not cfg.root.startswith("~")


def test_data_root_expands_env_var(monkeypatch):
    monkeypatch.setenv("REFINE_TEST_DATA_DIR", "/tmp/refine-test-data")
    from refine.config import DataConfig, LoaderConfig, AugmentConfig
    cfg = DataConfig(root="$REFINE_TEST_DATA_DIR/sub",
                     loader=LoaderConfig(), augment=AugmentConfig())
    assert cfg.root == "/tmp/refine-test-data/sub"


def test_default_axis_probs_uniform_half():
    """Proven on 2026-05-14: axis_probs uniform 0.5 outperforms the
    0.75/0.40 split (which caused multi-task contention)."""
    cfg = load_config(ROOT / "default.yaml", overrides={"data": {"root": "/tmp"}})
    ap = cfg.compound.axis_probs
    assert ap.colorize == 0.50
    assert ap.sharpen  == 0.50
    assert ap.denoise  == 0.50
    assert ap.dejpeg   == 0.50
    assert ap.deblur   == 0.50
