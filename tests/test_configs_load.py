"""Smoke-tests for the two shipped configs (local + b200) and base presets.

After the 2026-05-15 cleanup, only `default.yaml`, `large.yaml`, `local.yaml`,
and `b200.yaml` ship. Earlier per-experiment configs (tiny variants, phase
breakouts, h200/promptir) were dropped — see git history for the deleted
files if needed.
"""
from pathlib import Path

from restora_models.config import load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_large_yaml_loads():
    cfg = load_config(ROOT / "large.yaml", overrides={"data": {"root": "/tmp"}})
    assert cfg.model.size == "large"


def test_local_yaml_loads():
    """The local smoke config on RTX PRO 6000 Blackwell. GAN is intentionally
    disabled — the local schedule can't give the GAN enough warmup runway,
    and reconstruction losses are what we want to validate at this scale."""
    cfg = load_config(ROOT / "local.yaml")
    assert cfg.model.type == "nafnet"
    assert cfg.model.size == "large"
    assert cfg.model.adversarial_refine is True
    assert cfg.data.loader.batch_size == 16
    assert cfg.video.enabled is True
    # No GAN in the smoke loss stack
    loss_names = [l.name for l in cfg.losses]
    assert "gan" not in loss_names
    # GAN warmup fields are zeroed (sanity — they'd be no-ops anyway)
    assert cfg.train.gan_warmup_start == 0
    assert cfg.train.gan_warmup_steps == 0


def test_b200_yaml_loads():
    """The production config: 500k steps, bs=96, full pipeline, B200."""
    cfg = load_config(ROOT / "b200.yaml")
    assert cfg.model.type == "nafnet"
    assert cfg.model.size == "large"
    assert cfg.model.adversarial_refine is True
    assert cfg.train.total_steps == 500_000
    # bs=64 — cut from 96 after 2026-05-15 OOM on B200 (178 GB)
    assert cfg.data.loader.batch_size == 64
    assert cfg.train.gan_warmup_start == 50_000
    # Numbered ckpts written every 50k steps so eval_checkpoints can A/B
    assert cfg.train.ckpt_history_every == 50_000
    assert cfg.video.enabled is True


def test_local_and_b200_share_non_gan_losses():
    """The non-GAN portion of the loss stack must be identical so smoke
    reconstruction deltas predict production behavior. b200 additionally
    carries the GAN loss; local does not (see test_local_yaml_loads)."""
    local = load_config(ROOT / "local.yaml")
    b200 = load_config(ROOT / "b200.yaml")
    local_w = {l.name: l.weight for l in local.losses if l.name != "gan"}
    b200_w  = {l.name: l.weight for l in b200.losses  if l.name != "gan"}
    assert local_w == b200_w
    # And the b200 stack does carry GAN
    assert "gan" in {l.name for l in b200.losses}


def test_local_and_b200_share_axis_probs():
    """Same reason — smoke must use the same axis-sample distribution."""
    local = load_config(ROOT / "local.yaml")
    b200 = load_config(ROOT / "b200.yaml")
    assert local.compound.axis_probs.model_dump() == b200.compound.axis_probs.model_dump()


def test_default_axis_probs_uniform_half():
    """default.yaml ships uniform 0.50. Proven on 2026-05-14: uniform 0.5
    outperforms the 0.75/0.40 split (which caused multi-task contention).
    Production configs override colorize → 0.65 (hardest axis) but keep
    others at 0.5."""
    cfg = load_config(ROOT / "default.yaml", overrides={"data": {"root": "/tmp"}})
    ap = cfg.compound.axis_probs
    assert ap.colorize == 0.50
    assert ap.sharpen  == 0.50
    assert ap.denoise  == 0.50
    assert ap.dejpeg   == 0.50
    assert ap.deblur   == 0.50


def test_data_root_expands_tilde():
    """Regression: data.root in YAML can use ~ — it must expand to $HOME at
    load time so Path() / directory walks work."""
    import os
    from restora_models.config import DataConfig, LoaderConfig, AugmentConfig
    home = os.path.expanduser("~")
    cfg = DataConfig(root="~/data/laion-images",
                     loader=LoaderConfig(), augment=AugmentConfig())
    assert cfg.root == f"{home}/data/laion-images"
    assert not cfg.root.startswith("~")


def test_data_root_expands_env_var(monkeypatch):
    monkeypatch.setenv("REFINE_TEST_DATA_DIR", "/tmp/refine-test-data")
    from restora_models.config import DataConfig, LoaderConfig, AugmentConfig
    cfg = DataConfig(root="$REFINE_TEST_DATA_DIR/sub",
                     loader=LoaderConfig(), augment=AugmentConfig())
    assert cfg.root == "/tmp/refine-test-data/sub"
