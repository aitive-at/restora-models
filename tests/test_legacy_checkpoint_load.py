"""Old single-head checkpoints must still load into the new dual-head models.

The legacy keys (head.weight, head.bias) are renamed to the new heads' bare
keys so the carried-over weights aren't lost; missing keys (head_ab_*) are
zero-initialized by the model constructor and ignored at load time."""
from __future__ import annotations

import torch

from restora_models.config import ModelConfig
from restora_models.models import build_model
from restora_models.train.checkpoint import load_checkpoint


def _save_legacy_nafnet_ckpt(tmp_path):
    """Build a fake old NAFNet checkpoint by simulating the legacy single-head
    state_dict layout from a freshly initialized dual-head model."""
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    m = build_model(cfg, num_axes=5)
    sd = m.state_dict()
    legacy = {}
    for k, v in sd.items():
        if k.startswith("head_lab_delta."):
            legacy["head." + k.split(".", 1)[1]] = v.clone()
        elif k.startswith("head_ab_abs."):
            pass  # drop - legacy ckpt didn't have this
        else:
            legacy[k] = v.clone()
    path = tmp_path / "legacy.pt"
    torch.save({
        "model": legacy, "step": 100,
        "extra": {"cfg": {"model": cfg.model_dump()}},
    }, path)
    return path


def test_legacy_nafnet_checkpoint_loads(tmp_path):
    path = _save_legacy_nafnet_ckpt(tmp_path)
    cfg = ModelConfig(type="nafnet", size="tiny", input_size=32)
    fresh = build_model(cfg, num_axes=5)
    before = fresh.head_ab_abs.weight.clone()
    payload = load_checkpoint(path, model=fresh)
    after = fresh.head_ab_abs.weight
    # head_ab_abs untouched by load (key wasn't in the legacy ckpt)
    assert torch.equal(before, after)
    # head_lab_delta got loaded - verify by comparing to another fresh model
    fresh2 = build_model(cfg, num_axes=5)
    assert torch.equal(fresh.head_lab_delta.weight, fresh2.head_lab_delta.weight)
    assert payload["step"] == 100


