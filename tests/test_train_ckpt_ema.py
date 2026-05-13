import json

import torch
from torch import nn

from refine.train.checkpoint import load_checkpoint, save_checkpoint
from refine.train.ema import ModelEMA


def test_ema_converges():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0); m.bias.fill_(0.0)
    for _ in range(20):
        ema.update(m)
    assert torch.allclose(ema.module.weight, m.weight, atol=1e-3)


def test_ckpt_with_task_map(tmp_path):
    m = nn.Linear(4, 2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=m, step=10, extra={"foo": "bar"},
                    task_map={"tasks": {"colorize": 0, "sr_x4": 1}})
    sidecar = path.with_suffix(".task_map.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert data["tasks"]["colorize"] == 0

    m2 = nn.Linear(4, 2)
    payload = load_checkpoint(path, model=m2)
    assert payload["step"] == 10
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p.data, q.data)


def test_ema_works_with_torch_compile():
    """Regression: torch.compile wraps the model in OptimizedModule whose
    state_dict prefixes keys with '_orig_mod.'. EMA must unwrap before
    looking up parameters or it raises KeyError."""
    m = nn.Linear(4, 2)
    ema = ModelEMA(m, decay=0.9)
    # Simulate the trainer's order: build EMA, THEN compile the model.
    try:
        compiled = torch.compile(m)
    except Exception:
        import pytest
        pytest.skip("torch.compile unavailable in this environment")
    # The compiled module's state_dict has '_orig_mod.' prefixes.
    assert any(k.startswith("_orig_mod.") for k in compiled.state_dict()), \
        "compile didn't add the expected prefix — test premise broken"
    # EMA.update must still work — this is the failing path from the bug report.
    ema.update(compiled)
    # And the saved checkpoint state_dict must have bare keys (so it loads
    # back into either a compiled or non-compiled model).
    from refine.train.checkpoint import save_checkpoint
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ckpt.pt"
        save_checkpoint(path, model=compiled, step=1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert all(not k.startswith("_orig_mod.") for k in payload["model"]), \
            "saved checkpoint has _orig_mod. keys — won't load on non-compiled model"
