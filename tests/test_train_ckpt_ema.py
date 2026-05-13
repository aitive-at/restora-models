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
