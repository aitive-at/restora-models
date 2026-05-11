"""Tests for EMA, checkpoint, preview, and UI smoke."""
import numpy as np
import torch
from torch import nn

from coliraz.train.checkpoint import load_checkpoint, save_checkpoint
from coliraz.train.ema import ModelEMA
from coliraz.train.preview import render_preview_grid, write_png_atomic
from coliraz.train.ui import TrainUI


# ---------- EMA ----------

def test_ema_converges_to_model_after_many_updates():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.fill_(0.0)
    for _ in range(20):
        ema.update(m)
    assert torch.allclose(ema.module.weight, m.weight, atol=1e-3)


def test_ema_state_dict_round_trip():
    m = nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.9)
    sd = ema.state_dict()
    ema2 = ModelEMA(nn.Linear(2, 2), decay=0.9)
    ema2.load_state_dict(sd)
    for k in sd:
        assert torch.equal(ema2.state_dict()[k], sd[k])


# ---------- checkpoint ----------

def test_checkpoint_round_trip(tmp_path):
    m = nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=m, optimizer=opt, step=42, extra={"foo": "bar"})

    m2 = nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    payload = load_checkpoint(path, model=m2, optimizer=opt2)
    assert payload["step"] == 42
    assert payload["extra"]["foo"] == "bar"
    for p, q in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p.data, q.data)


# ---------- preview ----------

def test_render_preview_grid_returns_uint8_image():
    samples = []
    for _ in range(3):
        samples.append(
            {
                "original": torch.rand(3, 32, 32),
                "gray_rgb": torch.rand(3, 32, 32),
                "pred_rgb": torch.rand(3, 32, 32),
                "delta_ab": torch.rand(2, 32, 32),
            }
        )
    img = render_preview_grid(samples, caption="step 100", cell_size=32)
    assert img.dtype == np.uint8
    assert img.shape[1] == 32 * 4
    assert img.shape[2] == 3


def test_preview_write_atomic(tmp_path):
    img = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.png"
    write_png_atomic(p, img)
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()


# ---------- UI smoke ----------

def test_ui_can_render_one_frame():
    ui = TrainUI(run_name="t", total_steps=100, headless=True)
    ui.tick(step=1, losses={"l1_ab": 0.5}, lr=1e-4, throughput_imgs=10.0)
    ui.tick(step=2, losses={"l1_ab": 0.4}, lr=1e-4, throughput_imgs=12.0)
    frame = ui.render()
    assert frame is not None
