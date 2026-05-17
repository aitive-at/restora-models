"""Pre-training script for the FlowDistill static-unroll RAFT student.

Distills from torchvision's raft_large (12-iter, ~5.3M params) into our
static-unroll FlowDistill (4-iter, ~4.5M params) so the resulting weights
trace cleanly through ONNX export.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from restora_models.data.builders import build_video_window_dataset
from restora_models.models.flow_distill import FlowDistill


def _build_teacher(device: torch.device) -> torch.nn.Module:
    """Load torchvision RAFT-large with default Sintel weights."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.DEFAULT
    teacher = raft_large(weights=weights, progress=False)
    # nn.Module.eval() — sets eval mode; not Python eval().
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _sample_pairs(batch_frames: torch.Tensor) -> torch.Tensor:
    """Pick two adjacent frames per clip in a (B, T, 3, H, W) tensor.

    Returns (B, 2, 3, H, W) where the two frames are at indices (i, i+1)
    chosen uniformly at random per sample.
    """
    b, t, c, h, w = batch_frames.shape
    if t < 2:
        raise ValueError(f"need T>=2 frames per clip, got T={t}")
    offsets = torch.randint(0, t - 1, (b,), device=batch_frames.device)
    pairs = []
    for i in range(b):
        k = int(offsets[i])
        pairs.append(batch_frames[i, k:k + 2])
    return torch.stack(pairs, dim=0)


def run_flow_distill(
    *,
    out_dir: Path,
    config_path: Path | None,
    steps: int = 5000,
    batch_size: int = 4,
    lr: float = 3e-4,
    log_every: int = 100,
    device: str | None = None,
) -> Path:
    """Pre-train FlowDistill from RAFT-large teacher.

    Returns the path to the final student checkpoint at <out_dir>/final.pt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    if config_path is None:
        raise ValueError("config_path is required so the data sources are defined")
    from restora_models.config import load_config
    cfg = load_config(config_path)
    ds = build_video_window_dataset(cfg.data.sources)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=cfg.data.loader.num_workers,
                        pin_memory=(dev.type == "cuda"), drop_last=True)

    # Models
    teacher = _build_teacher(dev)
    student = FlowDistill(iters=4).to(dev).train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200.0))

    print(f"[flow-distill] teacher: raft_large  student: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    print(f"[flow-distill] training {steps} steps on {len(ds)} clips, bs={batch_size}, lr={lr}")

    step = 0
    start = time.time()
    iter_loader = iter(loader)
    ema_loss = None
    while step < steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
        frames = batch["frames"].to(dev, non_blocking=True)
        pairs = _sample_pairs(frames)
        # raft_large expects image1, image2 as separate (B, 3, H, W) inputs at [0,1] -> internally normalized
        img1 = pairs[:, 0]
        img2 = pairs[:, 1]
        with torch.inference_mode():
            # raft_large returns a list of flow predictions (one per iter); take the final
            target_flow = teacher(img1, img2)[-1]  # (B, 2, H, W)
        pred_flow = student(pairs)
        loss = F.l1_loss(pred_flow, target_flow)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_loss = loss.item() if ema_loss is None else 0.95 * ema_loss + 0.05 * loss.item()
        step += 1
        if step % log_every == 0:
            print(f"[flow-distill] step={step:5d}  loss={loss.item():.4f}  ema={ema_loss:.4f}  elapsed={time.time()-start:.0f}s")

    final = out_dir / "final.pt"
    torch.save({"model": student.state_dict()}, final)
    print(f"[flow-distill] saved {final}")
    return final
