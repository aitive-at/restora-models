"""Simple checkpoint save/load with atomic write."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    ema=None,
    scheduler=None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    discriminator: nn.Module | None = None,
    ema=None,
    scheduler=None,
    map_location="cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if optimizer_d is not None and "optimizer_d" in payload:
        optimizer_d.load_state_dict(payload["optimizer_d"])
    if discriminator is not None and "discriminator" in payload:
        discriminator.load_state_dict(payload["discriminator"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
