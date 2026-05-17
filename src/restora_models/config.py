"""Pydantic v2 config + YAML loader with chained defaults and !preset tag.

The schema is tailored to the temporal old-film remaster design:

- ``ModelConfig`` is minimal — model size is baked into ``type``
  (e.g. ``temporal_restora_small``); only forward-compat hyperparameters
  remain.
- ``DataConfig`` lists video sources via ``sources: list[dict]`` (consumed
  by ``data/builders.py``) and exposes the per-clip / film overlay knobs
  that the trainer uses to compose its degradation pipeline.

Older fields that belonged to the per-frame compound-degradation /
adversarial-refine era have been removed. Old YAMLs / checkpoints that
still mention them will continue to validate because pydantic v2 ignores
unknown fields by default (except where ``extra="allow"`` is set
explicitly).
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------- model ----------------------------------------------------------

class ModelConfig(BaseModel):
    """Model selector.

    ``type`` keys into ``MODEL_REGISTRY`` (e.g. ``temporal_restora_small``).
    For the temporal_restora_* family the size is encoded in the name, so
    no further hyperparameters are needed here. ``input_size`` and
    ``task_embed_dim`` stay for forward-compat with architectures that
    take explicit kwargs.
    """
    type: str = "temporal_restora_small"
    input_size: int = 256
    task_embed_dim: int = 128


# ---------- data -----------------------------------------------------------

class AugmentConfig(BaseModel):
    hflip: bool = True
    rotate90: bool = False


class LoaderConfig(BaseModel):
    batch_size: int | Literal["auto"] = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


class DataConfig(BaseModel):
    """Video data layer config.

    ``sources`` is consumed by ``data.builders.build_video_window_dataset``.
    Each entry is ``{"type": <builder_name>, ...kwargs, "weight": <float>}``.

    The film / clip degradation knobs are sampled per-batch in the trainer
    so the same composite dataset can serve both clean image-pair training
    and old-film simulation.
    """
    sources: list[dict] = Field(default_factory=list)
    val_fraction: float = 0.01
    num_fixed_preview_samples: int = 2
    num_random_preview_samples: int = 1
    augment: AugmentConfig = AugmentConfig()
    loader: LoaderConfig = LoaderConfig()
    # Path to the DeepRemaster noise_data.zip extraction directory.
    # When set, film-overlay degradation is enabled with `film_overlay_prob`.
    film_overlay_root: Path | None = None
    film_overlay_prob: float = 0.3
    film_color_cast_prob: float = 0.2
    gate_weave_prob: float = 0.2
    mpeg_transcode_prob: float = 0.1
    gate_weave_max_shift_px: float = 2.0


# ---------- losses ---------------------------------------------------------

class LossConfig(BaseModel):
    name: str
    weight: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)
    apply_to_axes: list[str] | None = None


_LOSS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "minimal": [{"name": "l1_rgb", "weight": 1.0}],
    # Proven on 2026-05-14 iter-6 (nafnet-large + balanced recipe): all 5
    # tasks improve (colorize +1.26 dB / sharpen +2.60 dB / dejpeg +1.61 dB /
    # deblur +1.69 dB / denoise +0.46 dB over 1000 steps). Earlier
    # experiments with stronger chroma_lab (0.25) crushed easy tasks; with
    # chroma_lab below 0.05 the colorize axis didn't move.
    "standard": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 0.10, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.05, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.30, "apply_to_axes": ["sharpen"]},
    ],
    "vivid": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 0.15, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.08, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.30, "apply_to_axes": ["sharpen"]},
    ],
    # Temporal old-film remaster recipe (Phase 11). Pairs the per-frame
    # spatial losses (l1_rgb / freq_l1 / chroma_lab / colorfulness) with
    # the new temporal losses introduced in Phase 8 (lpips_decoded for a
    # smoother perceptual signal than VGG, temporal_pair to penalise
    # frame-to-frame jitter, central_flicker to suppress single-frame
    # luminance pops in the predicted center frame).
    "temporal_v1": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "lpips_decoded", "weight": 0.4},
        # chroma + colorfulness bumped from 0.2/0.1 -> 0.5/0.3 after the
        # 2026-05-17 E2E validation showed colorize regressing under L1+LPIPS
        # pressure. The dual head's ab_abs path needs stronger supervision
        # to converge on the hardest ill-posed task.
        {"name": "chroma_lab", "weight": 0.5, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.3, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.4, "apply_to_axes": ["sharpen"]},
        {"name": "temporal_pair", "weight": 0.5},
        {"name": "central_flicker", "weight": 0.3},
    ],
}


def expand_loss_preset(name: str) -> list[LossConfig]:
    if name not in _LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; have {list(_LOSS_PRESETS)}")
    return [LossConfig(**d) for d in _LOSS_PRESETS[name]]


# ---------- optimizer / scheduler -----------------------------------------

class OptimConfig(BaseModel):
    type: Literal["AdamW", "Adam", "Muon"] = "Muon"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    fused: bool = True


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "multistep", "constant"] = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 100_000
    milestones: list[int] = Field(default_factory=list)
    gamma: float = 0.5


# ---------- training ------------------------------------------------------

class TrainConfig(BaseModel):
    total_steps: int = 100_000
    amp: Literal["bf16", "fp16", "fp32"] = "bf16"
    memory_format: Literal["channels_last", "contiguous"] = "channels_last"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    ema_decay: float = 0.999
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    seed: int = 0
    log_every: int = 50
    save_every: int = 5000
    # Real-time cadence for refreshing `<run>/samples/latest.png` — too
    # short and preview I/O steals from training; 60 s is comfortable on
    # the local Blackwell.
    preview_every_s: float = 60.0
    # Step-based cadence for archiving `<run>/samples/iter_NNNNNNN.png`
    # snapshots. Defaults to 1000 to match "preview every 1000 steps or so"
    # without flooding `samples/` on multi-hour runs.
    preview_history_every: int = 1000


class ExportConfig(BaseModel):
    on_finish: bool = True
    opset: int = 17
    simplify: bool = True
    dynamic_hw: bool = False


class RunConfig(BaseModel):
    name: str = "default"
    root: Path = Path("runs/")
    output_dir: str = ""
    seed: int = 0


class Config(BaseModel):
    run: RunConfig = RunConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    losses: list[LossConfig]
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()


# ---------- YAML loader ---------------------------------------------------

class _LoaderWithPresets(yaml.SafeLoader):
    pass


def _preset_constructor(loader, node):
    name = loader.construct_scalar(node)
    return [d.copy() for d in _LOSS_PRESETS[name]]


_LoaderWithPresets.add_constructor("!preset", _preset_constructor)


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _interpolate_date(value: str) -> str:
    while "${date:" in value:
        start = value.index("${date:")
        end = value.index("}", start)
        fmt = value[start + 7 : end]
        value = value[:start] + _dt.datetime.now().strftime(fmt) + value[end + 1 :]
    return value


def _load_yaml_chained(path: Path) -> dict:
    with path.open() as fh:
        d = yaml.load(fh, Loader=_LoaderWithPresets) or {}
    defaults = d.pop("defaults", None)
    if defaults:
        parent_path = (path.parent / defaults).resolve()
        parent = _load_yaml_chained(parent_path)
        d = deep_merge(parent, d)
    return d


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    raw = _load_yaml_chained(path)
    if overrides:
        raw = deep_merge(raw, overrides)

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _interpolate_date(x)
        return x

    return Config.model_validate(walk(raw))


# Also re-exported for any callers that still import os to expand paths
__all__ = [
    "AugmentConfig", "Config", "DataConfig", "ExportConfig", "LoaderConfig",
    "LossConfig", "ModelConfig", "OptimConfig", "RunConfig", "SchedulerConfig",
    "TrainConfig", "deep_merge", "expand_loss_preset", "load_config",
]
