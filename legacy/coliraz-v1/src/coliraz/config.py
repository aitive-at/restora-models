"""Pydantic v2 config models + YAML loader with chained defaults and !preset tag."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------- model ----------------------------------------------------------

class ModelConfig(BaseModel):
    size: Literal["tiny", "large"] = "tiny"
    input_size: int = 256
    num_queries: int = 100
    num_scales: int = 3
    dec_layers: int = 9
    nf: int = 512
    hidden_dim: int = 256
    refine_norm: Literal["spectral", "batch", "none"] = "spectral"
    encoder_variant: str | None = None


# ---------- data -----------------------------------------------------------

class AugmentConfig(BaseModel):
    hflip: bool = True
    rotate90: bool = False
    color_jitter: bool = False


class LoaderConfig(BaseModel):
    batch_size: int | Literal["auto"] = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


class DataConfig(BaseModel):
    root: str
    val_fraction: float = 0.01
    num_fixed_preview_samples: int = 4
    num_random_preview_samples: int = 2
    augment: AugmentConfig = AugmentConfig()
    loader: LoaderConfig = LoaderConfig()


# ---------- losses ---------------------------------------------------------

class LossConfig(BaseModel):
    name: str
    weight: float = 1.0
    config: dict[str, Any] = Field(default_factory=dict)


_LOSS_PRESETS: dict[str, list[dict[str, Any]]] = {
    "minimal": [{"name": "l1_ab", "weight": 1.0}],
    "standard": [
        {"name": "l1_ab", "weight": 0.1},
        {
            "name": "perceptual_vgg16bn",
            "weight": 5.0,
            "config": {
                "layer_weights": {
                    "conv1_1": 0.0625,
                    "conv2_1": 0.125,
                    "conv3_1": 0.25,
                    "conv4_1": 0.5,
                    "conv5_1": 1.0,
                },
                "criterion": "l1",
            },
        },
        {"name": "colorfulness", "weight": 0.5},
    ],
    "stable": [
        {"name": "charbonnier_ab", "weight": 0.1},
        {"name": "perceptual_vgg16bn", "weight": 5.0, "config": {"criterion": "l1"}},
    ],
    "ddcolor_full": [
        {"name": "l1_ab", "weight": 0.1},
        {
            "name": "perceptual_vgg16bn",
            "weight": 5.0,
            "config": {"criterion": "l1"},
        },
        {
            "name": "gan",
            "weight": 1.0,
            "config": {"gan_type": "hinge", "discriminator": {"type": "unet", "nf": 64}},
        },
        {"name": "colorfulness", "weight": 0.5},
    ],
}


def expand_loss_preset(name: str) -> list[LossConfig]:
    if name not in _LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; have {list(_LOSS_PRESETS)}")
    return [LossConfig(**d) for d in _LOSS_PRESETS[name]]


# ---------- optimizer / scheduler -----------------------------------------

class OptimConfig(BaseModel):
    type: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    fused: bool = True


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "multistep", "constant"] = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 400_000
    milestones: list[int] = Field(default_factory=list)
    gamma: float = 0.5


# ---------- training ------------------------------------------------------

class TrainConfig(BaseModel):
    total_steps: int = 400_000
    amp: Literal["bf16", "fp16", "fp32"] = "bf16"
    memory_format: Literal["channels_last", "contiguous"] = "channels_last"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    ema_decay: float = 0.999
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    preview_every_s: float = 10.0
    preview_history_every: int = 10
    ckpt_every_steps: int = 5000
    val_every_steps: int = 5000
    log_every_steps: int = 25
    color_enhance: bool = True
    color_enhance_factor: float = 1.2


class ExportConfig(BaseModel):
    on_finish: bool = True
    opset: int = 17
    simplify: bool = True
    # When true, training also emits model_dynamic.onnx with H/W as dynamic
    # axes in addition to model.onnx (fixed at model.input_size).
    dynamic_hw: bool = False


class RunConfig(BaseModel):
    name: str = ""
    output_dir: str = ""
    seed: int = 0


# ---------- root ----------------------------------------------------------

class Config(BaseModel):
    run: RunConfig = RunConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig
    losses: list[LossConfig]
    optim_g: OptimConfig = OptimConfig()
    optim_d: OptimConfig = OptimConfig(weight_decay=0.0)
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()


# ---------- loader --------------------------------------------------------

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
        stamp = _dt.datetime.now().strftime(fmt)
        value = value[:start] + stamp + value[end + 1 :]
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

    raw = walk(raw)
    return Config.model_validate(raw)
