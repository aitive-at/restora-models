"""Pydantic v2 config + YAML loader with chained defaults and !preset tag."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Literal

import os
import yaml
from pydantic import BaseModel, Field, field_validator


# ---------- model ----------------------------------------------------------

class ModelConfig(BaseModel):
    type: str = "nafnet"
    size: Literal["tiny", "large"] = "tiny"
    input_size: int = 256
    nf: int | None = None
    enc_depths: list[int] | None = None
    bottle_blocks: int | None = None
    hidden_dim: int | None = None
    task_embed_dim: int = 128
    # PromptIR-specific overrides (ignored when type != "promptir"):
    prompt_n: int | None = None
    prompt_dim: int | None = None
    prompt_hw: int | None = None
    # Adversarial refine head — optional residual generator that sits after
    # the deterministic dual-head output. Trained with adversarial + perceptual
    # losses for improved perceptual quality on the hard ill-posed tasks
    # (colorize, sharpen). Default OFF for backward compat.
    adversarial_refine: bool = False
    refine_hidden_dim: int | None = None    # default 128 if None
    refine_n_blocks: int | None = None      # default 8 if None


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
    root: str
    val_fraction: float = 0.01
    num_fixed_preview_samples: int = 2
    num_random_preview_samples: int = 1
    augment: AugmentConfig = AugmentConfig()
    loader: LoaderConfig = LoaderConfig()

    @field_validator("root")
    @classmethod
    def _expand_root(cls, v: str) -> str:
        """Expand ~ and $VAR in data.root so configs can be portable.

        Without this, Path('~/data/laion-images') is a literal, and the
        dataset scan finds zero files even when the directory exists.
        """
        return os.path.expandvars(os.path.expanduser(v))


# ---------- compound degradations -----------------------------------------

class AxisProbs(BaseModel):
    colorize: float = 0.5
    denoise: float = 0.5
    sharpen: float = 0.5
    dejpeg: float = 0.5
    deblur: float = 0.5


class CompoundDegradations(BaseModel):
    """Per-axis degradation parameters. Extra fields per-axis allowed for
    forward compatibility."""
    model_config = {"extra": "allow"}
    colorize: dict[str, Any] = Field(default_factory=dict)
    denoise:  dict[str, Any] = Field(default_factory=lambda: {"sigma_range": [0.005, 0.05]})
    sharpen:  dict[str, Any] = Field(default_factory=lambda: {"factor_choices": [2, 4, 8]})
    dejpeg:   dict[str, Any] = Field(default_factory=lambda: {"quality_range": [20, 70]})
    deblur:   dict[str, Any] = Field(default_factory=lambda: {"sigma_range": [1.0, 3.0], "motion_prob": 0.2})


class CompoundConfig(BaseModel):
    identity_prob: float = 0.05
    axis_probs: AxisProbs = AxisProbs()
    degradations: CompoundDegradations = CompoundDegradations()


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
    "full": [
        {"name": "l1_rgb", "weight": 1.0},
        {"name": "perceptual_vgg16bn", "weight": 0.5, "config": {"criterion": "l1"}},
        {"name": "chroma_lab", "weight": 0.10, "apply_to_axes": ["colorize"]},
        {"name": "colorfulness", "weight": 0.05, "apply_to_axes": ["colorize"]},
        {"name": "freq_l1", "weight": 0.30, "apply_to_axes": ["sharpen"]},
        # GAN added AFTER warmup, not from cold — observed to destabilize
        # training when introduced at step 0. Add via curriculum or
        # checkpoint resume; not via the standard preset.
        {"name": "gan", "weight": 0.05, "config": {"gan_type": "hinge"},
         "apply_to_axes": ["colorize", "sharpen"]},
    ],
}


def expand_loss_preset(name: str) -> list[LossConfig]:
    if name not in _LOSS_PRESETS:
        raise ValueError(f"unknown loss preset {name!r}; have {list(_LOSS_PRESETS)}")
    return [LossConfig(**d) for d in _LOSS_PRESETS[name]]


# ---------- optimizer / scheduler -----------------------------------------

class OptimConfig(BaseModel):
    type: Literal["AdamW", "Adam"] = "AdamW"
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
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    preview_every_s: float = 10.0
    preview_history_every: int = 50
    ckpt_every_steps: int = 5000
    val_every_steps: int = 5000
    log_every_steps: int = 25
    # GAN warmup: linearly ramp the gan loss weight from 0 to its configured
    # value over `gan_warmup_steps` steps starting at `gan_warmup_start`.
    # Critical for stability — adding GAN from cold has been observed to
    # destabilize training (see 2026-05-14 iter-N experiments). 0 disables.
    gan_warmup_start: int = 0
    gan_warmup_steps: int = 10000
    # If > 0, also save numbered checkpoints (iter_NNNNNN.pt) at this
    # interval. Unlike last.pt (which is overwritten), these accumulate so
    # you can A/B different training stages or restart from a specific
    # step. 0 disables.
    ckpt_history_every: int = 0


class ExportConfig(BaseModel):
    on_finish: bool = True
    opset: int = 17
    simplify: bool = True
    dynamic_hw: bool = False


class VideoConfig(BaseModel):
    """Optional video-pair training for temporal consistency.

    When `enabled`, the trainer builds a VideoPairDataset over `root` and
    on each training step draws Bernoulli(video_batch_prob): if True, the
    batch comes from the video loader (paired frames + flow), otherwise
    the regular image loader. Video batches populate the temporal_pair
    loss; image batches are unchanged.
    """
    enabled: bool = False
    root: str = ""
    max_skip: int = 5
    hflip_prob: float = 0.5
    require_flow: bool = True
    video_batch_prob: float = 0.25
    batch_size: int | Literal["auto"] = 0    # 0 = match image loader bs
    num_workers: int = 4

    @field_validator("root")
    @classmethod
    def _expand_root(cls, v: str) -> str:
        return os.path.expandvars(os.path.expanduser(v)) if v else v


class RunConfig(BaseModel):
    name: str = ""
    output_dir: str = ""
    seed: int = 0


class Config(BaseModel):
    run: RunConfig = RunConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig
    compound: CompoundConfig = CompoundConfig()
    losses: list[LossConfig]
    optim_g: OptimConfig = OptimConfig()
    optim_d: OptimConfig = OptimConfig(weight_decay=0.0)
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    export: ExportConfig = ExportConfig()
    video: VideoConfig = VideoConfig()


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
