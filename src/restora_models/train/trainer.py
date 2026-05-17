"""Trainer for the temporal restoration model.

Forward contract: model(frames [B,7,3,H,W], config [B,5]) -> pred [B,3,H,W]
where pred is the restored center frame (index 3 of the 7-frame window).

Composes degradations inline in `_degrade_batch` (per-frame axes +
per-clip film/codec layers) rather than via the legacy dataset wrapper,
because the per-clip layers (gate weave, mpeg) need the full clip.
"""
from __future__ import annotations

import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from restora_models.data.builders import build_video_window_dataset
from restora_models.data.compound import AXES, DEGRADE_ORDER
from restora_models.data.degradations.film_color_cast import FilmColorCastDegradation
from restora_models.data.degradations.film_overlay import FilmOverlayDegradation
from restora_models.data.degradations.gate_weave import GateWeaveDegradation
from restora_models.data.degradations.mpeg_transcode import MpegTranscodeDegradation
from restora_models.data.degradations.registry import Degradation, build_degradation
# Ensure each degradation module is imported so the registry is populated
from restora_models.data.degradations import colorization as _colorization  # noqa: F401
from restora_models.data.degradations import deblur as _deblur  # noqa: F401
from restora_models.data.degradations import denoise as _denoise  # noqa: F401
from restora_models.data.degradations import jpeg as _jpeg  # noqa: F401
from restora_models.data.degradations import superres as _superres  # noqa: F401
from restora_models.losses import LossSet
from restora_models.losses.metrics import psnr as psnr_per_sample
from restora_models.losses.registry import LossContext
from restora_models.models.registry import build_model
from restora_models.models.temporal_align_stem import TemporalAlignStem

from .checkpoint import save_checkpoint
from .ema import ModelEMA
from .preview import (
    make_temporal_preview_samples,
    render_multitask_grid,
    write_png_atomic,
)
from .ui import TrainUI

CENTER_INDEX = TemporalAlignStem.CENTER_INDEX  # 3
NUM_FRAMES = TemporalAlignStem.NUM_FRAMES      # 7

# Map task axis -> registry name for per-frame degradations.
_AXIS_TO_REG = {
    "colorize": "colorize",
    "denoise":  "denoise",
    "sharpen":  "sharpen",
    "dejpeg":   "jpeg",
    "deblur":   "deblur",
}


def _build_per_frame_degradations() -> dict[str, Degradation]:
    """Instantiate one Degradation per axis from the registry."""
    deg_cfg = {
        "colorize": {},
        "denoise":  {"sigma_range": [0.005, 0.05]},
        "sharpen":  {"factor_choices": [2, 4, 8]},
        "dejpeg":   {"quality_range": [20, 70]},
        "deblur":   {"sigma_range": [1.0, 3.0], "motion_prob": 0.2},
    }
    return {axis: build_degradation(_AXIS_TO_REG[axis], deg_cfg[axis]) for axis in AXES}


def _apply_per_frame_degradations(
    clip: torch.Tensor,
    active_axes: set[str],
    per_frame_degs: dict[str, Degradation],
    rng: random.Random,
) -> torch.Tensor:
    """Apply each active axis to every frame in the clip.

    clip: (T,3,H,W) float in [0,1] on CPU. Returns same-shape degraded clip.
    Degradations run in real-world causal order (blur -> noise -> downsample
    -> jpeg -> grayscale).
    """
    if not active_axes:
        return clip.clone()
    out_frames = []
    for k in range(clip.shape[0]):
        np_img = clip[k].permute(1, 2, 0).contiguous().numpy()
        for axis in DEGRADE_ORDER:
            if axis in active_axes:
                np_img = per_frame_degs[axis].degrade(np_img, rng)
        out_frames.append(torch.from_numpy(np_img.transpose(2, 0, 1)).contiguous())
    return torch.stack(out_frames, dim=0)


def _apply_per_frame_single(
    clip: torch.Tensor, deg: Degradation, rng: random.Random,
) -> torch.Tensor:
    """Run a single Degradation over every frame of the clip."""
    out_frames = []
    for k in range(clip.shape[0]):
        np_img = clip[k].permute(1, 2, 0).contiguous().numpy()
        np_img = deg.degrade(np_img, rng)
        out_frames.append(torch.from_numpy(np_img.transpose(2, 0, 1)).contiguous())
    return torch.stack(out_frames, dim=0)


def _make_config_vec(active: set[str]) -> torch.Tensor:
    vec = torch.zeros(len(AXES))
    for i, ax in enumerate(AXES):
        if ax in active:
            vec[i] = 1.0
    return vec


def _sample_axes(rng: random.Random, identity_prob: float = 0.15) -> set[str]:
    """Sample a task set with balanced single/compound/identity distribution.

    Targets (per user direction): the model must handle modern footage
    needing single restoration tasks, modern footage needing compound
    restoration, and clean-modern footage that needs NO changes (identity).

    Distribution:
      - identity_prob   (default 15%): no axes -> output must equal input
      - 35%             single random axis
      - 35%             2 axes  (compound light)
      - 15%             3+ axes (compound heavy, up to all 5)
    """
    r = rng.random()
    if r < identity_prob:
        return set()
    remaining = 1.0 - identity_prob
    # Renormalize the three buckets
    p_single = 0.35 / remaining
    p_two    = 0.35 / remaining
    r2 = rng.random()
    if r2 < p_single:
        n = 1
    elif r2 < p_single + p_two:
        n = 2
    else:
        n = rng.randint(3, len(AXES))
    return set(rng.sample(AXES, n))


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                     *, prefer_muon: bool = False):
    """AdamW by default; Muon opt-in via prefer_muon=True.

    Muon's Newton-Schulz update requires `view(N, -1)` on 4D conv kernels,
    which fails when the tensor is laid out as channels_last (non-contiguous
    strides). Until that interaction is fixed upstream, AdamW is the safe
    default. Set ``cfg.train.optimizer = "muon"`` to opt in once stable.
    """
    if not prefer_muon:
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        ), "adamw"

    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError:
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        ), "adamw"

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matmul = (p.ndim >= 2
                     and "norm" not in name.lower()
                     and "embed" not in name.lower())
        (muon_params if is_matmul else adam_params).append(p)

    param_groups = [
        dict(params=muon_params, lr=lr, momentum=0.95,
             weight_decay=weight_decay, use_muon=True),
        dict(params=adam_params, lr=lr * 0.5, betas=(0.9, 0.95),
             eps=1e-10, weight_decay=weight_decay, use_muon=False),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups), "muon"


@dataclass
class _BatchDegradations:
    """Output of `_degrade_batch`: tensors + the per-sample axis labels."""
    degraded: torch.Tensor      # (B, T, 3, H, W)
    config: torch.Tensor        # (B, num_axes)
    axes_active: list[str]


class Trainer:
    """Temporal restoration trainer.

    Lifecycle: ``Trainer(cfg, out_dir=...).fit()``. The output dir is
    created automatically; checkpoints land in ``<out_dir>/<run.name>/``.
    """

    def __init__(self, cfg, *, device: torch.device | None = None,
                 out_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # When the caller passes an explicit `out_dir`, treat it as the
        # final destination. Otherwise build `<run.root>/<run.name>/`.
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        else:
            self.out_dir = Path(cfg.run.root) / cfg.run.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.train.seed)

        # Model
        self.model = build_model(cfg.model, num_axes=len(AXES)).to(self.device)
        # channels_last is a 4D layout — the model's internal convs still
        # benefit from it after TemporalAlignStem flattens the 5D clip to
        # 4D features, but we don't apply it on the raw 5D batch tensor.
        if cfg.train.memory_format == "channels_last":
            self.model = self.model.to(memory_format=torch.channels_last)
        if cfg.train.compile:
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)

        # Loss aggregator
        self.loss_set = LossSet(cfg.losses)
        self.loss_set.to(self.device)

        # Optimizer (AdamW by default; Muon is opt-in via cfg.train.optimizer="muon")
        prefer_muon = getattr(cfg.train, "optimizer", "adamw").lower() == "muon"
        self.optimizer, self.optimizer_kind = _build_optimizer(
            self.model, cfg.train.lr, cfg.train.weight_decay,
            prefer_muon=prefer_muon)

        # LR scheduler: linear warmup + cosine decay to 1e-2 * base_lr.
        # Using LambdaLR for full control; no extra dep.
        import math
        warmup_steps = max(1, int(cfg.scheduler.warmup_steps))
        total_steps = max(warmup_steps + 1, int(cfg.scheduler.total_steps))
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            # cosine to 0.01 * base
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        # EMA shadow (optional)
        self.ema = (ModelEMA(self.model, decay=cfg.train.ema_decay)
                    if cfg.train.ema_decay > 0 else None)

        # Composite video dataset + loader
        self.train_ds = build_video_window_dataset(cfg.data.sources)
        loader_cfg = cfg.data.loader
        bs = loader_cfg.batch_size if loader_cfg.batch_size != "auto" else 8
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=int(bs),
            num_workers=loader_cfg.num_workers,
            shuffle=True,
            pin_memory=loader_cfg.pin_memory and self.device.type == "cuda",
            persistent_workers=(loader_cfg.num_workers > 0
                                and loader_cfg.persistent_workers),
            prefetch_factor=(loader_cfg.prefetch_factor
                             if loader_cfg.num_workers > 0 else None),
            drop_last=True,
        )

        # Per-frame degradations (one Degradation per axis)
        self.per_frame_degs = _build_per_frame_degradations()

        # Film overlay (optional — needs noise_data.zip extraction)
        self.film_overlay: FilmOverlayDegradation | None = None
        if cfg.data.film_overlay_root is not None:
            root = Path(cfg.data.film_overlay_root)
            if root.exists():
                self.film_overlay = FilmOverlayDegradation.from_dir(root)
        self.film_overlay_prob = float(cfg.data.film_overlay_prob)

        # Film color cast (per-frame, no asset dependency)
        self.film_color_cast = FilmColorCastDegradation()
        self.film_color_cast_prob = float(cfg.data.film_color_cast_prob)

        # Per-clip degradations
        self.gate_weave = GateWeaveDegradation(
            max_shift_px=cfg.data.gate_weave_max_shift_px)
        self.gate_weave_prob = float(cfg.data.gate_weave_prob)
        self.mpeg: MpegTranscodeDegradation | None = None
        if shutil.which("ffmpeg") is not None:
            self.mpeg = MpegTranscodeDegradation()
        self.mpeg_prob = float(cfg.data.mpeg_transcode_prob)

        self.step = 0

        # ---- live UI + preview wiring ----------------------------------
        # Headless when stdout isn't a TTY (e.g. `nohup > log 2>&1 &`) so
        # rich doesn't try to redraw into a redirected file. The dashboard
        # downgrades to one-line stdout updates in that mode.
        headless = not sys.stdout.isatty()
        self.ui = TrainUI(
            run_name=cfg.run.name or "run",
            total_steps=int(cfg.train.total_steps),
            headless=headless,
            task_names=list(AXES),
        )
        # Seed-shuffle a fixed set of dataset indices used for previews so
        # the same scenes show up each iteration — easy to eyeball progress
        # frame-over-frame.
        n_fixed = int(cfg.data.num_fixed_preview_samples)
        n_random = int(cfg.data.num_random_preview_samples)
        ds_len = max(1, len(self.train_ds))
        rng_prev = random.Random(cfg.train.seed)
        self._preview_indices: list[int] = [
            i % ds_len for i in range(min(n_fixed, ds_len))
        ]
        for _ in range(n_random):
            self._preview_indices.append(rng_prev.randrange(ds_len))
        self._preview_seed = int(cfg.train.seed)
        self._last_preview_t: float = 0.0
        self._last_preview_step: int = -1
        self._t_window: float = time.perf_counter()
        self._samples_window: int = 0
        self._last_per_axis_psnr: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Per-batch degradation pipeline
    # ------------------------------------------------------------------

    def _degrade_batch(self, clean_clips: torch.Tensor,
                       rng: random.Random) -> _BatchDegradations:
        """Sample axes + apply degradations per-sample.

        clean_clips: (B, T, 3, H, W) on `self.device`. The degradation
        pipeline runs on CPU (numpy / opencv heavy) and the result is
        moved back to the device.
        """
        b, t, c, h, w = clean_clips.shape
        degraded_out = torch.empty_like(clean_clips, device="cpu")
        config_out = torch.empty(b, len(AXES))
        axes_active: list[str] = []
        cpu_clips = clean_clips.detach().cpu()
        for i in range(b):
            clip = cpu_clips[i]
            active = _sample_axes(rng)
            # Per-frame degradations (the 5 standard axes)
            clip = _apply_per_frame_degradations(clip, active,
                                                 self.per_frame_degs, rng)
            # Optional per-frame film overlay (real grain / dust textures)
            if self.film_overlay is not None and rng.random() < self.film_overlay_prob:
                clip = _apply_per_frame_single(clip, self.film_overlay, rng)
            # Optional per-frame film color cast (sepia / cyan fade / etc.)
            if rng.random() < self.film_color_cast_prob:
                clip = _apply_per_frame_single(clip, self.film_color_cast, rng)
            # Per-clip degradations (need the full clip in one shot)
            if self.gate_weave_prob > 0 and rng.random() < self.gate_weave_prob:
                clip = self.gate_weave.apply_clip(clip)
            if (self.mpeg is not None
                    and self.mpeg_prob > 0
                    and rng.random() < self.mpeg_prob):
                clip = self.mpeg.apply_clip(clip)
            degraded_out[i] = clip
            config_out[i] = _make_config_vec(active)
            axes_active.append("+".join(sorted(active)) or "identity")
        return _BatchDegradations(
            degraded=degraded_out.to(clean_clips.device).to(clean_clips.dtype),
            config=config_out.to(clean_clips.device),
            axes_active=axes_active,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, rng: random.Random) -> dict[str, float]:
        clean_clips = batch["frames"].to(self.device, non_blocking=True)
        deg = self._degrade_batch(clean_clips, rng)
        target = clean_clips[:, CENTER_INDEX]

        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[self.cfg.train.amp]
        autocast_enabled = (amp_dtype != torch.float32
                            and self.device.type == "cuda")

        with torch.amp.autocast(self.device.type, dtype=amp_dtype,
                                enabled=autocast_enabled):
            pred = self.model(deg.degraded, deg.config)
            ctx = LossContext(
                pred_rgb=pred,
                clean_rgb=target,
                degraded_rgb=deg.degraded[:, CENTER_INDEX],
                config=deg.config,
                axes_active=deg.axes_active,
            )
            loss, loss_log = self.loss_set(ctx)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.train.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)

        # Per-axis PSNR for the live dashboard. Mask is taken from the
        # per-sample config vector so each sample contributes only to the
        # axes that were active for it (identity samples contribute to
        # nothing here — they have their own dedicated preview row).
        with torch.no_grad():
            per_sample = psnr_per_sample(pred, target)
            self._last_per_axis_psnr = {}
            for ax_idx, axis in enumerate(AXES):
                mask = deg.config[:, ax_idx] >= 0.5
                if mask.any():
                    self._last_per_axis_psnr[axis] = float(
                        per_sample[mask].mean().item())

        return {"loss": float(loss.detach()), "lr": self.scheduler.get_last_lr()[0], **loss_log}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> Path:
        """Run the full training loop. Returns the path to final.pt."""
        rng = random.Random(self.cfg.train.seed)
        total_steps = int(self.cfg.train.total_steps)
        log_every = int(self.cfg.train.log_every)
        save_every = int(self.cfg.train.save_every)
        preview_every_s = float(self.cfg.train.preview_every_s)

        loader_iter = iter(self.train_loader)
        bs = int(self.train_loader.batch_size or 1)
        with self.ui:
            # Skip the initial preview burst by seeding the timer to "now"
            # offset by the cadence — first preview lands after
            # `preview_every_s` real-time seconds, not on step 0.
            self._last_preview_t = time.perf_counter()
            self._t_window = time.perf_counter()
            self._samples_window = 0

            while self.step < total_steps:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                log = self._train_step(batch, rng)
                self.step += 1
                self._samples_window += bs

                if log_every > 0 and self.step % log_every == 0:
                    now = time.perf_counter()
                    dt = max(1e-6, now - self._t_window)
                    imgs_per_s = self._samples_window / dt
                    self._t_window = now
                    self._samples_window = 0
                    self.ui.tick(
                        step=self.step,
                        losses=log,
                        lr=float(log.get("lr", 0.0)),
                        throughput_imgs=imgs_per_s,
                        per_task_psnr=self._last_per_axis_psnr or None,
                    )

                if (preview_every_s > 0
                        and time.perf_counter() - self._last_preview_t
                        >= preview_every_s):
                    self._write_preview()

                if save_every > 0 and self.step % save_every == 0:
                    ck_path = self.out_dir / f"iter_{self.step:07d}.pt"
                    save_checkpoint(ck_path, model=self.model,
                                    optimizer=self.optimizer, ema=self.ema,
                                    step=self.step,
                                    extra={"cfg": self.cfg.model_dump(mode="json")})

            # Final checkpoint + one closing preview so the gallery has the
            # post-final-step state without waiting for the cadence timer.
            final_path = self.out_dir / "final.pt"
            save_checkpoint(final_path, model=self.model,
                            optimizer=self.optimizer, ema=self.ema,
                            step=self.step,
                            extra={"cfg": self.cfg.model_dump(mode="json")})
            self._write_preview(force_history=True)
            return final_path

    def _write_preview(self, *, force_history: bool = False) -> None:
        """Render `<run>/samples/latest.png` and optionally a step-tagged
        history snapshot. Errors are surfaced into the UI status line
        instead of crashing the trainer — the run keeps going even if
        previews fail (e.g. transient GPU OOM during inference)."""
        try:
            eval_model = self.ema.module if self.ema is not None else self.model
            was_training = eval_model.training
            eval_model.train(False)
            try:
                samples = make_temporal_preview_samples(
                    model=eval_model,
                    dataset=self.train_ds,
                    device=self.device,
                    sample_indices=self._preview_indices,
                    seed=self._preview_seed,
                )
            finally:
                eval_model.train(was_training)
            caption = f"step {self.step}  ts {time.strftime('%H:%M:%S')}"
            cell = int(getattr(self.cfg.model, "input_size", 256) or 256)
            grid = render_multitask_grid(samples, caption=caption, cell_size=cell)
            latest = self.out_dir / "samples" / "latest.png"
            write_png_atomic(latest, grid)
            phe = int(self.cfg.train.preview_history_every)
            if phe > 0 or force_history:
                crossed = (force_history
                           or self._last_preview_step < 0
                           or (self.step // max(1, phe))
                           > (max(0, self._last_preview_step) // max(1, phe)))
                if crossed:
                    hist = self.out_dir / "samples" / f"iter_{self.step:07d}.png"
                    write_png_atomic(hist, grid)
            self._last_preview_step = self.step
            try:
                rel = latest.relative_to(self.out_dir)
            except ValueError:
                rel = latest
            self.ui.note_preview(f"wrote {rel} @ step {self.step}")
        except Exception as exc:
            self.ui.note_preview(f"preview error: {exc}")
        finally:
            self._last_preview_t = time.perf_counter()


# ----------------------------------------------------------------------
# Programmatic entry points
# ----------------------------------------------------------------------

def run_train_stage(
    *,
    out_dir: Path,
    config_path: Path | None,
    flow_estimator_ckpt: Path | None = None,
    warm_start: Path | None = None,
    freeze: tuple[str, ...] = (),
    lr_scale: float = 1.0,
) -> Path:
    """Run one training stage. Returns the path to final.pt.

    Stub kwargs (`warm_start`, `flow_estimator_ckpt`, `freeze`, `lr_scale`)
    will be wired up in Phase 14 (distillation) and Phase 18 (orchestrator).
    For now the function ignores them and runs a vanilla `Trainer.fit()`.
    """
    from restora_models.config import load_config
    if config_path is None:
        raise ValueError("config_path is required for run_train_stage")
    cfg = load_config(config_path)
    trainer = Trainer(cfg, out_dir=out_dir)
    return trainer.fit()


def fit(cfg, *, device: torch.device | None = None) -> Path:
    """Legacy callable kept for back-compat with the existing CLI."""
    return Trainer(cfg, device=device).fit()
