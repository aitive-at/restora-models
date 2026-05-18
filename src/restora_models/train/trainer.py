"""Trainer for the temporal restoration model.

Forward contract: model(frames [B,7,3,H,W], config [B,5]) -> pred [B,3,H,W]
where pred is the restored center frame (index 3 of the 7-frame window).

Degradation pipeline runs **inside DataLoader workers** via
:class:`restora_models.data.compound_wrapper.CompoundDegradationWrapper`.
That move was a ~6x throughput win — the legacy in-trainer
``_degrade_batch`` was running ~250 numpy/opencv ops per step on the
main process between forward passes, leaving the GPU ~60% idle. See
``data/compound_wrapper.py`` if you need to inspect the actual
degradation logic; it used to live here.
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from restora_models.data.builders import build_compound_video_dataset
from restora_models.data.compound import AXES
from restora_models.data.compound_wrapper import (
    collate_compound, compound_worker_init_fn,
)
from restora_models.losses import LossSet
from restora_models.losses.metrics import (
    find_lpips_model, lpips_per_sample, psnr as psnr_per_sample,
)
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
from .tb import TensorBoardWriter
from .ui import TrainUI

CENTER_INDEX = TemporalAlignStem.CENTER_INDEX  # 3
NUM_FRAMES = TemporalAlignStem.NUM_FRAMES      # 7


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


class Trainer:
    """Temporal restoration trainer.

    Lifecycle: ``Trainer(cfg, out_dir=...).fit()``. The output dir is
    created automatically; checkpoints land in ``<out_dir>/<run.name>/``.
    """

    def __init__(self, cfg, *, device: torch.device | None = None,
                 out_dir: Path | None = None) -> None:
        # Startup logging — init is slow (model build + first-batch compile
        # warmup can easily take 5-10 min on big GPUs) and a silent trainer
        # is indistinguishable from a hung one. Each major step is logged
        # with cumulative wall-clock so a slow stage is obvious.
        t0 = time.perf_counter()
        def _ilog(msg: str) -> None:
            print(f"[trainer +{time.perf_counter() - t0:5.1f}s] {msg}", flush=True)

        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            name = torch.cuda.get_device_name(self.device)
            _ilog(f"device: {self.device} ({name}, sm_{cap[0]}{cap[1]}); "
                  f"torch={torch.__version__}, cuda={torch.version.cuda}")
        else:
            _ilog(f"device: {self.device} (CPU — no GPU available)")
        # When the caller passes an explicit `out_dir`, treat it as the
        # final destination. Otherwise build `<run.root>/<run.name>/`.
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        else:
            self.out_dir = Path(cfg.run.root) / cfg.run.name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _ilog(f"output dir: {self.out_dir}")
        torch.manual_seed(cfg.train.seed)

        # Model
        _ilog(f"building model: {cfg.model.type}")
        self.model = build_model(cfg.model, num_axes=len(AXES)).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        _ilog(f"  params: {n_params/1e6:.1f} M; moved to {self.device}")
        # channels_last is a 4D layout — the model's internal convs still
        # benefit from it after TemporalAlignStem flattens the 5D clip to
        # 4D features, but we don't apply it on the raw 5D batch tensor.
        if cfg.train.memory_format == "channels_last":
            self.model = self.model.to(memory_format=torch.channels_last)
            _ilog("  memory_format: channels_last")
        if cfg.train.compile:
            _ilog(f"  torch.compile(mode={cfg.train.compile_mode}) wrapper attached "
                  "— actual graph compile happens on first forward (can take "
                  "minutes; nvrtc errors here mean PyTorch's CUDA toolchain is "
                  "too old for this GPU's compute capability — try --no-compile)")
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)
        else:
            _ilog("  torch.compile: DISABLED (eager mode)")

        # Loss aggregator
        _ilog("building loss set + LPIPS model")
        self.loss_set = LossSet(cfg.losses)
        self.loss_set.to(self.device)
        # Reuse the LPIPS model from the loss set for per-axis LPIPS metric
        # rather than instantiating a second VGG (saves ~500 MB).
        self._lpips_model = find_lpips_model(self.loss_set)

        # Optimizer (AdamW by default; Muon is opt-in via cfg.train.optimizer="muon")
        prefer_muon = getattr(cfg.train, "optimizer", "adamw").lower() == "muon"
        self.optimizer, self.optimizer_kind = _build_optimizer(
            self.model, cfg.train.lr, cfg.train.weight_decay,
            prefer_muon=prefer_muon)
        _ilog(f"optimizer: {self.optimizer_kind}, lr={cfg.train.lr}")

        # LR scheduler: linear warmup + cosine decay to 1e-2 * base_lr.
        warmup_steps = max(1, int(cfg.scheduler.warmup_steps))
        total_steps = max(warmup_steps + 1, int(cfg.scheduler.total_steps))
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        # EMA shadow (optional)
        self.ema = (ModelEMA(self.model, decay=cfg.train.ema_decay)
                    if cfg.train.ema_decay > 0 else None)

        # Composite video dataset + loader. The wrapper runs the
        # degradation pipeline inside each worker (see
        # compound_wrapper.py), so the main process never sees raw
        # numpy/opencv work between forward passes.
        _ilog("building composite video dataset (scanning sources)…")
        self.train_ds = build_compound_video_dataset(
            cfg.data.sources, data_cfg=cfg.data, seed=cfg.train.seed)
        _ilog(f"  dataset: {len(self.train_ds)} windows total")
        loader_cfg = cfg.data.loader
        bs = loader_cfg.batch_size if loader_cfg.batch_size != "auto" else 8
        _ilog(f"building dataloader (bs={bs}, workers={loader_cfg.num_workers}, "
              f"prefetch={loader_cfg.prefetch_factor})")
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
            collate_fn=collate_compound,
            worker_init_fn=compound_worker_init_fn,
        )

        self.step = 0
        _ilog(f"init done in {time.perf_counter() - t0:.1f}s — "
              "first batch will trigger dataloader-worker spin-up "
              + ("+ torch.compile JIT (this is the slowest single step)"
                 if cfg.train.compile else "")
              + "; subsequent steps will be much faster")

        # ---- live UI + preview + tensorboard wiring --------------------
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
        self.tb = TensorBoardWriter(self.out_dir)

        # Preview indices: pull a fixed set of dataset indices for the
        # preview grid so the same scenes appear iteration-over-iteration.
        # We pull from the underlying clean dataset (no degradation) — the
        # preview helper applies its own deterministic per-row degradation.
        inner_ds = getattr(self.train_ds, "inner", self.train_ds)
        n_fixed = int(cfg.data.num_fixed_preview_samples)
        n_random = int(cfg.data.num_random_preview_samples)
        ds_len = max(1, len(inner_ds))
        rng_prev = random.Random(cfg.train.seed)
        self._preview_dataset = inner_ds
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
        self._last_per_axis_lpips: dict[str, float] = {}
        self._last_grad_norm: float = 0.0

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, *,
                    compute_metrics: bool = True) -> dict[str, float]:
        clean_clips = batch["clean"].to(self.device, non_blocking=True)
        degraded    = batch["degraded"].to(self.device, non_blocking=True)
        config      = batch["config"].to(self.device, non_blocking=True)
        axes_active = batch["axes_active"]
        target = clean_clips[:, CENTER_INDEX]

        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[self.cfg.train.amp]
        autocast_enabled = (amp_dtype != torch.float32
                            and self.device.type == "cuda")

        with torch.amp.autocast(self.device.type, dtype=amp_dtype,
                                enabled=autocast_enabled):
            pred = self.model(degraded, config)
            ctx = LossContext(
                pred_rgb=pred,
                clean_rgb=target,
                degraded_rgb=degraded[:, CENTER_INDEX],
                config=config,
                axes_active=axes_active,
            )
            loss, loss_log = self.loss_set(ctx)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        if self.cfg.train.clip_grad_norm:
            # clip_grad_norm_ returns the *unclipped* total grad norm,
            # which is the right thing to log: it shows the actual
            # gradient magnitude the optimizer is seeing.
            gn = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm)
            grad_norm = float(gn.detach()) if torch.is_tensor(gn) else float(gn)
        self.optimizer.step()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self._last_grad_norm = grad_norm

        # Per-axis PSNR + LPIPS are forced CPU-sync points: even with a
        # bulk transfer per metric, `.cpu().numpy()` flushes the CUDA
        # stream — blocking the main process from queueing the *next*
        # batch's forward pass until every preceding op (optimizer.step,
        # EMA.update, LPIPS forward) finishes. The UI only consumes
        # these every `log_every` steps, so we skip the computation
        # entirely on non-tick steps. Stale values stay in
        # ``_last_per_axis_*`` and the UI is none the wiser.
        if compute_metrics:
            with torch.no_grad():
                psnr_b = psnr_per_sample(pred, target).detach().cpu().numpy()
                if self._lpips_model is not None:
                    lpips_b = (lpips_per_sample(
                        self._lpips_model, pred, target).detach().cpu().numpy())
                else:
                    lpips_b = None
                config_cpu = config.detach().cpu().numpy()
                self._last_per_axis_psnr = {}
                self._last_per_axis_lpips = {}
                for ax_idx, axis in enumerate(AXES):
                    mask = config_cpu[:, ax_idx] >= 0.5
                    if mask.any():
                        self._last_per_axis_psnr[axis] = float(psnr_b[mask].mean())
                        if lpips_b is not None:
                            self._last_per_axis_lpips[axis] = float(
                                lpips_b[mask].mean())

        # Rename the aggregate to "total" so TB tags read `loss/total`
        # rather than `loss/loss`. The UI's headless path already prefers
        # "total" over "loss".
        return {
            "total": float(loss.detach()),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            **loss_log,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def fit(self) -> Path:
        """Run the full training loop. Returns the path to final.pt."""
        total_steps = int(self.cfg.train.total_steps)
        log_every = int(self.cfg.train.log_every)
        save_every = int(self.cfg.train.save_every)
        preview_every_s = float(self.cfg.train.preview_every_s)

        print(f"[trainer] starting fit() — total_steps={total_steps}, "
              f"log_every={log_every}, save_every={save_every}", flush=True)
        print("[trainer] requesting first batch (dataloader worker spin-up "
              "+ first model forward will be the slow step)…", flush=True)
        loader_iter = iter(self.train_loader)
        bs = int(self.train_loader.batch_size or 1)
        with self.ui, self.tb:
            self._last_preview_t = time.perf_counter()
            self._t_window = time.perf_counter()
            self._samples_window = 0

            _first_step_t0: float | None = None
            while self.step < total_steps:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                if self.step == 0:
                    _first_step_t0 = time.perf_counter()
                    print(f"[trainer] first batch received "
                          "— running first forward+backward (this is when "
                          "torch.compile JIT actually fires if enabled)…",
                          flush=True)
                # Predict whether the *next* step lands on a tick boundary
                # so the trainer only computes per-axis PSNR/LPIPS (each a
                # forced CPU sync) on the steps the UI actually consumes.
                will_tick = (log_every > 0
                             and (self.step + 1) % log_every == 0)
                log = self._train_step(batch, compute_metrics=will_tick)
                self.step += 1
                self._samples_window += bs
                if self.step == 1 and _first_step_t0 is not None:
                    print(f"[trainer] first step done in "
                          f"{time.perf_counter() - _first_step_t0:.1f}s "
                          "— subsequent steps will be much faster",
                          flush=True)

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
                        per_task_lpips=self._last_per_axis_lpips or None,
                        grad_norm=self._last_grad_norm,
                    )
                    self.tb.log_scalars(
                        self.step,
                        _tag_scalars(log, self._last_per_axis_psnr,
                                     self._last_per_axis_lpips, imgs_per_s),
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
        """Render `<run>/samples/latest.png`, optionally archive a
        step-tagged history snapshot, and mirror the same grid into
        TensorBoard under ``preview/grid``. Errors are surfaced into the
        UI status line — the run keeps going even if a preview fails."""
        try:
            eval_model = self.ema.module if self.ema is not None else self.model
            was_training = eval_model.training
            eval_model.train(False)
            try:
                samples = make_temporal_preview_samples(
                    model=eval_model,
                    dataset=self._preview_dataset,
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
            self.tb.log_image(self.step, "preview/grid", grid)
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


def _tag_scalars(log: dict[str, float],
                 per_axis_psnr: dict[str, float],
                 per_axis_lpips: dict[str, float],
                 imgs_per_s: float) -> dict[str, float]:
    """Build the flat ``{tag: value}`` dict the TB writer expects.

    Convention:
      - ``loss/<name>``   — every numeric in the loss-log dict
      - ``metric/psnr/<axis>`` and ``metric/lpips/<axis>``
      - ``train/lr``, ``train/grad_norm``, ``train/img_per_s``
    """
    out: dict[str, float] = {}
    for k, v in log.items():
        if not isinstance(v, (int, float)):
            continue
        if k == "lr":
            out["train/lr"] = float(v)
        elif k == "grad_norm":
            out["train/grad_norm"] = float(v)
        else:
            out[f"loss/{k}"] = float(v)
    out["train/img_per_s"] = float(imgs_per_s)
    for axis, val in (per_axis_psnr or {}).items():
        out[f"metric/psnr/{axis}"] = float(val)
    for axis, val in (per_axis_lpips or {}).items():
        out[f"metric/lpips/{axis}"] = float(val)
    return out


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
