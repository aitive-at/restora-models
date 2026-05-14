"""Single-class trainer wiring data + model + losses + UI + preview."""
from __future__ import annotations

import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from restora_models.config import Config
from restora_models.data.dataset import RecursiveImageDataset
from restora_models.data.compound import AXES, CompoundDegradationWrapper, collate_compound
from restora_models.data.degradations import colorization as _colorization  # noqa: F401
from restora_models.data.degradations import deblur as _deblur  # noqa: F401
from restora_models.data.degradations import denoise as _denoise  # noqa: F401
from restora_models.data.degradations import jpeg as _jpeg  # noqa: F401
from restora_models.data.degradations import superres as _superres  # noqa: F401
from restora_models.data.degradations.registry import build_degradation
from restora_models.data.video import VideoPairDataset
from restora_models.data.video_compound import (
    VideoCompoundDegradationWrapper, collate_video_compound,
)
from restora_models.losses import LossContext, LossSet
from restora_models.losses.gan import discriminator_loss
from restora_models.losses.metrics import psnr
from restora_models.models import build_model
from restora_models.models.discriminator import UNetDiscriminator

from .checkpoint import save_checkpoint
from .ema import ModelEMA
from .preview import render_multitask_grid, write_png_atomic
from .ui import TrainUI


def _build_optimizer(params, cfg) -> torch.optim.Optimizer:
    klass = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[cfg.type]
    kw: dict = {"lr": cfg.lr, "weight_decay": cfg.weight_decay, "betas": tuple(cfg.betas)}
    try:
        kw["fused"] = cfg.fused
    except TypeError:
        pass
    try:
        return klass(params, **kw)
    except (TypeError, ValueError):
        kw.pop("fused", None)
        return klass(params, **kw)


def _build_scheduler(opt, cfg, total_steps):
    if cfg.type == "cosine":
        warm = max(1, cfg.warmup_steps)
        def lr(step):
            if step < warm:
                return step / warm
            t = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr)
    if cfg.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg.milestones), gamma=cfg.gamma)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, cfg: Config, *, device: torch.device | None = None,
                 headless: bool = False) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.run.seed)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Build compound degradation config
        axis_probs = cfg.compound.axis_probs.model_dump()
        deg_params = cfg.compound.degradations.model_dump()
        identity_prob = cfg.compound.identity_prob

        # Model
        self.memory_format = (torch.channels_last if cfg.train.memory_format == "channels_last"
                              else torch.contiguous_format)
        self.model = build_model(cfg.model, num_axes=len(AXES)).to(
            self.device, memory_format=self.memory_format)

        # Optim + scheduler
        self.opt_g = _build_optimizer(self.model.parameters(), cfg.optim_g)
        self.scheduler_g = _build_scheduler(self.opt_g, cfg.scheduler, cfg.train.total_steps)

        # Losses
        self.loss_set = LossSet(cfg.losses)
        for _, loss, _ in self.loss_set.entries:
            loss.to(self.device)

        # Discriminator
        self.disc: nn.Module | None = None
        self.opt_d: torch.optim.Optimizer | None = None
        self.gan_type = "hinge"
        if self.loss_set.has_gan:
            dcfg = self.loss_set.discriminator_cfg or {}
            self.disc = UNetDiscriminator(in_ch=3, nf=int(dcfg.get("nf", 64))).to(self.device)
            self.opt_d = _build_optimizer(self.disc.parameters(), cfg.optim_d)
            for _, loss, _ in self.loss_set.entries:
                if hasattr(loss, "gan_type"):
                    self.gan_type = loss.gan_type
                    break

        # EMA
        self.ema = (ModelEMA(self.model, decay=cfg.train.ema_decay)
                    if cfg.train.ema_decay > 0 else None)

        # AMP
        amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        self.amp_dtype = amp_map[cfg.train.amp]
        self.scaler = (torch.amp.GradScaler("cuda")
                       if (cfg.train.amp == "fp16" and self.device.type == "cuda") else None)

        # Data
        clean = RecursiveImageDataset(
            cfg.data.root, target_size=cfg.model.input_size,
            val_fraction=cfg.data.val_fraction,
            split="train" if cfg.data.val_fraction > 0 else "all",
            augment_hflip=cfg.data.augment.hflip,
            augment_rotate90=cfg.data.augment.rotate90,
            seed=cfg.run.seed,
        )
        if len(clean) == 0:
            raise RuntimeError(f"no images under {cfg.data.root}")
        train_ds = CompoundDegradationWrapper(
            clean, axis_probs=axis_probs,
            identity_prob=identity_prob,
            degradation_params=deg_params,
            seed=cfg.run.seed,
        )
        bs = cfg.data.loader.batch_size if cfg.data.loader.batch_size != "auto" else 16
        self.train_loader = DataLoader(
            train_ds, batch_size=int(bs), shuffle=True,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.data.loader.persistent_workers and cfg.data.loader.num_workers > 0,
            prefetch_factor=cfg.data.loader.prefetch_factor if cfg.data.loader.num_workers > 0 else None,
            collate_fn=collate_compound, drop_last=True,
        )

        # Val for preview
        if cfg.data.val_fraction > 0:
            clean_val = RecursiveImageDataset(
                cfg.data.root, target_size=cfg.model.input_size,
                val_fraction=cfg.data.val_fraction, split="val",
                augment_hflip=False, seed=cfg.run.seed,
            )
            if len(clean_val) == 0:
                clean_val = clean
            self.val_ds = CompoundDegradationWrapper(
                clean_val, axis_probs=axis_probs,
                identity_prob=identity_prob,
                degradation_params=deg_params,
                seed=cfg.run.seed,
            )
        else:
            self.val_ds = train_ds

        # Video pair loader (optional)
        self.video_loader: DataLoader | None = None
        self._video_iter = None
        self.video_batch_prob = 0.0
        vcfg = cfg.video
        if vcfg.enabled:
            if not vcfg.root:
                raise RuntimeError("video.enabled=True but video.root is empty")
            video_clean = VideoPairDataset(
                root=Path(vcfg.root), target_size=cfg.model.input_size,
                max_skip=vcfg.max_skip, hflip_prob=vcfg.hflip_prob,
                seed=cfg.run.seed, require_flow=vcfg.require_flow,
            )
            video_train = VideoCompoundDegradationWrapper(
                video_clean, axis_probs=axis_probs,
                identity_prob=identity_prob,
                degradation_params=deg_params, seed=cfg.run.seed,
            )
            v_bs = vcfg.batch_size if (vcfg.batch_size and vcfg.batch_size != "auto") else int(bs)
            self.video_loader = DataLoader(
                video_train, batch_size=int(v_bs), shuffle=True,
                num_workers=vcfg.num_workers,
                pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
                persistent_workers=vcfg.num_workers > 0,
                prefetch_factor=2 if vcfg.num_workers > 0 else None,
                collate_fn=collate_video_compound, drop_last=True,
            )
            self._video_iter = _cycle(self.video_loader)
            self.video_batch_prob = float(vcfg.video_batch_prob)

        if cfg.train.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)

        self.ui = TrainUI(run_name=cfg.run.name or "run",
                          total_steps=cfg.train.total_steps, headless=headless,
                          task_names=list(AXES))
        self._iter = _cycle(self.train_loader)
        self.step = 0
        self._last_preview_t = 0.0
        self._t_window = time.perf_counter()
        self._samples_window = 0
        self._preview_lock = threading.Lock()
        self._consecutive_nan = 0
        self._last_preview_step = -1

    def _amp_ctx(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def _compute_weight_overrides(self) -> dict[str, float] | None:
        """Per-step loss-weight multipliers. Currently: linear GAN warmup.

        Returns {"gan": factor} where factor ∈ [0, 1]:
            - 0 before `gan_warmup_start`
            - linearly ramps from 0 to 1 over `gan_warmup_steps` steps
            - 1 after that
        Returns None if no overrides apply this step (avoids dict alloc).
        """
        warmup_steps = self.cfg.train.gan_warmup_steps
        if warmup_steps <= 0:
            return None
        start = self.cfg.train.gan_warmup_start
        if self.step < start:
            factor = 0.0
        elif self.step >= start + warmup_steps:
            return None    # full weight; no override needed
        else:
            factor = (self.step - start) / float(warmup_steps)
        return {"gan": factor}

    def run_one_step(self) -> dict[str, float]:
        # Per-step Bernoulli: with prob video_batch_prob, use a video pair
        # batch (dual forward + temporal_pair loss). Otherwise the regular
        # image batch. Image and video paths share the same loss stack and
        # the same model; only the LossContext differs.
        if (self._video_iter is not None
                and torch.rand(()).item() < self.video_batch_prob):
            return self._train_step_video(next(self._video_iter))
        return self._train_step(next(self._iter))

    def _g_backward_and_step(self, total_g: torch.Tensor
                              ) -> tuple[bool, torch.Tensor]:
        """Backward + clip + opt-step for the generator. Returns
        (step_skipped, grad_norm)."""
        step_skipped = False
        if self.scaler is not None:
            self.scaler.scale(total_g).backward()
            self.scaler.unscale_(self.opt_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.scaler.step(self.opt_g)
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)
            self.scaler.update()
        else:
            total_g.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.opt_g.step()
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)
        return step_skipped, grad_norm

    def _disc_step(self, clean: torch.Tensor, pred: torch.Tensor) -> float | None:
        """Train discriminator on one (clean, pred) batch. Returns d_loss
        or None when no discriminator is present."""
        if self.disc is None or self.opt_d is None:
            return None
        self.opt_d.zero_grad(set_to_none=True)
        with self._amp_ctx():
            d_loss = discriminator_loss(self.disc, clean.detach(),
                                         pred.detach(), self.gan_type)
        if self.scaler is not None:
            self.scaler.scale(d_loss).backward(); self.scaler.unscale_(self.opt_d)
            self.scaler.step(self.opt_d); self.scaler.update()
        else:
            d_loss.backward(); self.opt_d.step()
        return float(d_loss.detach())

    def _train_step(self, batch: dict) -> dict[str, float]:
        clean = batch["clean"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        degraded = batch["degraded"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        config = batch["config"].to(self.device, non_blocking=True)
        axes = batch["axes"]

        self.opt_g.zero_grad(set_to_none=True)
        with self._amp_ctx():
            pred = self.model(degraded, config)
            ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=degraded,
                              config=config, axes_active=axes, discriminator=self.disc)
            weight_overrides = self._compute_weight_overrides()
            total_g, log_g = self.loss_set(ctx, weight_overrides=weight_overrides)
            if not torch.isfinite(total_g):
                self.opt_g.zero_grad(set_to_none=True)
                self.scheduler_g.step()
                self.step += 1
                self._samples_window += degraded.shape[0]
                self._consecutive_nan += 1
                if self._consecutive_nan >= 20:
                    raise RuntimeError(
                        f"20 consecutive non-finite losses at step {self.step}. "
                        "Resume from last.pt and try amp=fp32 or lower lr.")
                return {"total_g": float(total_g.detach()), **log_g, "_skipped": 1.0}

        step_skipped, grad_norm = self._g_backward_and_step(total_g)

        if step_skipped:
            self._consecutive_nan += 1
        else:
            self._consecutive_nan = 0
        if self._consecutive_nan >= 20:
            raise RuntimeError(
                f"20 consecutive non-finite gradient steps at step {self.step}. "
                "Resume from last.pt and try amp=fp32 or lower lr.")

        log: dict[str, float] = {"total_g": float(total_g.detach()), **log_g,
                                 "grad_norm": float(grad_norm)}
        if step_skipped:
            log["_skipped_grad"] = 1.0

        with torch.no_grad():
            per_sample = psnr(pred, clean)
            axis_to_idx = {a: i for i, a in enumerate(AXES)}
            self._last_per_task_psnr = {}
            for axis in AXES:
                ax_idx = axis_to_idx[axis]
                mask = config[:, ax_idx] >= 0.5
                if mask.any():
                    self._last_per_task_psnr[axis] = float(per_sample[mask].mean().item())

        d_loss = self._disc_step(clean, pred)
        if d_loss is not None:
            log["d_total"] = d_loss

        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler_g.step()
        self.step += 1
        self._samples_window += degraded.shape[0]
        return log

    def _train_step_video(self, batch: dict) -> dict[str, float]:
        """Video pair step: forward model on both frames, compute losses
        with the temporal_pair signal populated, backprop through both
        outputs.

        The primary (frame_t) supplies all reconstruction losses; the
        secondary (frame_t+k) supplies the temporal target. Both forward
        passes are concatenated into a single 2B forward for efficiency.
        """
        ml = self.memory_format
        clean_t   = batch["clean_t"].to(self.device, non_blocking=True, memory_format=ml)
        deg_t     = batch["degraded_t"].to(self.device, non_blocking=True, memory_format=ml)
        clean_tk  = batch["clean_tk"].to(self.device, non_blocking=True, memory_format=ml)
        deg_tk    = batch["degraded_tk"].to(self.device, non_blocking=True, memory_format=ml)
        flow      = batch["flow_t_tk"].to(self.device, non_blocking=True)
        config    = batch["config"].to(self.device, non_blocking=True)
        axes      = batch["axes"]
        B = deg_t.shape[0]

        self.opt_g.zero_grad(set_to_none=True)
        with self._amp_ctx():
            deg_pair = torch.cat([deg_t, deg_tk], dim=0)
            cfg_pair = torch.cat([config, config], dim=0)
            pred_pair = self.model(deg_pair, cfg_pair)
            pred_t, pred_tk = pred_pair[:B], pred_pair[B:]

            ctx = LossContext(
                pred_rgb=pred_t, clean_rgb=clean_t, degraded_rgb=deg_t,
                config=config, axes_active=axes, discriminator=self.disc,
                secondary_pred_rgb=pred_tk, flow_t_to_secondary=flow,
            )
            weight_overrides = self._compute_weight_overrides()
            total_g, log_g = self.loss_set(ctx, weight_overrides=weight_overrides)
            if not torch.isfinite(total_g):
                self.opt_g.zero_grad(set_to_none=True)
                self.scheduler_g.step()
                self.step += 1
                self._samples_window += 2 * B
                self._consecutive_nan += 1
                if self._consecutive_nan >= 20:
                    raise RuntimeError(
                        f"20 consecutive non-finite losses at step {self.step}. "
                        "Resume from last.pt and try amp=fp32 or lower lr.")
                return {"total_g": float(total_g.detach()), **log_g, "_skipped": 1.0}

        step_skipped, grad_norm = self._g_backward_and_step(total_g)

        if step_skipped:
            self._consecutive_nan += 1
        else:
            self._consecutive_nan = 0
        if self._consecutive_nan >= 20:
            raise RuntimeError(
                f"20 consecutive non-finite gradient steps at step {self.step}. "
                "Resume from last.pt and try amp=fp32 or lower lr.")

        log: dict[str, float] = {"total_g": float(total_g.detach()), **log_g,
                                 "grad_norm": float(grad_norm), "_video": 1.0}
        if step_skipped:
            log["_skipped_grad"] = 1.0

        with torch.no_grad():
            per_sample = psnr(pred_t, clean_t)
            axis_to_idx = {a: i for i, a in enumerate(AXES)}
            self._last_per_task_psnr = {}
            for axis in AXES:
                ax_idx = axis_to_idx[axis]
                mask = config[:, ax_idx] >= 0.5
                if mask.any():
                    self._last_per_task_psnr[axis] = float(per_sample[mask].mean().item())

        # Discriminator sees both frames' predictions vs both cleans for
        # twice the data per video step.
        clean_pair = torch.cat([clean_t, clean_tk], dim=0)
        d_loss = self._disc_step(clean_pair, pred_pair)
        if d_loss is not None:
            log["d_total"] = d_loss

        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler_g.step()
        self.step += 1
        self._samples_window += 2 * B
        return log

    def fit(self) -> None:
        with self.ui:
            self._last_preview_t = time.perf_counter() - self.cfg.train.preview_every_s
            for _ in range(self.cfg.train.total_steps):
                log = self.run_one_step()
                if self.step % self.cfg.train.log_every_steps == 0:
                    now = time.perf_counter()
                    imgs_per_s = self._samples_window / max(1e-6, now - self._t_window)
                    self._t_window = now; self._samples_window = 0
                    self.ui.tick(step=self.step, losses=log,
                                 lr=self.opt_g.param_groups[0]["lr"],
                                 throughput_imgs=imgs_per_s,
                                 per_task_psnr=getattr(self, "_last_per_task_psnr", None))
                if (self.cfg.train.preview_every_s > 0
                        and time.perf_counter() - self._last_preview_t
                        >= self.cfg.train.preview_every_s):
                    self._write_preview()
                if (self.cfg.train.ckpt_every_steps > 0
                        and self.step % self.cfg.train.ckpt_every_steps == 0):
                    self._save_ckpt(name="last.pt")
            self._save_ckpt(name="final.pt")
            self._write_preview()
            if self.cfg.export.on_finish:
                self._maybe_export_onnx()

    @torch.inference_mode()
    def _build_preview_samples(self) -> dict[str, list[dict]]:
        n_fixed = self.cfg.data.num_fixed_preview_samples
        n_rand = self.cfg.data.num_random_preview_samples
        eval_model = self.ema.module if self.ema is not None else self.model
        was_training = eval_model.training
        eval_model.train(False)

        # 9 rows: identity + colorize + denoise + 3x sharpen + dejpeg + deblur + all-on.
        # The three sharpen rows force SR factors 2 / 4 / 8 so users see each scale.
        preview_configs: list[tuple[str, list[int], dict]] = [
            ("identity",      [0, 0, 0, 0, 0], {}),
            ("colorize-only", [1, 0, 0, 0, 0], {}),
            ("denoise-only",  [0, 1, 0, 0, 0], {}),
            ("sharpen-2x",    [0, 0, 1, 0, 0], {"sharpen_factor": 2}),
            ("sharpen-4x",    [0, 0, 1, 0, 0], {"sharpen_factor": 4}),
            ("sharpen-8x",    [0, 0, 1, 0, 0], {"sharpen_factor": 8}),
            ("dejpeg-only",   [0, 0, 0, 1, 0], {}),
            ("deblur-only",   [0, 0, 0, 0, 1], {}),
            ("all-on",        [1, 1, 1, 1, 1], {}),
        ]

        out: dict[str, list[dict]] = {label: [] for label, _, _ in preview_configs}
        n_total = len(self.val_ds.clean)
        idxs = list(range(min(n_fixed, n_total)))
        if n_rand > 0 and n_total > len(idxs):
            extra = torch.randint(len(idxs), n_total, (min(n_rand, n_total - len(idxs)),)).tolist()
            idxs += extra

        import random as _random
        from restora_models.data.compound import DEGRADE_ORDER
        from restora_models.data.degradations.registry import build_degradation

        for label, vec, opts in preview_configs:
            flags = dict(zip(AXES, vec))
            # When we want a fixed SR factor, build a dedicated SharpenDegradation
            # with single-element factor_choices and use it in place of the
            # dataset's stochastic one for this row only.
            sharpen_override = None
            if "sharpen_factor" in opts:
                sharpen_override = build_degradation(
                    "sharpen", {"factor_choices": [int(opts["sharpen_factor"])]}
                )

            for i in idxs:
                clean_t = self.val_ds.clean[i]
                rng = _random.Random((self.cfg.run.seed * 1_000_003) ^ i)
                rgb_np = clean_t.permute(1, 2, 0).numpy().copy()
                for axis in DEGRADE_ORDER:
                    if flags[axis]:
                        deg = sharpen_override if (axis == "sharpen" and sharpen_override is not None) \
                              else self.val_ds.degs[axis]
                        rgb_np = deg.degrade(rgb_np, rng)
                degraded_t = torch.from_numpy(rgb_np.transpose(2, 0, 1)).contiguous()
                cfg_t = torch.tensor([vec], dtype=torch.float32, device=self.device)
                pred = eval_model(degraded_t.unsqueeze(0).to(self.device), cfg_t)
                out[label].append({
                    "clean": clean_t, "degraded": degraded_t,
                    "predicted": pred.clamp(0, 1).squeeze(0).cpu(),
                })

        if was_training:
            eval_model.train(True)
        return out

    def _write_preview(self) -> None:
        with self._preview_lock:
            try:
                samples = self._build_preview_samples()
            except Exception as e:
                self.ui.note_preview(f"preview error: {e}")
                self._last_preview_t = time.perf_counter()
                return
            caption = f"step {self.step}  ts {time.strftime('%H:%M:%S')}"
            grid = render_multitask_grid(samples, caption=caption,
                                          cell_size=self.cfg.model.input_size)
            latest = self.output_dir / "samples" / "latest.png"
            write_png_atomic(latest, grid)
            # History snapshot: write iter_N.png when this preview straddles
            # a multiple of preview_history_every since the last preview.
            # The old "step % N == 0" check almost never fired (the chance
            # of a preview moment landing exactly on a multiple is small),
            # so most history files were silently dropped.
            phe = self.cfg.train.preview_history_every
            if phe > 0:
                crossed = (self.step // phe) > max(0, self._last_preview_step // phe)
                if crossed or self.step == self.cfg.train.total_steps:
                    hist = self.output_dir / "samples" / f"iter_{self.step:07d}.png"
                    write_png_atomic(hist, grid)
            self._last_preview_step = self.step
            try:
                rel = latest.relative_to(self.output_dir)
            except ValueError:
                rel = latest
            self.ui.note_preview(f"wrote {rel} @ step {self.step}")
            self._last_preview_t = time.perf_counter()

    def _axes_map(self) -> dict:
        return {
            "model_type": self.cfg.model.type,
            "model_size": self.cfg.model.size,
            "input_size": self.cfg.model.input_size,
            "config_axes": list(AXES),
            "version": "0.2.0",
        }

    def _save_ckpt(self, name: str) -> None:
        save_checkpoint(
            self.output_dir / "ckpt" / name,
            model=self.model, optimizer=self.opt_g, optimizer_d=self.opt_d,
            discriminator=self.disc, ema=self.ema, scheduler=self.scheduler_g,
            step=self.step, extra={"cfg": self.cfg.model_dump()},
            task_map=self._axes_map(),
        )

    def _maybe_export_onnx(self) -> None:
        try:
            from restora_models.export.onnx import export_onnx_from_model
        except Exception:
            return
        export_model = self.ema.module if self.ema is not None else self.model
        export_onnx_from_model(
            export_model, num_axes=len(AXES),
            input_size=self.cfg.model.input_size,
            export_path=self.output_dir / "model.onnx",
            opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
            task_map=self._axes_map(),
        )
        if self.cfg.export.dynamic_hw:
            export_onnx_from_model(
                export_model, num_axes=len(AXES),
                input_size=self.cfg.model.input_size,
                export_path=self.output_dir / "model_dynamic.onnx",
                opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
                dynamic_hw=True, task_map=self._axes_map(),
            )


def fit(cfg: Config, *, device: torch.device | None = None) -> None:
    Trainer(cfg, device=device).fit()
