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

from refine.config import Config
from refine.data.dataset import RecursiveImageDataset
from refine.data.degradations import colorization as _colorization  # noqa: F401
from refine.data.degradations import deblur as _deblur  # noqa: F401
from refine.data.degradations import denoise as _denoise  # noqa: F401
from refine.data.degradations import jpeg as _jpeg  # noqa: F401
from refine.data.degradations import superres as _superres  # noqa: F401
from refine.data.degradations.registry import build_degradation
from refine.data.multitask import MultiTaskWrapper, collate_multitask
from refine.losses import LossContext, LossSet
from refine.losses.gan import discriminator_loss
from refine.losses.metrics import per_task_average, psnr
from refine.models import build_model
from refine.models.discriminator import UNetDiscriminator

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

        # Build degradations + assign task IDs in YAML declaration order
        deg_items = list(cfg.degradations.items())
        self.degradations = []
        self.task_name_to_id: dict[str, int] = {}
        for i, (name, dcfg) in enumerate(deg_items):
            kwargs = {k: v for k, v in dcfg.model_dump(exclude_none=True).items() if k != "weight"}
            deg = build_degradation(name, kwargs)
            deg.task_id = i
            self.degradations.append(deg)
            self.task_name_to_id[name] = i
        weights = [cfg.degradations[name].weight for name, _ in deg_items]
        num_tasks = len(self.degradations)

        # Model
        self.memory_format = (torch.channels_last if cfg.train.memory_format == "channels_last"
                              else torch.contiguous_format)
        self.model = build_model(cfg.model, num_tasks=num_tasks).to(
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
        train_ds = MultiTaskWrapper(clean, self.degradations, weights, seed=cfg.run.seed)
        bs = cfg.data.loader.batch_size if cfg.data.loader.batch_size != "auto" else 16
        self.train_loader = DataLoader(
            train_ds, batch_size=int(bs), shuffle=True,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
            persistent_workers=cfg.data.loader.persistent_workers and cfg.data.loader.num_workers > 0,
            prefetch_factor=cfg.data.loader.prefetch_factor if cfg.data.loader.num_workers > 0 else None,
            collate_fn=collate_multitask, drop_last=True,
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
            self.val_ds = MultiTaskWrapper(clean_val, self.degradations, weights, seed=cfg.run.seed)
        else:
            self.val_ds = train_ds

        if cfg.train.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)

        self.ui = TrainUI(run_name=cfg.run.name or "run",
                          total_steps=cfg.train.total_steps, headless=headless,
                          task_names=list(self.task_name_to_id.keys()))
        self._iter = _cycle(self.train_loader)
        self.step = 0
        self._last_preview_t = 0.0
        self._t_window = time.perf_counter()
        self._samples_window = 0
        self._preview_lock = threading.Lock()
        self._consecutive_nan = 0

    def _amp_ctx(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def run_one_step(self) -> dict[str, float]:
        return self._train_step(next(self._iter))

    def _train_step(self, batch: dict) -> dict[str, float]:
        clean = batch["clean"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        degraded = batch["degraded"].to(self.device, non_blocking=True, memory_format=self.memory_format)
        task_id = batch["task_id"].to(self.device, non_blocking=True)
        task_names = batch["task_name"]

        self.opt_g.zero_grad(set_to_none=True)
        with self._amp_ctx():
            pred = self.model(degraded, task_id)
            ctx = LossContext(pred_rgb=pred, clean_rgb=clean, degraded_rgb=degraded,
                              task_ids=task_id, task_names=task_names, discriminator=self.disc)
            total_g, log_g = self.loss_set(ctx)
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

        step_skipped = False
        if self.scaler is not None:
            self.scaler.scale(total_g).backward()
            self.scaler.unscale_(self.opt_g)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.scaler.step(self.opt_g)
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)
            self.scaler.update()
        else:
            total_g.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.train.clip_grad_norm)
            if torch.isfinite(grad_norm):
                self.opt_g.step()
            else:
                step_skipped = True; self.opt_g.zero_grad(set_to_none=True)

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
            num_tasks = len(self.task_name_to_id)
            avg = per_task_average(per_sample.cpu(), task_id.cpu(), num_tasks=num_tasks)
        self._last_per_task_psnr = {
            name: float(avg[i]) for name, i in self.task_name_to_id.items()
        }

        if self.disc is not None and self.opt_d is not None:
            self.opt_d.zero_grad(set_to_none=True)
            with self._amp_ctx():
                d_loss = discriminator_loss(self.disc, clean.detach(), pred.detach(), self.gan_type)
            if self.scaler is not None:
                self.scaler.scale(d_loss).backward(); self.scaler.unscale_(self.opt_d)
                self.scaler.step(self.opt_d); self.scaler.update()
            else:
                d_loss.backward(); self.opt_d.step()
            log["d_total"] = float(d_loss.detach())

        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler_g.step()
        self.step += 1
        self._samples_window += degraded.shape[0]
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

        out: dict[str, list[dict]] = {name: [] for name in self.task_name_to_id}
        deg_by_name = {d.name: d for d in self.degradations}
        n_total = len(self.val_ds.clean)
        idxs = list(range(min(n_fixed, n_total)))
        if n_rand > 0 and n_total > len(idxs):
            extra = torch.randint(len(idxs), n_total, (min(n_rand, n_total - len(idxs)),)).tolist()
            idxs += extra
        import random as _random
        for task_name, deg in deg_by_name.items():
            for i in idxs:
                clean_t = self.val_ds.clean[i]
                rng = _random.Random((self.cfg.run.seed * 1_000_003) ^ (i + deg.task_id))
                degraded_np = deg.degrade(clean_t.permute(1, 2, 0).numpy(), rng)
                degraded_t = torch.from_numpy(degraded_np.transpose(2, 0, 1)).contiguous()
                pred = eval_model(degraded_t.unsqueeze(0).to(self.device),
                                  torch.tensor([deg.task_id], dtype=torch.long, device=self.device))
                out[task_name].append({
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
            if (self.cfg.train.preview_history_every > 0
                    and (self.step % self.cfg.train.preview_history_every == 0)):
                hist = self.output_dir / "samples" / f"iter_{self.step:07d}.png"
                write_png_atomic(hist, grid)
            try:
                rel = latest.relative_to(self.output_dir)
            except ValueError:
                rel = latest
            self.ui.note_preview(f"wrote {rel} @ step {self.step}")
            self._last_preview_t = time.perf_counter()

    def _task_map(self) -> dict:
        return {
            "tasks": dict(self.task_name_to_id),
            "input_size": self.cfg.model.input_size,
            "model_size": self.cfg.model.size,
            "version": "0.1.0",
        }

    def _save_ckpt(self, name: str) -> None:
        save_checkpoint(
            self.output_dir / "ckpt" / name,
            model=self.model, optimizer=self.opt_g, optimizer_d=self.opt_d,
            discriminator=self.disc, ema=self.ema, scheduler=self.scheduler_g,
            step=self.step, extra={"cfg": self.cfg.model_dump()},
            task_map=self._task_map(),
        )

    def _maybe_export_onnx(self) -> None:
        try:
            from refine.export.onnx import export_onnx_from_model
        except Exception:
            return
        export_model = self.ema.module if self.ema is not None else self.model
        export_onnx_from_model(
            export_model, num_tasks=len(self.task_name_to_id),
            input_size=self.cfg.model.input_size,
            export_path=self.output_dir / "model.onnx",
            opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
            task_map=self._task_map(),
        )
        if self.cfg.export.dynamic_hw:
            export_onnx_from_model(
                export_model, num_tasks=len(self.task_name_to_id),
                input_size=self.cfg.model.input_size,
                export_path=self.output_dir / "model_dynamic.onnx",
                opset=self.cfg.export.opset, simplify=self.cfg.export.simplify,
                dynamic_hw=True, task_map=self._task_map(),
            )


def fit(cfg: Config, *, device: torch.device | None = None) -> None:
    Trainer(cfg, device=device).fit()
