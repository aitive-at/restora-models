"""Single-class trainer wiring data, model, losses, AMP, EMA, UI, preview."""
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

from coliraz.config import Config
from coliraz.data.dataset import RecursiveImageDataset, collate
from coliraz.losses import LossContext, LossSet
from coliraz.losses.gan import discriminator_loss
from coliraz.models import build_ddcolor
from coliraz.models.discriminator import UNetDiscriminator
from coliraz.utils.color import lab_to_rgb

from .checkpoint import save_checkpoint
from .ema import ModelEMA
from .preview import render_preview_grid, write_png_atomic
from .ui import TrainUI


def _build_optimizer(model_params, cfg) -> torch.optim.Optimizer:
    klass = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[cfg.type]
    kw: dict = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.type != "SGD":
        kw["betas"] = tuple(cfg.betas)
        try:
            kw["fused"] = cfg.fused
        except TypeError:
            pass
    try:
        return klass(model_params, **kw)
    except (TypeError, ValueError):
        kw.pop("fused", None)
        return klass(model_params, **kw)


def _build_scheduler(opt, cfg, total_steps: int):
    if cfg.type == "cosine":
        warm = max(1, cfg.warmup_steps)

        def lr(step):
            if step < warm:
                return step / warm
            t = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr)
    if cfg.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=list(cfg.milestones), gamma=cfg.gamma
        )
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(
        self,
        cfg: Config,
        *,
        device: torch.device | None = None,
        pretrained_encoder: bool = True,
        headless: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = Path(cfg.run.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.run.seed)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Model
        self.memory_format = (
            torch.channels_last
            if cfg.train.memory_format == "channels_last"
            else torch.contiguous_format
        )
        self.model = build_ddcolor(cfg.model, pretrained=pretrained_encoder).to(
            self.device, memory_format=self.memory_format
        )

        # Optimizer + scheduler
        self.opt_g = _build_optimizer(self.model.parameters(), cfg.optim_g)
        self.scheduler_g = _build_scheduler(self.opt_g, cfg.scheduler, cfg.train.total_steps)

        # Losses
        self.loss_set = LossSet(cfg.losses)
        for _, loss in self.loss_set.entries:
            loss.to(self.device)

        # Discriminator (only if GAN loss is enabled)
        self.disc: nn.Module | None = None
        self.opt_d: torch.optim.Optimizer | None = None
        self.gan_type = "hinge"
        if self.loss_set.has_gan:
            dcfg = self.loss_set.discriminator_cfg or {}
            self.disc = UNetDiscriminator(in_ch=3, nf=int(dcfg.get("nf", 64))).to(self.device)
            self.opt_d = _build_optimizer(self.disc.parameters(), cfg.optim_d)
            for _, loss_mod in self.loss_set.entries:
                if hasattr(loss_mod, "gan_type"):
                    self.gan_type = loss_mod.gan_type
                    break

        # EMA
        self.ema = (
            ModelEMA(self.model, decay=cfg.train.ema_decay)
            if cfg.train.ema_decay > 0
            else None
        )

        # AMP
        amp_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        self.amp_dtype = amp_map[cfg.train.amp]
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (cfg.train.amp == "fp16" and self.device.type == "cuda")
            else None
        )

        # Data
        ds = RecursiveImageDataset(
            cfg.data.root,
            target_size=cfg.model.input_size,
            val_fraction=cfg.data.val_fraction,
            split="train" if cfg.data.val_fraction > 0 else "all",
            augment_hflip=cfg.data.augment.hflip,
            augment_rotate90=cfg.data.augment.rotate90,
            seed=cfg.run.seed,
        )
        if len(ds) == 0:
            raise RuntimeError(f"no images under {cfg.data.root}")
        bs = cfg.data.loader.batch_size if cfg.data.loader.batch_size != "auto" else 16
        self.train_loader = DataLoader(
            ds,
            batch_size=int(bs),
            shuffle=True,
            num_workers=cfg.data.loader.num_workers,
            pin_memory=cfg.data.loader.pin_memory and self.device.type == "cuda",
            persistent_workers=(
                cfg.data.loader.persistent_workers and cfg.data.loader.num_workers > 0
            ),
            prefetch_factor=(
                cfg.data.loader.prefetch_factor if cfg.data.loader.num_workers > 0 else None
            ),
            collate_fn=collate,
            drop_last=True,
        )

        # Val/preview dataset
        if cfg.data.val_fraction > 0:
            self.val_ds = RecursiveImageDataset(
                cfg.data.root,
                target_size=cfg.model.input_size,
                val_fraction=cfg.data.val_fraction,
                split="val",
                augment_hflip=False,
                seed=cfg.run.seed,
            )
            if len(self.val_ds) == 0:
                self.val_ds = ds
        else:
            self.val_ds = ds

        # Compile (opt-in)
        if cfg.train.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model, mode=cfg.train.compile_mode)

        # UI
        self.ui = TrainUI(
            run_name=cfg.run.name or "run",
            total_steps=cfg.train.total_steps,
            headless=headless,
        )
        self._iter = _cycle(self.train_loader)
        self.step = 0
        self._last_preview_t = 0.0
        self._t_window = time.perf_counter()
        self._samples_window = 0
        self._preview_lock = threading.Lock()

    # ------------------------------------------------------------------
    # one step (used by tests + main loop)
    # ------------------------------------------------------------------
    def run_one_step(self) -> dict[str, float]:
        batch = next(self._iter)
        return self._train_step(batch)

    def _amp_ctx(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def _train_step(self, batch: dict) -> dict[str, float]:
        gray_rgb = batch["gray_rgb"].to(
            self.device, non_blocking=True, memory_format=self.memory_format
        )
        gt_ab = batch["gt_ab"].to(self.device, non_blocking=True)
        # L_full is the target_size-resized L; trainer uses it as the L for loss-ctx lab->rgb
        L = batch["L_full"].to(self.device, non_blocking=True)
        # If L_full and gt_ab have different spatial sizes, downsample L to match
        if L.shape[-2:] != gt_ab.shape[-2:]:
            L = torch.nn.functional.interpolate(
                L, size=gt_ab.shape[-2:], mode="bilinear", align_corners=False
            )

        self.opt_g.zero_grad(set_to_none=True)
        with self._amp_ctx():
            pred_ab = self.model(gray_rgb)
            pred_lab = torch.cat([L, pred_ab], dim=1)
            pred_rgb = lab_to_rgb(pred_lab).clamp(0, 1)
            gt_lab = torch.cat([L, gt_ab], dim=1)
            gt_rgb = lab_to_rgb(gt_lab).clamp(0, 1)
            ctx = LossContext(
                pred_ab=pred_ab,
                gt_ab=gt_ab,
                pred_rgb=pred_rgb,
                gt_rgb=gt_rgb,
                gray_rgb=gray_rgb,
                discriminator=self.disc,
            )
            total_g, log_g = self.loss_set(ctx)

        if self.scaler is not None:
            self.scaler.scale(total_g).backward()
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm
            )
            self.scaler.step(self.opt_g)
            self.scaler.update()
        else:
            total_g.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.clip_grad_norm
            )
            self.opt_g.step()

        log: dict[str, float] = {"total_g": float(total_g.detach()), **log_g}
        if self.disc is not None and self.opt_d is not None:
            self.opt_d.zero_grad(set_to_none=True)
            with self._amp_ctx():
                d_loss = discriminator_loss(
                    self.disc, gt_rgb.detach(), pred_rgb.detach(), self.gan_type
                )
            if self.scaler is not None:
                self.scaler.scale(d_loss).backward()
                self.scaler.unscale_(self.opt_d)
                self.scaler.step(self.opt_d)
                self.scaler.update()
            else:
                d_loss.backward()
                self.opt_d.step()
            log["d_total"] = float(d_loss.detach())

        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler_g.step()
        self.step += 1
        self._samples_window += gray_rgb.shape[0]
        return log

    # ------------------------------------------------------------------
    # full loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        with self.ui:
            self._last_preview_t = time.perf_counter() - self.cfg.train.preview_every_s
            for _ in range(self.cfg.train.total_steps):
                log = self.run_one_step()
                if self.step % self.cfg.train.log_every_steps == 0:
                    now = time.perf_counter()
                    imgs_per_s = self._samples_window / max(1e-6, now - self._t_window)
                    self._t_window = now
                    self._samples_window = 0
                    lr = self.opt_g.param_groups[0]["lr"]
                    self.ui.tick(
                        step=self.step,
                        losses=log,
                        lr=lr,
                        throughput_imgs=imgs_per_s,
                    )
                if (
                    self.cfg.train.preview_every_s > 0
                    and time.perf_counter() - self._last_preview_t
                    >= self.cfg.train.preview_every_s
                ):
                    self._write_preview()
                if (
                    self.cfg.train.ckpt_every_steps > 0
                    and self.step % self.cfg.train.ckpt_every_steps == 0
                ):
                    self._save_ckpt(name="last.pt")
            self._save_ckpt(name="final.pt")
            self._write_preview()
            if self.cfg.export.on_finish:
                self._maybe_export_onnx()

    # ------------------------------------------------------------------
    # preview + ckpt + export helpers
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _build_preview_samples(self) -> list[dict]:
        n_fixed = self.cfg.data.num_fixed_preview_samples
        n_rand = self.cfg.data.num_random_preview_samples
        eval_model = self.ema.module if self.ema is not None else self.model
        was_training = eval_model.training
        eval_model.train(False)
        out: list[dict] = []
        n_total = len(self.val_ds)
        idxs = list(range(min(n_fixed, n_total)))
        if n_rand > 0 and n_total > len(idxs):
            extra = list(
                torch.randint(len(idxs), n_total, (min(n_rand, n_total - len(idxs)),)).tolist()
            )
            idxs += extra
        for i in idxs:
            s = self.val_ds[i]
            gray = s["gray_rgb"].unsqueeze(0).to(self.device)
            L_full = s["L_full"].unsqueeze(0).to(self.device)
            # resize L_full to gray spatial size for the preview lab merge
            L = torch.nn.functional.interpolate(
                L_full, size=gray.shape[-2:], mode="bilinear", align_corners=False
            )
            gt_ab = s["gt_ab"].unsqueeze(0).to(self.device)
            pred_ab = eval_model(gray)
            pred_lab = torch.cat([L, pred_ab], dim=1)
            pred_rgb = lab_to_rgb(pred_lab).clamp(0, 1).squeeze(0)
            orig_lab = torch.cat([L, gt_ab], dim=1)
            orig_rgb = lab_to_rgb(orig_lab).clamp(0, 1).squeeze(0)
            delta = (pred_ab - gt_ab).squeeze(0)
            out.append(
                {
                    "original": orig_rgb,
                    "gray_rgb": s["gray_rgb"],
                    "pred_rgb": pred_rgb,
                    "delta_ab": delta,
                }
            )
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
            grid = render_preview_grid(
                samples, caption=caption, cell_size=self.cfg.model.input_size
            )
            latest = self.output_dir / "samples" / "latest.png"
            write_png_atomic(latest, grid)
            if (
                self.cfg.train.preview_history_every > 0
                and (self.step % self.cfg.train.preview_history_every == 0)
            ):
                hist = self.output_dir / "samples" / f"iter_{self.step:07d}.png"
                write_png_atomic(hist, grid)
            try:
                rel = latest.relative_to(self.output_dir)
            except ValueError:
                rel = latest
            self.ui.note_preview(f"wrote {rel} @ step {self.step}")
            self._last_preview_t = time.perf_counter()

    def _save_ckpt(self, name: str) -> None:
        path = self.output_dir / "ckpt" / name
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.opt_g,
            optimizer_d=self.opt_d,
            discriminator=self.disc,
            ema=self.ema,
            scheduler=self.scheduler_g,
            step=self.step,
            extra={"cfg": self.cfg.model_dump()},
        )

    def _maybe_export_onnx(self) -> None:
        try:
            from coliraz.export.onnx import export_onnx_from_model
        except Exception:
            return
        path = self.output_dir / "model.onnx"
        export_onnx_from_model(
            self.ema.module if self.ema is not None else self.model,
            input_size=self.cfg.model.input_size,
            export_path=path,
            opset=self.cfg.export.opset,
            simplify=self.cfg.export.simplify,
        )


def fit(cfg: Config, *, device: torch.device | None = None) -> None:
    Trainer(cfg, device=device).fit()
