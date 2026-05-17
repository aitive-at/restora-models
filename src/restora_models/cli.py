"""restora-models CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="restora-models — multi-task image restoration", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """restora CLI."""


@app.command()
def version() -> None:
    from restora_models import __version__
    typer.echo(__version__)


@app.command(name="scan-data")
def scan_data(root: Path = typer.Option(..., "--root", exists=True, file_okay=False)) -> None:
    from restora_models.data.dataset import build_manifest
    paths = build_manifest(root, force=True)
    typer.echo(f"{len(paths)} images indexed under {root}")


@app.command(name="download")
def download(
    dataset: str = typer.Option(
        "relaion2B-multi-aesthetic", "--dataset",
        help="Which LAION-aesthetic subset: relaion2B-multi-aesthetic (17M, "
             "default) | laion2B-en-aesthetic (51M) | relaion1B-nolang-aesthetic "
             "(52M). NOTE: all are gated on HuggingFace; you must request "
             "access on each dataset page individually and have HF_TOKEN set "
             "(or have run `huggingface-cli login`)."),
    output_dir: Path = typer.Option(
        ..., "--output", "--out", "-o",
        help="Target dir; created if missing. Resume works by re-running with "
             "the same flags — both metadata and image steps skip completed work."),
    image_size: int = typer.Option(
        384, "--image-size",
        help="Longest-side cap for downloaded JPEGs. 384 matches the LAION "
             "default; bump higher if you want to keep originals (more disk)."),
    max_shards: Optional[int] = typer.Option(
        None, "--max-shards",
        help="Limit to first N of 128 parquet shards. Useful for partial "
             "downloads or smoke-testing on a small disk."),
    processes: int = typer.Option(
        16, "--processes",
        help="img2dataset worker processes. 16 is a good default for a single "
             "machine with a 1-10 Gbps link; bump up on 100Gbps cloud."),
    threads: int = typer.Option(
        64, "--threads",
        help="img2dataset threads per worker (HTTP fetchers). Higher = more "
             "concurrent connections; lower if you hit rate limits."),
    progress_every_s: float = typer.Option(
        5.0, "--progress-every-s",
        help="Print download progress (cumulative count + rate) every N seconds. "
             "Set higher for less log noise on long runs."),
    timeout_s: int = typer.Option(
        5, "--timeout-s",
        help="Per-URL HTTP timeout. LAION URLs are old; ~15-20%% are dead. "
             "Lower = drop dead URLs faster, higher throughput overall. "
             "Trade-off: too low and some slow-but-live servers get skipped. "
             "5s is the sweet spot in practice."),
    skip_metadata: bool = typer.Option(
        False, "--skip-metadata",
        help="Skip the HF parquet download step (assume metadata is already present)."),
    skip_images: bool = typer.Option(
        False, "--skip-images",
        help="Stop after metadata download (useful for staged deployments)."),
) -> None:
    """Download a LAION-aesthetic image dataset for training.

    Two-step pipeline (both resumable):
      1) Fetch the parquet metadata shards from HuggingFace.
      2) Run img2dataset to download the actual JPEGs.

    Final layout under --output:
        metadata/<dataset>/part-NNNNN-...parquet
        images/<dataset>/NNNNN/         (sharded JPEG tree)
        images/<dataset>/NNNNN.parquet  (per-shard metadata)
        images/<dataset>/NNNNN_stats.json
    """
    from restora_models.data.download import download_laion_aesthetic, list_datasets

    valid = list_datasets()
    if dataset not in valid:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; options: {valid}"
        )

    try:
        download_laion_aesthetic(
            dataset=dataset, output_dir=output_dir,
            image_size=image_size, max_shards=max_shards,
            processes=processes, threads=threads,
            timeout_s=timeout_s,
            progress_every_s=progress_every_s,
            skip_metadata=skip_metadata, skip_images=skip_images,
        )
    except RuntimeError as e:
        typer.secho(f"[download failed] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e


@app.command()
def info(model: Path = typer.Option(..., "--model", exists=True, dir_okay=False)) -> None:
    """Show model metadata (type, axes, input size) from the task-map sidecar."""
    import json
    sidecar = model.with_suffix(".task_map.json")
    if sidecar.exists():
        typer.echo(sidecar.read_text())
        return
    import torch
    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    tm = payload.get("task_map")
    if tm:
        typer.echo(json.dumps(tm, indent=2))
    else:
        typer.echo("no task_map found in checkpoint")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data: Optional[Path] = typer.Option(None, "--data"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    batch_size: Optional[str] = typer.Option(None, "--batch-size"),
    compile_: bool = typer.Option(False, "--compile/--no-compile"),
    amp: Optional[str] = typer.Option(None, "--amp"),
    total_steps: Optional[int] = typer.Option(None, "--total-steps"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
    video_root: Optional[Path] = typer.Option(None, "--video-root"),
    no_video: bool = typer.Option(False, "--no-video"),
    video_batch_prob: Optional[float] = typer.Option(None, "--video-batch-prob"),
) -> None:
    from restora_models.config import load_config
    from restora_models.train import Trainer
    from restora_models.train.checkpoint import load_checkpoint

    overrides: dict = {}
    if data is not None:
        overrides.setdefault("data", {})["root"] = str(data)
    if run_name is not None:
        overrides.setdefault("run", {})["name"] = run_name
    if batch_size is not None:
        bs: int | str = "auto" if batch_size == "auto" else int(batch_size)
        overrides.setdefault("data", {}).setdefault("loader", {})["batch_size"] = bs
    if amp is not None:
        overrides.setdefault("train", {})["amp"] = amp
    if total_steps is not None:
        overrides.setdefault("train", {})["total_steps"] = total_steps
        overrides.setdefault("scheduler", {})["total_steps"] = total_steps
    if compile_:
        overrides.setdefault("train", {})["compile"] = True
    if video_root is not None:
        overrides.setdefault("video", {})["root"] = str(video_root)
        overrides.setdefault("video", {})["enabled"] = True
    if no_video:
        overrides.setdefault("video", {})["enabled"] = False
    if video_batch_prob is not None:
        overrides.setdefault("video", {})["video_batch_prob"] = video_batch_prob

    cfg = load_config(config, overrides=overrides)
    trainer = Trainer(cfg)
    if resume is not None:
        load_checkpoint(resume, model=trainer.model, optimizer=trainer.opt_g,
                        optimizer_d=trainer.opt_d, discriminator=trainer.disc,
                        ema=trainer.ema, scheduler=trainer.scheduler_g)
    trainer.fit()


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    input_: Path = typer.Option(..., "--input", "--in", exists=True),
    output: Path = typer.Option(..., "--output", "--out"),
    color: bool = typer.Option(False, "--color/--no-color", help="apply colorize axis"),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise", help="apply denoise axis"),
    sharp: bool = typer.Option(False, "--sharp/--no-sharp", help="apply sharpen (SR) axis"),
    dejpeg: bool = typer.Option(False, "--dejpeg/--no-dejpeg", help="apply JPEG-restore axis"),
    deblur: bool = typer.Option(False, "--deblur/--no-deblur", help="apply deblur axis"),
    upsample_to: Optional[str] = typer.Option(
        None, "--upsample-to",
        help="WxH (e.g. 2048x2048) — bicubic pre-upsample before inference"),
) -> None:
    """Colorize / denoise / sharpen / dejpeg / deblur one image or a folder."""
    import cv2
    import torch
    from restora_models.infer.pipeline import load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(model, device=device)
    config = {"colorize": color, "denoise": denoise, "sharpen": sharp,
              "dejpeg": dejpeg, "deblur": deblur}

    def maybe_upsample(img):
        if not upsample_to:
            return img
        w, h = (int(x) for x in upsample_to.lower().split("x"))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    if input_.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(input_))
        if img is None:
            raise typer.BadParameter(f"could not read {input_}")
        cv2.imwrite(str(output), pipe.process(maybe_upsample(img), config=config))
    else:
        output.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        for p in sorted(input_.rglob("*")):
            if p.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            out_path = output / p.relative_to(input_)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), pipe.process(maybe_upsample(img), config=config))
    typer.echo(f"wrote {output}")


_TASK_CONFIGS: dict[str, list[float]] = {
    "colorize": [1.0, 0.0, 0.0, 0.0, 0.0],
    "denoise":  [0.0, 1.0, 0.0, 0.0, 0.0],
    "sharpen":  [0.0, 0.0, 1.0, 0.0, 0.0],
    "dejpeg":   [0.0, 0.0, 0.0, 1.0, 0.0],
    "deblur":   [0.0, 0.0, 0.0, 0.0, 1.0],
    "all":      [1.0, 1.0, 1.0, 1.0, 1.0],
}


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(256, "--input-size"),
    format_: str = typer.Option("onnx", "--format",
                                 help="onnx (default) | pnnx. ONNX is for "
                                      "server / cross-platform inference; "
                                      "PNNX produces ncnn-compatible files "
                                      "for mobile/edge deployment."),
    opset: int = typer.Option(17, "--opset",
                              help="ONNX opset (ignored for --format pnnx)"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify",
                                  help="ONNX-only; ignored for --format pnnx"),
    dynamic_hw: bool = typer.Option(False, "--dynamic-hw/--fixed-hw"),
    precision: str = typer.Option("fp32", "--precision",
                                  help="fp32 (default) | fp16 | fp8 | fp4. "
                                       "For --format pnnx, fp32/fp16 are "
                                       "honored (fp16 -> ncnn fp16 weights); "
                                       "fp8/fp4 raise an error for pnnx."),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Bake a specific task's config into the export (resulting "
             "graph has ONLY 'input' tensor, no 'config'). Options: "
             "colorize | denoise | sharpen | dejpeg | deblur | all"),
    bake_axes: Optional[str] = typer.Option(
        None, "--bake-axes",
        help="Comma-separated 5-axis vector to bake in, e.g. '1,0,1,0,0' "
             "for colorize+sharpen. Mutually exclusive with --task."),
    keep_pnnx_debug: bool = typer.Option(
        False, "--keep-pnnx-debug/--no-keep-pnnx-debug",
        help="(--format pnnx only) Keep the auto-generated _pnnx.py and "
             "_ncnn.py recreate scripts. Off by default — they have known "
             "generation bugs for some ops and aren't needed for deployment."),
    verify_ep: Optional[str] = typer.Option(
        None, "--verify-ep",
        help="(--format onnx only) After export, run inference through the "
             "named ORT execution provider with profiling and assert no "
             "tensor op fell back to CPU. Options: cuda | tensorrt. Skips "
             "cleanly if the matching provider isn't installed."),
) -> None:
    """Export a trained refine checkpoint to ONNX or PNNX/ncnn.

    Default emits a 2-input artifact (input, config) -> output that
    works for any of the 5 restoration axes by setting the config vector.

    Pass --task NAME or --bake-axes V1,V2,V3,V4,V5 to bake the config
    as a constant; that resulting file has only an `input` tensor —
    "RGB in, RGB out" — and is what deployment consumers usually want.

    --format pnnx is the ncnn deployment path. It produces several files
    alongside --output: .pnnx.bin / .pnnx.param / .ncnn.bin / .ncnn.param
    / .pnnx.onnx / .pt / Python recreate scripts. The .ncnn.* pair is
    what mobile / edge consumers load."""
    import torch
    from restora_models.config import ModelConfig
    from restora_models.data.compound import AXES
    from restora_models.models import build_model

    format_ = format_.lower()
    if format_ not in ("onnx", "pnnx"):
        raise typer.BadParameter(f"--format must be onnx or pnnx, got {format_!r}")
    if task is not None and bake_axes is not None:
        raise typer.BadParameter("--task and --bake-axes are mutually exclusive")
    fixed_config: list[float] | None = None
    if task is not None:
        if task not in _TASK_CONFIGS:
            raise typer.BadParameter(
                f"unknown task {task!r}; options: {sorted(_TASK_CONFIGS)}"
            )
        fixed_config = list(_TASK_CONFIGS[task])
    elif bake_axes is not None:
        try:
            fixed_config = [float(x) for x in bake_axes.split(",")]
        except ValueError as e:
            raise typer.BadParameter(
                f"--bake-axes must be comma-separated floats: {e}"
            ) from e
        if len(fixed_config) != len(AXES):
            raise typer.BadParameter(
                f"--bake-axes needs {len(AXES)} values; got {len(fixed_config)}"
            )

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    m = build_model(mcfg, num_axes=len(AXES))
    m.load_state_dict(payload["model"])
    task_map = payload.get("task_map") or {}
    if task is not None:
        task_map = dict(task_map); task_map["task"] = task

    if format_ == "pnnx":
        if precision not in ("fp32", "fp16"):
            raise typer.BadParameter(
                f"--format pnnx supports precision fp32 or fp16, got {precision!r}"
            )
        from restora_models.export.pnnx import export_pnnx_from_model
        export_pnnx_from_model(
            m, num_axes=len(AXES), input_size=input_size,
            export_path=output, dynamic_hw=dynamic_hw,
            fp16=(precision == "fp16"), fixed_config=fixed_config,
            task_map=task_map, keep_debug_scripts=keep_pnnx_debug,
        )
    else:
        if verify_ep is not None and verify_ep not in ("cuda", "tensorrt"):
            raise typer.BadParameter(f"--verify-ep must be cuda or tensorrt, got {verify_ep!r}")
        from restora_models.export.onnx import export_onnx_from_model
        effective_opset = max(opset, 19) if precision == "fp8" else opset
        export_onnx_from_model(
            m, num_axes=len(AXES), input_size=input_size,
            export_path=output, opset=effective_opset, simplify=simplify,
            dynamic_hw=dynamic_hw, task_map=task_map, precision=precision,
            fixed_config=fixed_config, verify_ep=verify_ep,
        )
    flavor = f"baked={task or fixed_config}" if fixed_config else "generic"
    typer.echo(f"wrote {output} ({format_}, {precision}, {flavor})")


# =========================================================================
# Model lifecycle: distill, bench, compare
# =========================================================================


_STUDENT_PRESET_CHOICES = "nano | tiny | small | medium"


@app.command()
def distill(
    teacher: Path = typer.Option(
        ..., "--teacher", exists=True, dir_okay=False,
        help="Path to teacher checkpoint (.pt). Typically a NAFNet-large + "
             "adversarial-refine model from a `restora train` run."),
    output: Path = typer.Option(
        ..., "--output", "--out", "-o",
        help="Where to write the student checkpoint. Periodic checkpoints "
             "are saved next to this path as <stem>_iter_NNNNNNN.pt."),
    data: Path = typer.Option(
        ..., "--data", exists=True, file_okay=False,
        help="Image dataset root (will be scanned recursively). Same layout "
             "the trainer uses."),
    video_root: Optional[Path] = typer.Option(
        None, "--video-root", exists=True, file_okay=False,
        help="(optional) Video frame-pair root, with precomputed flow under "
             "<video_root>/<video>/.flow/. Run `restora prepare-videos` or "
             "`restora make-synthetic-videos` to create one. When set, the "
             "student also gets a temporal_pair consistency term on "
             "video batches."),
    video_batch_prob: float = typer.Option(
        0.25, "--video-batch-prob",
        help="Fraction of steps that draw from the video loader instead of "
             "the image loader. Only active when --video-root is set."),
    student_preset: str = typer.Option(
        "small", "--student-preset",
        help=f"Student size preset: {_STUDENT_PRESET_CHOICES}. "
             "nano=5M (no quality), tiny=9M, small=14M (recommended; "
             "~3× faster than teacher), medium=29M (closest to teacher)."),
    nf: Optional[int] = typer.Option(
        None, "--nf",
        help="Override student base channel count (default depends on preset)."),
    enc_depths: Optional[str] = typer.Option(
        None, "--enc-depths",
        help="Override student encoder depths as 4 comma-separated ints, "
             "e.g. '2,2,2,4'."),
    bottle_blocks: Optional[int] = typer.Option(
        None, "--bottle-blocks",
        help="Override student bottleneck transformer-block count."),
    hidden_dim: Optional[int] = typer.Option(
        None, "--hidden-dim",
        help="Override student bottleneck hidden dim."),
    steps: int = typer.Option(50000, "--steps", help="Total training steps."),
    batch_size: int = typer.Option(16, "--batch-size"),
    num_workers: int = typer.Option(
        8, "--num-workers",
        help="DataLoader workers. Cap at ~8 on WSL2 to avoid RAM pressure."),
    input_size: int = typer.Option(256, "--input-size"),
    lr: float = typer.Option(2e-4, "--lr"),
    warmup: int = typer.Option(500, "--warmup"),
    weight_decay: float = typer.Option(0.01, "--weight-decay"),
    clip_grad_norm: float = typer.Option(1.0, "--clip-grad-norm"),
    lambda_chroma: float = typer.Option(
        0.10, "--lambda-chroma",
        help="Weight on the chroma_lab distillation term (colorize-masked). "
             "Default 0.10 matches the trainer's standard loss preset."),
    lambda_temporal: float = typer.Option(
        0.25, "--lambda-temporal",
        help="Weight on flow-warped temporal_pair consistency on video "
             "batches. Only active with --video-root."),
    amp: str = typer.Option(
        "bf16", "--amp",
        help="Mixed-precision dtype: bf16 (recommended on Blackwell/H100/B200), "
             "fp16 (older GPUs), or fp32."),
    ema_decay: float = typer.Option(
        0.0, "--ema-decay",
        help="If > 0, maintain an EMA of student weights. Distillation rarely "
             "needs EMA (teacher already provides smooth targets); off by default."),
    seed: int = typer.Option(0, "--seed"),
    log_every: int = typer.Option(50, "--log-every"),
    save_every: int = typer.Option(
        5000, "--save-every",
        help="Write a numbered checkpoint every N steps. 0 disables periodic saves."),
    device: Optional[str] = typer.Option(
        None, "--device", help="cuda or cpu. Auto-detected when omitted."),
    memory_format: str = typer.Option(
        "channels_last", "--memory-format",
        help="channels_last (recommended on modern GPUs) or contiguous."),
    no_augment_hflip: bool = typer.Option(
        False, "--no-augment-hflip",
        help="Disable horizontal-flip augmentation on the image loader."),
) -> None:
    """Distill a trained teacher into a smaller student.

    Response-based distillation: the teacher generates online RGB targets
    from the same (clean, degraded, config) batches the trainer uses;
    the student is trained to match them via L1 + chroma_lab (on the
    colorize axis) + temporal_pair (on video batches when --video-root
    is set).

    The student checkpoint is written in the same format the trainer
    produces, so `restora export --model OUT.pt` and `restora bench
    --ckpt OUT.pt` work without modification.

    Example (local smoke, ~30 min on Blackwell):
        restora distill \\
          --teacher runs/<run>/ckpt/final.pt \\
          --output runs/distilled/student.pt \\
          --data ~/data/laion-images \\
          --steps 10000 --batch-size 16

    Example (production, with video pairs):
        restora distill \\
          --teacher runs/int_b200/iter_0200000.pt \\
          --output runs/distilled/student.pt \\
          --data ~/data/laion-images \\
          --video-root ~/data/laion-videos \\
          --student-preset small --steps 100000 --batch-size 32
    """
    from restora_models.distill import run_distill, STUDENT_PRESETS

    if student_preset not in STUDENT_PRESETS:
        raise typer.BadParameter(
            f"--student-preset must be one of {list(STUDENT_PRESETS)}; "
            f"got {student_preset!r}")
    enc_depths_list: list[int] | None = None
    if enc_depths is not None:
        try:
            enc_depths_list = [int(x) for x in enc_depths.split(",")]
        except ValueError as e:
            raise typer.BadParameter(f"--enc-depths must be comma-separated ints: {e}") from e
        if len(enc_depths_list) != 4:
            raise typer.BadParameter(
                f"--enc-depths needs 4 values; got {len(enc_depths_list)}")
    if amp not in ("bf16", "fp16", "fp32"):
        raise typer.BadParameter(f"--amp must be bf16/fp16/fp32; got {amp!r}")
    if memory_format not in ("channels_last", "contiguous"):
        raise typer.BadParameter(
            f"--memory-format must be channels_last or contiguous; got {memory_format!r}")

    run_distill(
        teacher=teacher, output=output, data=data,
        video_root=video_root, video_batch_prob=video_batch_prob,
        student_preset=student_preset, nf=nf, enc_depths=enc_depths_list,
        bottle_blocks=bottle_blocks, hidden_dim=hidden_dim,
        steps=steps, batch_size=batch_size, num_workers=num_workers,
        input_size=input_size, lr=lr, warmup=warmup,
        weight_decay=weight_decay, clip_grad_norm=clip_grad_norm,
        lambda_chroma=lambda_chroma, lambda_temporal=lambda_temporal,
        amp=amp, ema_decay=ema_decay, seed=seed,
        log_every=log_every, save_every=save_every,
        device=device, memory_format=memory_format,
        augment_hflip=not no_augment_hflip,
    )


@app.command()
def bench(
    ckpt: Path = typer.Option(
        ..., "--ckpt", "--checkpoint", exists=True, dir_okay=False,
        help="Path to the .pt checkpoint to benchmark."),
    input_size: int = typer.Option(256, "--input-size"),
    batch_size: int = typer.Option(1, "--batch-size"),
    iters: int = typer.Option(100, "--iters"),
    warmup: int = typer.Option(10, "--warmup"),
    device: Optional[str] = typer.Option(None, "--device"),
    amp: str = typer.Option(
        "bf16", "--amp",
        help="fp32 | bf16 | fp16. bf16 is the production default on "
             "Blackwell / H100 / B200."),
    config: str = typer.Option(
        "1 1 1 1 1", "--config",
        help="5 axis flags as space-separated 0/1 (colorize denoise sharpen "
             "dejpeg deblur). Default: all-on."),
    no_ema: bool = typer.Option(
        False, "--no-ema",
        help="Bench the raw model weights instead of EMA shadow. By default "
             "we prefer EMA — same precedence the exporter uses."),
) -> None:
    """Benchmark inference speed of a checkpoint.

    Reports warm-up median, steady-state median + p99, throughput, and
    peak VRAM. EMA weights are loaded when present (override with --no-ema).

    Example:
        restora bench --ckpt runs/int_b200/iter_0200000.pt --iters 100
    """
    from restora_models.bench import run_bench

    flags = [int(c) for c in config.split()]
    if len(flags) != 5:
        raise typer.BadParameter("--config must be 5 0/1 values")
    if amp not in ("fp32", "bf16", "fp16"):
        raise typer.BadParameter(f"--amp must be fp32/bf16/fp16; got {amp!r}")

    run_bench(
        ckpt=ckpt, input_size=input_size, batch_size=batch_size,
        iters=iters, warmup=warmup, device=device, amp=amp,
        config=tuple(flags), use_ema=not no_ema,
    )


@app.command()
def gallery(
    ckpt: Path = typer.Option(
        ..., "--ckpt", "--checkpoint", exists=True, dir_okay=False,
        help="Path to the .pt checkpoint to run inference with."),
    data: Path = typer.Option(
        ..., "--data", exists=True, file_okay=False,
        help="Image dataset root (scanned recursively, same layout as the "
             "trainer uses)."),
    output_dir: Path = typer.Option(
        ..., "--out", "--output", "-o",
        help="Directory to write the side-by-side PNGs into. Created if missing."),
    n: int = typer.Option(
        100, "--n",
        help="Number of images to sample. Capped at len(dataset)."),
    axis: str = typer.Option(
        "colorize", "--axis",
        help="Which single-axis degradation to test. One of: "
             "colorize, denoise, sharpen, dejpeg, deblur."),
    input_size: int = typer.Option(
        256, "--input-size",
        help="Crop size for each sampled image. 256 matches training."),
    seed: int = typer.Option(
        0, "--seed",
        help="Deterministic sample selection + degradation seed."),
    device: Optional[str] = typer.Option(None, "--device"),
    no_labels: bool = typer.Option(
        False, "--no-labels",
        help="Omit the 'original | degraded | restored' caption strip "
             "above each triptych."),
    file_prefix: str = typer.Option(
        "sample", "--file-prefix",
        help="Prefix for output filenames. e.g. <prefix>_007.png."),
) -> None:
    """Build a qualitative inference gallery for a checkpoint.

    For each of N sampled clean images, applies the chosen axis's
    degradation (grayscale-via-Lab-L for `colorize`, Gaussian noise σ=0.03
    for `denoise`, 4× SR for `sharpen`, JPEG Q=40 for `dejpeg`, Gaussian
    blur σ=2.0 for `deblur`), runs the model with that one axis on, and
    writes a side-by-side PNG (original | degraded | restored) to the
    output dir — one file per sample.

    Use this to *eyeball* a checkpoint's behavior — `restora compare`
    tells you if PSNR went up, this tells you WHY (or why not).

    Example:
        restora gallery \\
          --ckpt trained/b200/iter_0300000.pt \\
          --data ~/data/laion-images \\
          --out /tmp/gallery-colorize-300k \\
          --n 100 --axis colorize
    """
    from restora_models.data.compound import AXES
    from restora_models.gallery import run_gallery

    if axis not in AXES:
        raise typer.BadParameter(
            f"--axis must be one of {list(AXES)}; got {axis!r}")
    run_gallery(
        ckpt=ckpt, data=data, out=output_dir,
        n=n, axis=axis, input_size=input_size,
        seed=seed, device=device,
        with_labels=not no_labels, file_prefix=file_prefix,
    )


@app.command()
def compare(
    ckpts: list[Path] = typer.Option(
        ..., "--ckpts", "--checkpoint",
        help="One or more checkpoint paths to compare. The first is the "
             "baseline; later ones are reported as deltas vs the first."),
    data: Path = typer.Option(
        Path("~/data/laion-images"), "--data",
        help="Image dataset root used to build the comparison batch."),
    n: int = typer.Option(
        32, "--n",
        help="How many images to sample for the comparison batch."),
    input_size: int = typer.Option(256, "--input-size"),
    seed: int = typer.Option(
        0, "--seed",
        help="Deterministic sample selection + degradation seed. Same seed = "
             "same batch across runs."),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    """Compare per-task PSNR for two or more checkpoints on a fixed batch.

    Builds a deterministic batch (same N images, same per-axis degradation
    parameters) and reports PSNR(pred, clean) for each restoration axis
    under each checkpoint, plus deltas vs the first checkpoint.

    Example:
        restora compare \\
          --ckpts runs/<smoke>/ckpt/iter_0001000.pt runs/<smoke>/ckpt/final.pt \\
          --data ~/data/laion-images --n 64
    """
    from restora_models.evaluate import run_compare

    run_compare(
        ckpts=list(ckpts), data=data.expanduser(), n=n,
        input_size=input_size, seed=seed, device=device,
    )


# =========================================================================
# Data pipeline: download-*, prepare-videos, precompute-flow, make-synthetic-videos
# =========================================================================


@app.command(name="download-davis")
def download_davis(
    output_dir: Path = typer.Option(
        Path("~/data/laion-videos"), "--out", "--output", "-o",
        help="Output root for video frames. Videos go to "
             "<out>/<video_name>/frame_NNNNN.jpg."),
    cache: Path = typer.Option(
        Path("~/.cache/davis"), "--cache",
        help="Where to put the downloaded DAVIS zip and extract staging."),
    keep_staging: bool = typer.Option(
        False, "--keep-staging/--no-keep-staging",
        help="Don't delete the unzipped staging dir after layout."),
) -> None:
    """Download DAVIS-2017 (480p, train+val).

    Lays out frames as `<out>/<video_name>/frame_NNNNN.jpg`, the format
    VideoPairDataset expects. Re-runs are idempotent — videos already
    present are skipped.
    """
    from restora_models.data.davis import run_download_davis
    run_download_davis(out=output_dir, cache=cache, keep_staging=keep_staging)


@app.command(name="download-imagenet")
def download_imagenet(
    output_dir: Path = typer.Option(
        ..., "--out", "--output", "-o",
        help="Output root. Images go to <out>/<split>/NNNNNNNN.jpg."),
    splits: str = typer.Option(
        "train,val", "--splits",
        help="Comma-separated splits to materialize. Options: train, val, "
             "validation (alias for val), test. Default: train,val."),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir",
        help="Where snapshot_download puts the parquet files. "
             "Default: <out>/.hf-cache (self-contained on the volume)."),
    workers: int = typer.Option(
        4, "--workers",
        help="Parallel parquet → JPG extraction workers. Each holds one "
             "parquet in memory (~600 MB), so cap at 4-8 on modest RAM."),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="List files in the HF repo and exit. Verifies auth and shows "
             "what would be downloaded."),
    keep_parquet: bool = typer.Option(
        False, "--keep-parquet/--no-keep-parquet",
        help="Don't delete parquet files after extraction. Useful for "
             "re-extracting or re-sharding without re-downloading."),
) -> None:
    """Download ImageNet-1k from HuggingFace and extract JPGs.

    Prereqs:
      1. Request access at https://huggingface.co/datasets/imagenet-1k
         (gated, but approval is usually instant).
      2. `huggingface-cli login` or set HF_TOKEN.

    Bundled-parquet path: much faster than img2dataset on bundled HF
    datasets. Full train+val is ~163 GB on disk and takes 30-60 min on
    a fast link.
    """
    from restora_models.data.imagenet1k import run_download_imagenet
    split_list = tuple(s.strip() for s in splits.split(",") if s.strip())
    valid = {"train", "val", "validation", "test"}
    bad = [s for s in split_list if s not in valid]
    if bad:
        raise typer.BadParameter(f"unknown split(s) {bad}; allowed: {sorted(valid)}")
    rc = run_download_imagenet(
        out=output_dir, splits=split_list, cache_dir=cache_dir,
        workers=workers, dry_run=dry_run, keep_parquet=keep_parquet,
    )
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command(name="download-openimages")
def download_openimages(
    output_dir: Path = typer.Option(
        ..., "--out", "--output", "-o",
        help="Output root. Images saved as <out>/<split>/<id>.jpg."),
    split: str = typer.Option(
        "validation", "--split",
        help="Which split to download: train (~9M, ~4 TB), validation "
             "(~41K, ~20 GB), test (~125K, ~60 GB). Default: validation."),
    limit: Optional[int] = typer.Option(
        None, "--limit",
        help="Max number of images to download. None = all. Use --limit 5 "
             "for a quick smoke test or --limit 1000000 for a 1M sample."),
    threads: int = typer.Option(
        64, "--threads",
        help="Concurrent download threads. Default 64. S3 happily serves "
             "thousands, but a single Python process plateaus around 128."),
    dry_run: bool = typer.Option(
        False, "--dry-run/--no-dry-run",
        help="List image keys but don't download anything."),
    print_every: int = typer.Option(
        100, "--print-every",
        help="Progress print cadence (every N completions)."),
) -> None:
    """Download Open Images from the public AWS S3 mirror.

    No auth, no rate limits, fast CDN. Discovery via the S3 XML bucket
    listing API; downloads via stdlib urllib + a thread pool. Resumable:
    existing files with non-zero size are skipped.
    """
    from restora_models.data.openimages import run_download_openimages
    rc = run_download_openimages(
        out=output_dir, split=split, limit=limit, threads=threads,
        dry_run=dry_run, print_every=print_every,
    )
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command(name="prepare-videos")
def prepare_videos(
    output_dir: Path = typer.Option(
        ..., "--out", "--output", "-o",
        help="Output root for video frames + flow. Use /workspace/data-videos "
             "on the B200 server (matches video.root in configs/b200.yaml)."),
    cache: Optional[Path] = typer.Option(
        None, "--cache",
        help="Where to put the downloaded DAVIS zip. "
             "Default: <out>/.davis-cache (self-contained)."),
    max_skip: int = typer.Option(
        5, "--max-skip",
        help="Max temporal skip for flow precompute (k=1..max_skip). "
             "Default 5 matches production video.max_skip."),
    resolution: int = typer.Option(
        256, "--resolution",
        help="Resolution for flow precompute. Default 256 = training res."),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="Device for RAFT. Default cuda if available, else cpu."),
    print_every: int = typer.Option(
        20, "--print-every",
        help="RAFT progress every N videos."),
    skip_davis: bool = typer.Option(
        False, "--skip-davis/--no-skip-davis",
        help="Skip DAVIS download/layout (assume frames are present)."),
    skip_flow: bool = typer.Option(
        False, "--skip-flow/--no-skip-flow",
        help="Skip RAFT flow precompute (frames only)."),
    keep_staging: bool = typer.Option(
        False, "--keep-staging/--no-keep-staging",
        help="Don't delete the unzipped DAVIS staging dir."),
) -> None:
    """One-shot video dataset prep: DAVIS download + RAFT flow precompute.

    Wraps download-davis + precompute-flow end-to-end. Both steps are
    idempotent; re-runs skip work that's already done.

    Expected runtime on B200:
      - DAVIS download + extract: ~5-10 min (~480 MB zip)
      - RAFT flow precompute:     ~20-40 min (~50k flow pairs)
    """
    from restora_models.data.video_prep import run_prepare_videos
    run_prepare_videos(
        out=output_dir, cache=cache, max_skip=max_skip, resolution=resolution,
        device=device, print_every=print_every,
        skip_davis=skip_davis, skip_flow=skip_flow, keep_staging=keep_staging,
    )


@app.command(name="precompute-flow")
def precompute_flow(
    root: Path = typer.Option(
        ..., "--root", exists=True, file_okay=False,
        help="Video root with one subdir per video, each containing "
             "frame_NNNNN.jpg files."),
    max_skip: int = typer.Option(
        5, "--max-skip",
        help="Max temporal skip k (compute flow for k=1..max_skip)."),
    resolution: int = typer.Option(
        256, "--resolution",
        help="Resolution to compute flow at (square). RAFT runs at any "
             "size; 256 is fastest and matches training."),
    device: Optional[str] = typer.Option(None, "--device"),
    print_every: int = typer.Option(20, "--print-every"),
) -> None:
    """Precompute RAFT optical flow for all video frame pairs.

    Writes backward flow tk→t into <video>/.flow/frame_NNNNN_skipK.npz
    (key: 'flow', shape (2,H,W)). Resumable: existing flow files are
    skipped.

    Use after `restora download-davis` when you want to control the
    flow-only step separately, or after manually populating frames.
    """
    from restora_models.data.flow_precompute import run_precompute_flow
    run_precompute_flow(
        root=root, max_skip=max_skip, resolution=resolution,
        device=device, print_every=print_every,
    )


@app.command(name="make-synthetic-videos")
def make_synthetic_videos(
    source: Path = typer.Option(
        ..., "--source", exists=True, file_okay=False,
        help="Source image dataset root (LAION-style layout)."),
    output_dir: Path = typer.Option(
        ..., "--out", "--output", "-o",
        help="Output root. Videos are created as <out>/vid_NNNNN/."),
    num_videos: int = typer.Option(
        200, "--num-videos", help="How many videos to generate."),
    frames_per_video: int = typer.Option(5, "--frames-per-video"),
    resolution: int = typer.Option(256, "--resolution"),
    max_skip: int = typer.Option(
        5, "--max-skip",
        help="Precompute flow for skip = 1..max_skip."),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Generate synthetic video pairs from existing image data.

    For each source image, generates K-frame "videos" by applying smooth
    per-frame affine transforms. Backward flow is computed analytically
    from the affines (zero estimation error — cleaner than RAFT).

    Useful for:
      - Validating the trainer's video path before real video data exists
      - Adding training-time temporal signal when no DAVIS-style dataset
        is available
    """
    from restora_models.data.synthetic_videos import run_make_synthetic_videos
    run_make_synthetic_videos(
        source=source, out=output_dir,
        num_videos=num_videos, frames_per_video=frames_per_video,
        resolution=resolution, max_skip=max_skip, seed=seed,
    )
