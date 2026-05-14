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
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    dynamic_hw: bool = typer.Option(False, "--dynamic-hw/--fixed-hw"),
    precision: str = typer.Option("fp32", "--precision",
                                  help="fp32 (default) | fp16 | fp8 | fp4"),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Bake a specific task's config into the ONNX (resulting graph "
             "has ONLY 'input' tensor, no 'config'). Options: colorize | "
             "denoise | sharpen | dejpeg | deblur | all"),
    bake_axes: Optional[str] = typer.Option(
        None, "--bake-axes",
        help="Comma-separated 5-axis vector to bake into the ONNX, e.g. "
             "'1,0,1,0,0' for colorize+sharpen. Mutually exclusive with --task."),
) -> None:
    """Export a trained refine checkpoint to ONNX.

    Default export emits a 2-input ONNX (input, config) -> output that
    works for any of the 5 restoration axes by setting the config vector.

    Pass --task NAME or --bake-axes V1,V2,V3,V4,V5 to emit a per-task
    ONNX with the config tensor BAKED in as a buffer; that resulting
    file has only an `input` tensor — "RGB in, RGB out" — and is what
    deployment consumers usually want."""
    import torch
    from restora_models.config import ModelConfig
    from restora_models.data.compound import AXES
    from restora_models.export.onnx import export_onnx_from_model
    from restora_models.models import build_model

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
    effective_opset = max(opset, 19) if precision == "fp8" else opset
    export_onnx_from_model(
        m, num_axes=len(AXES), input_size=input_size,
        export_path=output, opset=effective_opset, simplify=simplify,
        dynamic_hw=dynamic_hw, task_map=task_map, precision=precision,
        fixed_config=fixed_config,
    )
    flavor = f"baked={task or fixed_config}" if fixed_config else "generic"
    typer.echo(f"wrote {output} ({precision}, {flavor})")
