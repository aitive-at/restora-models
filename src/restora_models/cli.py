"""restora-models CLI for the temporal old-film remaster model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(
    help="restora — temporal multi-task video restoration (colorize / denoise / sharpen / dejpeg / deblur)",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """restora CLI."""


@app.command()
def version() -> None:
    """Print the package version."""
    from restora_models import __version__
    typer.echo(__version__)


# ---- prepare-data umbrella ------------------------------------------------

prepare_app = typer.Typer(help="Download / prepare training datasets.", no_args_is_help=True)
app.add_typer(prepare_app, name="prepare-data")


@prepare_app.command("film-overlays")
def prepare_film_overlays(
    output_dir: Path = typer.Option(..., "--out", "-o",
                                     help="Where to extract the DeepRemaster noise_data.zip"),
    keep_zip: bool = typer.Option(False, "--keep-zip", help="Don't delete the downloaded zip"),
) -> None:
    """Download + extract DeepRemaster's noise_data.zip (898 MB, ~6152 textures)."""
    from restora_models.cli_prepare import download_film_overlays
    download_film_overlays(output_dir, keep_zip=keep_zip)


@prepare_app.command("reds")
def prepare_reds(
    output_dir: Path = typer.Option(..., "--out", "-o",
                                     help="REDS root (expects sequences as subdirs)"),
) -> None:
    """Verify REDS dataset layout and print a manifest. REDS is gated; the
    user must download it manually first from
    https://seungjunnah.github.io/Datasets/reds.html.
    """
    from restora_models.cli_prepare import verify_reds
    verify_reds(output_dir)


@prepare_app.command("vimeo")
def prepare_vimeo(
    output_dir: Path = typer.Option(..., "--out", "-o", help="Vimeo Septuplet root"),
) -> None:
    """Verify Vimeo Septuplet layout and print a manifest."""
    from restora_models.cli_prepare import verify_vimeo
    verify_vimeo(output_dir)


# ---- train ---------------------------------------------------------------

@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False,
                                help="Training config YAML"),
    run_name: Optional[str] = typer.Option(None, "--run-name"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    total_steps: Optional[int] = typer.Option(None, "--total-steps"),
    compile_: bool = typer.Option(False, "--compile/--no-compile"),
    amp: Optional[str] = typer.Option(None, "--amp"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Override cfg.run.root"),
) -> None:
    """Train the temporal restoration model from a config."""
    from restora_models.config import load_config
    from restora_models.train.trainer import Trainer

    cfg = load_config(config)
    if run_name is not None:
        cfg.run.name = run_name
    if batch_size is not None:
        cfg.data.loader.batch_size = int(batch_size)
    if total_steps is not None:
        cfg.train.total_steps = int(total_steps)
        cfg.scheduler.total_steps = int(total_steps)
    if compile_:
        cfg.train.compile = True
    if amp is not None:
        cfg.train.amp = amp
    trainer = Trainer(cfg, out_dir=out_dir)
    final = trainer.fit()
    typer.echo(f"final checkpoint: {final}")


# ---- train-flow-distill (Phase 17 will implement) -------------------------

@app.command(name="train-flow-distill")
def train_flow_distill(
    config: Optional[Path] = typer.Option(None, "--config"),
    out_dir: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Pre-train the static-unroll FlowDistill student. Implemented in Phase 17."""
    try:
        from restora_models.train.flow_distill import run_flow_distill
    except ImportError:
        typer.secho("train-flow-distill: not implemented yet (Phase 17 pending)",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)
    run_flow_distill(out_dir=out_dir, config_path=config)


# ---- train-pipeline (Phase 18 orchestrator) -------------------------------

@app.command(name="train-pipeline")
def train_pipeline(
    config: Optional[Path] = typer.Option(None, "--config"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
    run_root: Optional[Path] = typer.Option(None, "--run-root"),
    extend_from: Optional[str] = typer.Option(None, "--extend-from"),
) -> None:
    """End-to-end multi-stage training pipeline. Implemented in Phase 18."""
    try:
        from restora_models.train.pipeline import run_pipeline
        from restora_models.train.pipeline_state import STAGE_ORDER
    except ImportError:
        typer.secho("train-pipeline: not implemented yet (Phase 18 pending)",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)
    if resume is not None and run_root is None:
        run_root = resume
    if run_root is None:
        raise typer.BadParameter("--run-root or --resume is required")
    if extend_from is not None and extend_from not in STAGE_ORDER:
        raise typer.BadParameter(f"--extend-from must be one of {list(STAGE_ORDER)}")
    if config is None and resume is None:
        raise typer.BadParameter("--config is required when starting fresh")
    run_pipeline(run_root=run_root, config_path=config, extend_from=extend_from)


# ---- inference / export / lifecycle (later phases fill these in) ---------

@app.command()
def infer(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    input_: Path = typer.Option(..., "--input", "--in", exists=True),
    output: Path = typer.Option(..., "--output", "--out"),
    color: bool = typer.Option(False, "--color/--no-color"),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise"),
    sharp: bool = typer.Option(False, "--sharp/--no-sharp"),
    dejpeg: bool = typer.Option(False, "--dejpeg/--no-dejpeg"),
    deblur: bool = typer.Option(False, "--deblur/--no-deblur"),
) -> None:
    """Run inference on a single image or a directory."""
    import cv2
    import torch
    from restora_models.infer.pipeline import VideoPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = VideoPipeline.from_checkpoint(model, device=device)
    config = {"colorize": color, "denoise": denoise, "sharpen": sharp,
              "dejpeg": dejpeg, "deblur": deblur}
    if input_.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(input_))
        if img is None:
            raise typer.BadParameter(f"could not read {input_}")
        cv2.imwrite(str(output), pipe.process_image(img, config=config))
    else:
        output.mkdir(parents=True, exist_ok=True)
        pipe.process_directory(input_, output, config=config)
    typer.echo(f"wrote {output}")


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(256, "--input-size"),
    precision: str = typer.Option("fp16", "--precision"),
    task: Optional[str] = typer.Option(
        None, "--task",
        help="Bake a single-task config: colorize/denoise/sharpen/dejpeg/deblur/all"),
    dynamic_hw: bool = typer.Option(True, "--dynamic-hw/--fixed-hw"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    verify_ep: Optional[str] = typer.Option(None, "--verify-ep"),
) -> None:
    """Export a checkpoint to ONNX."""
    import torch
    from restora_models.config import ModelConfig
    from restora_models.export.onnx import export_onnx_from_model, TASK_CONFIGS
    from restora_models.models.registry import build_model

    if task is not None and task not in TASK_CONFIGS:
        raise typer.BadParameter(f"--task must be in {sorted(TASK_CONFIGS)}")

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    # Try to recover model config from sidecar 'cfg'; fall back to default
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mtype = (cfg_dict.get("model") or {}).get("type", "temporal_restora_small")
    mcfg = ModelConfig(type=mtype)
    m = build_model(mcfg, num_axes=5)
    m.load_state_dict(payload["model"])

    fixed_config = TASK_CONFIGS[task] if task is not None else None
    task_map = {"task": task} if task is not None else None

    out_path = export_onnx_from_model(
        m, num_axes=5, input_size=input_size, export_path=output,
        opset=opset, simplify=simplify, dynamic_hw=dynamic_hw,
        task_map=task_map, precision=precision, fixed_config=fixed_config,
        verify_ep=verify_ep,
    )
    typer.echo(f"wrote {out_path} ({precision}, {'baked='+task if task else 'generic'})")


@app.command()
def distill(
    teacher: Path = typer.Option(..., "--teacher", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    student_preset: str = typer.Option("small", "--student-preset"),
) -> None:
    """SLKD-style distillation. Implemented in Phase 14."""
    try:
        from restora_models.train.distill import run_distill
    except ImportError:
        typer.secho("distill: not implemented yet (Phase 14 pending)",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=2)
    typer.echo("distill: see Phase 14 implementation")


@app.command()
def bench(
    ckpt: Path = typer.Option(..., "--ckpt", exists=True, dir_okay=False),
    iters: int = typer.Option(100, "--iters"),
) -> None:
    """Benchmark inference latency / throughput on a checkpoint."""
    from restora_models.train.bench import run_bench
    run_bench(ckpt=ckpt, iters=iters)


@app.command()
def compare(
    ckpts: list[Path] = typer.Option(..., "--ckpts"),
    n: int = typer.Option(32, "--n"),
    data: Path = typer.Option(Path("~/data/reds").expanduser(), "--data",
                              help="REDS root for the holdout sampler"),
) -> None:
    """Compare per-axis PSNR across one or more checkpoints."""
    from restora_models.train.evaluate import run_compare
    run_compare(ckpts=list(ckpts), data=data, n=n)


@app.command()
def gallery(
    ckpt: Path = typer.Option(..., "--ckpt", exists=True, dir_okay=False),
    data: Path = typer.Option(..., "--data", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out", "-o"),
    n: int = typer.Option(16, "--n"),
    axis: str = typer.Option("colorize", "--axis",
                             help="Restoration axis to visualize"),
) -> None:
    """Generate qualitative triptych gallery (clean | degraded | restored)."""
    from restora_models.train.gallery import run_gallery
    run_gallery(ckpt=ckpt, data=data, out=out, n=n, axis=axis)


if __name__ == "__main__":
    app()
