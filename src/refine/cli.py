"""refine CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="refine — multi-task image restoration", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """refine CLI."""


@app.command()
def version() -> None:
    from refine import __version__
    typer.echo(__version__)


@app.command(name="scan-data")
def scan_data(root: Path = typer.Option(..., "--root", exists=True, file_okay=False)) -> None:
    from refine.data.dataset import build_manifest
    paths = build_manifest(root, force=True)
    typer.echo(f"{len(paths)} images indexed under {root}")


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
    from refine.config import load_config
    from refine.train import Trainer
    from refine.train.checkpoint import load_checkpoint

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
    from refine.infer.pipeline import load_pipeline

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


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(256, "--input-size"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    dynamic_hw: bool = typer.Option(False, "--dynamic-hw/--fixed-hw"),
) -> None:
    import torch
    from refine.config import ModelConfig
    from refine.data.compound import AXES
    from refine.export.onnx import export_onnx_from_model
    from refine.models import build_model

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg = ModelConfig(**(cfg_dict.get("model") or {}))
    m = build_model(mcfg, num_axes=len(AXES))
    m.load_state_dict(payload["model"])
    task_map = payload.get("task_map") or {}
    export_onnx_from_model(
        m, num_axes=len(AXES), input_size=input_size,
        export_path=output, opset=opset, simplify=simplify,
        dynamic_hw=dynamic_hw, task_map=task_map,
    )
    typer.echo(f"wrote {output}")
