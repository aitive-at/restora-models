"""coliraz CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="coliraz — image colorization", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """coliraz CLI."""


@app.command()
def version() -> None:
    from coliraz import __version__
    typer.echo(__version__)


@app.command(name="scan-data")
def scan_data(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
) -> None:
    """Build/refresh the recursive image manifest under ROOT."""
    from coliraz.data.dataset import build_manifest

    paths = build_manifest(root, force=True)
    typer.echo(f"{len(paths)} images indexed under {root}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    data: Optional[Path] = typer.Option(None, "--data", help="override data.root"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="override run.name"),
    batch_size: Optional[str] = typer.Option(None, "--batch-size"),
    compile_: bool = typer.Option(False, "--compile/--no-compile", help="torch.compile model"),
    amp: Optional[str] = typer.Option(None, "--amp", help="bf16|fp16|fp32"),
    total_steps: Optional[int] = typer.Option(None, "--total-steps"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
) -> None:
    """Train a colorization model from images recursively rooted at --data."""
    from coliraz.config import load_config
    from coliraz.train import Trainer
    from coliraz.train.checkpoint import load_checkpoint

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
        load_checkpoint(
            resume,
            model=trainer.model,
            optimizer=trainer.opt_g,
            optimizer_d=trainer.opt_d,
            discriminator=trainer.disc,
            ema=trainer.ema,
            scheduler=trainer.scheduler_g,
        )
    trainer.fit()


@app.command()
def infer(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    input_: Path = typer.Option(..., "--input", "--in", exists=True),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(512, "--input-size"),
) -> None:
    """Colorize a single image or a directory (recursive)."""
    import cv2
    import torch

    from coliraz.infer.pipeline import load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(model, input_size=input_size, device=device)
    if input_.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(input_))
        if img is None:
            raise typer.BadParameter(f"could not read {input_}")
        cv2.imwrite(str(output), pipe.process(img))
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
            cv2.imwrite(str(out_path), pipe.process(img))
    typer.echo(f"wrote {output}")


@app.command()
def export(
    model: Path = typer.Option(..., "--model", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "--out"),
    input_size: int = typer.Option(512, "--input-size"),
    opset: int = typer.Option(17, "--opset"),
    simplify: bool = typer.Option(True, "--simplify/--no-simplify"),
    dynamic_hw: bool = typer.Option(
        False,
        "--dynamic-hw/--fixed-hw",
        help=(
            "Export with dynamic height and width axes so the ONNX model "
            "accepts any (B, 3, H, W) fp32 input. --input-size is used as "
            "the tracing shape only. Transformer attention is O((HW)^2) so "
            "memory grows with resolution. Default is fixed H/W."
        ),
    ),
) -> None:
    """Export a checkpoint to ONNX with parity verification."""
    import torch

    from coliraz.config import ModelConfig
    from coliraz.export.onnx import export_onnx_from_model
    from coliraz.models import build_ddcolor

    payload = torch.load(str(model), map_location="cpu", weights_only=False)
    cfg_dict = (payload.get("extra") or {}).get("cfg", {})
    mcfg_dict = cfg_dict.get("model") or {"input_size": input_size}
    mcfg = ModelConfig(**mcfg_dict)
    m = build_ddcolor(mcfg, pretrained=False)
    m.load_state_dict(payload["model"])
    export_onnx_from_model(
        m,
        input_size=input_size,
        export_path=output,
        opset=opset,
        simplify=simplify,
        dynamic_hw=dynamic_hw,
    )
    typer.echo(f"wrote {output}")
