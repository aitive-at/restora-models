"""Data preparation helpers (download + verify layout)."""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import typer


# DeepRemaster scratch/dust/grain texture pack (898 MB).
# Public, no auth, slow but reliable single-source download.
FILM_OVERLAY_URL = "http://iizuka.cs.tsukuba.ac.jp/projects/remastering/data/noise_data.zip"


def download_film_overlays(output_dir: Path, *, keep_zip: bool = False) -> None:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "noise_data.zip"
    if zip_path.exists():
        typer.echo(f"already downloaded: {zip_path}")
    else:
        typer.echo(f"downloading {FILM_OVERLAY_URL} -> {zip_path} (~898 MB)")
        with urllib.request.urlopen(FILM_OVERLAY_URL) as src, open(zip_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        typer.echo(f"download complete: {zip_path.stat().st_size / 1e6:.1f} MB")
    typer.echo("extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    n = sum(1 for _ in output_dir.rglob("*.png"))
    typer.echo(f"extracted {n} PNG textures to {output_dir}")
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        typer.echo("removed zip; pass --keep-zip to retain")


def verify_reds(root: Path) -> None:
    root = root.expanduser()
    if not root.exists():
        typer.secho(f"REDS root not found: {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    # Look for sequences either flat or under train_sharp/
    candidates = [root, root / "train_sharp"]
    seq_root = next((c for c in candidates if c.is_dir() and any(c.iterdir())), None)
    if seq_root is None:
        typer.secho(f"no sequences found in {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    seqs = [p for p in seq_root.iterdir() if p.is_dir()]
    typer.echo(f"REDS root: {seq_root}")
    typer.echo(f"sequences: {len(seqs)}")
    if seqs:
        sample = seqs[0]
        frames = sorted(sample.glob("[0-9]*.png"))
        typer.echo(f"sample {sample.name}: {len(frames)} frames")


def verify_vimeo(root: Path) -> None:
    root = root.expanduser()
    if not root.exists():
        typer.secho(f"Vimeo root not found: {root}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    train_list = root / "sep_trainlist.txt"
    if not train_list.exists():
        typer.secho(f"missing sep_trainlist.txt at {train_list}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    n_train = len([ln for ln in train_list.read_text().splitlines() if ln.strip()])
    typer.echo(f"Vimeo Septuplet: {n_train} train clips")
