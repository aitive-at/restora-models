"""Data preparation helpers (download + verify layout)."""
from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import typer


# DeepRemaster scratch/dust/grain texture pack. The original
# `iizuka.cs.tsukuba.ac.jp` host is dead as of mid-2026; we keep the URL
# here for posterity but the workflow falls back to synthesis when the
# fetch fails.
FILM_OVERLAY_URL = "http://iizuka.cs.tsukuba.ac.jp/projects/remastering/data/noise_data.zip"


def download_film_overlays(output_dir: Path, *, keep_zip: bool = False,
                            n_synthetic: int = 600) -> None:
    """Try the DeepRemaster fetch; fall back to local synthesis if the
    source URL is unreachable. Either way ``output_dir`` ends up with a
    pile of grayscale PNG textures the wrapper can load."""
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "noise_data.zip"
    fetched = False
    if zip_path.exists():
        typer.echo(f"already downloaded: {zip_path}")
        fetched = True
    else:
        try:
            typer.echo(f"trying upstream {FILM_OVERLAY_URL} -> {zip_path} (~898 MB)")
            with urllib.request.urlopen(FILM_OVERLAY_URL, timeout=30) as src, \
                    open(zip_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            typer.echo(f"download complete: {zip_path.stat().st_size / 1e6:.1f} MB")
            fetched = True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, OSError) as exc:
            zip_path.unlink(missing_ok=True)
            typer.secho(
                f"upstream fetch failed ({exc.__class__.__name__}: {exc}); "
                f"falling back to synthetic textures.",
                fg=typer.colors.YELLOW, err=True,
            )

    if fetched:
        typer.echo("extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        n = sum(1 for _ in output_dir.rglob("*.png"))
        typer.echo(f"extracted {n} PNG textures to {output_dir}")
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
            typer.echo("removed zip; pass --keep-zip to retain")
        return

    # Synthesis path — generate `n_synthetic` PNGs (~5 categories blended).
    typer.echo(f"synthesizing {n_synthetic} film-overlay textures to {output_dir}")
    synthesize_film_overlays(output_dir, n=n_synthetic, seed=0)
    n_total = sum(1 for _ in output_dir.rglob("*.png"))
    typer.secho(f"done: {n_total} PNGs at {output_dir}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------
# Synthetic film-overlay generation
# ---------------------------------------------------------------------
#
# The trainer's FilmOverlayDegradation blends textures additively:
# ``out = rgb + alpha * crop`` with alpha ∈ [0.1, 0.4]. So dark pixels
# leave RGB unchanged and bright pixels look like scratches or dust on
# the print. We bias every synthetic texture dark (mean ≈ 0.05) and
# scatter bright artifacts on top.
#
# Categories (per 600 default):
#   - 40% fine grain        — multi-scale Gaussian noise (subtle film texture)
#   - 25% vertical scratches — 1-3 white vertical lines, slight blur
#   - 15% dust spots         — 5-30 small Gaussian blobs
#   - 10% hair / curly lines — short Bezier-sampled curves
#   - 10% heavy mixed        — combinations / dense dust + scratches


def _grain_only(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Fine-grained low-amplitude noise biased toward black."""
    # Two octaves of noise summed for slight roughness.
    coarse = rng.normal(0.0, 1.0, (size // 4, size // 4)).astype(np.float32)
    coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_LINEAR)
    fine = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    n = 0.65 * fine + 0.35 * coarse
    # Half-wave rectify and scale — only positive (brightening) parts
    # survive, which matches the additive blend semantics.
    n = np.clip(n - 0.5, 0.0, None)
    n = n / max(n.max(), 1e-6) * rng.uniform(0.3, 0.8)
    return n.astype(np.float32)


def _vertical_scratches(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """1-3 vertical lines, sometimes interrupted, sometimes feathered."""
    img = np.zeros((size, size), dtype=np.float32)
    n_lines = int(rng.integers(1, 4))
    for _ in range(n_lines):
        x = int(rng.integers(2, size - 2))
        width = int(rng.integers(1, 4))
        brightness = float(rng.uniform(0.6, 1.0))
        y0 = int(rng.integers(0, size // 2))
        y1 = int(rng.integers(size // 2, size))
        img[y0:y1, x:x + width] = brightness
        # Random small breaks
        for _ in range(int(rng.integers(0, 4))):
            yb = int(rng.integers(y0, y1))
            img[yb:yb + int(rng.integers(1, 6)), x:x + width] = 0.0
    # Soft feathering so the scratches don't look pixel-perfect.
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=float(rng.uniform(0.3, 0.9)))
    return img


def _dust_spots(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """5-30 scattered bright blobs, occasionally clustered."""
    img = np.zeros((size, size), dtype=np.float32)
    n_spots = int(rng.integers(5, 31))
    for _ in range(n_spots):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        r = int(rng.integers(1, 6))
        brightness = float(rng.uniform(0.5, 1.0))
        cv2.circle(img, (cx, cy), r, brightness, thickness=-1, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=float(rng.uniform(0.5, 1.2)))
    return img


def _curly_hairs(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """0-2 thin curvy lines drawn as polyline through random control points."""
    img = np.zeros((size, size), dtype=np.float32)
    n_hairs = int(rng.integers(1, 3))
    for _ in range(n_hairs):
        n_pts = int(rng.integers(6, 14))
        # Random walk from a starting edge across the frame
        x = float(rng.uniform(0, size))
        y = float(rng.uniform(0, size))
        pts = [(int(x), int(y))]
        for _ in range(n_pts):
            x += float(rng.normal(0, 12))
            y += float(rng.normal(0, 12))
            pts.append((int(np.clip(x, 0, size - 1)),
                        int(np.clip(y, 0, size - 1))))
        thickness = int(rng.integers(1, 3))
        brightness = float(rng.uniform(0.4, 0.9))
        cv2.polylines(img, [np.array(pts, dtype=np.int32)],
                      isClosed=False, color=brightness,
                      thickness=thickness, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=float(rng.uniform(0.4, 0.9)))
    return img


def _heavy_mixed(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Combinations for dirtier 'used print' frames."""
    a = _grain_only(rng, size) * float(rng.uniform(0.5, 1.0))
    b = _dust_spots(rng, size)
    img = np.clip(a + b, 0.0, 1.0)
    if rng.random() < 0.5:
        img = np.clip(img + _vertical_scratches(rng, size), 0.0, 1.0)
    return img


_CATEGORIES = [
    ("grain",       _grain_only,        0.40),
    ("scratch",     _vertical_scratches, 0.25),
    ("dust",        _dust_spots,        0.15),
    ("hair",        _curly_hairs,       0.10),
    ("heavy",       _heavy_mixed,       0.10),
]


def synthesize_film_overlays(out_dir: Path, *, n: int = 600,
                              size: int = 256, seed: int = 0) -> None:
    """Fill ``out_dir`` with ``n`` synthetic grayscale-PNG film overlays.

    Output PNGs are 8-bit grayscale, named ``synth_<category>_<idx>.png``.
    Format matches what ``FilmOverlayDegradation.from_dir`` consumes:
    grayscale, additive blend ``out = rgb + alpha * crop`` where dark
    pixels ≈ no effect and bright pixels ≈ scratch/dust/grain visible.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Build a list of (category, generator) draws weighted by share
    cats = [c[0] for c in _CATEGORIES]
    gens = {c[0]: c[1] for c in _CATEGORIES}
    weights = np.array([c[2] for c in _CATEGORIES])
    weights = weights / weights.sum()
    picks = rng.choice(cats, size=n, p=weights)
    counts = {c: 0 for c in cats}
    for i, cat in enumerate(picks):
        img = gens[cat](rng, size)
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        idx = counts[cat]
        counts[cat] += 1
        path = out_dir / f"synth_{cat}_{idx:04d}.png"
        ok = cv2.imwrite(str(path), img_u8,
                         [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed: {path}")
    typer.echo("category counts: " + ", ".join(
        f"{c}={counts[c]}" for c in cats))


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
