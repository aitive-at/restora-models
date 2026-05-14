"""Download LAION-aesthetic image datasets via img2dataset.

Two-step pipeline:

  1) Fetch the parquet metadata shards from HuggingFace (gated; needs an HF
     token + access granted on the dataset page). Resumable: existing
     parquet files are skipped on re-run.
  2) Run img2dataset to download the actual JPEGs from the parquet URLs
     into a sharded directory tree. img2dataset is resumable by design —
     it skips shards whose output dir already exists with a completion
     sentinel file.

The HF token is read from `~/.cache/huggingface/token` by `huggingface_hub`
automatically, or you can set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`.

See `docs/integration/laion-download.md` for the deployment recipe and
B200 server checklist.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Three LAION-aesthetic subsets exposed on HuggingFace. Each is sharded
# into 128 numbered parquet "part" files; the UUID is dataset-specific
# and embedded in every filename. UUIDs verified 2026-05-14 against the
# upstream HF dataset listings.
@dataclass(frozen=True)
class LaionDataset:
    name: str
    uuid: str
    approx_images_m: float          # millions of samples in this subset
    description: str


# Note (2026-05-14): After the 2023 LAION takedown and 2024 republish,
# two of the three subsets were renamed on HuggingFace with `re` prefix and
# a fresh UUID. The English subset kept its original name+UUID.
# Names + UUIDs verified live against the HF dataset API.
_DATASETS: dict[str, LaionDataset] = {
    "relaion2B-multi-aesthetic": LaionDataset(
        name="relaion2B-multi-aesthetic",
        uuid="2ec10f02-51eb-4e2e-9b77-103ee7982d99",
        approx_images_m=17.0,
        description="Multi-language (non-English) re-released subset, aesthetic>=7",
    ),
    "laion2B-en-aesthetic": LaionDataset(
        name="laion2B-en-aesthetic",
        uuid="9230b837-b1e0-4254-8b88-ed2976e9cee9",
        approx_images_m=51.0,
        description="English subset (not renamed; original name+UUID), aesthetic>=7",
    ),
    "relaion1B-nolang-aesthetic": LaionDataset(
        name="relaion1B-nolang-aesthetic",
        uuid="a718cdfa-8fa6-4f99-a950-2ffa6b13c6c4",
        approx_images_m=52.0,
        description="No language tag, re-released subset, aesthetic>=7",
    ),
}

NUM_PARQUET_SHARDS = 128
"""Every LAION-aesthetic subset is split into exactly 128 parquet files
(part-00000 through part-00127). This is upstream-fixed."""


def list_datasets() -> list[str]:
    return list(_DATASETS.keys())


def _parquet_filename(ds: LaionDataset, shard_idx: int) -> str:
    """Reconstruct the exact HF filename for a given shard.

    Format: `part-{NNNNN}-{uuid}-c000.snappy.parquet`
    """
    return f"part-{shard_idx:05d}-{ds.uuid}-c000.snappy.parquet"


def _parquet_repo_path(ds: LaionDataset, shard_idx: int) -> str:
    """Path within the HF dataset repo — the file lives at the repo root."""
    return _parquet_filename(ds, shard_idx)


def download_parquets(
    dataset: str,
    metadata_dir: Path,
    *,
    max_shards: int | None = None,
    print_every: int = 10,
) -> list[Path]:
    """Download missing parquet shards into `metadata_dir`.

    Returns the list of all expected parquet paths (whether freshly downloaded
    or already present). Skips files that already exist and have non-zero size.

    `max_shards` caps the download to the first N shards (useful for partial
    deployments or local testing).
    """
    if dataset not in _DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; options: {sorted(_DATASETS)}"
        )
    ds = _DATASETS[dataset]
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    n_shards = max_shards if max_shards is not None else NUM_PARQUET_SHARDS
    n_shards = min(n_shards, NUM_PARQUET_SHARDS)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub not installed. Run: uv sync"
        ) from e

    paths: list[Path] = []
    skipped = 0
    downloaded = 0

    for i in range(n_shards):
        fname = _parquet_filename(ds, i)
        target = metadata_dir / fname
        paths.append(target)

        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            if (i + 1) % print_every == 0:
                print(f"[{i+1}/{n_shards}] {downloaded} downloaded, {skipped} skipped (resume)",
                      flush=True)
            continue

        # hf_hub_download downloads to a cache, then we move/copy.
        # local_dir mode places it directly at `local_dir/<filename>`.
        local_path = hf_hub_download(
            repo_id=f"laion/{ds.name}",     # explicit name, not user-passed key
            filename=_parquet_repo_path(ds, i),
            repo_type="dataset",
            local_dir=str(metadata_dir),
        )
        # hf_hub_download returns the local path
        if Path(local_path).resolve() != target.resolve():
            # Some hf_hub versions place under a different filename — normalize
            shutil.move(local_path, target)
        downloaded += 1
        if (i + 1) % print_every == 0 or (i + 1) == n_shards:
            print(f"[{i+1}/{n_shards}] {downloaded} downloaded, {skipped} skipped (resume)",
                  flush=True)

    return paths


def run_img2dataset(
    metadata_dir: Path,
    output_dir: Path,
    *,
    image_size: int = 384,
    processes: int = 16,
    threads: int = 64,
    timeout_s: int = 10,
    enable_wandb: bool = False,
    save_additional_columns: tuple[str, ...] = (
        "similarity", "hash", "punsafe", "pwatermark", "aesthetic",
    ),
) -> None:
    """Invoke `img2dataset` as a subprocess. Resumes by skipping completed shards.

    img2dataset reads every `*.parquet` under `metadata_dir`, dispatches URL
    downloads via a pool of `processes` workers each with `threads` threads,
    resizes to at most `image_size` (longest side), and writes the JPEGs in
    a sharded "files" layout into `output_dir`:

        output_dir/
            00000/000000000.jpg
            00000/000000001.jpg
            ...
            00000.parquet                <- per-shard metadata
            00000_stats.json             <- per-shard completion stats (the
                                            resume sentinel)
            ...
    """
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any(metadata_dir.glob("*.parquet")):
        raise RuntimeError(
            f"no .parquet files in {metadata_dir} — run download_parquets first"
        )

    additional_cols_json = (
        "[" + ",".join(f'"{c}"' for c in save_additional_columns) + "]"
    )

    cmd = [
        # Invoke via `uv run` so the right venv is activated regardless of
        # whether the user has the venv's bin on PATH. Mirrors the pattern
        # in ~/laion-download/run.sh (the user's original setup).
        "uv", "run", "img2dataset",
        "--url_list", str(metadata_dir),
        "--input_format", "parquet",
        "--url_col", "URL",
        "--caption_col", "TEXT",
        "--output_format", "files",
        "--output_folder", str(output_dir),
        "--processes_count", str(processes),
        "--thread_count", str(threads),
        "--image_size", str(image_size),
        "--resize_only_if_bigger=True",
        "--resize_mode=keep_ratio",
        "--skip_reencode=True",
        "--timeout", str(timeout_s),
        "--save_additional_columns", additional_cols_json,
        "--enable_wandb", str(enable_wandb),
    ]
    print(f"[img2dataset] {' '.join(cmd)}", flush=True)
    # Propagate exit code; stream stdout/stderr live for visibility
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"img2dataset exited {result.returncode}; check output above for the failing shard"
        )


def download_laion_aesthetic(
    dataset: str,
    output_dir: Path,
    *,
    image_size: int = 384,
    max_shards: int | None = None,
    processes: int = 16,
    threads: int = 64,
    skip_metadata: bool = False,
    skip_images: bool = False,
) -> None:
    """End-to-end download.

    Layout under `output_dir`:
        output_dir/
            metadata/<dataset>/part-NNNNN-...parquet   <- the URL+caption shards
            images/<dataset>/00000/, 00001/, ...        <- the actual JPEGs

    Resume: just re-run with the same flags. Both steps skip work that's
    already done.
    """
    if dataset not in _DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; options: {sorted(_DATASETS)}"
        )
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata" / dataset
    images_dir = output_dir / "images" / dataset

    ds = _DATASETS[dataset]
    n_to_fetch = max_shards if max_shards is not None else NUM_PARQUET_SHARDS
    approx_m = ds.approx_images_m * (n_to_fetch / NUM_PARQUET_SHARDS)
    print(
        f"[download] {dataset} — ~{approx_m:.1f}M images expected for "
        f"{n_to_fetch}/{NUM_PARQUET_SHARDS} shards (resize {image_size}px, "
        f"~{approx_m * 25:.0f} GB on disk at 25KB/avg JPEG)", flush=True
    )

    if not skip_metadata:
        print(f"[1/2] fetching parquet metadata to {metadata_dir}", flush=True)
        download_parquets(dataset, metadata_dir, max_shards=max_shards)
    else:
        print(f"[1/2] skipped (--skip-metadata)", flush=True)

    if not skip_images:
        print(f"[2/2] running img2dataset → {images_dir}", flush=True)
        run_img2dataset(
            metadata_dir, images_dir,
            image_size=image_size, processes=processes, threads=threads,
        )
    else:
        print(f"[2/2] skipped (--skip-images)", flush=True)

    print(f"[done] {dataset} ready under {output_dir}", flush=True)


def __getattr__(name: str):    # noqa: D401  -- module-level helper
    """Expose the DATASETS dict as `DATASETS` for backward compat."""
    if name == "DATASETS":
        return _DATASETS
    raise AttributeError(name)
