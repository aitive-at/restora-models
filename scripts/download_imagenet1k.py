#!/usr/bin/env python
"""Download ImageNet-1k from HuggingFace, extract JPGs to disk.

Why this exists (vs `restora download` which uses img2dataset):
  The img2dataset tool is built for URL-list datasets (one HTTP request per
  image, scaled to billions). ImageNet-1k on HuggingFace is a *bundled*
  dataset — the entire training corpus ships as ~257 parquet files on HF's
  CDN. Pulling those is hundreds of MB/s sustained; making 1.28M individual
  HTTP requests would be orders of magnitude slower and trigger rate limits.

Prerequisites:
  1. Request access at https://huggingface.co/datasets/imagenet-1k
     (gated, but approval is instant in most cases).
  2. `huggingface-cli login` or set HF_TOKEN.

Outputs:
  <out>/train/<NNNNNN>.jpg     (~1,281,167 files, ~145 GB)
  <out>/val/<NNNNNN>.jpg       (~50,000 files, ~6 GB)
  <out>/test/<NNNNNN>.jpg      (~100,000 files, ~12 GB)

Each split is in its own subdirectory so `RecursiveImageDataset` walking
`<out>` (or just `<out>/train`) finds the right images. Filenames are
zero-padded indices — labels are dropped since restoration training
doesn't use them.

Resumable: both snapshot_download (the file fetch) and the extract step
skip already-completed work.

Usage examples:
  # Full dataset (~163 GB extracted, ~30-60 min on a fast link):
  uv run python scripts/download_imagenet1k.py --out /workspace/data/imagenet

  # Validation only for a quick test (~6 GB, ~2-5 min):
  uv run python scripts/download_imagenet1k.py \\
    --out /workspace/data/imagenet --splits val

  # List files without downloading (auth + repo sanity check):
  uv run python scripts/download_imagenet1k.py \\
    --out /workspace/data/imagenet --dry-run
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATASET_REPO = "ILSVRC/imagenet-1k"   # canonical; old "imagenet-1k" redirects here


def _hf_list_files(repo_id: str) -> list[str]:
    """List all files in the dataset repo (no download). Used for --dry-run
    and for deciding which splits to target."""
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo_id, repo_type="dataset")


def _filter_parquet_for_splits(files: list[str], splits: list[str]) -> dict[str, list[str]]:
    """Return {split: [parquet_paths_in_repo]} keyed by 'train'/'val'/'test'.

    HF imagenet-1k uses these on-disk prefixes (verified 2026-05-15):
      data/train-NNNNN-of-00294.parquet      (294 files, ~155 GB)
      data/validation-NNNNN-of-00014.parquet (14 files, ~6 GB)
      data/test-NNNNN-of-00028.parquet       (28 files, ~13 GB)

    The CLI accepts 'val' as an alias for 'validation' so users don't
    have to type the long form. The internal split key stays 'val' for
    the output directory name (shorter path).

    Returns only splits that actually have files in the repo.
    """
    # alias: cli-input  -> internal-split-key
    alias = {"validation": "val", "val": "val", "train": "train", "test": "test"}
    # repo-prefix -> internal-split-key
    repo_prefix = {"train": "train", "validation": "val", "test": "test"}
    want = {alias[s] for s in splits if s in alias}
    out: dict[str, list[str]] = {s: [] for s in want}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        stem = f.split("/")[-1]
        for prefix, split_key in repo_prefix.items():
            if stem.startswith(prefix + "-") and split_key in want:
                out[split_key].append(f)
                break
    return {k: sorted(v) for k, v in out.items() if v}


def _download_parquets(repo_id: str, allow_patterns: list[str],
                        cache_dir: Path) -> Path:
    """huggingface_hub.snapshot_download wrapper. Returns the local snapshot
    directory containing the matched parquet files."""
    from huggingface_hub import snapshot_download
    print(f"[hf] snapshot_download {repo_id}: {len(allow_patterns)} parquet pattern(s)",
          flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=str(cache_dir),
        # HF Hub picks the right network backend (xet for v1+, hf_transfer if
        # available) automatically; no further tuning needed.
    )
    return Path(snap)


def _extract_one_parquet(parquet_path: str, split_out_dir: str,
                          start_idx: int, image_col: str) -> tuple[str, int]:
    """Iterate over a single parquet file and write each image's bytes
    to disk as a JPG. Returns (path, n_written).

    Idempotent: skips files that already exist with non-zero size.
    """
    import pyarrow.parquet as pq

    out = Path(split_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet_path, columns=[image_col])
    images = table.column(image_col)

    n_written = 0
    for i in range(len(images)):
        idx = start_idx + i
        fname = out / f"{idx:08d}.jpg"
        if fname.exists() and fname.stat().st_size > 0:
            continue
        item = images[i].as_py()
        # The HF imagenet-1k schema stores image as {'bytes': b'...', 'path': ...}
        if isinstance(item, dict) and "bytes" in item:
            payload = item["bytes"]
        elif isinstance(item, (bytes, bytearray)):
            payload = bytes(item)
        else:
            # Last-ditch: hope it's a PIL-decodeable raw image
            raise RuntimeError(
                f"unexpected image cell type in {parquet_path}: {type(item)}"
            )
        # Validate it's a JPEG header (cheap sanity check)
        if len(payload) < 4 or payload[:3] != b"\xff\xd8\xff":
            # Some rows may be PNG or other; we just persist the bytes as-is
            # with .jpg extension. RecursiveImageDataset accepts mixed exts;
            # for max strictness, rename here based on magic bytes.
            pass
        fname.write_bytes(payload)
        n_written += 1
    return (parquet_path, n_written)


def _extract_split(split_name: str, parquet_paths: list[Path],
                    out_root: Path, *, workers: int) -> int:
    """Extract images from a list of parquet files in parallel processes."""
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Probe the first parquet for the image column name (most HF image
    # datasets use 'image', but be defensive).
    import pyarrow.parquet as pq
    schema = pq.read_schema(str(parquet_paths[0]))
    image_col = next(
        (n for n in ("image", "img", "jpg", "jpeg") if n in schema.names),
        None,
    )
    if image_col is None:
        raise RuntimeError(
            f"no image column found in {parquet_paths[0]}; "
            f"got fields: {schema.names}"
        )

    # Pre-compute per-file row offsets so filenames don't collide across
    # parquets (each parquet writes a contiguous index range).
    offsets: list[int] = []
    cumulative = 0
    for pq_path in parquet_paths:
        offsets.append(cumulative)
        n_rows = pq.read_metadata(str(pq_path)).num_rows
        cumulative += n_rows

    print(f"[extract:{split_name}] {len(parquet_paths)} parquet files, "
          f"{cumulative:,} rows total, image col={image_col!r}",
          flush=True)

    t0 = time.time()
    total_written = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_extract_one_parquet, str(pq_path),
                      str(split_dir), offset, image_col)
            for pq_path, offset in zip(parquet_paths, offsets)
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            pq_path, n = fut.result()
            total_written += n
            elapsed = time.time() - t0
            rate = total_written / max(elapsed, 1e-3)
            print(f"[extract:{split_name}] file {done}/{len(parquet_paths)} "
                  f"done (+{n:,}, total {total_written:,}, "
                  f"{rate:.0f} img/s)", flush=True)
    return total_written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True,
                   help="Output root. Images go to <out>/<split>/NNNNNNNN.jpg")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=["train", "val", "validation", "test"],
                   help="Which splits to materialize. Default: train + val")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where snapshot_download puts the parquet files. "
                        "Default: <out>/.hf-cache")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel parquet → JPG extraction workers. Default 4. "
                        "Each worker holds one parquet in memory (~600 MB).")
    p.add_argument("--dry-run", action="store_true",
                   help="List files in the HF repo and exit. Verifies auth "
                        "and shows what would be downloaded.")
    p.add_argument("--keep-parquet", action="store_true",
                   help="Don't delete parquet files after extraction. "
                        "Useful if you want to re-extract or re-shard.")
    args = p.parse_args()

    out_root = args.out.expanduser().resolve()
    cache_dir = (args.cache_dir or (out_root / ".hf-cache")).expanduser().resolve()

    # Step 1: list repo files (this works without download-access on most
    # gated repos, so we can't rely on it for an access check)
    print(f"[hf] listing files in {DATASET_REPO}...", flush=True)
    try:
        all_files = _hf_list_files(DATASET_REPO)
    except Exception as e:
        print(f"\n[error] could not list {DATASET_REPO}: {e}", file=sys.stderr)
        print("  Probably an HF token / network issue. Run "
              "`huggingface-cli login` and retry.", file=sys.stderr)
        return 2

    # Step 1b: access probe — do a HEAD-only check on the FIRST parquet
    # we plan to download. README.md isn't gated on this repo (only the
    # data/*.parquet files are), so we have to probe a real data file.
    # get_hf_file_metadata does a HEAD request, no body — costs nothing
    # but surfaces GatedRepoError if access is denied.
    if not args.dry_run:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import GatedRepoError
        # Pick the first parquet from the requested splits (small probe target)
        sample_parquets = [
            f for f in all_files
            if f.endswith(".parquet") and (
                f.split("/")[-1].startswith("train-")
                or f.split("/")[-1].startswith("validation-")
                or f.split("/")[-1].startswith("test-")
            )
        ]
        if not sample_parquets:
            print("[error] no parquets found in repo — schema may have changed",
                  file=sys.stderr)
            return 2
        probe_file = sample_parquets[0]
        try:
            probe_url = hf_hub_url(repo_id=DATASET_REPO,
                                    filename=probe_file, repo_type="dataset")
            _ = get_hf_file_metadata(probe_url)
        except GatedRepoError as e:
            print(
                "\n[error] HuggingFace gated-access denied for "
                f"{DATASET_REPO}.\n"
                "  1. Visit https://huggingface.co/datasets/"
                f"{DATASET_REPO} and click 'Request access'.\n"
                "  2. Approval is usually instant.\n"
                "  3. Then re-run this script.\n"
                f"  Underlying error: {e}",
                file=sys.stderr,
            )
            return 2
        except Exception as e:
            msg = str(e)
            if "401" in msg or "token" in msg.lower():
                print(
                    "\n[error] HuggingFace token problem. Run "
                    "`huggingface-cli login` and retry.\n"
                    f"  Underlying error: {e}",
                    file=sys.stderr,
                )
                return 2
            raise

    by_split = _filter_parquet_for_splits(all_files, args.splits)
    if not by_split:
        print(f"[hf] no parquet files found for requested splits {args.splits}",
              file=sys.stderr)
        print(f"[hf] repo contains {len(all_files)} files; first 20:",
              file=sys.stderr)
        for f in all_files[:20]:
            print(f"      {f}", file=sys.stderr)
        return 2

    total_files = sum(len(v) for v in by_split.values())
    print(f"[hf] matched {total_files} parquet files across "
          f"splits {list(by_split.keys())}:", flush=True)
    for sp, files in by_split.items():
        print(f"      {sp}: {len(files)} files (first: {files[0]})", flush=True)

    if args.dry_run:
        print("[dry-run] not downloading. Re-run without --dry-run to fetch.")
        return 0

    # Step 2: download parquets
    patterns = sorted({f for files in by_split.values() for f in files})
    snap_dir = _download_parquets(DATASET_REPO, patterns, cache_dir)
    print(f"[hf] snapshot at {snap_dir}", flush=True)

    # Step 3: extract images per split
    t0 = time.time()
    for split_name, repo_paths in by_split.items():
        parquet_paths = [snap_dir / p for p in repo_paths]
        for p in parquet_paths:
            if not p.exists():
                raise RuntimeError(f"expected parquet missing after download: {p}")
        n = _extract_split(split_name, parquet_paths, out_root,
                            workers=args.workers)
        print(f"[done] {split_name}: {n:,} images under {out_root / split_name}",
              flush=True)
        if not args.keep_parquet:
            for p in parquet_paths:
                try:
                    p.unlink()
                except OSError:
                    pass

    print(f"[done] total elapsed: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
