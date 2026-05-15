#!/usr/bin/env python
"""Download Open Images from the public AWS S3 mirror.

Why this exists (vs `restora download` which uses img2dataset):
  img2dataset CAN be pointed at Open Images via the official CSV, but the
  URLs in that CSV are Flickr — the same rot/throttle problem that just
  stalled the LAION download. The Open Images team mirrors the entire
  dataset at `s3://open-images-dataset/`, a public AWS Open Data bucket
  with no auth, no rate limits, and a fast CDN. This script downloads
  from there directly.

Layout in the bucket (verified 2026-05-15):
  s3://open-images-dataset/train/<image_id>.jpg          (~9M files)
  s3://open-images-dataset/validation/<image_id>.jpg     (~41K files)
  s3://open-images-dataset/test/<image_id>.jpg           (~125K files)

These map to HTTPS URLs of the form:
  https://open-images-dataset.s3.amazonaws.com/<split>/<image_id>.jpg

We use the S3 XML bucket listing API to discover image IDs (no AWS CLI
or boto3 needed) and download in parallel via httpx + a thread pool.

Outputs:
  <out>/<split>/<image_id>.jpg

Resumable: skips files that already exist with non-zero size.

Usage examples:
  # Just download the validation set (~41K files, ~20 GB):
  uv run python scripts/download_openimages.py \\
    --out /workspace/data/openimages --split validation

  # Sample 1M training images (good size for a 500k-step run):
  uv run python scripts/download_openimages.py \\
    --out /workspace/data/openimages --split train --limit 1000000

  # Quick smoke test (5 images, ~5 MB, no need to commit to a big run):
  uv run python scripts/download_openimages.py \\
    --out /tmp/oi-test --split validation --limit 5
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

BUCKET = "open-images-dataset"
S3_HOST = f"https://{BUCKET}.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix: str, *, limit: int | None = None,
              max_keys_per_page: int = 1000) -> list[str]:
    """Stream the bucket's keys under `prefix` via the public S3 XML API.

    Uses `marker=<last_key>` pagination — no auth, no SDK. Returns at most
    `limit` keys (None = all).

    Each key looks like 'train/00000abcdef123.jpg'.
    """
    out: list[str] = []
    marker: str | None = None
    pages = 0
    while True:
        url = f"{S3_HOST}/?prefix={prefix}&max-keys={max_keys_per_page}"
        if marker is not None:
            url += f"&marker={urllib.parse.quote(marker)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml = resp.read()
        root = ET.fromstring(xml)
        contents = root.findall(f"{S3_NS}Contents")
        if not contents:
            break
        for c in contents:
            key_el = c.find(f"{S3_NS}Key")
            if key_el is None or not key_el.text:
                continue
            key = key_el.text
            if not key.endswith(".jpg"):
                continue
            out.append(key)
            if limit is not None and len(out) >= limit:
                return out
        pages += 1
        # Pagination: S3 sets IsTruncated=true if more pages exist.
        truncated_el = root.find(f"{S3_NS}IsTruncated")
        if truncated_el is None or truncated_el.text.lower() != "true":
            break
        # Marker for the next page = last key on this page
        marker = contents[-1].find(f"{S3_NS}Key").text
        if pages % 10 == 0:
            print(f"[list] paginated {pages} pages, {len(out)} keys so far",
                  flush=True)
    return out


def _download_one(key: str, out_root: Path, *, retries: int = 2) -> tuple[str, str]:
    """Download one S3 key over HTTPS. Returns (key, status):
      status in {'ok', 'skip', 'http_<NNN>', 'err_<class>:<msg>'}.

    Uses stdlib urllib.request — simple, thread-safe, no shared-client
    bugs. httpx Client sharing across a thread pool with .stream() has
    known edge cases in 0.28; we don't need streaming for ~500 KB images.

    Retries on transient connection resets / timeouts (not on HTTP 4xx).
    """
    url = f"{S3_HOST}/{key}"
    local = out_root / key
    if local.exists() and local.stat().st_size > 0:
        return (key, "skip")
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "restora-models/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    return (key, f"http_{resp.status}")
                data = resp.read()
            tmp.write_bytes(data)
            tmp.rename(local)
            return (key, "ok")
        except urllib.error.HTTPError as e:
            # 4xx is a real "this image isn't there" — don't retry
            try: tmp.unlink()
            except OSError: pass
            return (key, f"http_{e.code}")
        except Exception as e:
            # Connection reset / timeout / DNS hiccup — retry
            last_err = f"err_{type(e).__name__}:{e}"
            try: tmp.unlink()
            except OSError: pass
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))   # backoff: 0.5s, 1.0s
                continue
    return (key, last_err or "err_unknown")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True,
                   help="Output root. Images saved as <out>/<split>/<id>.jpg")
    p.add_argument("--split", default="validation",
                   choices=["train", "validation", "test"],
                   help="Which split to download. Default: validation "
                        "(smallest at ~41K images).")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of images to download. None = all. "
                        "Useful for testing with --limit 5 or capping a "
                        "training subset at e.g. --limit 1000000.")
    p.add_argument("--threads", type=int, default=64,
                   help="Concurrent download threads. Default 64. AWS S3 "
                        "happily serves thousands but a single Python "
                        "process gets diminishing returns past ~128.")
    p.add_argument("--dry-run", action="store_true",
                   help="List image keys but don't download anything.")
    p.add_argument("--print-every", type=int, default=100,
                   help="Progress print cadence (every N successes).")
    args = p.parse_args()

    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: discover image keys
    prefix = f"{args.split}/"
    print(f"[list] enumerating s3://{BUCKET}/{prefix} (limit={args.limit})",
          flush=True)
    t_list = time.time()
    keys = list_keys(prefix, limit=args.limit)
    print(f"[list] {len(keys):,} keys discovered in {time.time() - t_list:.0f}s",
          flush=True)
    if not keys:
        print("[list] no keys returned — check --split spelling", file=sys.stderr)
        return 2

    if args.dry_run:
        print("[dry-run] first 5 keys:")
        for k in keys[:5]:
            print(f"   {S3_HOST}/{k}")
        return 0

    # Step 2: parallel download via stdlib urllib + thread pool
    t0 = time.time()
    ok = skip = http_err = exc_err = 0
    err_samples: list[str] = []   # capture first few error messages for diagnostics
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(_download_one, k, out_root): k for k in keys}
        for done, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status.startswith("http_"):
                http_err += 1
                if len(err_samples) < 5: err_samples.append(status)
            else:
                exc_err += 1
                if len(err_samples) < 5: err_samples.append(status)
            if done % args.print_every == 0 or done == len(keys):
                elapsed = time.time() - t0
                rate = (ok + skip) / max(elapsed, 1e-3)
                print(f"[dl] {done:,}/{len(keys):,}  "
                      f"ok={ok:,} skip={skip:,} http_err={http_err} "
                      f"exc_err={exc_err}  {rate:.0f} img/s  "
                      f"{elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    print(f"\n[done] {ok:,} new, {skip:,} skip, "
          f"{http_err + exc_err} failed, in {elapsed:.0f}s "
          f"({(ok + skip) / max(elapsed, 1e-3):.0f} img/s avg)", flush=True)
    if http_err + exc_err > 0:
        # Non-fatal: a handful of 404s is normal (deleted images).
        pct = 100.0 * (http_err + exc_err) / max(len(keys), 1)
        print(f"[note] {pct:.2f}% failure rate", flush=True)
        if err_samples:
            print("[note] sample errors:", flush=True)
            for s in err_samples:
                print(f"       {s}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
