#!/usr/bin/env python
"""One-shot video dataset preparation for production training.

Wraps the two existing steps end-to-end:
  1) Download DAVIS-2017 (480p, train+val) and lay out frames at
     <out>/<video_name>/frame_NNNNN.jpg  (download_davis logic inlined).
  2) Precompute RAFT optical flow for every frame pair (skip = 1..max_skip)
     into <out>/<video_name>/.flow/frame_NNNNN_skipK.npz.

Both steps are idempotent — re-running skips work that's already done.
Designed to be the production analog of `restora download` for video data.

Usage on the server:
    tmux new -s prep-video
    cd /workspace/code/restora-models
    uv run python scripts/prepare_video_dataset.py --out /workspace/data-videos
    # Ctrl-B then D to detach

Expected runtime on B200:
  - DAVIS download + extract: ~5-10 min (net + disk bound, ~480 MB zip)
  - RAFT flow precompute:     ~20-40 min (GPU bound, ~50k flow pairs)
  - Total: under an hour. Re-runs skip both stages if outputs exist.

Optional flags:
  --skip-davis      Skip step 1 (assume frames are already laid out)
  --skip-flow       Skip step 2 (frames only, no flow)
  --max-skip K      Compute flow for skip = 1..K. Default 5; matches video.max_skip
                    in production configs.
  --resolution N    Flow precompute resolution. Default 256 matches training.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"


# ---------- step 1: DAVIS download + layout -------------------------------

def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[davis] archive present, skip download "
              f"({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    print(f"[davis] downloading {url}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_pct = [-1]

    def hook(chunk_i: int, chunk_sz: int, total: int) -> None:
        if total <= 0:
            return
        downloaded = chunk_i * chunk_sz
        pct = int(100 * downloaded / total)
        if pct != last_pct[0]:
            print(
                f"\r[davis]   {pct:3d}%  "
                f"({downloaded / 1e6:.0f} / {total / 1e6:.0f} MB)",
                end="", flush=True,
            )
            last_pct[0] = pct

    urllib.request.urlretrieve(url, tmp, hook)
    print(flush=True)
    tmp.rename(dest)


def _unzip(zip_path: Path, staging: Path) -> Path:
    """Extract only JPEGImages/480p/* (saves time + disk). Returns the
    path to DAVIS/JPEGImages/480p inside staging."""
    print(f"[davis] unzipping into {staging}", flush=True)
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n for n in zf.namelist()
            if "/JPEGImages/480p/" in n and n.endswith(".jpg")
        ]
        for n in names:
            zf.extract(n, staging)
    return staging / "DAVIS" / "JPEGImages" / "480p"


def _layout(jpegs_root: Path, out_root: Path) -> int:
    """Move each video's frames into <out_root>/<video>/frame_NNNNN.jpg.
    Returns count of videos newly written (existing ones are left alone)."""
    out_root.mkdir(parents=True, exist_ok=True)
    new = 0
    for video_dir in sorted(jpegs_root.iterdir()):
        if not video_dir.is_dir():
            continue
        target = out_root / video_dir.name
        if target.exists() and any(target.glob("frame_*.jpg")):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for src in sorted(video_dir.iterdir()):
            if src.suffix.lower() != ".jpg":
                continue
            try:
                fi = int(src.stem)
            except ValueError:
                continue
            shutil.move(str(src), str(target / f"frame_{fi:05d}.jpg"))
        new += 1
    return new


def step_davis(out_root: Path, cache: Path, keep_staging: bool = False) -> None:
    zip_path = cache / "DAVIS-2017-trainval-480p.zip"
    _download(DAVIS_URL, zip_path)

    staging = cache / "extract"
    if staging.exists():
        shutil.rmtree(staging)
    jpegs_root = _unzip(zip_path, staging)

    new = _layout(jpegs_root, out_root)
    total = sum(
        1 for p in out_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    print(f"[davis] done. {new} new videos, {total} total under {out_root}",
          flush=True)
    if not keep_staging and staging.exists():
        shutil.rmtree(staging)
        print(f"[davis] cleaned up staging {staging}", flush=True)


# ---------- step 2: RAFT flow precompute (delegates to existing script) ---

def step_flow(out_root: Path, *, max_skip: int, resolution: int,
              device: str, print_every: int) -> None:
    """Shell out to scripts/precompute_video_flow.py.

    Rationale: that script already loads RAFT_Large via torchvision and
    handles resumability. Re-implementing it here would duplicate ~80
    lines and another model-load. The shell-out is cheaper to maintain.
    """
    here = Path(__file__).resolve().parent
    flow_script = here / "precompute_video_flow.py"
    if not flow_script.exists():
        raise SystemExit(f"missing helper: {flow_script}")

    cmd = [
        sys.executable, str(flow_script),
        "--root", str(out_root),
        "--max-skip", str(max_skip),
        "--resolution", str(resolution),
        "--device", device,
        "--print-every", str(print_every),
    ]
    print(f"[flow] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"flow precompute exited with code {rc}")


# ---------- entry point ---------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True,
                   help="Output root for video frames. Use /workspace/data-videos "
                        "on the B200 server (matches video.root in "
                        "configs/b200-phase12v-nafnet-large.yaml).")
    p.add_argument("--cache", type=Path, default=None,
                   help="Where to put the downloaded DAVIS zip. "
                        "Default: <out>/.davis-cache (self-contained on the volume)")
    p.add_argument("--max-skip", type=int, default=5,
                   help="Max temporal skip for flow precompute (k=1..max_skip). "
                        "Default 5 — matches production video.max_skip.")
    p.add_argument("--resolution", type=int, default=256,
                   help="Resolution for flow precompute. Default 256 = training res.")
    p.add_argument("--device", default=None,
                   help="Device for RAFT. Default cuda if available, else cpu.")
    p.add_argument("--print-every", type=int, default=20,
                   help="RAFT progress every N videos. Default 20.")
    p.add_argument("--skip-davis", action="store_true",
                   help="Skip DAVIS download/layout (assume frames are present).")
    p.add_argument("--skip-flow", action="store_true",
                   help="Skip RAFT flow precompute (frames only).")
    p.add_argument("--keep-staging", action="store_true",
                   help="Don't delete the unzipped DAVIS staging dir.")
    args = p.parse_args()

    out_root = args.out.expanduser().resolve()
    cache = (args.cache or (out_root / ".davis-cache")).expanduser().resolve()

    if args.device is None:
        try:
            import torch  # noqa: PLC0415  — only needed when picking default
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    t0 = time.time()
    print(f"[prep-video] out={out_root}", flush=True)
    print(f"[prep-video] cache={cache}", flush=True)
    print(f"[prep-video] device={args.device}", flush=True)

    if not args.skip_davis:
        step_davis(out_root, cache, keep_staging=args.keep_staging)
    else:
        print("[prep-video] --skip-davis: assuming frames are already laid out",
              flush=True)

    if not args.skip_flow:
        step_flow(
            out_root,
            max_skip=args.max_skip, resolution=args.resolution,
            device=args.device, print_every=args.print_every,
        )
    else:
        print("[prep-video] --skip-flow: leaving flow tree empty", flush=True)

    print(f"[prep-video] all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
