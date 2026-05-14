# LAION-aesthetic dataset download

Two-step pipeline that fetches the parquet metadata shards from HuggingFace and
then downloads the actual JPEGs via `img2dataset`. Both steps are resumable —
re-run with the same flags to continue from where you stopped.

## Prerequisites

1. **HuggingFace token** with access to the gated LAION datasets. Get one
   from <https://huggingface.co/settings/tokens>, then either:
   - `huggingface-cli login` (writes to `~/.cache/huggingface/token`), OR
   - `export HF_TOKEN=hf_...` (the env var the download module reads)

2. **Access granted on each dataset page**. Each subset is independently
   gated; visit the HF page and click "Request access":
   - <https://huggingface.co/datasets/laion/relaion2B-multi-aesthetic>  (17M images)
   - <https://huggingface.co/datasets/laion/laion2B-en-aesthetic>        (51M images)
   - <https://huggingface.co/datasets/laion/relaion1B-nolang-aesthetic>  (52M images)
   Approval is usually instant but can take a day. If a 403 fires during
   download, that's why.

3. **`uv` installed and `uv sync` complete** in the repo root. The download
   command uses `uv run img2dataset` under the hood, so the right venv is
   activated transparently.

## Quick start (defaults)

```sh
# 17M images, ~700 GB on disk at 384px JPEG, 1-10 Gbps link
uv run refine download --output ~/data/laion --dataset relaion2B-multi-aesthetic
```

That's all. The command:
- Creates `~/data/laion/metadata/relaion2B-multi-aesthetic/` and downloads
  128 parquet shards (~63 MB each, ~8 GB total).
- Then runs `img2dataset` to fetch JPEGs into
  `~/data/laion/images/relaion2B-multi-aesthetic/<NNNNN>/`.

If interrupted, re-run the exact same command — both steps skip work that's
already complete.

## Common variations

```sh
# Partial download (first 8 of 128 parquet shards = ~1M images, ~40 GB)
uv run refine download --output ~/data/laion -d relaion2B-multi-aesthetic --max-shards 8

# Higher resolution (keep up to 768px instead of 384px)
uv run refine download --output ~/data/laion -d relaion2B-multi-aesthetic --image-size 768

# Beefy server: more parallelism (B200 / 100Gbps link)
uv run refine download --output /mnt/data/laion -d relaion2B-multi-aesthetic \
                       --processes 32 --threads 128

# Just download metadata, defer images (e.g. on a temp box, copy to server later)
uv run refine download --output ./laion-meta -d relaion2B-multi-aesthetic --skip-images
```

## Output layout

```
<output>/
  metadata/<dataset>/
    part-00000-<uuid>-c000.snappy.parquet
    part-00001-<uuid>-c000.snappy.parquet
    ... (128 files)
  images/<dataset>/
    00000/                            # ~10000 images per shard
      000000000.jpg
      000000001.jpg
      ...
      000000000.json                  # caption + metadata sidecar
      ...
    00000.parquet                     # per-shard manifest
    00000_stats.json                  # completion sentinel (presence = done)
    00001/
    ...
```

For training, point `data.root` (in your training YAML) at
`<output>/images/<dataset>/`. The dataset scanner walks recursively and
indexes all `*.jpg` files.

## Disk + bandwidth estimates

| Dataset | Images | Disk @ 384px | Disk @ 768px | Time @ 1Gbps | Time @ 10Gbps |
|---|---|---|---|---|---|
| relaion2B-multi-aesthetic | 17M | ~430 GB | ~1.3 TB | ~3 days | ~8 hours |
| laion2B-en-aesthetic | 51M | ~1.3 TB | ~3.9 TB | ~9 days | ~22 hours |
| relaion1B-nolang-aesthetic | 52M | ~1.3 TB | ~3.9 TB | ~9 days | ~22 hours |

Estimates assume ~25 KB average JPEG at 384px and reasonable hit rate on the
source URLs (~85% — some links 404). Real throughput is usually
bandwidth-limited, not CPU-limited.

## Troubleshooting

- **`Access to dataset ... is restricted`** — request access on the HF page
  (link above). The token alone isn't enough.
- **`401 Unauthorized`** — token missing/invalid. Run `huggingface-cli login`.
- **High dropout rate (>20% URLs failed)** — many original LAION URLs are dead
  by now. Adjust `--timeout 5` to fail faster on dead URLs and move on.
  Expect ~80-90% hit rate on a fresh download.
- **Out of disk** — img2dataset writes per-shard as it goes. If you run out,
  just free disk and re-run; existing shards are skipped.
