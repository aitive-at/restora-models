# refine

Multi-task image restoration: one model trained jointly on colorization,
super-resolution, denoising, deblurring, and JPEG-artifact removal.

## Quick start

```sh
uv sync
uv run refine scan-data --root /path/to/images
uv run refine train --config configs/laion-multitask.yaml --data /path/to/images
```

See `docs/superpowers/specs/2026-05-13-refine-multitask-design.md` for the design.

The previous colorization-only project lives in `legacy/coliraz-v1/`.
