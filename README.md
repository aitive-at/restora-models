# coliraz

Modern PyTorch port of DDColor (ICCV 2023) for image colorization.
Single CLI for training (with live UI + periodic sample previews),
inference, and ONNX export. Managed with `uv`.

## Quick start

```bash
uv sync
uv run coliraz scan-data --root /path/to/images
uv run coliraz train --config configs/tiny.yaml --data /path/to/images
```

See `docs/superpowers/specs/2026-05-11-coliraz-modern-port-design.md` for the design.
