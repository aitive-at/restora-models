# restora-models

Temporal multi-task video restoration: one model that handles colorization,
super-resolution, denoising, deblurring, and JPEG-artifact removal on either
modern footage or damaged old film. The same model handles single-task and
compound-task requests — a 5-axis config vector tells the model what to fix.

## Architecture

- **Contract:** `forward(frames [B,7,3,H,W], config [B,5]) → rgb [B,3,H,W]`
- **Backbone:** TemporalNAFNet — fully convolutional (works at any input
  resolution divisible by 16). FiLM-conditioned on the task vector at every
  block.
- **Temporal stem:** FlowDistill (static-unroll RAFT, ~5M params) estimates
  flow between each neighbor and the center frame, then bilinear-warp +
  visibility mask align the 7-frame window into a 28-channel input.
- **Refine head:** RSDRefineHead — one-step residual-shift diffusion in RGB.
  No external VAE. Single-pass ONNX-exportable.
- **Output:** Lab dual head (delta + ab-abs gated by colorize axis).
  Hard-wired identity preservation: when config is all-zeros, output equals
  the input center frame exactly.

Sizes:

| Size                       | Params     | Use case                  |
|----------------------------|------------|---------------------------|
| `temporal_restora_nano`    | ~9M        | Edge / mobile             |
| `temporal_restora_small`   | ~28M       | Desktop GPU               |
| `temporal_restora_medium`  | ~75M       | High-end workstation      |
| `temporal_restora_large`   | ~183M      | Server / batch processing |

All sizes share the same forward contract and ONNX I/O.

## Quick start

```sh
# 1. Install
uv sync --extra dev

# 2. Prepare data
#    REDS (primary): download manually from
#    https://seungjunnah.github.io/Datasets/reds.html
#    then place sequences at ~/data/reds/000, ~/data/reds/001, ...
uv run restora prepare-data reds --out ~/data/reds

#    Old-film overlay textures (DeepRemaster pack, 898 MB):
uv run restora prepare-data film-overlays --out ~/data/film-overlays

# 3. Train (local smoke or B200 production)
uv run restora train --config configs/local-temporal.yaml  # ~30 min on RTX 6000
uv run restora train --config configs/b200-temporal.yaml --compile  # ~50 h B200

# 4. Export to ONNX (dynamic spatial dims; runs at any input resolution)
uv run restora export --model runs/<run>/final.pt \
                      --output restora.onnx \
                      --precision fp16 --task colorize

# 5. Run inference on a clip directory
uv run restora infer --model runs/<run>/final.pt \
                     --input ./input_frames \
                     --output ./restored \
                     --color --denoise --sharp
```

## End-to-end training pipeline

The orchestrator runs every stage (flow-distill → backbone → RSD refine →
end-to-end FT → multi-size distillation) with state persistence so it can
resume from any point.

```sh
# Start from scratch
uv run restora train-pipeline --config configs/local-temporal.yaml \
                              --run-root runs/local

# Resume an interrupted pipeline
uv run restora train-pipeline --resume runs/local

# Continue training after adding new data (re-runs backbone onwards)
uv run restora train-pipeline --resume runs/local --extend-from backbone
```

## Training data

Composite `VideoWindowDataset` interleaves any number of `VideoSubDataset`
sources by weight. Currently registered:

- `reds` — REDS (270 sequences × 100 frames at 720p, modern content)
- `vimeo_septuplet` — Vimeo Septuplet (7-frame clips at 256p)

Add a new source by implementing the `VideoSubDataset` protocol
(`__len__`, `__getitem__ → {"frames": (7,3,H,W), "source": str, "key": str}`)
and registering it in `src/restora_models/data/builders.py`.

Per-sample degradations are sampled with a balanced single-task / compound /
identity distribution (15% identity, 35% single axis, 35% two axes, 15%
three+ axes). Film-specific layers (scratch/dust overlays, color cast, gate
weave, MPEG transcode) are applied with low probability on top.

## Inference contract

Consumers (Python, C#, ORT, TensorRT) all see the same ONNX:

- Input 1: `frames` — `(B, 7, 3, H, W)` float32 in [0, 1]
- Input 2: `config` — `(B, 5)` float32 in {0, 1} (axes: colorize, denoise,
  sharpen, dejpeg, deblur)
- Output: `output` — `(B, 3, H, W)` float32 in [0, 1] (restored center frame)

For per-task baked exports (`--task colorize`), the config tensor is folded
into the graph as a constant; the resulting ONNX has only the `frames` input.

For single still-image inference, replicate the image 7× to form the
window — the bundled inference pipeline does this automatically.

## CLI cheat sheet

```
restora train                  Train from a config (single stage)
restora train-pipeline         End-to-end multi-stage pipeline
restora train-flow-distill     Pre-train the FlowDistill RAFT student
restora infer                  Inference: single image OR clip directory
restora export                 Export checkpoint to ONNX
restora distill                Distill a teacher into a smaller student
restora bench                  Benchmark inference speed
restora compare                Compare per-axis PSNR across checkpoints
restora gallery                Generate qualitative triptychs (clean | degraded | restored)
restora prepare-data           Download + verify training datasets
restora version                Print package version
```

## Docs

- `docs/superpowers/specs/2026-05-17-temporal-old-film-remaster-design.md`
  — full architecture spec
- `docs/superpowers/plans/2026-05-17-temporal-old-film-remaster.md`
  — implementation plan
- `docs/integration/onnx-inference-guide.md` — ONNX consumer guide
- `docs/integration/csharp-video-inference.md` — C# integration recipe
