# ONNX Inference Guide

This guide covers loading and running the restora ONNX models from any
ONNX Runtime consumer (Python, C++, C#, Rust, etc.).

## Model contract

The exported ONNX has either 1 or 2 inputs depending on how it was created.

**Generic export** (`--task` flag NOT used):

| Tensor name | Shape                  | Dtype   | Range  |
|-------------|------------------------|---------|--------|
| `frames`    | `(B, 7, 3, H, W)`      | float32 | [0, 1] |
| `config`    | `(B, 5)`               | float32 | {0, 1} |
| `output`    | `(B, 3, H, W)`         | float32 | [0, 1] |

**Per-task baked export** (`--task colorize|denoise|sharpen|dejpeg|deblur|all`):

| Tensor name | Shape                  | Dtype   | Range  |
|-------------|------------------------|---------|--------|
| `frames`    | `(B, 7, 3, H, W)`      | float32 | [0, 1] |
| `output`    | `(B, 3, H, W)`         | float32 | [0, 1] |

## Dynamic spatial dimensions

Models are exported with dynamic axes on `B`, `H`, `W`. The same ONNX runs
at any resolution where H and W are divisible by 16. The consumer is
responsible for padding non-multiple-of-16 inputs (mirror or replicate is
fine) and cropping back after inference.

## TensorRT

```sh
trtexec --onnx=restora.onnx \
        --fp16 \
        --minShapes=frames:1x7x3x64x64 \
        --optShapes=frames:1x7x3x256x256 \
        --maxShapes=frames:1x7x3x1080x1920 \
        --saveEngine=restora.trt
```

For per-task baked exports, drop the `config` input from the shape profiles
(only `frames` exists).

## Single image inference

For a single still, replicate the image 7× to fill the temporal window:

```python
import numpy as np
import onnxruntime as ort
import cv2

sess = ort.InferenceSession("restora.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

img_bgr = cv2.imread("photo.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
h, w = img_rgb.shape[:2]
# Pad to multiple of 16
ph = (16 - h % 16) % 16
pw = (16 - w % 16) % 16
if ph or pw:
    img_rgb = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
frame = img_rgb.transpose(2, 0, 1)  # (3, H, W)
window = np.tile(frame, (7, 1, 1, 1))  # (7, 3, H, W)
frames = window[None]                   # (1, 7, 3, H, W)
config = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)  # colorize only

out = sess.run(None, {"frames": frames, "config": config})[0]
if ph or pw:
    out = out[..., :h, :w]
```

## Video inference (sliding window)

For a sequence of N frames, the model is run N times with a sliding
7-frame window centered on each output frame. At the boundaries, the
edge frames are replicated.

The bundled `restora infer --input <dir> --output <dir>` does this.
For custom consumers, see the reference implementation in
`src/restora_models/infer/pipeline.py::VideoPipeline.process_directory`.
