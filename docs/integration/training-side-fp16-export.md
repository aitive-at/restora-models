# FP16 Re-Export Handoff

**Audience:** the engineer maintaining `restora_models` (the PyTorch training project).
**Authored by:** the C# inference team (Aitive.Restora.\*).
**Goal:** ship a working fp16 ONNX so TensorRT can compile a real fp16 engine.

---

## 1. Why this matters

The current fp16 export (`iter_0030000_ema_generic_fp16.onnx`) is dead on arrival
on the inference side:

* **TensorRT EP** rejects the graph at engine-compile time with a hard error
  (see §2 below). Falls through to the CUDA EP fallback layer.
* **CUDA EP fallback** completes the run without raising, but the output is
  uniformly **(0, 0, 0)** — verifiable via our `RestoraOutputDiagnosticTests.DiagnosticInput`
  with `modelIdOverride: "iter_0030000_ema_generic_fp16"`. So the model
  silently produces garbage today.

The fp32 sibling (`iter_0030000_ema_generic.onnx`) is healthy and runs fine, so
we're shipping with that. But fp16 would roughly **halve our per-frame
inference cost** (41 ms → ~20 ms on RTX-class hardware with tensor cores) and is
the single biggest perf win still on the table.

---

## 2. The exact symptom

When ORT/TRT tries to compile the current fp16 ONNX, the model importer
explodes in `align_stem`:

```
[E:onnxruntime:CSharpOnnxRuntime, tensorrt_execution_provider.h]
    [ERROR] ITensor::getDimensions: Error Code 4: API Usage Error
    (/model/backbone/align_stem/GridSample_10:
     IGridSampleLayer `input` and `grid` must be of same type.
     `input` type is Half and `grid` type is Float.
     In nvinfer1::builder::GridSampleNode::validateTypes
     at C:\_src\optimizer\common\nodes\gridSampleNode.cpp:28)

[E:onnxruntime] ModelImporter.cpp:150: ERROR: ModelImporter.cpp:506
    In function parseNode:
[6] Invalid Node - /model/backbone/align_stem/GridSample_10
```

This error fires for **every** `GridSample_*` op in the `align_stem` module
(we counted ~12 of them in the dump). After hitting all of them TRT marks the
whole subgraph unfit, ORT routes it to CUDA EP fallback, and the chained
Cast(fp32→fp16) / Cast(fp16→fp32) shims between TRT and CUDA partitions hit
an edge case that produces zeros.

For full output of the TRT importer error sequence: re-run our diagnostic test
locally and inspect `cache/bench-yuv-parallel.txt` (or any of the `bench-*.txt`
files post-rename). Search for `GridSampleNode`.

---

## 3. Root cause hypothesis

The export currently casts the **network weights** and **most activations** to
fp16, but the **flow / sampling-grid tensors** feeding `F.grid_sample` are
still fp32.

The likely reason is that those grids are computed from index arithmetic
(`torch.meshgrid` over integer coordinates, plus a learned residual flow), and
when you cast the model with `model.half()` the integer-arithmetic intermediates
stay fp32 — only the **leaf weight tensors** get the dtype change, not the
synthetic coordinate tensors built during forward.

So inside the temporal-alignment block you have roughly:

```python
# inside AlignStem.forward (approximate):
flow = self.flow_estimator(features)   # fp16 — has learned weights, gets cast
grid = make_grid(flow)                 # fp32! built from torch.arange / meshgrid
aligned = F.grid_sample(features_fp16, grid_fp32, ...)
#                       ^^^^^^^^^^^^   ^^^^^^^^^
#                       Half           Float  →  TRT explodes
```

When PyTorch exports this to ONNX, the Cast nodes around `grid` don't get
inserted (PyTorch sees no dtype mismatch in eager mode for `grid_sample`
because libcudnn/aten promotes internally), but TRT's `IGridSampleLayer`
demands matching dtypes.

We can't confirm this without seeing the wrapper source — it might be a
slightly different structural cause (see §6 alternatives) — but the fix below
covers all the plausible variants.

---

## 4. The fix

### 4a. Minimal patch — explicit dtype match before each GridSample

In `src/restora_models/export/wrapper.py` (and any module the wrapper
imports that calls `F.grid_sample`):

```python
# Before each F.grid_sample call:
if grid.dtype != input.dtype:
    grid = grid.to(input.dtype)
output = F.grid_sample(
    input, grid,
    mode='bilinear',
    align_corners=True,        # keep whatever the current export uses
    padding_mode='zeros',
)
```

This is one line per GridSample call and is the **lowest-risk** fix. PyTorch
will export the `.to(...)` as an ONNX `Cast` node, which TRT can fuse away
during engine compile (so no runtime cost).

If the GridSample calls are inside a shared helper (e.g. `_warp_features` in
`align_stem.py`), put the cast there once and every call site is fixed.

### 4b. Belt-and-braces — autocast wrapper for the entire forward

If `align_stem` isn't the only place this pattern occurs, the safer global
fix is to wrap the export's forward pass in `torch.autocast`:

```python
class ONNXExportWrapper(nn.Module):
    def __init__(self, model, fp16=False):
        super().__init__()
        self.model = model
        self.fp16 = fp16

    def forward(self, frames, config):
        if self.fp16:
            # autocast inserts Cast nodes consistently. The output Cast back to
            # fp16 happens automatically since the model returns fp16 tensors.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return self.model(frames, config)
        return self.model(frames, config)
```

Then export with `frames.half()` and `config.half()` as the dummy inputs.

### 4c. What NOT to do

* **Don't** rely on `onnxconverter-common.float16.convert_float_to_float16` for
  this model. We tried it: it crashes on the model's `Loop` / `Sequence`
  subgraphs (the temporal-attention modules). See our memory note
  `feedback_restora_fp16_local_conversion_fails.md` if you want the trace.

* **Don't** cast only the weights with `model.half()` and re-export. That's
  what produced the current broken file — same problem.

* **Don't** strip the dynamic-axes annotations from the export to "simplify".
  We need dynamic H, W, B (see §5).

---

## 5. The contract you must preserve

The C# inference side has been built against the existing 2-input contract.
**This contract must not change** — only the dtype across the graph.

### 5a. Inputs

| Name      | Shape                  | Dtype             | Range  |
|-----------|------------------------|-------------------|--------|
| `frames`  | `(B, 7, 3, H, W)`      | **float16** (was float32) | [0, 1] |
| `config`  | `(B, 5)`               | **float16** (was float32) | [0, 1] |

Both inputs must be the **same dtype** — ORT rejects mixed-dtype binding on
the consumer side. So when you re-export, the config tensor's dummy input
should also be `.half()`.

### 5b. Output

| Name     | Shape            | Dtype     | Range            |
|----------|------------------|-----------|------------------|
| `output` | `(B, 3, H, W)`   | **float16** | [0, 1] (clamped) |

The `clamp(0, 1)` baked into the export wrapper (currently at
`src/restora_models/export/wrapper.py:35` per the integration guide) must
**stay**. fp16 rounding can push the unclamped network output slightly outside
[0, 1] and we rely on the in-graph clamp for the uint8 quantisation downstream.

### 5c. Dynamic axes

Required:

* `frames`: dims `[0, 3, 4]` dynamic (batch B, height H, width W). The
  temporal axis T must remain **fixed at 7**.
* `config`: dim `[0]` dynamic (batch B).
* `output`: dims `[0, 2, 3]` dynamic (B, H, W).

H and W must remain **divisible by 16** at runtime (we pad to multiple-of-16
on the consumer side). The export should NOT bake in any specific H/W.

Confirm with `torch.onnx.export(..., dynamic_axes={...})`:

```python
torch.onnx.export(
    wrapper,
    (dummy_frames_fp16, dummy_config_fp16),
    out_path,
    input_names=['frames', 'config'],
    output_names=['output'],
    dynamic_axes={
        'frames':  {0: 'batch', 3: 'height', 4: 'width'},
        'config':  {0: 'batch'},
        'output':  {0: 'batch', 2: 'height', 3: 'width'},
    },
    opset_version=17,     # keep whatever the current export uses
    do_constant_folding=True,
)
```

### 5d. Identity invariant (the canary)

With `config = [0, 0, 0, 0, 0]`, the model must return **the centre frame of
the 7-frame input window, unchanged**. Specifically:

* `output[b, c, h, w] == frames[b, 3, c, h, w]` (PSNR ≥ 65 dB for fp16, per
  the integration guide §8 thresholds).

If this invariant breaks during re-export (e.g. somebody removes the residual
shortcut in the wrapper), our diagnostic test catches it immediately.

### 5e. Config axis order

Positional float[5] in this order (do not reshuffle):

| Index | Axis        |
|-------|-------------|
| 0     | `colorize`  |
| 1     | `denoise`   |
| 2     | `sharpen`   |
| 3     | `dejpeg`    |
| 4     | `deblur`    |

---

## 6. Alternative root causes (less likely)

If the GridSample fix doesn't make TRT compile cleanly, check these next:

* **fp16 Pow with large exponents.** Earlier we found that the fp32 export has
  constants outside ±65504 that TRT clips when `trt_fp16_enable=1` is on; we
  worked around it by pinning `trt_fp16_enable=0` on the fp32 path. A true
  fp16 export with the same large constants will hit the same clipping at
  engine-build time — the warning is `[Constant] contains out-of-range
  weights when cast to Half, clipping to +/- 65504`. If you see those at
  engine build, find the offending constants (likely softmax scale factors
  inside temporal-attention `exp(x - max)` numerical stabilisers) and
  refactor them so the constant stays representable in fp16.

* **`Loop` subgraph dtype propagation.** PyTorch's ONNX exporter is known to
  not always thread dtype changes into `Loop` body subgraphs. If GridSample
  isn't the only failing op, `torch.onnx.checker.check_model` after export
  will name the inconsistent edges.

---

## 7. Pre-flight checks before handing the file back

Run these in your repo, before zipping the file over:

### 7a. ORT loads it on CPU

```python
import onnxruntime as ort
sess = ort.InferenceSession(
    'iter_0030000_ema_generic_fp16.onnx',
    providers=['CPUExecutionProvider'],
)
print('inputs:', [(i.name, i.type, i.shape) for i in sess.get_inputs()])
print('output:', [(o.name, o.type, o.shape) for o in sess.get_outputs()])
```

Expected output (the key bits):

```
inputs: [
  ('frames', 'tensor(float16)', ['batch', 7, 3, 'height', 'width']),
  ('config', 'tensor(float16)', ['batch', 5]),
]
output: [
  ('output', 'tensor(float16)', ['batch', 3, 'height', 'width']),
]
```

Any `tensor(float)` (i.e. fp32) in those lists means a dtype slipped through —
re-export.

### 7b. Identity-gate canary, CPU EP

```python
import numpy as np
H, W = 128, 128
frames = np.random.rand(1, 7, 3, H, W).astype(np.float16)
config = np.zeros((1, 5), dtype=np.float16)

out = sess.run(['output'], {'frames': frames, 'config': config})[0]

# Check: output should match centre frame within fp16 rounding.
centre = frames[:, 3]
mse = ((out.astype(np.float32) - centre.astype(np.float32)) ** 2).mean()
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 200
print(f'identity PSNR: {psnr:.1f} dB (expected ≥ 65 dB for fp16)')
```

If PSNR < 60 dB the identity shortcut is broken — investigate before shipping.

### 7c. TensorRT compile (the actual check)

```python
import onnxruntime as ort
opts = ort.SessionOptions()
sess = ort.InferenceSession(
    'iter_0030000_ema_generic_fp16.onnx',
    sess_options=opts,
    providers=[
        ('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 4 * 1024**3,
            'trt_fp16_enable': '1',
            'device_id': 0,
        }),
        'CUDAExecutionProvider',
    ],
)
# Just constructing the session triggers TRT engine compile.
print('TRT session created — no errors means the graph is acceptable.')
```

**No `ITensor::getDimensions: Error Code 4` lines** in the TRT logger output
during this construct is the pass criterion. If TRT compiles cleanly here,
we're 95% of the way home.

### 7d. Per-axis differential

Each axis must produce a **visibly different** output from the identity.

```python
for axis_idx in range(5):
    cfg = np.zeros((1, 5), dtype=np.float16)
    cfg[0, axis_idx] = 1.0
    out = sess.run(['output'], {'frames': frames, 'config': cfg})[0]
    diff = np.abs(out.astype(np.float32) - centre.astype(np.float32)).max()
    name = ['colorize', 'denoise', 'sharpen', 'dejpeg', 'deblur'][axis_idx]
    print(f'{name:9s}: max diff vs centre = {diff:.3f}')
```

Expected: each line shows `max diff ≥ 0.05`. If any axis prints `0.000`, the
config wiring isn't reaching the network for that axis.

---

## 8. What we'll do on our side once you hand it back

Drop the new file at `models/iter_0030000_ema_generic_fp16.onnx`, replacing
the current broken one. Then we run:

```
dotnet test tests/E2E/Aitive.Restora.E2E.Tests/Aitive.Restora.E2E.Tests.csproj \
    --filter "FullyQualifiedName~RestoraOutputDiagnosticTests.DiagnosticInput"
```

Expected diff in test output vs today:

**Today** (broken fp16 line):
```
Model: iter_0030000_ema_generic_fp16
Input  stats: R mean=0.5112  G mean=0.5071  B mean=0.5033
Output stats: R mean=0.0000  G mean=0.0000  B mean=0.0000     ← all zeros, bug
```

**After fix:**
```
Model: iter_0030000_ema_generic_fp16
Input  stats: R mean=0.5112  G mean=0.5071  B mean=0.5033
Output stats: R mean=0.5112  G mean=0.5071  B mean=0.5033     ← identity, fixed
```

Then we run the benchmark:

```
dotnet test --filter "FullyQualifiedName~Benchmark_NightOfTheLivingDead"
```

And expect the `fp16-cuda` line to go from its current 5.4 fps (broken — runs
slowly because TRT falls back) to **~25 fps** end-to-end. If we don't see that
speedup, something else is off and we'll loop back with measurements.

---

## 9. Files to deliver

1. `iter_0030000_ema_generic_fp16.onnx` — the re-exported model.
2. The patched `src/restora_models/export/wrapper.py` (or a diff against the
   current revision) — so we know what changed.
3. Optional but appreciated: the **same checkpoint re-exported at fp32** with
   any wrapper-level changes you made for the fp16 fix. If the wrapper now
   conditionally casts grids in fp16-only mode, the fp32 export should be
   byte-identical to what's in `models/iter_0030000_ema_generic.onnx` today
   (so we can verify there's no regression).

---

## 10. Open questions / things we don't know

* **Is `model.half() + autocast` together overkill?** §4a alone might suffice
  if GridSample is the only mixed-dtype offender; §4b is the safer global
  fix. Your call.
* **Does the residual shortcut handle fp16 cleanly?** The identity invariant
  works on fp32 because the model's output wrapper does a `clamp(network(x) +
  x_center, 0, 1)`-style shortcut. In fp16, this addition might saturate
  differently — the canary in §7b catches this.
* **Is the `Loop` subgraph dtype-correct?** Beyond TRT's GridSample complaint,
  there may be other ops in the temporal-attention `Loop` that PyTorch's
  exporter didn't promote consistently. `onnx.checker.check_model` after
  export is the cheapest probe.

Reach out if anything in §7 misbehaves — we'd rather iterate on the export
together than ship a second broken file. Thanks!
