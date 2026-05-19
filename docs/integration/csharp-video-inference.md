# C# Integration Guide

End-to-end recipe for running the temporal restoration model from a
C# application that already has tensor / ONNX-Runtime plumbing. This
guide focuses on the **algorithmic contract** — the exact data
transformations, the sliding-window construction, the UI ↔ config
mapping, and the edge cases — rather than ONNX-Runtime API boilerplate.

Companion reference: `docs/integration/onnx-inference-guide.md` covers
the cross-language contract and TensorRT engine builds.

---

## 1. The exported models

Two variants ship per checkpoint, both **generic 2-input** ONNX with
dynamic spatial axes:

| File suffix                 | Inputs           | Precision | Size   | Use when                           |
|-----------------------------|------------------|-----------|--------|------------------------------------|
| `_ema_generic.onnx`         | `frames, config` | fp32      | ~108 MB | Quality benchmarks, CPU inference  |
| `_ema_generic_fp16.onnx`    | `frames, config` | fp16      | ~52 MB  | GPU production (default)           |

Both are **the same model with the same 2-input contract** — the only
difference is tensor precision. The C# wiring is identical for both;
just bind `Float16` instead of `Float32` for the fp16 file. The fp16
graph is ~2× faster on most GPUs at a max ~1.9e-3 visual difference
(well below perceptual threshold).

Per the project's design, **both axes are toggled at inference time
via the `config` vector** — no separate models per axis, no rebuild
when the user changes which restoration steps to apply. A single
forward pass produces a result with any subset of the 5 axes enabled.

---

## 2. Model contract (C# perspective)

### Inputs

| Name      | Shape                  | Dtype                     | Range  |
|-----------|------------------------|---------------------------|--------|
| `frames`  | `(B, 7, 3, H, W)`      | float32 *or* float16      | [0, 1] |
| `config`  | `(B, 5)`               | float32 *or* float16      | [0, 1] |

The `frames` tensor is **5-dimensional** (batch × temporal × channels
× height × width). PyTorch's NCHW convention with an extra `T = 7`
axis injected after batch. This is the most common shape mistake in
C# consumers — some `DenseTensor` helper libraries default to 4D; you
need explicit 5D.

`H` and `W` must be **multiples of 16**. The graph has dynamic
spatial axes so any divisible-by-16 size works at runtime; non-
multiples must be padded by the consumer (see §3).

`config` is a 5-element vector indexing the active restoration axes:

| Index | Axis        | Meaning when set to 1.0                        |
|-------|-------------|------------------------------------------------|
| 0     | `colorize`  | Predict color for grayscale/desaturated input  |
| 1     | `denoise`   | Remove noise / grain                           |
| 2     | `sharpen`   | Restore high-frequency detail                  |
| 3     | `dejpeg`    | Remove JPEG / MPEG block artifacts             |
| 4     | `deblur`    | Reverse motion / camera blur                   |

Values are floats in [0, 1]. For UI checkboxes use `{0.0, 1.0}`; for
intensity sliders the model accepts intermediate values, though
training was conducted only at the 0/1 endpoints — intermediates are
graceful but unverified.

**Identity invariant** (verification canary): `config = [0,0,0,0,0]`
returns the *center frame* of the input window unchanged. PSNR between
output and `frames[:, 3]` will be ≥120 dB on fp32 and ~65-70 dB on
fp16. See §8.

### Output

| Name     | Shape           | Dtype                | Range  |
|----------|-----------------|----------------------|--------|
| `output` | `(B, 3, H, W)`  | matches input dtype  | [0, 1] |

A single RGB frame — the **restored center frame** (index 3 of the
7-frame input window). It is **not** a 7-frame clip; the temporal
window is purely input context.

---

## 3. UI ↔ config mapping

Wire UI checkboxes straight to the `config` vector. The model performs
**one forward pass per output frame** regardless of how many checkboxes
are on, so toggling axes is free at inference time — no model swap, no
rebuild.

```csharp
public sealed class AxisConfig
{
    public bool Colorize { get; set; }
    public bool Denoise  { get; set; }
    public bool Sharpen  { get; set; }
    public bool Dejpeg   { get; set; }
    public bool Deblur   { get; set; }

    public float[] ToVector() => new[]
    {
        Colorize ? 1f : 0f,
        Denoise  ? 1f : 0f,
        Sharpen  ? 1f : 0f,
        Dejpeg   ? 1f : 0f,
        Deblur   ? 1f : 0f,
    };
}
```

Binding to UI (WinForms / WPF / Avalonia / Maui all work the same):

```csharp
// In the checkbox's CheckedChanged handler:
void OnAxisToggled(object sender, EventArgs e)
{
    _currentConfig = new AxisConfig
    {
        Colorize = chkColorize.Checked,
        Denoise  = chkDenoise.Checked,
        Sharpen  = chkSharpen.Checked,
        Dejpeg   = chkDejpeg.Checked,
        Deblur   = chkDeblur.Checked,
    };
    // Trigger re-render of the preview frame with the new config
    _engine.RenderPreview(_currentFrameIndex, _currentConfig);
}
```

**Live-preview pattern:** because each toggle changes only the
`config` tensor (a 5-element vector), the per-frame work to re-render
is dominated by the inference itself. There is *no* expensive setup
when the user toggles a checkbox — you can re-run inference on the
same frame instantly. This is a deliberate design choice over per-axis
ONNX models.

**Default UI state recommendation:** all axes off (config=zeros) at
startup. By the identity invariant, this displays the raw center
frame — the user immediately sees the source content with no
modification, then enables axes to introduce restoration. Lower
cognitive load than "everything is restored by default and you have
to figure out what's enhancement vs source."

---

## 4. Per-frame preprocessing

Input source frames are typically delivered as **BGR uint8** from
OpenCV, FFmpeg, or `System.Drawing`. The model wants **RGB in [0,1]
NCHW layout, padded to multiple-of-16**.

Apply these transforms in order — order matters for pad / convert:

```
BGR uint8 (H, W, 3)
  └─ swap channel order to RGB         → RGB uint8 (H, W, 3)
  └─ cast to float32 and divide by 255 → RGB float32 (H, W, 3), [0, 1]
  └─ pad H and W to next multiple of 16 with edge-replicate
                                       → RGB float32 (Hp, Wp, 3)
  └─ transpose HWC → CHW               → RGB float32 (3, Hp, Wp)
  └─ (fp16 only) cast float32 → float16 at the end
```

Reasoning on edge-replicate padding: zero-padding injects a hard
black border that the conv stems treat as image content and color/
denoise around — producing visible halos. Edge-replicate (copy the
boundary pixel outward) matches what the trainer's data loader does
on REDS clips.

Record `(H, W)` so you can crop the output back at the end.

### Pad math

```csharp
int Hp = H + ((16 - H % 16) % 16);
int Wp = W + ((16 - W % 16) % 16);
```

`(16 - x % 16) % 16` evaluates to 0 when `x` is already a multiple of
16, so it's a no-op on aligned inputs. 1080p (1920×1080) needs 8 rows
of padding on H. 720p (1280×720) needs 0.

### fp16 caveat

Cast to float16 *after* the divide-by-255 and pad steps — keep
intermediates in fp32, only the final tensor binding is fp16. Both
`frames` and `config` must be float16 for the fp16 ONNX file (mixing
fp32 and fp16 inputs is rejected by ORT).

---

## 5. Building the 7-frame input window

The model expects 7 contiguous frames *centered on* the frame you
want to restore. To produce output frame `i`, feed input frames:

```
[i-3, i-2, i-1, i, i+1, i+2, i+3]
```

with each index **clamped** to `[0, N-1]` (edge-replicate at clip
boundaries):

```csharp
int Clamp(int idx, int n) => Math.Max(0, Math.Min(n - 1, idx));

int[] WindowIndices(int center, int n)
{
    var indices = new int[7];
    for (int k = 0; k < 7; k++)
        indices[k] = Clamp(center - 3 + k, n);
    return indices;
}
```

For a 100-frame clip:

| Output frame `i` | Input indices                  |
|------------------|--------------------------------|
| 0                | `[0, 0, 0, 0, 1, 2, 3]`        |
| 1                | `[0, 0, 0, 1, 2, 3, 4]`        |
| 50               | `[47, 48, 49, 50, 51, 52, 53]` |
| 98               | `[95, 96, 97, 98, 99, 99, 99]` |
| 99               | `[96, 97, 98, 99, 99, 99, 99]` |

Stack the 7 preprocessed CHW tensors along a new leading dim →
`(7, 3, Hp, Wp)`, prepend a batch dim → `(1, 7, 3, Hp, Wp)`. The flat
float buffer is row-major `[B, T, C, H, W]`.

### Single-image use case

Treat a still as a 1-frame clip with `N = 1`: every window index
clamps to 0, so the 7-frame buffer is just the image replicated 7×.
This is the supported single-frame path; the temporal-attention
components degenerate gracefully when all frames are identical.

---

## 6. The inference call

```csharp
// Reuse this DenseTensor across frames — only the underlying buffer
// changes. Allocating 7×3×Hp×Wp floats per frame is the #1 cause of
// stuttery video playback.
var framesTensor = new DenseTensor<float>(framesBuffer,
    new[] { 1, 7, 3, Hp, Wp });

var configTensor = new DenseTensor<float>(
    currentConfig.ToVector(), new[] { 1, 5 });

var inputs = new[]
{
    NamedOnnxValue.CreateFromTensor("frames", framesTensor),
    NamedOnnxValue.CreateFromTensor("config", configTensor),
};

using var results = session.Run(inputs);
var output = results.First().AsTensor<float>();   // (1, 3, Hp, Wp)
```

For fp16 (`_ema_generic_fp16.onnx`): bind both `frames` and `config`
as `DenseTensor<Float16>`. Output is also `DenseTensor<Float16>` —
cast back to float32 before postprocessing arithmetic.

Note that the **`config` tensor is rebuilt every frame** (it's only 5
floats — trivial cost). This is what makes UI-toggle response instant:
the next frame after a checkbox change uses the new config without
any other plumbing.

---

## 7. Post-processing

Inverse of preprocessing, in order:

```
output (1, 3, Hp, Wp), fp32 or fp16
  └─ (fp16 only) cast → fp32           → (1, 3, Hp, Wp) fp32
  └─ squeeze batch                     → (3, Hp, Wp)
  └─ clamp to [0, 1]                   → (3, Hp, Wp), clean
  └─ transpose CHW → HWC               → (Hp, Wp, 3)
  └─ crop to original (H, W)           → (H, W, 3)
  └─ multiply by 255, cast to uint8    → (H, W, 3) uint8 RGB
  └─ swap RGB → BGR                    → (H, W, 3) uint8 BGR
```

The clamp is *belt-and-suspenders* — the exported wrapper has
`clamp(0, 1)` baked in (`src/restora_models/export/wrapper.py:35`) so
fp32 output is already clean. But fp16 can produce values
fractionally outside [0, 1] (1.0001, -0.0003) due to half-precision
rounding, and uint8 casting would wrap-around those. Always clamp.

---

## 8. Verifying your integration

Run this **identity-gate canary** before trusting any output:

1. Load the `_ema_generic.onnx` file (fp32 for clearer thresholds).
2. Build a random `(1, 7, 3, 128, 128)` `frames` tensor with values
   in [0, 1].
3. Set `config = [0, 0, 0, 0, 0]` (all axes off).
4. Run inference → `output` shape `(1, 3, 128, 128)`.
5. Extract `frames[:, 3, :, :, :]` — the center frame.
6. Compute MSE between `output` and the center frame.
7. **Expected:**
   - fp32 ONNX (any EP): PSNR ≥ 120 dB (clamp ceiling, essentially perfect)
   - fp16 ONNX on CPU EP: PSNR ~65-70 dB (CPU emulates fp16 with fp32
     accumulators internally, so quantization is only at the I/O boundary)
   - fp16 ONNX on CUDA / TensorRT EP: PSNR ~55-60 dB (intrinsic native-fp16
     lab round-trip precision floor on GPU silicon; the identity path
     goes through `lab_to_rgb(rgb_to_lab(center))` and native fp16
     can't represent the round-trip more accurately than that —
     verified directly on Blackwell sm_120, expect similar on sm_100/103)

   The CPU vs GPU fp16 gap is *not* a bug — it's the difference between
   fp32-emulated and native-silicon fp16 precision. Both are visually
   lossless (fp16 noise floor is ~3-4 decimal digits; 55 dB is well
   below any perceptual threshold for restored video).

PSNR-to-bug lookup if the canary fails on fp32:

| Observed PSNR | Likely root cause                              |
|---------------|------------------------------------------------|
| < 10 dB       | Channel order swap missing (BGR vs RGB)        |
| 15–25 dB      | HWC ↔ CHW transpose missing or mis-axed        |
| 30–45 dB      | uint8/float scale wrong (forgot `/255`)        |
| 50–80 dB      | fp16 file loaded as if fp32, or vice versa     |
| 110–119 dB    | Working as expected (minor fp32 noise)         |
| 120 dB        | Perfect — clamp saturation                     |

Once the canary passes, the **single-axis canary** confirms config
wiring:

1. Same random `frames` tensor.
2. Set `config = [1, 0, 0, 0, 0]` (colorize only).
3. Output should be **visibly different** from the center frame —
   typically max-diff ~0.5-0.9 on random input (the colorize axis
   is the most aggressive).
4. Try `config = [0, 0, 0, 0, 1]` (deblur only) on the same frames:
   different result. If both configs give *identical* outputs, your
   `config` tensor isn't reaching the graph (check the input name
   matches `"config"` exactly).

---

## 9. Performance — what actually matters

In rough order of impact (highest first):

1. **Reuse the input tensor buffer across frames.** The 7×3×Hp×Wp
   `float[]` is the largest allocation per frame. Manage it as a
   persistent buffer filled in place each iteration. Same for the
   output buffer. This is *the* difference between 25 fps and 60 fps
   at 1080p.
2. **Use the GPU execution provider.** `AppendExecutionProvider_CUDA`
   on NVIDIA, `AppendExecutionProvider_DML` on DirectX. CPU works
   but is 20-50× slower for this model size.
3. **Use the fp16 ONNX** (`_ema_generic_fp16.onnx`) when on GPU. Half
   the transfer cost, half the compute, no meaningful quality loss.
4. **Incremental window update.** For video, the window for frame
   `i+1` shares 6 frames with the window for frame `i`. Instead of
   rebuilding from scratch, shift the buffer by `3×Hp×Wp` floats and
   only preprocess the one new frame. Cuts per-frame preprocessing
   cost by ~85%.
5. **Pin host memory.** ONNX-Runtime CUDA EP benefits from pinned
   (page-locked) host buffers for the H2D transfer. ORT exposes this
   via `OrtMemoryInfo` with pinned allocation.
6. **Skip preprocessing when input is already aligned.** If your
   pipeline upstream already yields RGB float NCHW padded, skip the
   conversion stages — measure the bytes-already-correct fast path.

The `config` tensor is 5 floats; rebuilding it per frame is free, do
not optimize it.

Don't bother with: pre-loading the whole video into memory (RAM bound
for >1 min clips), or running multiple frames in a batch (the
sliding-window structure makes this awkward; pipeline parallelism
across CPU preprocessing + GPU inference is the cheaper win).

---

## 10. Recommended starting code structure

```
class TemporalRestoreEngine
{
    InferenceSession _session;
    float[]  _framesBuffer;    // reusable, sized to 7*3*Hp*Wp
    int      _Hp, _Wp;
    DenseTensor<float> _framesTensor;
    DenseTensor<float> _configTensor;

    void LoadModel(string onnxPath, bool useGpu);

    // Single-frame entry point — for previews, scrubbing, single
    // images. Builds a clip from one frame.
    byte[] ProcessSingleFrame(byte[] bgrFrame, int H, int W,
                              AxisConfig axes);

    // Video entry point — owns the sliding-window state.
    IEnumerable<byte[]> ProcessClip(IReadOnlyList<byte[]> bgrFrames,
                                    int H, int W, AxisConfig axes);

    // Live-preview entry point — re-runs inference on a frame
    // already in the window. For checkbox-toggle UI updates.
    byte[] RerunCurrentFrame(AxisConfig axes);
}
```

`ProcessClip` owns the sliding-window state, fills `_framesBuffer`
in place per output frame, rebuilds the 5-element `config` from
`axes`, and yields BGR uint8 results. `RerunCurrentFrame` skips the
window-advance step and re-runs inference on the existing buffer
contents with a new config — that's the live-preview toggle path.

Wire it to your existing tensor abstraction by replacing the
"build a DenseTensor" step with your equivalent constructor.

---

## Reference

- ONNX export source: `src/restora_models/export/wrapper.py`
  (`ONNXExportWrapper`)
- Python reference inference: `src/restora_models/infer/pipeline.py`
  (`VideoPipeline.process_directory`) — the exact algorithm this
  guide mirrors. Use it as a Python-side equivalent when verifying
  outputs against your C# integration.
- Identity-gate canary numerical thresholds were measured directly
  on the 30k checkpoint exports (`runs/iter_0030000_ema_generic*.onnx`).
