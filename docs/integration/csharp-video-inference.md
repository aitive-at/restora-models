# Restora Models — C# Inference Guide

> **Audience:** an engineer (or LLM agent) integrating a restora-models
> ONNX into a C# / .NET application that already has its own video
> decode / encode pipeline producing image tensors. This guide is purely
> about **how to fill the model's inputs and read its outputs** — no
> video I/O, no image library bindings. Bring your own tensors.

---

## 1. What the model does

One model performs **five image-restoration tasks** in a single forward
pass, gated by a 5-axis configuration vector you supply per call:

| Axis index | Task | What it does |
|---|---|---|
| 0 | **colorize** | Black-and-white → color |
| 1 | **denoise** | Removes Gaussian noise |
| 2 | **sharpen** | Super-resolution / detail recovery |
| 3 | **dejpeg** | Removes JPEG compression artifacts |
| 4 | **deblur** | Removes Gaussian + motion blur |

Any subset can be active per call. Inference cost is identical
regardless of which axes are on (the model always runs the full graph
and the config vector gates internal pathways).

Architecture: NAFNet-large encoder/decoder + Lab-native dual output
head (Lab-delta + ab-abs gated by colorize axis) + adversarial refine
head. Convolutional throughout; trained at 256×256 with a video-pair
temporal-consistency loss, so frame-by-frame application to video is
stable without explicit cross-frame state.

---

## 2. ONNX export — which variant to consume

The training side can emit two flavors:

| Flavor | Inputs | When to use |
|---|---|---|
| **Generic** | `input` (image) + `config` (5-axis vector) | Runtime task toggles ← **this is what you want** |
| **Per-task baked** | `input` only (config baked as buffer) | Single fixed task; no UI toggles |

Ask the training side for:

```sh
uv run restora export \
  --model runs/<run>/ckpt/final.pt \
  --output restora_generic_fp16.onnx \
  --precision fp16 \
  --dynamic-hw
```

Result: one ONNX file (~85 MB at fp16) with:
- `input` shape `(B, 3, H, W)`, dynamic H/W
- `config` shape `(B, 5)`
- `output` shape `(B, 3, H, W)`, same H/W as input

For very old GPUs without fp16 acceleration, `--precision fp32` instead
(~340 MB).

---

## 3. Tensor contract

This is the entire interface. Everything you put on the wire must match.

### Input 1 — `input` (the image)

| Property | Value |
|---|---|
| Name | `"input"` |
| Type | `float32` |
| Layout | `(B, 3, H, W)` — **batch first, channels second, then spatial** (CHW per sample, not HWC) |
| Channel order | **RGB** — index 0 = R, index 1 = G, index 2 = B |
| Value range | `[0.0, 1.0]` — if your source is `uint8 [0, 255]`, divide by 255 |
| H, W constraint | Both must be **multiples of 32** (5 downsample stages in NAFNet). Pad upstream if needed. |

If your decode pipeline gives you tensors in any other layout (e.g.
HWC, BGR, [0, 255]), permute / swap / scale before feeding in.

### Input 2 — `config` (the task toggles)

| Property | Value |
|---|---|
| Name | `"config"` |
| Type | `float32` |
| Layout | `(B, 5)` — one row per image in the batch |
| Values | Either `0.0` (axis off) or `1.0` (axis on). Soft values in `(0, 1)` blend, but aren't meaningful for production. |
| Axis order | `[colorize, denoise, sharpen, dejpeg, deblur]` |

**Every image in the batch must carry its own config row.** If you batch
N frames with the same toggles, broadcast the same 5-vector N times into
shape `(N, 5)`.

### Output — `output` (the restored image)

| Property | Value |
|---|---|
| Name | `"output"` |
| Type | `float32` |
| Layout | `(B, 3, H, W)` — same as input |
| Channel order | **RGB**, channel-first |
| Value range | Approximately `[0.0, 1.0]` — clamp before converting back to `uint8`. The model can produce small overshoots near saturated pixels (~0.01 above/below); always clamp. |

---

## 4. Building the config tensor from UI toggles

Map your task-selection state into the 5-vector. The exact UI representation
is up to you; here's a clean pattern:

```csharp
[Flags]
public enum RestoraTask
{
    None     = 0,
    Colorize = 1 << 0,   // axis 0
    Denoise  = 1 << 1,   // axis 1
    Sharpen  = 1 << 2,   // axis 2  (super-resolution / detail recovery)
    DeJpeg   = 1 << 3,   // axis 3
    Deblur   = 1 << 4,   // axis 4
}

public static float[] ConfigVector(RestoraTask tasks) => new[]
{
    tasks.HasFlag(RestoraTask.Colorize) ? 1f : 0f,
    tasks.HasFlag(RestoraTask.Denoise)  ? 1f : 0f,
    tasks.HasFlag(RestoraTask.Sharpen)  ? 1f : 0f,
    tasks.HasFlag(RestoraTask.DeJpeg)   ? 1f : 0f,
    tasks.HasFlag(RestoraTask.Deblur)   ? 1f : 0f,
};
```

> **The `[Flags]` enum's bit values (1, 2, 4, 8, 16) are not the same as
> the tensor's axis order.** Bit-position is just for the C# enum's
> `HasFlag()`; the tensor expects positional order `[colorize, denoise,
> sharpen, dejpeg, deblur]`. The `ConfigVector()` method does the mapping
> explicitly so the order is one place to change.

For batched inference where every frame uses the same toggles:

```csharp
public static DenseTensor<float> ConfigTensorForBatch(RestoraTask tasks, int batchSize)
{
    var row = ConfigVector(tasks);
    var packed = new float[batchSize * 5];
    for (int i = 0; i < batchSize; i++)
        Array.Copy(row, 0, packed, i * 5, 5);
    return new DenseTensor<float>(packed, new[] { batchSize, 5 });
}
```

If different frames in the batch have different toggles, build each row
independently (rare in practice — usually a UI commits one toggle state
per video).

---

## 5. Running inference

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public sealed class RestoraSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;   // resolved from model metadata
    private readonly string _configName;  // resolved from model metadata
    private readonly string _outputName;  // resolved from model metadata

    public RestoraSession(string onnxPath, SessionOptions options)
    {
        _session = new InferenceSession(onnxPath, options);

        // Don't hardcode names — read them from the loaded model so a
        // re-export with different naming still works.
        var inputs = _session.InputMetadata.Keys.ToList();
        _inputName  = inputs.FirstOrDefault(n => n == "input")  ?? inputs[0];
        _configName = inputs.FirstOrDefault(n => n == "config") ?? inputs[1];
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Run a single forward pass.
    ///   image  -- shape (B, 3, H, W) float32 RGB in [0, 1]; H,W multiples of 32
    ///   config -- shape (B, 5)       float32 toggles
    /// Returns a DenseTensor (B, 3, H, W) float32 RGB in ~[0, 1] (clamp before use).
    /// </summary>
    public DenseTensor<float> Run(DenseTensor<float> image, DenseTensor<float> config)
    {
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor(_inputName,  image),
            NamedOnnxValue.CreateFromTensor(_configName, config),
        };

        using var results = _session.Run(inputs);
        // Materialize the output tensor before disposing the result collection.
        var output = results.First(r => r.Name == _outputName).AsTensor<float>();
        return ToDense(output);
    }

    private static DenseTensor<float> ToDense(Tensor<float> t)
    {
        if (t is DenseTensor<float> dense) return dense;
        // Output from the EP can be a non-dense view; copy to a fresh dense tensor.
        var copy = new DenseTensor<float>(t.Dimensions.ToArray());
        t.Buffer.CopyTo(copy.Buffer);
        return copy;
    }

    public void Dispose() => _session.Dispose();
}
```

Usage:

```csharp
var sessionOptions = new SessionOptions
{
    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
};
sessionOptions.AppendExecutionProvider_CUDA(deviceId: 0);
// or: sessionOptions.AppendExecutionProvider_DML(deviceId: 0);
// or: leave it for CPU-only fallback.

using var model = new RestoraSession("restora_generic_fp16.onnx", sessionOptions);

DenseTensor<float> imageTensor  = /* (B, 3, H, W) RGB float32 in [0,1] from your pipeline */;
DenseTensor<float> configTensor = ConfigTensorForBatch(RestoraTask.Colorize | RestoraTask.Sharpen, batchSize: imageTensor.Dimensions[0]);

DenseTensor<float> result = model.Run(imageTensor, configTensor);
// result.Dimensions == imageTensor.Dimensions; clamp values to [0,1] before
// converting back to whatever your downstream expects (uint8, bf16, etc.)
```

That's the entire physical interface. Two named tensor inputs go in,
one named tensor output comes out.

---

## 6. Constraints to enforce upstream

Things your tensor pipeline must do **before** calling `Run`:

1. **H and W are multiples of 32.** NAFNet has 5 sequential `/2`
   downsamples; non-multiple sizes fail at the first non-divisible stage.
   Pad with reflect-border (recommended) or zero-pad, then crop the
   result back. Pad amounts: `padH = (32 - H % 32) % 32` and likewise
   for W.

2. **Layout is CHW per sample, not HWC.** If your decode pipeline
   produces HWC tensors, permute to CHW before feeding in. Wrong layout
   typically shows up as completely garbled output rather than an error.

3. **Channel order is RGB.** If your pipeline gives you BGR (common
   from OpenCV / some video decoders), swap the channel axis before
   packing the tensor. Wrong order shows up as a hue shift on
   colorize-enabled outputs (looks roughly correct but tinted) and as
   subtle texture errors elsewhere.

4. **Values are in `[0, 1]`.** Scale your `uint8 [0, 255]` source by
   `1/255`. If you feed `[0, 255]` floats by mistake, output is
   saturated.

5. **dtype is `float32`.** ONNX Runtime will reject anything else for
   these inputs. (The model weights are fp16 internally if you exported
   with `--precision fp16`, but the I/O contract is still fp32 — ORT
   converts at the boundary.)

---

## 7. Performance

### Reuse the session

`InferenceSession` construction loads the ONNX file, optimizes the
graph, and allocates execution-provider resources. It's slow (hundreds
of ms to seconds). **Build one session per (model, EP) and keep it
alive for the app lifetime.**

### Pre-warm

The first `Run()` call triggers kernel autotune / JIT compilation and
is 3–10× slower than subsequent calls. After session creation, fire one
dummy call with a representative shape:

```csharp
var dummyImage  = new DenseTensor<float>(new[] { 1, 3, 256, 256 });
var dummyConfig = ConfigTensorForBatch(RestoraTask.None, 1);
model.Run(dummyImage, dummyConfig);  // discard result; first call done.
```

Do this at app startup, ideally on a background thread before the user
clicks anything.

### Batch frames

`Run` with batch=N is ~3× faster than N sequential calls with batch=1
(kernel-launch overhead amortizes). If your pipeline buffers N frames
per chunk, feed all N in one `Run`. Output shape is `(N, 3, H, W)`;
slice rows out for downstream consumers.

Practical batch sizes on a single GPU (rough rules of thumb at 1080p
fp16): 12 GB GPU ≈ batch 4, 24 GB ≈ batch 8, 40+ GB ≈ batch 16+.

### Execution provider selection

| EP | When |
|---|---|
| `AppendExecutionProvider_CUDA` | NVIDIA GPU, CUDA 12+ runtime, Linux or Windows. Fastest. |
| `AppendExecutionProvider_DML` | DirectX 12 (any GPU on Windows incl. integrated). Slower than CUDA on NVIDIA but cross-vendor. |
| `AppendExecutionProvider_CoreML` | macOS / Apple Silicon. Use with the CoreML EP NuGet. |
| (default CPU) | Fallback. 10–50× slower; only viable for batch=1, small images. |

Don't mix EPs — one session, one EP.

### fp16 vs fp32 trade-off

| Precision | Quality | Speed | Memory |
|---|---|---|---|
| **fp16** | Visually identical | ~2× faster on modern GPUs | ~half |
| fp32 | Reference | 1× | 1× |

Use fp16 unless you're targeting GPUs without fp16 throughput
acceleration (some integrated and pre-Pascal NVIDIA cards).

---

## 8. Multi-model selection

If you ship multiple ONNX files (different training runs, different
precisions, different baked task subsets), keep them as a registry:

```csharp
public sealed class RestoraRegistry : IDisposable
{
    private readonly Dictionary<string, RestoraSession> _sessions = new();

    public void Register(string name, string onnxPath, SessionOptions options) =>
        _sessions[name] = new RestoraSession(onnxPath, options);

    public RestoraSession this[string name] => _sessions[name];
    public IReadOnlyCollection<string> Names => _sessions.Keys;

    public void Dispose()
    {
        foreach (var s in _sessions.Values) s.Dispose();
        _sessions.Clear();
    }
}
```

In your UI: dropdown lists `registry.Names`, selection sets
`currentSession = registry[name]`. Switching between pre-loaded
sessions is instant; constructing fresh sessions per switch is not.

**Memory note:** each loaded fp16 session reserves ~85 MB ONNX file +
~few hundred MB of EP working memory. If you offer 10 model variants in
the UI, only pre-load the 2–3 you expect to be common; lazy-load the
rest on first selection.

---

## 9. Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Output identical to input | Config vector all zeros | At least one axis must be 1.0 to trigger restoration |
| Output completely scrambled | Layout wrong (HWC sent as CHW, or B/H/W transposed) | Verify `Dimensions == (B, 3, H, W)` before `Run` |
| Output has hue shift (greenish, magenta) on colorize | Channel order BGR instead of RGB | Swap channel dim before packing |
| Output clipped / saturated | Values fed as `[0, 255]` instead of `[0, 1]` | Divide by 255 upstream |
| First call after model load is 5–30× slower | EP not pre-warmed | Run a dummy at startup (§7) |
| Run fails with "shape mismatch" on config | Config shape `(5,)` not `(B, 5)`, or B≠image batch | Config must be 2-D, first dim must match image's batch |
| GPU OOM on large frames at fp32 | One sample > VRAM budget | Use fp16, smaller H/W, or batch=1 |
| Output flickers across video frames | Per-frame inference variance | Model is already temporal-trained; further smoothing must be done outside (e.g. exponential moving average across frame outputs in your pipeline) |

---

## Summary cheat sheet

```csharp
// 1. Once per app:
var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
opts.AppendExecutionProvider_CUDA(0);
using var model = new RestoraSession("restora_generic_fp16.onnx", opts);

// 2. Pre-warm (once, at startup):
model.Run(
    new DenseTensor<float>(new[] { 1, 3, 256, 256 }),
    ConfigTensorForBatch(RestoraTask.None, 1));

// 3. Per call: feed your tensors, read result.
//    Caller guarantees: image is float32 (B, 3, H, W) RGB in [0,1], H&W % 32 == 0.
DenseTensor<float> image  = /* from your pipeline */;
DenseTensor<float> config = ConfigTensorForBatch(currentTasks, image.Dimensions[0]);
DenseTensor<float> result = model.Run(image, config);
// result has the same dimensions as image; clamp to [0, 1] before
// converting back to your downstream representation.
```

Three things to remember:
- **CHW, RGB, [0, 1], float32, H/W % 32 == 0** for the image tensor.
- **(B, 5) float32 axis-positional** for the config tensor.
- **Reuse session, pre-warm, batch, fp16** for performance.

Everything else — codecs, I/O, color spaces, padding, scaling — is yours.
