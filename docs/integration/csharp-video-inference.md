# Restora Models — C# Video Inference Guide

> **Audience:** an engineer (or LLM agent) integrating a restora-models
> ONNX into a C# / .NET downstream application that processes **video
> frame-by-frame**, with a UI that lets the user **select a model file**
> and **toggle individual restoration tasks at runtime**.

This guide is self-contained — you don't need to read the training repo.
Everything you need (model contract, pre/post processing, full C# class,
video pipeline, perf tips) is below.

---

## 1. What the model does

One model performs **five image-restoration tasks** in a single forward
pass, gated by a 5-axis configuration vector:

| Axis index | Task | What it does |
|---|---|---|
| 0 | **colorize** | Black-and-white → color |
| 1 | **denoise** | Removes Gaussian noise |
| 2 | **sharpen** | Super-resolution / detail recovery |
| 3 | **dejpeg** | Removes JPEG compression artifacts |
| 4 | **deblur** | Removes Gaussian + motion blur |

You can enable any subset per frame (e.g. `[1, 1, 0, 0, 0]` = colorize +
denoise; `[1, 0, 0, 1, 0]` = colorize + dejpeg). Inference cost is
identical regardless of which axes are on.

The model is convolutional (NAFNet-large + adversarial refine head). It
was trained on natural images at 256×256, and on video pairs with a
temporal-consistency loss — so applying it frame-by-frame to video
produces stable output without explicit cross-frame state.

---

## 2. ONNX export — which variant to use

The training side can emit two flavors of ONNX:

| Flavor | Inputs | When to use |
|---|---|---|
| **Generic** | `input` (image) + `config` (5-axis vector) | UI with runtime task toggles ← **this is what you want** |
| **Per-task baked** | `input` only (config baked as buffer) | Deployment of one fixed task (e.g. always colorize) |

For your use case (user toggles tasks at runtime), ask for the **generic
ONNX with dynamic spatial dims**. The Python export command is:

```sh
uv run restora export \
  --model runs/<run>/ckpt/final.pt \
  --output restora_generic_fp16.onnx \
  --precision fp16 \
  --dynamic-hw
```

This produces a single ONNX file with:
- `input` shape `(B, 3, H, W)`, dynamic H/W
- `config` shape `(B, 5)`
- `output` shape `(B, 3, H, W)`, same H/W as input
- fp16 weights (~85 MB), ~2× faster than fp32 on consumer GPUs

For low-end GPUs (no fp16 acceleration), use `--precision fp32` instead
(~340 MB, native float speed).

---

## 3. Tensor contract — exact details

### Input 1: `input`

| Property | Value |
|---|---|
| Name | `"input"` |
| Type | `float32` |
| Shape | `(B, 3, H, W)` — channel-first, H/W any positive int |
| Channel order | **RGB** (NOT BGR — OpenCV/Mat default is BGR; you must swap) |
| Range | `[0.0, 1.0]` — divide by 255 from the uint8 source |
| H, W | Must be multiples of 32 (NAFNet has 5 downsample stages). Pad up if needed. |

### Input 2: `config`

| Property | Value |
|---|---|
| Name | `"config"` |
| Type | `float32` |
| Shape | `(B, 5)` |
| Values | Either 0.0 or 1.0 per axis. (Soft 0.5 values work but produce blended outputs that aren't meaningful for production.) |
| Axis order | `[colorize, denoise, sharpen, dejpeg, deblur]` |

### Output

| Property | Value |
|---|---|
| Name | `"output"` |
| Type | `float32` |
| Shape | `(B, 3, H, W)` — same spatial dims as input |
| Channel order | **RGB**, channel-first |
| Range | `[0.0, 1.0]` — clamp + multiply by 255 to get back to uint8 |

---

## 4. Prerequisites

NuGet packages (.NET 6 or newer):

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.21.*" />
<PackageReference Include="OpenCvSharp4" Version="4.10.0.*" />
<PackageReference Include="OpenCvSharp4.runtime.win" Version="4.10.0.*" /> <!-- or .ubuntu, .osx -->
```

Pick **one** ONNX Runtime package:
- `Microsoft.ML.OnnxRuntime.Gpu` — CUDA on NVIDIA (Linux + Windows). Requires CUDA 12.x runtime.
- `Microsoft.ML.OnnxRuntime.DirectML` — DirectX 12 on Windows (any GPU vendor).
- `Microsoft.ML.OnnxRuntime` (CPU only) — slow but zero install friction.

For the video use case on Windows-with-any-GPU, **DirectML** is the
lowest-friction choice. On Linux with NVIDIA, use CUDA.

---

## 5. Core inference class

Full working code. Drop into your project, adjust namespaces.

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Restora.Inference;

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

public enum ExecutionBackend { Cpu, Cuda, DirectML }

/// <summary>
/// Wraps a generic restora-models ONNX (2 inputs: image + config).
/// Thread-safety: Run() is NOT thread-safe on a single session. Use one
/// instance per worker thread, OR serialize calls via lock.
/// </summary>
public sealed class RestoraSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _configName;
    private readonly string _outputName;

    public RestoraSession(string onnxPath, ExecutionBackend backend = ExecutionBackend.Cuda, int deviceId = 0)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };

        switch (backend)
        {
            case ExecutionBackend.Cuda:
                options.AppendExecutionProvider_CUDA(deviceId);
                break;
            case ExecutionBackend.DirectML:
                options.AppendExecutionProvider_DML(deviceId);
                break;
            case ExecutionBackend.Cpu:
                // No EP needed — falls through to CPU.
                break;
        }

        _session = new InferenceSession(onnxPath, options);

        // Discover input/output names (don't hardcode — re-exports may
        // change them in edge cases).
        var inputs = _session.InputMetadata.Keys.ToList();
        _inputName  = inputs.First(n => n == "input"  || n.Contains("rgb")   || n == inputs[0]);
        _configName = inputs.First(n => n == "config" || n.Contains("axis")  || n == inputs[1]);
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Run inference on a single BGR Mat (OpenCV default). Returns a new
    /// BGR Mat at the same resolution. The frame's H and W must each be
    /// multiples of 32 — call PadTo32() first if needed.
    /// </summary>
    public Mat InferBgr(Mat bgrFrame, RestoraTask tasks)
    {
        if (bgrFrame.Width % 32 != 0 || bgrFrame.Height % 32 != 0)
            throw new ArgumentException(
                $"Input must be multiple of 32; got {bgrFrame.Width}x{bgrFrame.Height}. " +
                "Call PadTo32(frame) first.");

        // 1. BGR Mat -> RGB float32 CHW tensor in [0, 1]
        var inputTensor = BgrMatToChwTensor(bgrFrame);

        // 2. Build config tensor
        var configTensor = TaskToConfigTensor(tasks);

        // 3. Run
        var inputs = new List<NamedOnnxValue>(2)
        {
            NamedOnnxValue.CreateFromTensor(_inputName,  inputTensor),
            NamedOnnxValue.CreateFromTensor(_configName, configTensor),
        };

        using var results = _session.Run(inputs);
        var outputTensor = results.First(r => r.Name == _outputName).AsTensor<float>();

        // 4. Tensor (1, 3, H, W) RGB float [0,1] -> BGR Mat uint8
        return ChwTensorToBgrMat(outputTensor, bgrFrame.Height, bgrFrame.Width);
    }

    private static DenseTensor<float> BgrMatToChwTensor(Mat bgr)
    {
        // bgr is uint8 HxWx3 BGR. We want float32 1x3xHxW RGB normalized.
        int h = bgr.Height, w = bgr.Width;
        var data = new float[3 * h * w];

        // Mat.GetGenericIndexer<Vec3b>() gives BGR pixels.
        var indexer = bgr.GetGenericIndexer<Vec3b>();
        int planeSize = h * w;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                var px = indexer[y, x];
                // Swap BGR -> RGB while packing into CHW planes.
                data[0 * planeSize + y * w + x] = px.Item2 / 255f; // R
                data[1 * planeSize + y * w + x] = px.Item1 / 255f; // G
                data[2 * planeSize + y * w + x] = px.Item0 / 255f; // B
            }
        }

        return new DenseTensor<float>(data, new[] { 1, 3, h, w });
    }

    private static Mat ChwTensorToBgrMat(Tensor<float> chw, int h, int w)
    {
        var bgr = new Mat(h, w, MatType.CV_8UC3);
        var indexer = bgr.GetGenericIndexer<Vec3b>();
        int planeSize = h * w;

        // chw is (1, 3, H, W) RGB float in [0, 1]; clamp + pack into BGR uint8.
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float r = chw[0, 0, y, x];
                float g = chw[0, 1, y, x];
                float b = chw[0, 2, y, x];
                indexer[y, x] = new Vec3b(
                    Clamp01ToByte(b),
                    Clamp01ToByte(g),
                    Clamp01ToByte(r));
            }
        }
        return bgr;
    }

    private static byte Clamp01ToByte(float v) =>
        (byte)Math.Clamp((int)Math.Round(v * 255f), 0, 255);

    private static DenseTensor<float> TaskToConfigTensor(RestoraTask tasks)
    {
        var vec = new float[5]
        {
            tasks.HasFlag(RestoraTask.Colorize) ? 1f : 0f,
            tasks.HasFlag(RestoraTask.Denoise)  ? 1f : 0f,
            tasks.HasFlag(RestoraTask.Sharpen)  ? 1f : 0f,
            tasks.HasFlag(RestoraTask.DeJpeg)   ? 1f : 0f,
            tasks.HasFlag(RestoraTask.Deblur)   ? 1f : 0f,
        };
        return new DenseTensor<float>(vec, new[] { 1, 5 });
    }

    /// <summary>
    /// Pad a frame so H and W are multiples of 32 using reflect-border.
    /// Crop the output back to the original size with CropTo().
    /// </summary>
    public static Mat PadTo32(Mat src, out int padBottom, out int padRight)
    {
        int targetH = ((src.Height + 31) / 32) * 32;
        int targetW = ((src.Width  + 31) / 32) * 32;
        padBottom = targetH - src.Height;
        padRight  = targetW - src.Width;
        if (padBottom == 0 && padRight == 0) return src.Clone();

        var padded = new Mat();
        Cv2.CopyMakeBorder(src, padded, 0, padBottom, 0, padRight, BorderTypes.Reflect101);
        return padded;
    }

    public static Mat CropTo(Mat src, int targetH, int targetW) =>
        new Mat(src, new Rect(0, 0, targetW, targetH));

    public void Dispose() => _session.Dispose();
}
```

### Usage from your UI

```csharp
using var session = new RestoraSession(
    onnxPath: @"C:\models\restora_generic_fp16.onnx",
    backend: ExecutionBackend.DirectML);

// User toggles in UI:
var enabledTasks = RestoraTask.Colorize | RestoraTask.Denoise | RestoraTask.Sharpen;

using var bgr = Cv2.ImRead(@"C:\input\frame.png");
using var padded = RestoraSession.PadTo32(bgr, out int pb, out int pr);
using var restoredPadded = session.InferBgr(padded, enabledTasks);
using var restored = RestoraSession.CropTo(restoredPadded, bgr.Height, bgr.Width);
restored.SaveImage(@"C:\output\frame.png");
```

---

## 6. Video pipeline

OpenCvSharp gives you `VideoCapture` for reading and `VideoWriter` for
writing. Process frames one at a time, route each through `InferBgr`:

```csharp
public static void ProcessVideo(
    string inputPath,
    string outputPath,
    RestoraSession session,
    RestoraTask tasks,
    IProgress<double>? progress = null)
{
    using var reader = new VideoCapture(inputPath);
    if (!reader.IsOpened()) throw new IOException($"cannot open {inputPath}");

    int w = reader.FrameWidth;
    int h = reader.FrameHeight;
    double fps = reader.Fps;
    int total = (int)reader.Get(VideoCaptureProperties.FrameCount);

    // Always re-encode at the same resolution as the source.
    var fourcc = FourCC.FromString("mp4v");
    using var writer = new VideoWriter(outputPath, fourcc, fps, new Size(w, h));
    if (!writer.IsOpened()) throw new IOException($"cannot open {outputPath} for writing");

    using var frame = new Mat();
    int frameIdx = 0;
    while (reader.Read(frame))
    {
        if (frame.Empty()) break;

        // Pad to multiple of 32 (NAFNet downsample requirement)
        using var padded = RestoraSession.PadTo32(frame, out int pb, out int pr);
        using var processedPadded = session.InferBgr(padded, tasks);
        using var processed = RestoraSession.CropTo(processedPadded, h, w);

        writer.Write(processed);

        frameIdx++;
        progress?.Report((double)frameIdx / total);
    }
}
```

### Notes on video I/O

- **Codec choice:** `mp4v` is the lowest-friction. For H.264 (smaller
  files), pass `FourCC.FromString("H264")` — requires OpenCvSharp built
  with ffmpeg support, which the standard NuGet package provides on
  Windows but is platform-dependent on Linux/macOS.
- **Audio is dropped.** OpenCV's VideoWriter is video-only. If you need
  to preserve audio, post-process with FFmpeg:
  ```
  ffmpeg -i video_out.mp4 -i video_in.mp4 -c:v copy -map 0:v -map 1:a -shortest final.mp4
  ```
- **Color space:** OpenCV's VideoCapture decodes to BGR by default,
  matching `InferBgr`'s expected channel order. Don't convert.

---

## 7. Resolution / tiling strategy

NAFNet was trained at **256×256 patches**. It works at any resolution
that's a multiple of 32, but quality is best near the training scale.
Three options, increasing in quality + cost:

### A. Full-frame inference (simple, works for ≤ ~1024 px frames)

What the code above does. For HD (1920×1080) you pad to 1920×1088 and
run a single forward. Fast, simple, generally good quality.

Memory cost on GPU: roughly **per-megapixel × 1 GB at fp16, × 2 GB at fp32**.
1080p ≈ 2.1 MP → ~2 GB at fp16. 4K ≈ 8.3 MP → ~9 GB at fp16. Fits comfortably
on 12 GB+ GPUs.

### B. Tiled inference (better quality on very large frames, ≥ 2K)

Split the frame into 256×256 (or 512×512) tiles with overlap, process
each, blend at the seams with a feathered mask. More code, ~1.2× slowdown,
no resolution limit.

A reference tiler:

```csharp
public static Mat InferTiled(this RestoraSession session, Mat bgr, RestoraTask tasks,
    int tileSize = 512, int overlap = 32)
{
    int H = bgr.Height, W = bgr.Width;
    var output = new Mat(H, W, MatType.CV_32FC3, Scalar.All(0));
    var weight = new Mat(H, W, MatType.CV_32FC1, Scalar.All(0));

    int stride = tileSize - overlap;
    for (int y = 0; y < H; y += stride)
    {
        for (int x = 0; x < W; x += stride)
        {
            int x0 = Math.Min(x, W - tileSize);
            int y0 = Math.Min(y, H - tileSize);
            x0 = Math.Max(0, x0);
            y0 = Math.Max(0, y0);

            using var tile = new Mat(bgr, new Rect(x0, y0, tileSize, tileSize));
            using var padded = RestoraSession.PadTo32(tile, out _, out _);
            using var restoredP = session.InferBgr(padded, tasks);
            using var restored = RestoraSession.CropTo(restoredP, tileSize, tileSize);

            BlendInto(output, weight, restored, x0, y0, tileSize, overlap);

            if (x0 + tileSize >= W) break;
        }
        if (y + tileSize >= H) break;
    }

    // Normalize accumulated values by accumulated weights
    var result = new Mat(H, W, MatType.CV_8UC3);
    Mat[] outChannels = output.Split();
    for (int c = 0; c < 3; c++) outChannels[c] = outChannels[c] / weight;
    using var merged = new Mat();
    Cv2.Merge(outChannels, merged);
    merged.ConvertTo(result, MatType.CV_8UC3, 1.0);
    return result;
}
// BlendInto is left as an exercise — a Gaussian-weighted feather mask works well.
```

### C. Downscale → infer → upscale (lowest quality, fastest)

For low-end hardware. Downscale frame to ≤ 512 px on the long side,
infer, bicubic upscale back. Cheap but loses detail.

**Recommendation:** start with **A** (full-frame). Switch to **B** if
you're seeing artifacts at very high resolutions.

---

## 8. Performance tips

### Batch multiple frames at once

The first dimension of `input` is batch. Processing 4 frames in one
`Run()` call is ~3× faster than 4 sequential calls (kernel-launch
amortization). Modify `BgrMatToChwTensor` to take `Mat[]` and pack
into `(B, 3, H, W)`.

### Pin host memory for transfer speed (CUDA backend)

ONNX Runtime allocates pinned host memory automatically for tensors
created from arrays. No special API call needed; just don't copy
tensors around unnecessarily before `Run()`.

### Pre-warm the session

The first `Run()` call triggers kernel compilation / autotuning and is
3-10× slower than subsequent calls. Run a dummy frame at startup:

```csharp
using var dummy = new Mat(256, 256, MatType.CV_8UC3, Scalar.All(0));
session.InferBgr(dummy, RestoraTask.None);  // discard result
```

### fp16 vs fp32 trade-off

| Precision | Quality | Speed | Memory |
|---|---|---|---|
| fp16 | Visually identical | ~2× faster | ~half |
| fp32 | Reference | 1× | 1× |

Use fp16 unless you're targeting very old GPUs without fp16 acceleration.

### Reuse the session

`InferenceSession` construction is expensive (loads ONNX, optimizes
graph, allocates EP resources). Build once at app startup, hold the
reference for the app lifetime, dispose on exit.

---

## 9. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Output is grayscale even with `Colorize` enabled | Channel order swapped (BGR fed as RGB or vice versa) | Verify channel pack order in `BgrMatToChwTensor` — must be `R, G, B` planes |
| Output has weird borders | Frame size not multiple of 32 | Call `PadTo32` first; crop result back |
| "Invalid Argument: Got: ... Expected: ..." on config | Config tensor shape wrong | Must be exactly `(1, 5)` float32 for single-frame inference |
| First frame is 10× slower than subsequent | EP not pre-warmed | Run dummy frame at startup (§8) |
| GPU OOM on large frames | Frame > 1080p at fp32 | Use fp16 ONNX, or switch to tiled inference (§7B) |
| Output values >1 or <0 (washed-out or clipped) | Forgot to clamp before `* 255` | `Math.Clamp((int)Math.Round(v * 255f), 0, 255)` |
| Output flickers across frames | Independent per-frame inference | Train (or re-export) with temporal_pair loss — actually already in this model, but the per-frame inference is still per-frame |

---

## 10. Model selection UI

You'll typically have multiple ONNX files (different training runs,
different precisions). Treat each as a switchable backend:

```csharp
public sealed class RestoraModelRegistry
{
    private readonly Dictionary<string, RestoraSession> _sessions = new();
    public IReadOnlyList<string> Names => _sessions.Keys.ToList();

    public void Register(string name, string onnxPath, ExecutionBackend backend)
    {
        _sessions[name] = new RestoraSession(onnxPath, backend);
    }

    public RestoraSession Get(string name) => _sessions[name];

    public void Dispose()
    {
        foreach (var s in _sessions.Values) s.Dispose();
        _sessions.Clear();
    }
}
```

In the UI:
- Dropdown lists `registry.Names`
- Selection → `currentSession = registry.Get(name)`
- Checkboxes for the 5 tasks build a `RestoraTask` flags value
- "Process video" button: `ProcessVideo(input, output, currentSession, tasks)`

Don't construct sessions inside the hot loop. Keep them loaded; switching
between pre-loaded sessions is instant.

---

## Summary cheat sheet

```csharp
// 1. Load (once at startup)
using var session = new RestoraSession("model.onnx", ExecutionBackend.DirectML);

// 2. Build task flags from UI
var tasks = RestoraTask.Colorize | RestoraTask.Sharpen;

// 3. Per-frame
using var bgr      = Cv2.ImRead("frame.png");
using var padded   = RestoraSession.PadTo32(bgr, out _, out _);
using var restored = session.InferBgr(padded, tasks);
using var cropped  = RestoraSession.CropTo(restored, bgr.Height, bgr.Width);

// 4. Video: do the same in a VideoCapture loop, write with VideoWriter.
```

Tensor shapes you'll see in flight: `input (1, 3, H, W)`, `config (1, 5)`,
`output (1, 3, H, W)`. H and W must be multiples of 32. All channels are
RGB (swap from OpenCV's BGR before feeding in; swap back before
displaying).
