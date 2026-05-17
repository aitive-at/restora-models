# C# Integration Recipe

The new temporal model uses ONNX exclusively (PNNX/ncnn support was
removed in the 2026-05 redesign). All consumer-side integration is via
ONNX Runtime; the C# integration is the same as for any ONNX consumer.

## Setup

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.19.0" />
```

## Single image inference

```csharp
using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CUDA();
using var session = new InferenceSession("restora.onnx", sessionOptions);

// Load image to (H, W, 3) byte array via your preferred image library.
// Convert to (1, 7, 3, H, W) float32 in [0, 1].
// Pad to multiple of 16 along H, W.
// Replicate the single image 7 times along the temporal axis.

var framesTensor = new DenseTensor<float>(framesData, new[] { 1, 7, 3, H, W });
var configTensor = new DenseTensor<float>(new[] { 1f, 0, 0, 0, 0 }, new[] { 1, 5 });

var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("frames", framesTensor),
    NamedOnnxValue.CreateFromTensor("config", configTensor),
};

using var results = session.Run(inputs);
var output = results.First().AsTensor<float>();
```

## Video inference

For a frame sequence, the C# consumer:
1. Buffers 7 frames at a time
2. Builds the sliding window with edge-replicate at clip boundaries
3. Calls `session.Run(...)` per output frame
4. Writes the result

See `docs/integration/onnx-inference-guide.md` for the model contract and
TensorRT engine build.
