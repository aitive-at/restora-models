# Brief for the ML-side: re-export the unified Restora model for ncnn

> Paste this verbatim to the Claude instance working on the training / export side
> (the one with access to the PyTorch source, the `.pt` checkpoint, and the pnnx
> CLI). It is self-contained — no other context from this repo is required.

---

You are working on the training / export pipeline of **Aitive Restora**, a
video-restoration project. The C# inference side just finished wiring up an
ncnn-backed runtime (Tencent's Vulkan-capable framework) so we can run the
unified restoration model on any GPU vendor — NVIDIA, AMD, Intel, integrated,
mobile. The C# code is complete and tested.

The blocker is the `.ncnn.param`/`.ncnn.bin` pair you produced earlier. Three
pnnx-extension layer types in the param are not in the stock ncnn layer
registry, so ncnn rejects the file the moment it tries to parse the .param
header.

---

## What I need from you

A re-exported pair of files where **every layer type referenced in the .param
is in ncnn's stock registry**, with the same model behaviour as today.

**Output filenames (the C# side already points at these exact names)**:

```
models/restora_generic_fp16_dynamic.ncnn.param
models/restora_generic_fp16_dynamic.ncnn.bin
```

---

## Concrete failure observed today

`ncnn.dll` version `1.0.20260113` (Jan 2026 build) bundled with the consuming
app rejects the param immediately:

```
layer torch.le not exists or registered
ncnn_net_load_param_memory(...) returned -1.
```

I enumerated the param's unique layer types and queried each against
`ncnn_layer_type_to_index` on the same dll. Result:

**Unknown to this ncnn build (need to disappear from the new .param)**:

| Layer type    | Where it shows up                               |
|---------------|-------------------------------------------------|
| `torch.le`    | element-wise `x ≤ const` — sRGB-segment compare |
| `torch.gt`    | element-wise `x > const` — same color math      |
| `torch.where` | element-wise ternary `cond ? a : b`             |

There is also a cosmetic mismatch — `nn.GroupNorm` in the .param vs `GroupNorm`
in ncnn's registry. The C# side rewrites that one in memory at load time, so
it's already addressed, but if you can emit `GroupNorm` directly the in-memory
patch becomes redundant.

**Known to this ncnn build (already loading fine — keep using them freely)**:

```
BinaryOp, Clip, Concat, Convolution, ConvolutionDepthWise, Crop, Einsum,
ExpandDims, GELU, Gemm, InnerProduct, Input, LayerNorm, MemoryData,
MultiHeadAttention, Permute, PixelShuffle, Pooling, Reshape, Slice, Split,
Swish, GroupNorm
```

---

## Where the three `torch.*` ops come from (probably)

Looking at the failing .param at lines ~7–14, the pattern is:

```
MemoryData               pnnx_fold_194       0 1 8  0=3 1=3
MemoryData               pnnx_fold_w.7       0 1 9  0=3
torch.le                 torch.le_52         1 1 1 10
BinaryOp                 div_0               1 1 2 11  0=3 1=1 2=12.92
BinaryOp                 add_1               1 1 3 12  0=0 1=1 2=0.055
BinaryOp                 div_2               1 1 12 13 0=3 1=1 2=1.055
BinaryOp                 pow_3               1 1 13 14 0=6 1=1 2=2.4
torch.where              torch.where_555     3 1 10 11 14 15
```

That's the **sRGB → linear-RGB** conversion baked inline into the graph:

```
linear = x ≤ 0.04045  ?  x / 12.92  :  ((x + 0.055) / 1.055)^2.4
```

(`torch.gt` appears in the symmetric linear→sRGB path further along the same
encoder.) The comparison thresholds (`0.04045`, `0.0031308`) are stored
somewhere in the .bin and loaded by `torch.le`/`torch.gt`'s `load_model`
callback — they're not in the .param text directly.

---

## Fix options, in priority order

### 1. Strip the sRGB↔linear conversion out of the exported graph (preferred)

The C# pipeline already feeds the model RGB float in `[0, 1]` and treats the
output as RGB float in `[0, 1]`. If the PyTorch source has a flag along the
lines of `if cfg.input_is_srgb: x = srgb_to_linear(x)` at the start of
`forward`, set it to **False** at export time. The training did its own
sRGB→linear; inference doesn't need it baked into the deployed graph.

Why this is best:

- The three offending ops vanish completely — they're an artifact of the
  in-graph color conversion, nothing else in the model uses them.
- Removes pnnx's dependence on the comparison constants being preserved
  through pnnx's IR (a known weak spot).
- The C# inference layer can apply the sRGB roundtrip outside the model in
  ~30 LOC of vectorized math if it turns out to be needed (it already has
  helpers — `VectorizedColorMath.RgbToLab` etc. — that include the gamma
  curve).

### 2. Newer pnnx with stricter "ncnn-stock-only" lowering

Recent pnnx versions (2024+) can lower `torch.where(cond, a, b)` algebraically:

```
out = a * cond + b * (1 - cond)
```

— i.e. two `BinaryOp` muls + a `BinaryOp` sub + a `BinaryOp` add. The
threshold-vs-tensor compare from `torch.le`/`torch.gt` can be replaced with
a `Threshold`-style fold using `MemoryData(const)` + `BinaryOp(SUB)` +
`Sign`/`Clamp`, all stock layers.

Run `pnnx --help` (or the appropriate Python entry point's `--help`) and look
for flags along the lines of `--ncnn-lower-comparison`,
`--customop=stock-only`, or `--target=ncnn-stock`. The exact name depends on
your pnnx version. Worst case `pip install -U pnnx` for the latest.

### 3. Replace the conversion in the PyTorch source with a sigmoid-blended approximation

```python
# Hand-rolled, only uses ops every ncnn build has.
def srgb_to_linear_approx(x, sharpness=1000.0):
    threshold = 0.04045
    soft_mask = torch.sigmoid(sharpness * (threshold - x))   # ~1 below, ~0 above
    linear_low  = x / 12.92
    linear_high = ((x + 0.055) / 1.055) ** 2.4
    return soft_mask * linear_low + (1 - soft_mask) * linear_high
```

The result is approximate near the threshold, but the sRGB transfer function
is itself a piecewise approximation of a true exponential, so a smooth
blend is visually indistinguishable. **Less preferred than option 1** — it
adds runtime work to handle something C# can do trivially outside the graph.

---

## Validation

After re-exporting, run this verifier in the model directory. It should print
nothing (no "MISSING" lines):

```bash
awk 'NR>2 {print $1}' restora_generic_fp16_dynamic.ncnn.param | sort -u | \
  awk '/^(torch\.|nn\.)/ { print "MISSING: " $0 }'
```

If you have a Python environment with `ncnn`'s Python bindings (or ctypes
against the same ncnn.dll), this catches any other registry gaps:

```python
import ctypes
ncnn = ctypes.CDLL("./ncnn.dll")
ncnn.ncnn_layer_type_to_index.restype = ctypes.c_int
ncnn.ncnn_layer_type_to_index.argtypes = [ctypes.c_char_p]

with open("restora_generic_fp16_dynamic.ncnn.param") as f:
    next(f); next(f)   # skip magic + header
    types = {line.split()[0] for line in f if line.strip()}

for t in sorted(types):
    if ncnn.ncnn_layer_type_to_index(t.encode()) < 0:
        print(f"MISSING: {t}")
```

`ncnn.dll` is the Windows native build the consuming app ships at
`runtimes/win-x64/native/ncnn.dll`. If you don't have that path, any
recent vanilla ncnn build (compiled without pnnx-only extensions) will
exhibit the same set of unknown layers, which is the property we're testing for.

---

## Model contract (unchanged — keep it identical to today)

- **Input `in0`**: `(B, 3, H, W)` float32 RGB in `[0, 1]`, channel-first
  (CHW per sample). H and W must be multiples of 32 (NAFNet's 5 downsample
  stages). The C# side reflect-pads any non-multiple to the nearest 32 and
  crops the output back.
- **Input `in1`**: `(B, 5)` float32 task config, positional order
  `[colorize, denoise, sharpen, dejpeg, deblur]`. Values 0 or 1. The model
  gates internal pathways on this; cost is the same regardless of which
  axes are on.
- **Output `out0`**: `(B, 3, H, W)` float32 RGB in `[0, 1]`. The graph ends
  with a `Clip [0, 1]` so values are pre-clamped.

Model architecture: NAFNet-large 5-task unified, fp16 weights, dynamic H/W.
**The model itself does not need any change** — only the export tool flags
and possibly the in-graph color-space conversion.

---

## Acceptance checklist

- [ ] `restora_generic_fp16_dynamic.ncnn.param` exists in `models/`.
- [ ] `restora_generic_fp16_dynamic.ncnn.bin` exists in `models/`.
- [ ] `awk` validator script above prints nothing (no `torch.*` / `nn.*` layers).
- [ ] The .param's layer count line (line 2) is unchanged or smaller — we should
      not be adding layers, only collapsing them.
- [ ] A round-trip inference sanity check on a single test image produces output
      visually equivalent to the existing ONNX exports of the same model
      (`models/restora_generic_fp16_dynamic.onnx`).

Once those four boxes are ticked, drop the two new files in `models/` and the
consuming C# project's E2E test will load them on first run. No code changes
on this side.
