# Coliraz ONNX Inference Guide — LEGACY (colorize-only v1)

> ⚠️  **This document describes the legacy v1 model** (DDColor port,
> colorize-only, LAB ab output, ConvNeXt encoder). The current
> restora-models v2 model is a 5-task RGB-in / RGB-out architecture
> (NAFNet-large + dual-output head + adversarial refine head).
>
> **For v2 ONNX inference in C# / video applications, see
> [csharp-video-inference.md](csharp-video-inference.md) instead.**
>
> Below is preserved for historical reference; do NOT implement against it
> for new integrations.

> **Audience (historical):** an engineer (or LLM teammate) integrating a
> coliraz-trained `.onnx` model into another language/runtime (C#, C++,
> JS, …). This file documents *exactly* the pre-/post-processing the
> model expects, the architectural constraints baked into the graph, and
> the recipe the reference Python pipeline runs.
>
> If you only read one section, read **§3 (Inference Recipe)** and
> **§4 (Critical Gotchas)**.

---

## 1. What the model is

A modernized port of DDColor (ICCV 2023): given a grayscale photo, predict
the missing color. The exported network is a fully-convolutional encoder
(timm ConvNeXt) → UNet pixel decoder → DETR-style transformer color
decoder → 1×1 spectral-norm refinement.

The model **only predicts the two chroma channels (`a` and `b`)** of LAB
color space. The luminance channel (`L`) is reused verbatim from the input
photo. The C# side has to do the LAB↔RGB plumbing.

---

## 2. Input / Output Tensor Contract

### Input

| Property | Value |
|---|---|
| Name | `"input"` |
| Type | `float32` |
| Shape (fixed-shape ONNX) | `(B, 3, input_size, input_size)` — H/W baked at export time |
| Shape (dynamic-shape ONNX) | `(batch, 3, height, width)` — any size |
| Channel order | **RGB** (not BGR) |
| Range | `[0.0, 1.0]` — sRGB-gamma values divided by 255 |
| Content | The grayscale-as-RGB image (see §4.1 — **do not just send a plain grayscale**) |

### Output

| Property | Value |
|---|---|
| Name | `"output"` |
| Type | `float32` |
| Shape (fixed) | `(B, 2, input_size, input_size)` |
| Shape (dynamic) | `(B, 2, 32 * (H // 32), 32 * (W // 32))` |
| Range | LAB `a, b` in roughly `[-128, +128]` (cv2 convention), but predictions usually fall in `[-30, +30]` |
| Channel order | `[a, b]` in that order, channel-first |

### Two export variants exist

- **Fixed shape (`model.onnx`)**: simpler, faster on some ORT execution
  providers, picks one resolution at export time. Use this if you always
  process at the same size.
- **Dynamic shape (`model_dynamic.onnx`)**: same weights, height/width are
  dynamic axes. Accepts arbitrary `(B, 3, H, W)` but
  **rounds H/W down to multiples of 32** (because ConvNeXt downsamples 4×2×2×2 = 32×).

---

## 3. Inference Recipe (do exactly this)

This is the algorithm the reference Python pipeline implements. **Match it
byte-for-byte** to get matching output quality.

```
INPUT:  H×W color or grayscale image, any dtype/range
OUTPUT: H×W colorized RGB image, uint8

Step 1. Decode image, convert to RGB if not already, normalize to float32 [0, 1].
Step 2. Convert the full-resolution image to LAB. KEEP the L channel (call it L_full).
        This is what supplies the final image's luminance — do NOT throw it away.
Step 3. Resize the RGB to the model's working size:
          - fixed ONNX: resize to input_size × input_size
          - dynamic ONNX: pad-or-crop H/W so each is a multiple of 32, or
            resize to a chosen target multiple of 32
Step 4. Convert the resized RGB to LAB. Extract L_low (1-channel).
Step 5. Build the model input: concat(L_low, 0, 0) in LAB, then convert
        LAB → RGB. This produces a *3-channel grayscale-as-RGB* tensor.
        See §4.1 for why this is NOT the same as just stacking L_low 3 times.
Step 6. Permute to (1, 3, H_lr, W_lr), divide by 255, dtype=float32.
Step 7. Run ORT inference → output (1, 2, H_lr, W_lr) = predicted [a, b].
Step 8. Resize the predicted [a, b] back to H×W using bilinear interpolation.
        (You upsample chroma — keeps luminance perfectly sharp.)
Step 9. Reassemble LAB = concat(L_full, a_upsampled, b_upsampled).
Step 10. Convert LAB → RGB, clip to [0, 1], multiply by 255, cast to uint8.
```

### Pseudocode (C#-flavored)

```csharp
// Step 1
using var bmp = Image.Load(path).CloneAs<Rgb24>();       // RGB uint8
int H = bmp.Height, W = bmp.Width;
float[,,] rgbFull01 = ToFloatHwc(bmp);                   // (H, W, 3) in [0,1]

// Step 2 — full-res LAB, keep L
float[,] L_full = RgbToLab(rgbFull01).GetChannel(0);      // (H, W) — perceptual L, range [0, 100]

// Step 3 — resize RGB to model size (here: model exports at 256x256)
const int LR = 256;                                       // or a multiple of 32 for dynamic
float[,,] rgb_lr = ResizeBilinear(rgbFull01, LR, LR);     // (LR, LR, 3)

// Step 4 — LAB of resized
float[,,] lab_lr = RgbToLab(rgb_lr);                      // (LR, LR, 3)
float[,] L_low = GetChannel(lab_lr, 0);                   // (LR, LR)

// Step 5 — build gray-as-RGB by routing through LAB with a=b=0
float[,,] grayLab = StackChannels(L_low, Zeros(LR,LR), Zeros(LR,LR));
float[,,] grayRgb = LabToRgb(grayLab);                    // (LR, LR, 3) in [0,1]

// Step 6 — to tensor (1, 3, LR, LR)
float[] inputTensor = HwcToChw(grayRgb, batch:1);

// Step 7 — ORT
var input = new DenseTensor<float>(inputTensor, new[] { 1, 3, LR, LR });
using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", input) });
var ab_lr = results.First().AsTensor<float>();             // shape (1, 2, LR, LR)

// Step 8 — upsample a,b to full res
float[,] a_lr = TakeChannel(ab_lr, 0);                     // (LR, LR)
float[,] b_lr = TakeChannel(ab_lr, 1);
float[,] a_full = ResizeBilinear(a_lr, H, W);
float[,] b_full = ResizeBilinear(b_lr, H, W);

// Step 9 — reassemble LAB with full-res L
float[,,] labOut = StackChannels(L_full, a_full, b_full);

// Step 10 — LAB → RGB → uint8
float[,,] rgbOut = LabToRgb(labOut);
SaveAsPng(ClipToUint8(rgbOut, scale: 255f));
```

---

## 4. Critical Gotchas

### 4.1 Grayscale must be derived via LAB-L, not via Y-luma

The most common bug. The model was **trained** on inputs derived this way:

```
RGB → LAB → take L → set a=b=0 → LAB → RGB
```

This is **not** the same as `0.299·R + 0.587·G + 0.114·B` (Rec.601 luma).
LAB-L is a perceptually-weighted lightness computed through the CIE pipeline
(cube root of relative luminance, with the D65 white point). The two differ
by 5–15 % per pixel on saturated colors — enough to noticeably degrade
output quality if you skip the LAB roundtrip and just feed `Y` repeated three
times.

If your input is already a grayscale photo (single channel), build the LAB
input as `L=that_channel_scaled_to_[0,100], a=0, b=0` and convert that to
RGB to feed the model.

### 4.2 LAB convention matches OpenCV float32

The Python side uses `cv2.cvtColor(img_f32, cv2.COLOR_RGB2LAB)`. Ranges:

- `L ∈ [0, 100]`
- `a, b ∈ ~[-128, +128]` (centered on 0)
- White point: D65
- The L → linear-luminance step uses sRGB gamma (the standard 2.4 power
  curve with the 0.04045 linear segment, **not** a simple gamma 2.2).

If your C# LAB conversion uses a different range (e.g. `L ∈ [0, 255]` or
non-sRGB gamma), you'll get a hue shift and possibly clamping issues.

### 4.3 Dynamic-shape ONNX rounds H/W to a multiple of 32

`ConvNeXt` has four downsampling stages: stride 4 + stride 2 + stride 2 +
stride 2 = 32×. The decoder upsamples symmetrically. So a 250×250 input
produces a 224×224 output (`32 × (250 // 32) = 224`), and you'll get a
shape mismatch when you try to stack it with full-res L.

Two safe approaches:

- **Resize to a target multiple of 32** before inference (e.g. fixed 512 or
  768). This matches the Python pipeline and is what the reference uses.
- **Pad H/W up to the next multiple of 32**, run inference, crop the
  predicted `[a, b]` back to the unpadded shape, then upsample to full res.

### 4.4 Output AB is **unbounded** — clip after LAB→RGB, not the AB

Do not clip the predicted `a` and `b` to `[-128, +128]` before the LAB→RGB
step. The model may legitimately output values slightly outside that range
on saturated regions; the subsequent LAB→RGB conversion will produce RGB
values which you then clip to `[0, 1]` before quantizing to uint8. Clipping
AB directly desaturates the output.

### 4.5 Don't normalize the input with ImageNet mean/std externally

The model's first layer applies the `0.485/0.456/0.406` mean and
`0.229/0.224/0.225` std internally (it's baked into the graph as a buffer
subtraction/division). **Send raw `[0, 1]` RGB**. Don't pre-normalize.

### 4.6 BGR vs RGB

The model expects **RGB** at the tensor level. If your image library returns
BGR (most C# image libs hand back RGBA or RGB, but if you're using OpenCV
wrappers it'll be BGR), swap channels before tensorization.

---

## 5. LAB ↔ RGB Math (in case you need to implement it)

The Python implementation is in `src/coliraz/utils/color.py` and matches
`cv2.cvtColor` to within `atol=1.0` for LAB (and `atol=0.02` for the
round-trip). The math:

### RGB → LAB

```
1. Apply sRGB → linear:
   c_lin = c <= 0.04045 ? c/12.92 : ((c + 0.055)/1.055)^2.4    (per channel)

2. Linear RGB → XYZ:                      [0.4124564 0.3575761 0.1804375]
   xyz = M · rgb_lin       where M =      [0.2126729 0.7151522 0.0721750]
                                          [0.0193339 0.1191920 0.9503041]

3. Normalize by D65 white: xyz /= [0.95047, 1.0, 1.08883]

4. f(t) = t > (6/29)^3 ? t^(1/3) : t/(3·(6/29)^2) + 4/29     (per channel)

5. L = 116·f(Y) - 16
   a = 500·(f(X) - f(Y))
   b = 200·(f(Y) - f(Z))
```

### LAB → RGB

Run those four steps in reverse. The inverse matrix is

```
[ 3.2404542 -1.5371385 -0.4985314]
[-0.9692660  1.8760108  0.0415560]
[ 0.0556434 -0.2040259  1.0572252]
```

And the inverse of `f`:

```
f_inv(t) = t > 6/29 ? t^3 : 3·(6/29)^2 · (t - 4/29)
```

Then linear → sRGB:

```
c_srgb = c_lin <= 0.0031308 ? c_lin · 12.92 : 1.055 · c_lin^(1/2.4) - 0.055
```

Clip the result to `[0, 1]` *only at the very end*, just before quantizing
to uint8.

---

## 6. Memory and performance characteristics

- **Encoder + decoder cost is roughly linear in H·W.**
- **Color decoder transformer self-attention is O((H/32 · W/32)²).** So at
  256² that's 8×8 = 64 attention tokens (cheap); at 1024² it's 32×32 =
  1024 tokens — 256× more compute and quadratic memory. Practical ceiling
  on a 24 GB consumer GPU is ~1024×1024; on CPU it gets slow fast above 512.
- **bf16/fp16** at runtime: the exported graph is fp32. If ORT supports it,
  you can enable fp16 conversion at session-build time; quality difference
  vs. fp32 is below the perceptual threshold but throughput roughly doubles
  on GPU.
- **Batch size > 1** is supported (batch is a dynamic axis in every
  export). Useful for processing many small images.

---

## 7. Verifying your C# port against the reference

To prove your implementation matches the Python pipeline:

1. Take any RGB image, save as PNG.
2. Run the reference: `coliraz infer --model <ckpt> --input photo.png --output ref.png --input-size 256`.
3. Run your C# implementation against `model.onnx` (or `model_dynamic.onnx`).
4. Compute `max(abs(ref_pixel - your_pixel))` over all pixels in 8-bit
   space. **A correct implementation should be within ±3** on every channel.
   Differences within ±3 are bilinear-resize / float precision; differences
   above suggest one of these mismatches:
   - You forgot the LAB-L grayscale derivation (§4.1)
   - Your LAB/RGB conversion uses a different gamma or white point (§4.2)
   - You pre-normalized with ImageNet mean/std (§4.5)
   - You're feeding BGR (§4.6)
   - You clipped AB before LAB→RGB (§4.4)

If your output is everywhere darker or lighter than the reference, suspect
the gamma (§4.2). If colors are shifted (e.g. greens look yellowish), the
LAB white point or matrix is wrong. If the output is mostly grayscale,
you're probably not running the model at all or hitting §4.5.

---

## 8. Tiling for high-resolution inputs (4K and beyond)

### 8.0 Read this first

**You probably don't need to tile.** The default `coliraz infer` pipeline
already handles arbitrary input sizes by:

1. running the model on a low-res (e.g. 256² or 512²) view of the image, and
2. upsampling the predicted AB chroma to full resolution, then
3. merging that AB with the **untouched full-resolution L** from the original.

This is the same principle as chroma subsampling in JPEG/H.264 — the human
visual system is far more sensitive to luminance detail than to chroma
detail, so bilinear upsampling of AB is essentially invisible. The pipeline
gives pixel-perfect L for free and globally-coherent color decisions because
the model sees the whole image at once at low res.

**Implement the simple recipe in §3 first.** Only add tiling if you have a
concrete use case the simple recipe fails on (see §8.1).

### 8.1 When tiling is the right answer

Three scenarios:

1. **The host hardware truly cannot run the model at a moderate fixed
   resolution.** E.g. you're on a small embedded GPU with <2 GB VRAM and
   even 512² OOMs. *Tiling can solve this.*
2. **You need locally-varying chroma at greater than ~512² resolution.**
   Examples: brand logos, small flags, fine textile patterns that span a
   few pixels at 4K and get smeared by AB upsampling. *Tiling can solve this.*
3. **The image contains spatially distinct regions where the global low-res
   pass picks a compromise color** (e.g. half the image is a beach scene,
   half is a forest, and the low-res pass picks a desaturated middle ground
   for both). *Tiling can sometimes help — see §8.2 for the caveat.*

For "I want to colorize a normal 4K photo," scenarios 1–3 rarely apply.
Use the simple pipeline.

### 8.2 The fundamental caveat: this model is a *global* operator

The transformer color decoder uses **learnable color queries** that
cross-attend to the entire feature map. These queries represent
image-level color decisions ("what's the dominant skin tone in this
image", "is the sky cloudy or sunny"). When you tile and process each tile
independently:

- Each tile's queries see only that tile's content.
- A tile of pure sky predicts everything as blue, including an unfortunate
  object in that tile.
- A tile of pure face has no idea there's a red car nearby that should
  inform shadow color.

You will see:

- **Color drift at tile boundaries** (one tile picks slightly warmer sky than
  the adjacent tile).
- **Contextually wrong colors** in tiles that lack distinguishing content
  (a tile that's all texture but no recognizable object).

Overlap + feather (§8.4) hides the first symptom but **does not fix the
second**. The only real fix for the second is a two-pass approach
(§8.5) or a different architecture entirely.

### 8.3 Tile chroma on top of the pipeline (recommended hybrid)

The cleanest practical recipe — combines global coherence from the
pipeline with locally-detailed chroma where it matters:

```
1. Run the simple pipeline (§3) to get a globally-coherent
   AB_global at full resolution.
2. Identify regions that need detailed chroma. Heuristics:
   - high-frequency edges in L_full (Sobel/Canny → dilate)
   - manual user-drawn masks
   - all pixels (if you want refinement everywhere)
3. For each region, extract overlapping tiles at full resolution
   from the gray-RGB input (built per §4.1).
4. Run the model on each tile → AB_tile.
5. Blend AB_tile with AB_global using a soft window
   (raised-cosine; see §8.4 for the kernel) and a mask that
   restricts the blend to the chosen region.
6. Final AB = mask·AB_blended + (1 - mask)·AB_global.
7. Continue at step 9 of §3 (LAB merge → RGB).
```

This is more work than §8.4 but produces the best results: global color
coherence is preserved everywhere, local refinement kicks in where you
choose. The C# implementer can ship §8.4 first and add §8.3 as a follow-up.

### 8.4 Standalone tiling with overlap + feather

The minimum viable tiling approach. Each tile is processed independently
and blended with a raised-cosine window.

#### Parameters

| Parameter | Recommendation | Notes |
|---|---|---|
| Tile size `T` | 512 (or 256 if memory-bound) | Must be a multiple of 32 |
| Overlap `O` | `T / 4` (= 128 for T=512) | Must be a multiple of 32; smaller = more seams, larger = more compute |
| Stride `S` | `T - O` (= 384) | |
| Padding mode | reflect | Edge tiles use mirrored padding |

#### Algorithm

```
INPUT:  H×W full-res RGB image
OUTPUT: H×W full-res AB chroma map (then continue per §3 step 9)

# Pad H/W so tile grid covers the image
pad_H = align_up(H, S) + O    # last tile reaches H+O if needed
pad_W = align_up(W, S) + O
rgb_padded = reflect_pad(rgb_full, top=O/2, bottom=pad_H-H-O/2, ...)

# Build the gray-as-RGB input over the padded canvas (per §4.1)
gray_rgb_padded = derive_gray_rgb(rgb_padded)

# Pre-compute the 2D blend weight: raised cosine in both dims
# w(i) = 0.5 * (1 - cos(2π · i / (T - 1)))   for i in [0, T-1]
# 2D weight W2D[i,j] = w(i) * w(j)
weight2d = raised_cosine_2d(T, T)

# Accumulator buffers (float32)
ab_sum     = zeros(2, pad_H, pad_W)
weight_sum = zeros(1, pad_H, pad_W)   # 1 channel, broadcast to AB

# Iterate over tile origins
for ty in range(0, pad_H - T + 1, S):
    for tx in range(0, pad_W - T + 1, S):
        tile_in = gray_rgb_padded[ty:ty+T, tx:tx+T, :]          # (T, T, 3)
        tile_in = to_chw_batch1_float32(tile_in)                 # (1, 3, T, T)
        tile_ab = ort.run(tile_in)[0]                            # (1, 2, T, T)
        tile_ab = squeeze_batch(tile_ab)                         # (2, T, T)

        # accumulate with the 2D feather weights
        ab_sum[:, ty:ty+T, tx:tx+T]     += tile_ab * weight2d    # broadcast over channel
        weight_sum[:, ty:ty+T, tx:tx+T] += weight2d

# Normalize (avoid div-by-zero at the corners, which can happen if you
# stride doesn't perfectly cover the canvas; add a tiny epsilon)
ab_padded = ab_sum / clamp(weight_sum, min=1e-6)

# Unpad
ab_full = ab_padded[:, O/2:O/2+H, O/2:O/2+W]

# Continue at §3 step 9 (concat L_full + ab_full → LAB → RGB)
```

#### Notes on the weights

- **Raised-cosine** (Hann window): `w(i) = 0.5 * (1 - cos(2π·i/(T-1)))`.
  Peaks at the tile center, smoothly approaches 0 at the edges.
- The 2D weight is the **outer product** of two 1D windows. At a pixel
  covered by multiple overlapping tiles, the sum of all tile weights at
  that pixel will be approximately constant if `S = T/2` or `S = T/4`.
- **Why not just average?** Each tile's predictions are less reliable near
  its edges (the model has less context there). Center-weighted blending
  trusts the high-confidence interior of each tile.

#### Edge cases to handle

- **Image already smaller than `T`**: skip tiling entirely and just run
  one inference at the (32-multiple-padded) original size.
- **Boundary tiles**: reflect-padding (mirrored) gives much better results
  than zero-padding for natural images. Most image libraries have it.
- **Output rounding**: per §4.3 the model rounds H/W down to a multiple of
  32. With tiles of size 512 = 16×32, this is exact. With other tile
  sizes, ensure your tile size is a multiple of 32.

### 8.5 Two-pass with low-res guidance (best quality, most work)

If the host can afford it, this gives results indistinguishable from a
single global pass at moderate resolution while supporting locally-detailed
chroma:

```
Pass 1 (global, cheap):
   - Run the simple pipeline §3 at e.g. 512².
   - You now have a globally-consistent ab_global at full res
     (or at 512² ready to be upsampled).

Pass 2 (local, per tile):
   - For each overlapping tile at the desired refinement resolution,
     build the gray-RGB input.
   - Before running the model, *blend* the tile's gray RGB with the
     pass-1 colorized version of the same region, e.g.:
        cond_rgb = 0.5 * gray_rgb_tile + 0.5 * lab_to_rgb(L_tile, ab_global_tile)
   - Run model(cond_rgb) → ab_tile_refined.
   - Compose into ab_full with overlap+feather (§8.4).
```

The "blend the input with the pass-1 colorization" trick is what lets the
local pass see the global color decision and refine it instead of starting
from gray. Without it, you're back in §8.2's "each tile picks its own
context" problem.

This is more complex; ship §8.4 first.

### 8.6 Performance ballpark

Times below assume the **tiny** model variant on a Blackwell-class GPU
(RTX 6000 Blackwell, 96 GB) running fp16:

| Resolution | Approach | Inferences | Wall time (estimate) | Notes |
|---|---|---|---|---|
| 4K (3840×2160) | Simple pipeline (256²) | 1 | <100 ms | Recommended default |
| 4K | Simple pipeline (512²) | 1 | ~150 ms | Higher chroma detail |
| 4K | Direct dynamic-shape ONNX | 1 | ~1.5–3 s | If memory allows, often beats tiling |
| 4K | Tiling 512² with O=128 (§8.4) | 35 | ~3–4 s | Use when memory-constrained |
| 4K | Two-pass (§8.5) | 1 + 35 | ~3.5–4 s | Use when colors look wrong at 4K direct |

For the *large* model variant, multiply by ~3–4.

CPU times will be 10–30× slower across the board.

### 8.7 Verifying your tiling implementation

Match these properties:

1. **Tile-grid coverage**: every output pixel must be touched by at least
   one tile. With `S = T - O` and reflect-padding on the canvas, this is
   guaranteed.
2. **Weight normalization**: `weight_sum` must be strictly positive
   everywhere before division. If you hit a division by zero, your tile
   stride is too large relative to tile size.
3. **No seams visible**: rendering the result and comparing against the
   simple pipeline's output should produce subtly different colors but
   **no visible grid lines or sudden hue jumps**. If you see seams, increase
   overlap or check the window function.
4. **Pixel-budget check**: `pad_H × pad_W * num_tile_origins` should match
   `T² × num_tile_origins`. Off-by-one errors in the stride loop are the
   most common bug.
5. **Spot-check a uniform image**: feed a constant-gray image; every tile
   should produce the same AB, and after blending the output should be
   uniform with no banding. If you see banding, the window function is
   wrong (probably not summing to ~constant over overlapping regions).

---

## 9. File reference (source of truth)

In the coliraz repo:

- `src/coliraz/utils/color.py` — tensor `rgb_to_lab` / `lab_to_rgb`
  implementations. Pure PyTorch but the math is identical to the formulas
  in §5.
- `src/coliraz/data/grayscale.py` — `derive_pair()` produces the
  training input. The same routing is used at inference.
- `src/coliraz/infer/pipeline.py` — `ColorizationPipeline.process()` is
  the canonical Python implementation of §3 above. Read this as the
  reference algorithm.
- `src/coliraz/export/onnx.py` — `export_onnx_from_model()` shows
  exactly what's in the graph (no hidden pre/post-processing). Includes
  the dynamic-hw path.

Note: tiling is not currently implemented in the Python reference — §8
documents the algorithm so the C# port can implement it cleanly without
needing a Python reference to chase. If you need a Python reference for
parity, ping the upstream maintainer.
