# FP16 Re-Export — Follow-up #1

**Status update on the May 19 re-export.**

Read the original brief first: `docs/training-side-fp16-export.md`. This is the
diff-against-current — what got fixed, what's still broken, and the new (much
narrower) ask.

---

## TL;DR

* **GridSample import error: FIXED.** ✓ TRT compiles a real engine for the
  new fp16 ONNX. No `ITensor::getDimensions: Error Code 4` lines anywhere.
* **fp32 export: unchanged, still healthy.** ✓ All three EPs (CPU / plain CUDA /
  TRT) produce identical output to the previous revision.
* **fp16 export: still broken at runtime on every GPU path.** ✗ CPU EP gives
  correct output, but plain CUDA EP returns **NaN** and TRT EP returns **zero**.

The model is now structurally importable — your earlier fix landed correctly —
but the network as exported isn't numerically stable inside the fp16 range
when it runs on real GPU hardware. Almost certainly a softmax / attention
overflow.

---

## What I ran and what I measured

Bisection via our `RawModelCanaryTests` (identity-gate canary: random input,
`config = [0, 0, 0, 0, 0]`, expect output == centre frame). For each combo of
{model variant} × {execution provider}:

| Variant | EP                              | Identity PSNR | Output                |
|---------|---------------------------------|---------------|------------------------|
| fp32    | CPU                              | **131.94 dB** ✓ | input ≈ output         |
| fp32    | plain CUDA (no graph, no TRT)    | **131.91 dB** ✓ | input ≈ output         |
| fp32    | TRT (via session cache)          | **130.78 dB** ✓ | input ≈ output         |
| fp16    | **CPU**                          | **67.82 dB** ✓ | input ≈ output (fp16 round-trip noise floor — exactly what the integration guide §8 promises for fp16) |
| fp16    | **plain CUDA**                   | **NaN dB**    ✗ | `min=∞ max=-∞ mean=NaN` |
| fp16    | **TRT**                          | **4.78 dB**   ✗ | uniform 0.0            |

The 67.82 dB CPU-EP result on fp16 is the headline. It means:

* The graph topology is sound.
* The Cast nodes inserted between fp16 and fp32 sections are consistent.
* The identity shortcut (`output ≈ frames[:, 3]` when config is zero) survives
  the fp16 round-trip.
* If we could just run this CPU graph on GPU bit-equivalently, we'd be done.

But we can't, because the GPU EPs don't fake-promote intermediates to fp32 the
way CPU does. They run the network arithmetic in actual native fp16, and
**something inside the network produces values outside the fp16 representable
range (±65504) or denormals that round to zero**. CPU's slower-but-emulated
fp16 path tolerates this; GPU silicon doesn't.

---

## Hypothesis: softmax / scale overflow in attention

The two GPU failure modes line up with classic fp16 arithmetic-stability bugs:

* **NaN on CUDA EP** = `exp(x)` of a too-large `x` overflows to +∞, then `+∞ −
  +∞` in the softmax normaliser produces NaN. Subsequent ops propagate NaN.
* **All zeros on TRT** = TRT's fp16 fusion pass replaces the unstable softmax
  with a tighter kernel that saturates the offending values to a constant
  (often 0). The output looks "valid" (no NaN) but is entirely wrong.

Both symptoms appear when an intermediate exceeds fp16 range. Things to audit
in `restora_models`, ranked by likelihood:

### 1. Attention softmax (almost certainly the culprit)

The temporal-attention module computes something like:

```python
# inside TemporalAttention.forward (suspected shape):
attn_logits = (q @ k.transpose(-2, -1)) * self.scale          # shape (B, H, T, T)
attn_weights = F.softmax(attn_logits, dim=-1)
out = attn_weights @ v
```

In fp16, `q @ k.transpose(...)` can produce values in the thousands; multiplied
by a `self.scale` of 1.0 (i.e. nothing) the logits stay there. Then `exp(2000)`
is +∞ in fp16. The numerically-stable softmax form solves this:

```python
# Numerically-stable softmax — required for fp16:
attn_logits = (q @ k.transpose(-2, -1)) * self.scale
attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values   # ← add this
attn_weights = F.softmax(attn_logits, dim=-1)
```

If the module uses `nn.functional.softmax` or `F.scaled_dot_product_attention`,
PyTorch does this subtraction internally **only on the CPU path** — on CUDA
it relies on libcudnn's softmax which already does it natively. But once the
graph is exported to ONNX and run on TRT, the explicit subtraction isn't
preserved unless the wrapper code does it.

If you're using `torch.nn.functional.scaled_dot_product_attention`, switch
to the explicit form (q @ k → scale → subtract max → softmax → @ v) so the
max-subtraction lands in the exported graph as ONNX `Sub` + `ReduceMax` nodes.

### 2. Scale factors > 65504 in constant initialisers

Search the wrapper / module sources for any literal constant > 65504. Common
offenders:

* Position embeddings with raw frequency multipliers (`10000.0` itself is
  fine; `10000.0 ** (2 * i / d)` can blow up for small `i`/`d`).
* Numerical-stability "epsilon-inverses" (`1.0 / eps`-style values).
* Mask values for masked attention (people often use `-1e9` or `-65504` to
  zero out masked positions after softmax — fp16 can't represent `-1e9`).

Replace `-1e9`/`-1e10` with `-65504.0` (the fp16 -∞) or `float("-inf")` which
PyTorch maps to `-65504` automatically when exporting fp16.

### 3. LayerNorm / GroupNorm variance accumulator

`LayerNorm` computes `var = E[(x - mean)^2]` which can overflow in fp16 even
if the input is in range. PyTorch's nn.LayerNorm internally upcasts to fp32
for the variance computation **when running eagerly**, but the upcast doesn't
always survive ONNX export.

If you have `nn.LayerNorm` modules, either:
* Override their forward to `.to(torch.float32).layernorm(...).to(dtype)`, or
* Keep LayerNorm in fp32 by NOT calling `.half()` on those specific modules.

### 4. RMS / mean-pool reductions

Same story as LayerNorm — any reduction over a large axis in fp16 can
accumulate beyond range. Common in global-average pooling. Audit any
`reduce_mean` / `reduce_sum` over axes of size > ~256.

---

## What we'd like in the next iteration

The single most useful debugging step on your side is **running the network
forward pass with hooks that record per-layer min/max activations, in fp16,
on GPU**:

```python
import torch

ranges = {}
def hook(name):
    def _h(module, inputs, output):
        t = output if isinstance(output, torch.Tensor) else output[0]
        ranges[name] = (t.min().item(), t.max().item(), t.abs().max().item())
    return _h

model = model.half().cuda()
for name, m in model.named_modules():
    m.register_forward_hook(hook(name))

with torch.no_grad():
    _ = model(dummy_frames_fp16.cuda(), dummy_config_fp16.cuda())

# Print the worst offenders:
for name, (lo, hi, abs_max) in sorted(ranges.items(), key=lambda kv: -kv[1][2]):
    if abs_max > 1000:        # anything inside ±1000 is comfortably in fp16 range
        print(f"{name:60s}  min={lo:.2f}  max={hi:.2f}  |abs|_max={abs_max:.2f}")
```

Anything that prints with `|abs|_max > 50000` is the smoking gun. Either fix
that op's stability (cases 1-4 above) or upcast that specific module to fp32.

If you don't have GPU access to repro the failure: even running on CPU **with
the fp16 dtype explicitly** (`model.to(torch.float16)` on a CPU build) will
expose any value that overflows the fp16 representable range. The hooks above
work identically there.

---

## What we'd like to verify on our side

We'll run the same three GPU configs again — plain CUDA, TRT, and our
production session cache (TRT+CUDA fallback) — and look for:

1. **No NaN/Inf** in the output anywhere.
2. **Identity PSNR ≥ 60 dB** for `config = [0, 0, 0, 0, 0]` (fp16 noise floor;
   matches the integration guide §8 thresholds).
3. **Per-axis differential**: each non-zero axis produces ≥ 0.05 max diff
   from the centre frame.

Drop the new file at `models/iter_0030000_ema_generic_fp16.onnx` and we'll
re-run within minutes.

---

## What's NOT the problem (so you can skip these)

We've verified:

* TRT engine *compilation* works fine. The earlier GridSample dtype mismatch
  is gone — your previous fix landed correctly.
* The fp16 ONNX file *loads* correctly into ORT (input/output dtypes are
  `tensor(float16)` as expected).
* The identity shortcut at the wrapper level (the `frames[:, 3]` residual)
  is preserved — CPU EP gives 67.82 dB identity match, which is exactly the
  fp16 round-trip noise floor and matches the §8 spec.
* The `clamp(0, 1)` at graph output is preserved (no out-of-range values from
  CPU EP).
* Our consumer-side fp16 binding (the IoBinding + Float16 OrtValue allocator)
  works — fp32 paths through the same pipeline all produce correct output.

So this is **specifically a network-internal fp16 numerical-stability problem
that only manifests under native GPU fp16**. Should be a tractable hunt with
the activation-range hook script above.

---

## Files to deliver again

Just the one — `iter_0030000_ema_generic_fp16.onnx`, post-fix. If you'd like
to leave the fp32 file as-is, perfect; we don't need a new one.

Reach out if any of §1-§4 turns up something interesting — happy to look at
the activation-range dump if you produce one.
