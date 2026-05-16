# Follow-up to ML-side: one more layer needs fixing for ncnn

The previous export (with `torch.le` / `torch.gt` / `torch.where` lowered) loads
fine through `ncnn_net_load_param` now — that step is solved. But
`ncnn_net_load_model` (the .bin weight load) fails on the FIRST `nn.GroupNorm`
layer at index 1287:

```
layer load_model 1287 model.refine_head.blocks.0.adaln1.norm failed
ncnn_net_load_model(...) returned -1
```

There are **16** `nn.GroupNorm` layers in the param (one `adaln1.norm` and one
`adaln2.norm` per refine_head block × 8 blocks). All look like:

```
nn.GroupNorm  model.refine_head.blocks.N.adalnK.norm  1 1 <in> <out>
```

No `0=...` / `1=...` / `2=...` params on the line — meaning pnnx is encoding
`groups` / `channels` / `eps` somewhere inside the .bin in **its own
pnnx-extension layout** for the `nn.GroupNorm` extension layer. The
consuming ncnn.dll doesn't have that extension; it only has stock `GroupNorm`,
which expects those three values on the .param line and just two contiguous
weight arrays in the .bin.

I tried rewriting the layer-type token `nn.GroupNorm` → `GroupNorm` at parse
time on the consumer side. That makes `load_param` accept the file, but then
`load_model` fails because ncnn's stock `GroupNorm.load_model` doesn't know
to skip pnnx's extra header bytes for this layer.

## What I need

Export `nn.GroupNorm` as stock `GroupNorm` with the three required parameters
on the .param line, and standard gamma+beta weights in the .bin.

The expected param-line shape for stock `GroupNorm`:

```
GroupNorm  <instance_name>  1 1 <in> <out>  0=<groups> 1=<channels> 2=<eps>
```

(`affine=1` is implied if gamma+beta are present in the .bin, which they
should be since these layers come from PyTorch's `nn.GroupNorm(..., affine=True)`.)

## Fix options (priority order)

### 1. Newer / different pnnx flag

The same `--ncnn-stock-layers-only` (or whatever flag fixed the `torch.*`
issue) almost certainly has a counterpart for normalisation layers. Try:

- `pnnx --help` and look for `--ncnn-norm-as-stock`, `--lower-groupnorm`,
  `--target=ncnn-stock` (likely affects both comparison ops and norms).
- If the flag that fixed `torch.where` is `--customop=lower`, try
  `--customop=lower-all` or similar — the existing flag may already cover
  norms with a stricter setting.

### 2. Reach into pnnx's source

If no flag handles `nn.GroupNorm` directly, pnnx's source has a layer
converter for it (look for `class GroupNorm` in `pnnx-to-ncnn/passes/`).
Patch the converter to emit `GroupNorm` with the three params instead of
`nn.GroupNorm` with bin-baked params. The PyTorch `nn.GroupNorm` module
exposes `.num_groups`, `.num_channels`, `.eps`, `.affine` — those are exactly
what the stock ncnn layer wants on the .param line.

### 3. Post-process the .param on the export side

Far less clean than 1 or 2, but mechanical. After pnnx runs, parse pnnx's
private `nn.GroupNorm` bin header (whatever its format is — read pnnx's
loader source to find out), and rewrite the .param to be stock `GroupNorm`
with the params extracted, plus a parallel rewrite of the .bin to strip the
header bytes.

## What to keep doing right (already correct in the new export)

- The three `torch.*` layers are gone — that part of the fix held.
- `BinaryOp`-only lowering of `torch.where` looks clean (15 new layers
  added, all stock).
- All other layer types (Convolution, LayerNorm, Einsum, MultiHeadAttention,
  etc.) load.

So this is the *only* remaining gap before the C# side runs end-to-end on
Vulkan.

## Same validation as before, plus a load-model check

```bash
# Layer-name check (should print nothing):
awk 'NR>2 {print $1}' restora_generic_fp16_dynamic.ncnn.param | sort -u | \
  awk '/^(torch\.|nn\.)/ { print "MISSING: " $0 }'
```

And, with ncnn.dll on the path:

```python
import ctypes
ncnn = ctypes.CDLL("./ncnn.dll")
# Probe one of the GroupNorm lines must parse:
# Look for "GroupNorm  ...  0=<int> 1=<int> 2=<float>" — three params.
```

If both the name check is clean AND there's at least one `GroupNorm` line
with `0=` / `1=` / `2=` params declared, the export is complete.

## Layer count for sanity

Current re-export has 1465 layers (up from 1423 in the previous one — the
`torch.where` algebraic lowering added ~42 new BinaryOps, as expected). The
GroupNorm rewrite shouldn't change the layer count further (each
`nn.GroupNorm` line becomes a `GroupNorm` line — same line count).
