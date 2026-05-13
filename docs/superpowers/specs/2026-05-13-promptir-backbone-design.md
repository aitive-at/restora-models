# PromptIR Backbone — Config-Driven Prompt Variant

**Status:** Approved
**Date:** 2026-05-13
**Owner:** bglueck
**Lineage:** Adds a second backbone alongside the existing
`nafnet` model from the 2026-05-13-refine-compound-design spec. Does
not modify NAFNet. Shares the same `(rgb, config) → rgb` forward
contract, the same `ConfigEmbed`, and every downstream component
(trainer, losses, inference pipeline, ONNX export, sidecar JSON, CLI).

## 1. Goal

Implement Potlapalli et al.'s PromptIR (NeurIPS 2023) as a drop-in
alternative backbone in this framework. After this change, a config
selects which backbone trains:

```yaml
model:
  type: promptir   # or "nafnet"
  size: large
```

Everything else — data, losses, CLI, ONNX export, the 5-axis
conditioning contract — stays bit-identical to today.

## 2. Non-goals

- Replacing NAFNet. Both backbones live in the registry; user picks via
  `model.type` in YAML.
- Faithful reproduction of PromptIR's paper-original "blind" prompt
  selection. We deliberately replace the blind self-attention branch
  with config-driven attention, so model behavior stays deterministic
  from the caller's 5-axis vector. (Paper-faithful blind variant is a
  possible future addition; not in this scope.)
- Reproducing the paper's training curriculum / progressive-resolution
  schedule. We use the existing trainer.

## 3. Architecture

4-level Restormer-style encoder/decoder U-Net:

```
input (B,3,H,W)
   │
   ├─ stem 3 → C ──────────────────────────────────────────┐
   │                                                        │
   │  enc_l1: TransformerBlock×N1  @ C       ──skip──►──┐  │
   │   ↓ down (PixelUnshuffle, 2C)                       │  │
   │  enc_l2: TransformerBlock×N2  @ 2C      ──skip──►─┐│  │
   │   ↓ down (4C)                                      ││  │
   │  enc_l3: TransformerBlock×N3  @ 4C      ──skip──►┐││  │
   │   ↓ down (8C)                                     │││  │
   │  latent: TransformerBlock×NL  @ 8C                │││  │
   │   │                                                │││  │
   │   ▼                                                │││  │
   │  PromptBlock (config-driven, scale 8C)             │││  │
   │   ↓ up (PixelShuffle, 4C)                          │││  │
   │  dec_l3 ◄─── skip ────────────────────────────────┘││  │
   │  PromptBlock (config-driven, scale 4C)              ││  │
   │   ↓ up (2C)                                          ││  │
   │  dec_l2 ◄─── skip ──────────────────────────────────┘│  │
   │  PromptBlock (config-driven, scale 2C)                │  │
   │   ↓ up (C)                                             │  │
   │  dec_l1 ◄─── skip ────────────────────────────────────┘  │
   │   │                                                       │
   │  refinement: TransformerBlock×NR @ C                      │
   │   │                                                       │
   │  head C → 3                                               │
   │   │                                                       │
   └───┴── + input (global residual) ──► output (B,3,H,W) ◄───┘
```

A `TransformerBlock` is a Restormer block:

- **MDTA** — Multi-Dconv head Transposed Attention. Q/K/V from 1×1 conv
  + depthwise 3×3 conv. Attention is computed along the channel axis
  (`softmax(K · Qᵀ / τ)` over channels, then `V · attn`), so cost is
  `O(C² · HW)` instead of the `O((HW)²)` of vanilla attention. Critical
  for 256² training resolution.
- **GDFN** — Gated-Dconv Feed-Forward. Two-branch: depthwise-conv +
  GELU on one branch, depthwise-conv on the other, element-wise
  multiplied, projected back. Replaces standard FFN.
- **AdaLN modulation** from the task vector. The same modulation hook
  NAFNet's TransformerBlock uses, so the config influences every block.

PixelUnshuffle/PixelShuffle for resolution change (no information-loss
strided convs).

## 4. The PromptBlock

```python
class PromptBlock(nn.Module):
    """Config-driven prompt selection. Replaces PromptIR's blind
    feature-pooling-attention with a router driven by config_embed.

    prompts: (N, P_c, P_h, P_w) learnable parameter
    router:  Linear(C_cond → N)
    fuse:    Conv1x1(feat_c + P_c → feat_c)
    """
    def __init__(self, *, feat_c: int, prompt_n: int = 5,
                 prompt_dim: int, prompt_hw: int, cond_dim: int):
        ...

    def forward(self, feat: Tensor[B, feat_c, H, W],
                cond: Tensor[B, cond_dim]) -> Tensor[B, feat_c, H, W]:
        alpha = F.softmax(self.router(cond), dim=-1)     # (B, N)
        # Broadcast-mix the prompt bank:
        mix = (alpha[:, :, None, None, None] * self.prompts[None]).sum(dim=1)
        mix = F.interpolate(mix, size=feat.shape[-2:], mode="bilinear",
                            align_corners=False)
        return self.fuse(torch.cat([feat, mix], dim=1))
```

**Design property — config determinism.** `alpha` is determined
entirely by `cond`. The same input image with a different config
produces a different mix and therefore different output. With
`config = [1,0,0,0,0]`, `alpha` peaks at one prompt (the one the model
has learned for "colorize"). With `config = [1,1,1,1,1]`, all 5
prompts mix. With `config = [0,0,0,0,0]`, the router output reduces to
its bias term — model learns to produce a near-zero mix and the global
residual carries the identity.

Default `prompt_n = 5` (matches the 5 axes — natural prior for
training to converge to "one prompt per axis", though nothing in the
loss forces this).

## 5. Conditioning

Reuse `ConfigEmbed(num_axes=5, dim=task_embed_dim)` from
`src/refine/models/task_embed.py` unchanged. The same `(B, C_cond)`
vector feeds:

- AdaLN scale/shift inside every TransformerBlock
- The router Linear inside every PromptBlock

One shared embedding, routed two ways. No parameter duplication.

## 6. Size presets

```python
_SIZE_PRESETS = {
    "tiny": {
        "dim": 32, "depths": [2, 2, 2, 2], "refinement": 2,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 32, "prompt_hw": 16,
    },
    "large": {
        "dim": 48, "depths": [4, 6, 6, 8], "refinement": 4,
        "heads": [1, 2, 4, 8],
        "prompt_n": 5, "prompt_dim": 64, "prompt_hw": 16,
    },
}
```

`depths` = blocks per encoder level (4 entries: enc_l1, enc_l2,
enc_l3, latent). Decoder mirrors: `dec_l3` uses `depths[2]` blocks,
`dec_l2` uses `depths[1]`, `dec_l1` uses `depths[0]`. `heads[i]` is the
MDTA head count at the i-th level (encoder and the corresponding
decoder share the same head count). `refinement` = block count of the
final-resolution refinement stack (head count = `heads[0]`).

Skip connections: encoder feature at level i is concatenated with the
upsampled decoder feature at that level, then projected back to the
decoder's channel count with a 1×1 conv before the decoder
TransformerBlocks consume it.

Approximate parameter counts: tiny ≈ 6M, large ≈ 33M (matches paper
PromptIR-large). Tiny is a smoke-test preset that fits CPU CI.

## 7. ModelConfig additions

`ModelConfig` (in `src/refine/config.py`) gains three optional
override fields, all defaulting to `None` so size-preset values are
used:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `prompt_n` | `int \| None` | `None` | Override prompt-bank size |
| `prompt_dim` | `int \| None` | `None` | Override prompt channel dim |
| `prompt_hw` | `int \| None` | `None` | Override prompt spatial size |

Existing fields stay. NAFNet ignores the new fields; PromptIR ignores
NAFNet-specific ones. No breaking change to any existing config or
checkpoint.

## 8. Files added / modified

| File | Status | Purpose |
|---|---|---|
| `src/refine/models/restormer_block.py` | NEW | MDTA + GDFN transformer block with AdaLN modulation |
| `src/refine/models/prompt_block.py` | NEW | Config-driven PromptBlock (sec. 4) |
| `src/refine/models/promptir.py` | NEW | Backbone class, `@register_model("promptir")`, `_SIZE_PRESETS` |
| `src/refine/models/__init__.py` | MOD | Import `promptir` to register |
| `src/refine/config.py` | MOD | Add `prompt_n`, `prompt_dim`, `prompt_hw` optional overrides |
| `tests/test_restormer_block.py` | NEW | Block forward + grad |
| `tests/test_prompt_block.py` | NEW | Config-determinism + shape + grad |
| `tests/test_promptir.py` | NEW | Full backbone shape, identity-config, param sanity, ONNX parity (slow) |
| `configs/promptir-tiny.yaml` | NEW | Smoke-test config |
| `configs/promptir-large.yaml` | NEW | Production-size config |
| `configs/promptir-laion.yaml` | NEW | LAION-specific (defaults: promptir-large) |

No changes to: `trainer.py`, `compound.py`, any loss file, `cli.py`,
`infer/pipeline.py`, `export/onnx.py`. Drop-in by construction.

## 9. Tests

| Test | Speed | What it proves |
|---|---|---|
| `test_restormer_block::test_shape_grad` | fast | MDTA/GDFN block accepts `(B, C, H, W)` + `(B, cond)`, output same shape, backward succeeds |
| `test_prompt_block::test_shape` | fast | Output shape matches input feature shape |
| `test_prompt_block::test_config_determinism` | fast | Two different configs on same input produce different outputs; same config produces identical outputs |
| `test_prompt_block::test_grad` | fast | All trainable params (`prompts`, `router`, `fuse`) get gradients |
| `test_promptir::test_forward_shape` | fast | `(2, 3, 128, 128) + (2, 5)` → `(2, 3, 128, 128)`, finite |
| `test_promptir::test_backward_smoke` | fast | One SGD step decreases L1 loss on a constant target |
| `test_promptir::test_identity_config_passes_through` | fast | With `config = 0` and untrained weights, output is close to input (global-residual sanity) |
| `test_promptir::test_param_count_sanity` | fast | tiny < 10M, large < 50M |
| `test_promptir::test_onnx_export_parity_all_configs` | slow (`REFINE_SLOW=1`) | Existing exporter writes a working 2-input ONNX; 7 reference configs match PyTorch within `1e-3` |

The ONNX parity test is the single most important integration check —
it exercises the forward signature, dynamic config behavior, dynamic
batch dim, and the existing exporter all at once. If it passes, the
trainer, inference pipeline, and sidecar contract are guaranteed to
work without modification.

## 10. Numerical / training contract

- Output range: same as NAFNet — model outputs are added to input, so
  the final tensor is in `[0, 1] ± small overshoot`. No clamp inside
  the model (callers / loss / metrics handle this exactly as today).
- AMP compatibility: MDTA's channel-axis softmax is fp32-stable; we
  match Restormer's reference and dispatch the softmax in fp32 even
  under bf16/fp16 training. GDFN's gating is fine in any precision.
- Gradient checkpointing: not enabled by default. Reservable later via
  a `model.gradient_checkpoint: bool` field if memory pressure forces
  it on `large` at 256² with batch 12.

## 11. Inference / export / sidecar

Unchanged. The exporter (`src/refine/export/onnx.py`) is
model-agnostic — it reads `model_type` from the checkpoint config and
emits an ONNX with two inputs and 7-config parity. After this change,
checkpoints sidecar JSON will simply have `"model_type": "promptir"`
where NAFNet runs have `"model_type": "nafnet"`. Downstream consumers
(C# integration, ORT-Web) need no code changes — the contract is the
contract.

## 12. Open questions

None blocking.
