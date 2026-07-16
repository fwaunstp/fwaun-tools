# fwaun-model-tools

Rust utilities for working with diffusion model checkpoints (safetensors).

## `merge-diff` — task-vector merge

Transfers a full fine-tune delta onto another checkpoint, key by key:

```
output[k] = target[k] + multiplier * (tuned[k] - base[k])
```

This is a Rust port of musubi-tuner's `krea2_merge_diff.py`, generalized to also
handle **Anima** checkpoints. Unlike a LoRA extraction it is full-rank and touches
every key, so norm / modulation / scale tensors a Linear-only low-rank LoRA cannot
represent are carried over too.

The intended Krea 2 use is to move a full fine-tune done on the RAW model onto the
distilled Turbo model, without the lossy SVD LoRA extraction:

```sh
fwaun-model-tools merge-diff \
  --base   krea2_raw_bf16.safetensors \
  --tuned  krea2-<your>-ft.safetensors \
  --target krea2_turbo_bf16.safetensors \
  --output krea2_turbo_<your>-ft.safetensors \
  --multiplier 0.8 \
  --model krea2
```

For Anima (DiT tensors namespaced under `net.`):

```sh
fwaun-model-tools merge-diff --model anima \
  --base anima_base.safetensors --tuned anima_ft.safetensors \
  --target anima_target.safetensors --output anima_merged.safetensors
```

### Options

| Flag | Description |
| --- | --- |
| `--base` | Original model the fine-tune started from |
| `--tuned` | Fine-tuned model (the full fine-tune output) |
| `--target` | Model to receive the delta (bf16) |
| `--output`, `-o` | Output safetensors path |
| `--multiplier` | Delta strength (default `1.0`; lower if it over-applies) |
| `--save-dtype` | Force merged-key output dtype: `bf16`, `fp16`, `fp32` (default: keep target's) |
| `--model` | Key-prefix convention: `auto` (default), `krea2`, or `anima` |

### Notes

- **Key matching** normalizes away DiT prefixes so a prefixed target lines up with
  bare base/tuned keys: `model.diffusion_model.` / `diffusion_model.` (Krea 2,
  ComfyUI/Civitai) and `net.` (Anima). `--model auto` accepts all of them.
- **Output keys and per-key dtype follow the target**, so the merged file stays in
  the target's namespace/format. `--save-dtype` only overrides keys that receive a
  delta; pass-through keys keep the target's dtype.
- **fp8_scaled checkpoints are rejected** (a bf16 delta cannot be cleanly added to
  fp8 weights that carry separate per-tensor scales). Use a bf16 target.
- **Memory**: all three inputs are memory-mapped and the output is streamed key by
  key, so peak RAM stays at a few tensors — much lower than the model size. Math
  runs on CPU in f32.

## `extract-lora` — LoRA from a full fine-tune

Extracts a low-rank LoRA from the difference between a base model and its full
fine-tune, so you can publish a LoRA (e.g. on Civitai) built from the *same*
richer fine-tune you already train — instead of training a separate, rank-limited
LoRA on the dataset. For every shared 2-D linear weight it forms the same
`tuned − base` delta as `merge-diff` and factorizes it by SVD:

```
ΔW = W_tuned − W_base  ≈  U_r · S_r · V_rᵀ
lora_up   (B) = U_r · √S_r          [out, r]
lora_down (A) = √S_r · V_rᵀ         [r, in]
```

Output uses the kohya-ss / ComfyUI convention (`lora_unet_<module>` with
`.lora_up.weight` / `.lora_down.weight` / `.alpha`) that FLUX-family LoRAs on
Civitai use, e.g. `double_blocks.0.img_attn.qkv` →
`lora_unet_double_blocks_0_img_attn_qkv`.

```sh
fwaun-model-tools extract-lora \
  --base  krea2_raw_bf16.safetensors \
  --tuned krea2-<your>-ft.safetensors \
  --output krea2-<your>-lora.safetensors \
  --rank 32 \
  --model krea2
```

### Options

| Flag | Description |
| --- | --- |
| `--base` | Original model the fine-tune started from |
| `--tuned` | Fine-tuned model (the full fine-tune output) |
| `--output`, `-o` | Output LoRA safetensors path |
| `--rank` | LoRA rank / network dim (default `32`; higher = closer to the fine-tune, larger file) |
| `--alpha` | Nominal alpha to store (default: each module's own rank, so multiplier 1 reproduces the truncated delta) |
| `--save-dtype` | LoRA weight dtype: `bf16`, `fp16` (default), `fp32` |
| `--model` | Key-prefix convention: `auto` (default), `krea2`, or `anima` |
| `--include RE` | Regex; only bare module paths matching this are extracted |
| `--exclude RE` | Regex; matching bare module paths are skipped |
| `--niter` | Power iterations in the randomized SVD (default `2`; more = more accurate, slower) |
| `--oversample` | Extra sampling columns above the rank (default `8`) |

### Notes

- **Lossy by nature.** A full fine-tune is generally full-rank; a rank-`r` LoRA can
  only approximate it. The tool reports the per-module **energy captured**
  (`Σσ²_kept / ‖ΔW‖²_F`) and warns when any module falls below 90% — raise `--rank`
  for a closer match at the cost of file size. This is exactly why a LoRA *extracted*
  from a data-rich fine-tune can carry more than one trained from scratch at the same
  rank: it approximates the fine-tune's full-rank delta rather than being constrained
  to low rank during training.
- **Scope.** Only 2-D linear weights are extracted; 1-D norms, biases and modulation
  scales (which a Linear-only LoRA cannot represent) are skipped. Use `--include` /
  `--exclude` on the bare module path to refine the set.
- **`alpha = dim`** is stored per module by default, so `up @ down` at multiplier 1
  reproduces the truncated delta exactly. `--alpha` rescales the factors to a fixed
  nominal alpha without changing the reconstruction.
- **fp8_scaled inputs are rejected** — extract from a bf16/fp16/fp32 base + fine-tune.
- **Compute**: randomized SVD (with subspace iteration) on CPU/f32, parallelized
  across cores, one module at a time so peak RAM stays near a single weight matrix.
  Extraction is markedly heavier than `merge-diff` — expect it to be compute-bound.

## `quant-int8` — INT8 + ConvRot quantization

Quantizes a bf16/fp16 checkpoint to the comfy-kitchen `int8_tensorwise` + ConvRot
layout the ComfyUI loader reads. It auto-detects the per-token block linears
(attention + FFN), rotates each with a block-Hadamard at the best power-of-4 group
size, and stores int8 weights with per-channel scales:

```
output[k].weight       = int8    (block-Hadamard rotated, per-channel absmax)
output[k].weight_scale = f32      per-channel scale
output[k].comfy_quant  = uint8    JSON config {format, convrot, convrot_groupsize}
```

This is a CPU/f32 Rust port of comfy-model-tools' `quant_int8_convrot.py`, so it
frees the GPU for training while quantization runs. Producing the file needs no
CUDA; only inference-time dequant does (unchanged, on the comfy-kitchen side). The
Hadamard rotation runs in parallel across cores.

Dry-run first on an unfamiliar architecture — it prints the plan and writes nothing:

```sh
fwaun-model-tools quant-int8 model_bf16.safetensors --dry-run
```

Then quantize (output path derived from the source if omitted):

```sh
fwaun-model-tools quant-int8 model_bf16.safetensors model_int8_convrot.safetensors
```

### Options

| Flag | Description |
| --- | --- |
| `--dry-run` | Print the plan (quantize list + skip reasons), write nothing |
| `--exclude RE` | Regex; matching layers are forced to passthrough |
| `--include RE` | Regex; matching eligible layers are forced to quantize |
| `--min-gemm N` | Skip a layer if `min(N,K) < N` (default `256`; `0` disables) |
| `--downcast-fp32` | Downcast stray fp32 passthrough linears to the compute dtype |
| `--warn-thresh F` | Warn on any quantized layer whose relerr% exceeds this (default `2.0`) |
| `--verify-report PATH` | Write the full per-layer `(relerr, cosine, gs)` table |

### Notes

- **Detection** quantizes every eligible 2-D block linear (an integer-indexed block
  with a valid group size and `N ≥ 8`) minus a name denylist (norms, rope/pos-embed,
  input embedders, gate/router logits, timestep MLPs, output head, adapters). Use
  `--dry-run` and `--include` / `--exclude` to adjust on a new arch.
- **Reconstruction error** (relerr = `‖dequant − source‖ / ‖source‖`, and cosine) is
  reported per layer and summarized worst-first. Because the Hadamard is orthogonal,
  these are computed in the rotated space at no extra cost and equal the true
  original-space error.
- **fp8_scaled sources are rejected** — quantize from a bf16/fp16/fp32 checkpoint.
- **Parity**: byte-for-byte identical to the Python reference except for the
  unavoidable CPU-vs-CUDA floating-point noise in the Hadamard reduction (per-channel
  scales differ by ~1e-10; the odd int8 element sitting on a rounding boundary may
  differ by 1 LSB).
