# anima-tagger

A Rust workspace for managing tags and captions on a local Stable Diffusion
LoRA dataset. Combines three sources of tags — manual edits, an automatic
WD14-family tagger, and Danbooru API fetches — under a single editing surface
so the user doesn't have to think about provenance while curating.

This is developer documentation: the goal is to let a future contributor
(quite possibly Future You) pick the codebase back up without re-deriving the
design decisions from scratch.

---

## Why this project exists

The user trains LoRAs locally for the [ANIMA preview model][anima] (booru tags
+ natural-language captions in one model). Their pre-existing pain points,
which drive every decision below:

1. Hand-editing per-image `.txt` tag files for multi-element LoRAs (a costume
   trigger plus character names plus the standard tagger output) gets tedious
   fast.
2. Re-running auto-taggers in append mode duplicates tags, forcing dataset
   folder splits to keep the deltas separate.
3. Threshold tweaking with most existing tools means re-running inference on
   the whole dataset.
4. Re-tagging with a different model loses any deletions the user previously
   curated.
5. For images that already exist on Danbooru, regenerating tags loses the
   human-curated metadata that's already there.

[anima]: https://civitai.com/models/anima-preview

---

## Workspace layout

```
crates/
  core/        ── data model (Sidecar, ProjectConfig, ExportProfile),
                  RON sidecar I/O, export logic, image walker. No ML deps.
  tagger/      ── WD14-family ONNX tagger via ort 2.0.0-rc.12.
  captioner/   ── Florence-2 ONNX captioner (4-session pipeline).
  booru/       ── Danbooru md5-lookup tag fetcher (ureq + md5).
  cli/         ── `anima-tagger` binary. Thin layer over the above.
  gui/         ── `anima-tagger-gui` binary (egui / eframe).
```

Each ML/network surface is its own crate so the GUI doesn't pull ort just to
render a thumbnail grid, and so build times stay reasonable when iterating on
one piece.

---

## Data model

### Sidecar (`<basename>.ron`)

Every image gets a sibling RON file holding everything we know about it.

```rust
pub struct Sidecar {
    pub manual_tags: Vec<String>,    // positives ("foo") + suppressions ("-foo")
    pub auto_tags: Vec<AutoTag>,     // tagger output (verbatim, never edited)
    pub booru_tags: Vec<BooruTag>,   // booru API output (verbatim, never edited)
    pub caption: Option<String>,         // captioner output (verbatim)
    pub manual_caption: Option<String>,  // user-prepended caption fragment
    pub tagger: Option<TaggerInfo>,      // provenance: which model, when
    pub captioner: Option<CaptionerInfo>,
    pub booru: Option<BooruInfo>,
}
```

- **RON, not JSON.** `<image>.json` would collide with arbitrary 3rd-party
  metadata sidecars (training tools, image organizers, etc.); `<image>.ron`
  has effectively zero collision surface.
- Auto/booru records store **raw scores and category labels**, so threshold
  tweaks at export time are a re-filter rather than a re-inference.
- `manual_tags` is one flat list with two semantics: `"foo"` is a positive
  tag, `"-foo"` is a suppression marker. There is no separate
  `negative_tags` field on purpose — keeping it flat makes the format
  greppable and direct hand-editing obvious.

### Three tag sources, one mental model

The GUI deliberately surfaces all three sources in one chip list,
distinguished only by color:

| Source       | Color  | Editable from GUI                                  |
| ------------ | ------ | -------------------------------------------------- |
| manual       | blue   | yes — add by typing, remove by ×                   |
| auto         | gray   | indirectly — × adds `-foo` to manual (suppression) |
| booru        | green  | indirectly — × adds `-foo` to manual (suppression) |
| (suppressed) | strike | × removes the `-foo` to un-suppress                |

The user said explicitly: *they don't want to think about manual vs auto
while editing*. The chip color is feedback, not a separate workflow.

### Negative-tag suppression

The killer property: a `-foo` entry suppresses any auto/booru tag with stem
`foo` from export, **regardless of which tagger or booru produced it**.
Re-running the tagger overwrites `auto_tags` but leaves `manual_tags`
untouched, so deletion decisions persist across model swaps.

Implementation: `Sidecar::suppressed_set()` returns the lowercase stems of
all `-foo` entries. `export::build_tags` filters every non-manual source
through it.

Negative entries are also never emitted as positive tags.

### Manual caption

Same philosophy on the caption side. `caption` is whatever Florence-2 spat
out. `manual_caption` is the user's prepend (typically character names,
scene context). On export the merged caption is `"{manual} {auto}"`.

---

## Configuration: `anima-tagger.toml`

Lives in the dataset directory. **Entirely optional** — with no config file at
all, the CLI/GUI fall back to built-in tagger and captioner profiles
(`SmilingWolf/wd-eva02-large-tagger-v3` and `onnx-community/Qwen3-4B-VL-ONNX`,
4B vision-fp32 / text-int4 variant), auto-downloaded into the shared
HuggingFace cache (`~/.cache/huggingface/hub`) on first run.

```toml
default_profile   = "anima"
default_tagger    = "wd-eva02-large-v3"
default_captioner = "florence2-base"

# How tags are formatted/filtered when written to .txt or meta.json.
[export.anima]
threshold = 0.35
shuffle = false                    # sd-scripts shuffles at training time
exclude_categories = []
category_prefixes = { artist = "@" }   # ANIMA's `@artist_name` convention

[export.plain]
threshold = 0.35
shuffle = false

# Tagger profile — HuggingFace repo holding model.onnx + selected_tags.csv.
[tagger.wd-eva02-large-v3]
repo = "SmilingWolf/wd-eva02-large-tagger-v3"
# revision = "main"                # optional; pins a branch/tag/commit
input_size = 448
storage_threshold = 0.10           # filter tags below this when *storing*

# Captioner profile — HuggingFace repo with the Qwen3-VL ONNX layout
# (`<subdir>/{qwen3vl-vision,qwen3vl-embedding,model}.onnx` +
# `model.onnx.data` + `tokenizer.json`).
[captioner.qwen3-vl-4b]
repo = "onnx-community/Qwen3-4B-VL-ONNX"
subdir = "qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu"
prompt = "Describe this image in detail."
max_pixels = 589824                # smart_resize area cap (≈768x768)
max_new_tokens = 1024
```

Models are fetched via [`hf-hub`][hf-hub] into the same cache directory the
Python `huggingface_hub` / sd-scripts / diffusers use, so any models already
downloaded by other tools are reused for free.

Key idea: model-specific quirks (ANIMA's `@artist`, alternate trainers'
prefixes) are encoded as **export profiles**, not hardcoded.

[hf-hub]: https://crates.io/crates/hf-hub

---

## CLI commands

```
anima-tagger tag <dir> [--model NAME] [--threshold X] [--force]
anima-tagger caption <dir> [--model NAME] [--force]
anima-tagger booru <dir> [--source danbooru] [--force]
anima-tagger export <dir> [--profile NAME] [--threshold X]
anima-tagger metadata <dir> [--profile NAME] [--threshold X] [--output PATH]
anima-tagger status <dir>
```

- `tag` / `caption` / `booru` — populate sidecars. Skip already-populated
  images unless `--force`.
- `export` — write one `<image>.txt` per image (DreamBooth/LoRA mode).
- `metadata` — write a single `meta.json` (sd-scripts fine-tune mode):
  `{ "<abs_path>": { "tags": "...", "caption": "..." }, ... }`. Tags use
  the same merge/dedup/filter logic as `export`. Caption is the merged
  manual + auto string.
- `status` — quick `[TCB] manual=N <path>` table per image (T=auto-tagged,
  C=captioned, B=booru-fetched).

---

## GUI

`cargo run -p anima-tagger-gui --release`

Toolbar: open folder, filter dropdown, Select visible / Clear sel., Run
tagger / Run captioner / Fetch booru. The model loads lazily on first run
(downloading the ONNX weights from HuggingFace if not already cached) and
is reused for the app's lifetime; opening a different folder invalidates
the cache (config might point at a different model).

Detail panel:
- 0 selected: hint.
- 1 selected: filename, full chip list (manual/auto/booru with strikethrough
  for suppressed), manual caption textarea (commits on blur), auto caption
  display, booru post link.
- N selected: bulk-edit mode with text input. Type `foo` to add a positive
  tag to all selected, `-foo` to suppress.

The GUI uses egui via eframe. The first cut was dioxus-desktop (chosen
to share a framework with a separate mobile project), but webkit2gtk
prevents Linux single-binary distribution and dioxus has no
mobile-rendering benefit on this side anyway, so v0.2 swapped to
eframe — same dataset, native windowing, single binary on every
target.

---

## Crate-specific notes

### crates/tagger (WD14)

Single ONNX model + `selected_tags.csv` (SmilingWolf format). Preprocessing:
square-pad with white → resize CatmullRom-bicubic to model input size →
BGR uint8→f32 → NHWC. Output is sigmoid probabilities mapped 1:1 to CSV
rows, filtered by threshold, sorted by score descending.

The tagger uses ort's **positional** `inputs![tensor]` because WD14 models
have only one input and naming varies between exports.

### crates/captioner (Qwen3-VL)

Three ONNX sessions in series, with a merged decoder:

```
pixel_values + image_grid_thw ──► qwen3vl-vision.onnx     ──► vision_hidden_states [Nv, 2560]
input_ids + vision_hidden_states ──► qwen3vl-embedding.onnx ──► inputs_embeds [1, S, 2560]
                                                  │
                                                  ▼ greedy decode loop with KV cache
                                          model.onnx (merged prefill / decode)
```

Default repo: `onnx-community/Qwen3-4B-VL-ONNX`, subdir
`qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu`. Picked because it ships
a 3-session ONNX layout that matches the existing pattern, has a healthy
abliterated / NSFW-tolerant finetune ecosystem (huihui-ai, prithivMLmods), and
the INT4 decoder keeps the total download in the ~5 GB range.

Things to remember on a re-read or before swapping models:

1. **Embedding session is fused.** `qwen3vl-embedding.onnx` does
   `embed_tokens(input_ids)` *and* in-graph `masked_scatter` of
   `vision_hidden_states` into the rows where `input_ids == <|image_pad|>`
   (151655). The caller only needs to expand the single `<|image_pad|>` token
   from the chat template to `n_vision_tokens = (grid_h*grid_w)/4` copies and
   pass both tensors in.

2. **Patch geometry: 16/2/2.** `patch_size=16`, `merge_size=2`,
   `temporal_patch_size=2`, so smart_resize aligns to `factor=32`. Each
   pixel_values row is `3 * 2 * 16 * 16 = 1536`. Vision output hidden dim is
   2560 — same as the LLM hidden_size, no projection needed.

3. **Plain (0.5, 0.5, 0.5) mean/std**, not CLIP / not ImageNet. Easy to miss.

4. **Decoder is INT4-quantized.** External weights live in `model.onnx.data`
   (~2.4 GB). ort needs to resolve the external-data path relative to
   `model.onnx`, so the decoder is loaded via `commit_from_file` (requires
   the `std` feature on `ort` — disabled by `default-features = false`, so
   we add it back explicitly in the workspace toml).

5. **3D MROPE position_ids.** `[3, 1, S]`: row 0 = temporal index, row 1 =
   h-index, row 2 = w-index. For text tokens all three rows hold the running
   text position. For image tokens at position k, rows hold `(t_idx, h_idx,
   w_idx)` within the merged-patch grid offset by where the image span
   starts. Text after the image resumes at `max(image positions) + 1`. See
   `Qwen2VLForConditionalGeneration.get_rope_index` in HF transformers for
   the canonical derivation; the algorithm is unchanged from Qwen2-VL.

6. **GQA-shaped KV cache.** 36 layers × 2 (key, value), shape
   `[1, 8, past_seq, 128]` (8 KV heads vs 32 attention heads). Empty initial
   tensors (`past_seq = 0`) are how we tell the merged decoder it's a
   prefill call.

7. **Chat template has no system message** by default (different from
   Qwen2-VL). One user turn, one assistant turn.

The build_session helper logs each session's actual input/output names on
load — keep this if you ever swap to a different Qwen3-VL export, since the
input names occasionally drift between builders.

### crates/booru

Computes the image's md5 hash, hits
`https://danbooru.donmai.us/posts.json?tags=md5:<hex>&limit=1`, parses the
response into `BooruTag`s grouped by category (artist / copyright /
character / general / meta).

Only Danbooru is wired up. Adding Gelbooru, Konachan, etc. is a matter of
adding adapter functions that return the same `(Vec<BooruTag>, BooruInfo)`
tuple — the trait is sketched but not formalized.

The md5 lookup only matches **bit-identical** files. Re-encoded or edited
copies won't hit. For LoRA datasets where the user has the original
booru-sourced files, this is fine.

### crates/core

`config.rs` — TOML-backed `ProjectConfig` with `export`, `tagger`, and
`captioner` profile maps plus the optional `default_*` selectors.
Resolvers: `resolve_profile`, `resolve_tagger`, `resolve_captioner`.

`sidecar.rs` — RON I/O via atomic write (`<file>.ron.tmp` + rename).
Includes the suppression helpers and the `manual_caption` accessor.

`export.rs` — `build_tags()` is the main entry; `export_image()` writes
the resulting comma-joined string to `<image>.txt`. Tests cover suppression
across sources, profile prefix dedup, threshold/category filters, and
manual ordering.

`walk.rs` — recursive walk yielding image-extension files.

---

## ort version notes

Pinned to `ort = "2.0.0-rc.12"` with `download-binaries` + `tls-rustls`.
Specific gotchas (see `crates/tagger/src/lib.rs` and
`crates/captioner/src/lib.rs`):

- `download-binaries` requires exactly one TLS feature alongside it.
- `Tensor::from_array` does not accept ndarray directly. Use the
  `(shape: [i64; N], data: Vec<T>)` tuple form.
- `try_extract_tensor::<f32>()` returns `(&Shape, &[T])`.
- `Session::builder().commit_from_file(path)` does not exist on rc.12;
  read bytes manually and use `commit_from_memory(&bytes)`.
- `ort::Error<F>` is parameterized — a single `#[from] ort::Error`
  variant via thiserror only catches one phantom. Both crates use a
  blanket `impl<F> From<ort::Error<F>> for MyError` to flatten via
  `e.to_string()`.

Re-verify these on any version bump.

---

## Output formats

| Mode      | Files written                       | Use case                              |
| --------- | ----------------------------------- | ------------------------------------- |
| `export`  | one `<image>.txt` per image         | sd-scripts DreamBooth / LoRA caption-file mode |
| `metadata`| one `meta.json` for the whole dir   | sd-scripts fine-tune mode (tags + caption together) |

Default `shuffle` is **off** — sd-scripts shuffles at training time, and a
non-shuffled metadata file diffs cleanly across runs.

---

## Known limitations / TODO

- **Captioner is INT4-only.** Only the int4 4B variant is published as a
  prebuilt ONNX. fp16 / fp32 require running the upstream `builder.py` to
  re-export. Quality is good enough for caption-style outputs but worse than
  fp16 would be, especially on non-photographic / illustrative inputs.
- **No caption-side beam search.** Greedy decoding only. With the merged
  decoder's KV cache already wired in, beam-3 would be cheap to add.
- **GUI freezes during long ops**. Tagger / captioner / booru runs block
  the egui event loop synchronously inside `update()`. Move to a
  background thread + `mpsc` channel so the user can see progress and
  cancel.
- **Single image at a time** for tagger and captioner. Batching would
  reduce per-image overhead, especially for the vision and embedding
  sessions.
- **No NSFW-finetune helper.** Switching to e.g.
  `prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it` requires manually
  re-exporting that model to the same 3-session ONNX layout (the upstream
  repo only ships safetensors). Once that's done it's a `repo = "..."` line
  swap, but the export step itself isn't automated here.
- **Bulk caption editing**. Currently single-image only — by design for
  per-character names, but a "append this fragment to all selected" might
  be useful.
- **Only Danbooru**. Gelbooru / Konachan / Yande.re need adapters.
- **No undo**. Edits commit straight to disk on blur / click. A dirty
  tracker + Ctrl+Z would help.
- **No image preview pane**. The thumbnail is the only visual. For
  detailed inspection users open the file externally.
- **Folder switch invalidates the model cache**, but doesn't preload the
  new folder's models. First action after switch always pays the load
  cost.
- **No async tagger/captioner I/O**. Reading 200MB ONNX files synchronously
  on app startup-ish is fine but not ideal.

---

## Testing

```bash
cargo test -p anima-tagger-core    # 9 unit tests covering export logic
cargo check --workspace            # everything compiles (~10s incremental)
cargo build -p anima-tagger-cli    # binary builds (~40s first time, ort download)
cargo build -p anima-tagger-gui    # GUI builds (egui + ort, ~1min first time)
```

The captioner and tagger have no automated tests because they require model
files. Manual testing is via `cargo run -p anima-tagger-cli -- caption <dir>`
on a small sample and inspecting the resulting `.ron` sidecars.

The annotated config example used by both `core` (snapshot test) and
`gui` (Config… modal default text) lives at
[`crates/core/anima-tagger.toml.example`](crates/core/anima-tagger.toml.example).
Re-exported as `anima_tagger_core::config::CONFIG_EXAMPLE`. Keep it
inside the core crate dir — `include_str!` paths must stay within the
package on crates.io.

---

## Background reading

When picking the captioner back up:

- Qwen3-VL model card and the `onnx-community/Qwen3-4B-VL-ONNX` repo —
  particularly `builder.py` (defines the exact ONNX I/O signatures) and
  `qwen3vl-oga-inference.py` (end-to-end reference using onnxruntime-genai;
  note its preprocessor skips the smart_resize step that the upstream HF
  processor does, so we still need to do that ourselves).
- HF transformers' `Qwen2VLForConditionalGeneration.get_rope_index` for the
  canonical 3D MROPE position_ids algorithm — unchanged for Qwen3-VL even
  though the internal `mrope_section` splits differ.
- HF transformers' `Qwen2VLImageProcessorFast.smart_resize` for the resize
  algorithm (Qwen3-VL inherits it; only `factor` changes from 28 to 32).

When picking the tagger back up:

- SmilingWolf's WD14 tagger HF repos (e.g. wd-eva02-large-tagger-v3) for
  the canonical preprocessing reference.
- The `selected_tags.csv` format: 4 columns, `category` is the danbooru
  numeric category (0=general, 1=artist, 3=copyright, 4=character,
  5=meta, 9=rating).
