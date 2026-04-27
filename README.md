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
  gui/         ── `anima-tagger-gui` binary (dioxus-desktop).
```

Each ML/network surface is its own crate so the GUI doesn't pull ort just to
render a thumbnail grid, and so build times stay reasonable when iterating on
one piece.

---

## Data model

### Sidecar (`<image>.png.ron`)

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

Lives in the dataset directory. Everything is optional except tagger /
captioner profiles you actually intend to invoke.

```toml
default_profile  = "anima"
default_tagger   = "wd-eva02-large-v3"
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

# Tagger profile — points at WD14-family ONNX + the matching CSV.
[tagger.wd-eva02-large-v3]
model_path = "/path/to/wd-eva02-large-tagger-v3/model.onnx"
tags_path  = "/path/to/wd-eva02-large-tagger-v3/selected_tags.csv"
input_size = 448
storage_threshold = 0.10           # filter tags below this when *storing*

# Captioner profile — model_dir holds the 4 ONNX submodels + tokenizer.json.
[captioner.florence2-base]
model_dir = "/path/to/Florence-2-base-ft/onnx"
prompt = "<MORE_DETAILED_CAPTION>"
input_size = 768
max_new_tokens = 1024
```

Key idea: model-specific quirks (ANIMA's `@artist`, alternate trainers'
prefixes) are encoded as **export profiles**, not hardcoded.

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
and is cached for the app's lifetime; opening a different folder
invalidates the cache (config might point at a different model).

Detail panel:
- 0 selected: hint.
- 1 selected: filename, full chip list (manual/auto/booru with strikethrough
  for suppressed), manual caption textarea (commits on blur), auto caption
  display, booru post link.
- N selected: bulk-edit mode with text input. Type `foo` to add a positive
  tag to all selected, `-foo` to suppress.

The dioxus crate is used because the user is also building a separate mobile
app with dioxus and wants one framework to learn for review purposes.

---

## Crate-specific notes

### crates/tagger (WD14)

Single ONNX model + `selected_tags.csv` (SmilingWolf format). Preprocessing:
square-pad with white → resize CatmullRom-bicubic to model input size →
BGR uint8→f32 → NHWC. Output is sigmoid probabilities mapped 1:1 to CSV
rows, filtered by threshold, sorted by score descending.

The tagger uses ort's **positional** `inputs![tensor]` because WD14 models
have only one input and naming varies between exports.

### crates/captioner (Florence-2)

Four ONNX submodels in series:

```
pixel_values ──► vision_encoder.onnx ────► image_features [1, V, D]
input_ids   ──► embed_tokens.onnx   ────► text_embeds   [1, T, D]
                          │
                          ▼ concat along seq axis
encoder_input  ──► encoder_model.onnx  ────► encoder_hidden_states
                          │
                          ▼ greedy decode loop (no KV cache yet)
                  decoder_model.onnx
```

Several non-obvious things had to be figured out empirically. They're worth
remembering because re-discovering them is painful:

1. **Florence-2's user-facing task names are NOT tokens.** `<CAPTION>`,
   `<MORE_DETAILED_CAPTION>` and friends are sugar that HF's
   `Florence2Processor` expands to natural-language questions
   ("Describe with a paragraph what is shown in the image.") *before*
   tokenization. The model is trained on the expanded text. We mirror the
   table in `expand_task_prompt`. The added_tokens.json contains only
   `<loc_*>` coordinate tokens and structural tokens (`<od>`, `<seg>`,
   `<cap>`, `<dcap>`, etc.), not task tokens.

2. **The decoder takes `inputs_embeds`, not `input_ids`.** This particular
   ONNX export expects pre-embedded decoder tokens, and emits a full set
   of `present.*.{decoder,encoder}.{key,value}` KV cache tensors that we
   currently discard. Each decode step we re-embed the running token
   sequence through `embed_tokens.onnx` — O(n²) in caption length but
   correct. Wiring `decoder_with_past_model.onnx` into the loop would
   restore O(n) (see TODO).

3. **Force `<s>` (id 0) at decode step 0.** BART-style models have
   `forced_bos_token_id=0` in their generation config. Without forcing,
   greedy decoding regularly collapses into degenerate output (e.g.
   echoing the prompt characters back as the "caption"). We override the
   step-0 argmax unconditionally.

4. **ImageNet normalization, RGB, NCHW, 768×768, CatmullRom resize.** PIL
   bicubic is what HF uses; `image::imageops::FilterType::CatmullRom` is
   the closest equivalent. `Triangle` (bilinear) is subtly different and
   may degrade quality.

5. **Decoder start = EOS = id 2.** BART convention: decoder starts at
   `</s>` (id 2), and stops when `</s>` (also id 2) is generated again.
   Step 0 is forced past this with `<s>` (id 0).

The build_session helper logs each session's actual input/output names on
load — keep this if you ever swap to a different Florence-2 export, since
the input names vary.

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

- **Captioner is O(n²)**. The `decoder_with_past_model.onnx` from the same
  HF repo, plus the `present.*.key/value` tensors we currently discard,
  would let us run the decoder incrementally. Big payoff for caption
  quality (could afford beam search) and throughput.
- **GUI freezes during long ops**. Tagger / captioner / booru runs block
  the dioxus event loop. Move to `spawn_blocking` + a progress signal so
  the user can see progress and cancel.
- **Single image at a time** for tagger and captioner. Batching would
  reduce per-image overhead, especially for the encoder/embed sessions
  in the captioner pipeline.
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
cargo build -p anima-tagger-gui    # GUI builds (~3min first time, webkit2gtk)
```

The captioner and tagger have no automated tests because they require model
files. Manual testing is via `cargo run -p anima-tagger-cli -- caption <dir>`
on a small sample and inspecting the resulting `.ron` sidecars.

---

## Background reading

When picking the captioner back up:

- Florence-2 paper and HF model card (microsoft/Florence-2-base-ft and
  the onnx-community mirror).
- HF transformers' `Florence2Processor` source — particularly
  `_construct_prompts` and `task_prompts_without_inputs`. The
  expand_task_prompt table here mirrors that dict.
- HF transformers' BART decoder, especially `forced_bos_token_id` handling
  in `LogitsProcessor`.

When picking the tagger back up:

- SmilingWolf's WD14 tagger HF repos (e.g. wd-eva02-large-tagger-v3) for
  the canonical preprocessing reference.
- The `selected_tags.csv` format: 4 columns, `category` is the danbooru
  numeric category (0=general, 1=artist, 3=copyright, 4=character,
  5=meta, 9=rating).
