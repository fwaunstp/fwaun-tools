# fwaun-tools

Tooling for training the fwaun model family. Two command groups under one
binary:

- **`dataset`** — a tag and caption editor for local Stable Diffusion LoRA
  datasets. Combines manual edits, an automatic WD14-family tagger, Qwen3-VL
  captioning, and Danbooru API fetches under one editing surface so the user
  doesn't have to think about provenance while curating.
- **`model`** — diffusion-checkpoint utilities over safetensors files:
  task-vector `merge-diff`, LoRA extraction (`extract-lora`), and INT8+ConvRot
  quantization (`quant-int8`). Pure-Rust/CPU, available in every build.

Built primarily for [ANIMA preview][anima] and Krea 2 LoRA training, but the
data model and export profiles are not ANIMA-specific.

> [日本語版 README](README.ja.md)

[anima]: https://civitai.com/models/anima-preview

## Features

- **One chip list, three sources.** Manual tags, auto-tagger output, and
  Danbooru tags live in the same list — color-coded, but the user
  curates without thinking about provenance.
- **Negative-tag suppression that survives model swaps.** Mark `-foo`
  once; re-running the tagger preserves the suppression even with a
  different model.
- **Curation-only organizational tags.** A positive manual tag starting
  with an underscore (`_foo`) is kept in the data and counted for
  tag-group sorting but never exported — so you can mark "reviewed,
  none of these" distinctly from "not yet reviewed".
- **Tag groups + Kanban view.** Declare mutually-exclusive tag sets
  (e.g. costume variants) in `fwaun-tools.toml`; the GUI shows one
  column per tag with drag-and-drop to switch.
- **Per-folder configuration via `fwaun-tools.toml`.** Pick the tagger
  model, captioner, export profile and threshold per dataset.
- **Two output modes.** `export` writes one `<image>.txt` per image
  (sd-scripts DreamBooth/LoRA caption-file mode); `metadata` writes a
  single `meta.json` (sd-scripts fine-tune mode).
- **Bilingual GUI.** English / 日本語 toggle, defaults to host locale.
- **CLI for batch operations**, GUI for curation.

## Install

### macOS (Apple Silicon) / Linux (x64 or arm64)

```sh
curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh
```

### Windows (x64)

```powershell
irm https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.ps1 | iex
```

Both installers download the latest GitHub release, verify SHA256, and
drop both binaries side-by-side:

| Platform | CLI                                        | GUI                                                |
| -------- | ------------------------------------------ | -------------------------------------------------- |
| macOS    | `~/.local/bin/fwaun-tools`                | `~/.local/bin/fwaun-tools-gui`                    |
| Linux    | `~/.local/bin/fwaun-tools`                | `~/.local/bin/fwaun-tools-gui`                    |
| Windows  | `%USERPROFILE%\bin\fwaun-tools.exe`       | `%USERPROFILE%\bin\fwaun-tools-gui.exe`           |

Pin a specific version with `--version v0.2.1` (or `-Version v0.2.1` on
PowerShell).

The GUI is a single self-contained binary (built with [egui][egui]) —
no `.app`, no AppImage, no MSI. On Linux it depends on the standard
X11 / Wayland system libraries that ship with every desktop
distribution, but no extra runtime install is required.

On macOS the binary is **not notarized**. The installer clears the
`com.apple.quarantine` attribute, but if Gatekeeper still blocks it
when launched from Finder, run it from Terminal once
(`~/.local/bin/fwaun-tools-gui`).

[egui]: https://github.com/emilk/egui

### Linux glibc requirement (full build only)

The Linux **release** binaries are *full* builds (see [Build variants](#build-variants)),
so they link against the glibc shipped on **Ubuntu 24.04 (glibc 2.39)**.
They will not run on Ubuntu 22.04, Debian 12, or earlier — the prebuilt
ONNX Runtime that the local tagger / captioner depend on references
`__isoc23_*` symbols introduced in glibc 2.38. Upgrade, or build a
*light* binary from source (no ONNX Runtime, no glibc floor — it runs on
those older distros).

### Windows support caveat

The maintainer develops on macOS and Linux. Windows builds are produced
by CI but not regularly exercised — please file an issue if anything
breaks.

### Build from source

Requires Rust 1.85+ (edition 2024). On Linux, install standard X11 /
Wayland dev headers (`libx11-dev`, `libxcb1-dev`, `libxkbcommon-dev`,
`libwayland-dev`, `libgl1-mesa-dev`, or your distro's equivalents) for
the GUI:

```sh
git clone https://github.com/fwaunstp/fwaun-tools
cd fwaun-tools
# light build (default) — no local ONNX inference, runs anywhere
cargo build --release -p fwaun-tools-cli
cargo build --release -p fwaun-tools-gui
# full build — adds the local WD14 tagger + Qwen3-VL captioner (glibc 2.38+)
cargo build --release -p fwaun-tools-cli --features full
cargo build --release -p fwaun-tools-gui --features full
```

### Build variants

Two build flavors, selected with the `full` cargo feature:

| | light (default) | full (`--features full`) |
| --- | --- | --- |
| Local WD14 tagger (`tag`) | ✗ | ✓ |
| Local Qwen3-VL captioner | ✗ | ✓ |
| OpenAI-compatible captioner (`caption`) | ✓ | ✓ |
| booru / export / metadata / tag / manual editing / tag groups | ✓ | ✓ |
| ONNX Runtime linked | no | yes |
| Linux glibc floor | none (runs on old distros) | 2.38+ |
| Approx. CLI size | ~11 MB | ~35 MB |

The published **release** binaries are *full*. A *light* binary drops the
two local ONNX models (WD14 tagging, Qwen3-VL captioning) but keeps
everything else — including captioning via any OpenAI-compatible endpoint
(llama.cpp, Ollama, LM Studio, vLLM, …). Running a local-ONNX-only command
in a light build fails fast with a message telling you to install the full
build. Use light when you caption over an API and/or need to run on an
older-glibc host.

## Quick start

1. Launch `fwaun-tools-gui` (or run the CLI directly — see below).
2. **Open folder…** → pick a directory of images.
3. (Optional) **Config…** → write `fwaun-tools.toml` for the dataset.
   Sensible defaults apply if you skip this.
4. Select images, then click **Run tagger** / **Run captioner** /
   **Fetch booru**. The first run downloads the relevant ONNX models
   into the HuggingFace cache (`~/.cache/huggingface/hub`). Set `HF_HOME`
   to relocate the cache, or `HF_ENDPOINT` (e.g.
   `https://hf-mirror.com`) if you cannot reach `huggingface.co` directly.
5. Curate: add manual tags, suppress unwanted auto/booru tags
   (`×` strikes them through), edit captions.
6. Export to disk:

   ```sh
   fwaun-tools dataset export <dir>          # one .txt per image
   fwaun-tools dataset metadata <dir>        # single meta.json
   ```

## Configuration overview

`fwaun-tools.toml` lives in the dataset directory. Everything is
optional — without it, defaults kick in. See
[`crates/core/fwaun-tools.toml.example`](crates/core/fwaun-tools.toml.example) for the
annotated full schema. Highlights:

```toml
default_profile   = "anima"
default_tagger    = "wd-eva02-large-v3"
default_captioner = "qwen3-vl-4b"

[export.anima]
threshold = 0.35
shuffle = false
category_prefixes = { artist = "@" }

[tagger.wd-eva02-large-v3]
repo = "SmilingWolf/wd-eva02-large-tagger-v3"
input_size = 448
storage_threshold = 0.10

[captioner.qwen3-vl-4b]
repo = "onnx-community/Qwen3-4B-VL-ONNX"
subdir = "qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu"
prompt = "Describe this image in detail."
```

## CLI commands

Dataset curation (`fwaun-tools dataset <verb>`):

```
fwaun-tools dataset tag <dir>      [--model NAME] [--threshold X] [--force]
fwaun-tools dataset caption <dir>  [--model NAME] [--force]
fwaun-tools dataset booru <dir>    [--source danbooru] [--force]
fwaun-tools dataset export <dir>   [--profile NAME] [--threshold X]
fwaun-tools dataset metadata <dir> [--profile NAME] [--threshold X] [--output PATH]
fwaun-tools dataset add-tag <dir>    --tags TAG[,...] [--dry-run]
fwaun-tools dataset remove-tag <dir> --tags TAG[,...] [--dry-run]
fwaun-tools dataset mv <dir> <dest>  --tags TAG[,...] [--dry-run]
fwaun-tools dataset status <dir>
fwaun-tools dataset tokens <dir>
fwaun-tools dataset validate-tag-group <dir> --group NAME [--problems-only] [--json]
```

Checkpoint tools (`fwaun-tools model <verb>`) — operate on safetensors files,
not a dataset directory:

```
fwaun-tools model merge-diff   --base B --tuned T --target G -o OUT [--multiplier M] [--model krea2|anima|auto] [--save-dtype bf16|fp16|fp32]
fwaun-tools model extract-lora --base B --tuned T -o OUT [--rank R] [--alpha A] [--model krea2|anima|auto] [--include RE] [--exclude RE]
fwaun-tools model quant-int8   SRC [DST] [--dry-run] [--include RE] [--exclude RE] [--min-gemm N] [--verify-report PATH]
```

`merge-diff` transfers a full fine-tune delta (`tuned − base`) onto another
checkpoint; `extract-lora` factorizes that delta into a kohya-ss/ComfyUI LoRA
by SVD; `quant-int8` writes the comfy-kitchen `int8_tensorwise` + ConvRot
layout. All three are CPU/f32 and stream key-by-key, so peak RAM stays small.

`add-tag` / `remove-tag` bulk-edit the manual tag layer across a
directory: `add-tag` appends each tag verbatim (`foo` positive, `-foo`
suppression marker), `remove-tag` deletes matching manual entries
case-insensitively (pass `--tags=-foo` to drop a suppression marker).
Together they perform a directory-wide tag rename
(`remove-tag <dir> --tags old` then `add-tag <dir> --tags new`).

## Tag groups

Declare named groups of tags that should be mutually exclusive on each
image. The CLI's `validate-tag-group` reports each image as one of the
group's tags, "unset", or "violation" (multiple group tags coexist —
informational, not an error). The GUI's **View → Kanban** mode renders
the same buckets as columns; thumbnails are draggable between them, and
each drop rewrites `manual_tags` to record the new state.

Costume separation for a character LoRA — every image lands in exactly
one column, so it's easy to spot stragglers:

```toml
[tag_group.official_costumes]
tags = ["official_school_uniform", "official_lounge_wear"]
```

Single-tag groups are valid too — handy as a "is tag X set?" sanity
pass on a dataset:

```toml
[tag_group.solo_check]
tags = ["solo"]
```

### Curation-only (organizational) tags

A positive manual tag starting with an underscore (`_foo`) is an
*organizational* tag: it's kept in the data and counted for tag-group
classification, but never written to the exported `.txt`. (Suppression
markers use `-foo`; organizational tags use `_foo`.)

This solves a common ambiguity in character/style tagging: an image with
no group tag could mean either "not reviewed yet" or "reviewed, and it's
deliberately none of these". Add an organizational tag as a group member
to give the latter its own Kanban column, separate from "unset":

```toml
[tag_group.character]
tags = ["character_a", "character_b", "_no_character"]
```

Images you drag into `_no_character` are marked reviewed without the
underscore tag ever leaking into the training caption.

```sh
fwaun-tools dataset validate-tag-group ./dataset --group official_costumes
```

## Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** — architecture, crate layout,
  ONNX session shapes, ort version notes. Read this before
  contributing.
- **[crates/core/fwaun-tools.toml.example](crates/core/fwaun-tools.toml.example)** —
  annotated configuration example.

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
