# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-16

### Changed

- **Installer auto-detects headless Linux.** `install.sh` now installs both
  binaries by default but drops the GUI on a headless Linux host (no
  `$DISPLAY` / `$WAYLAND_DISPLAY`), where it can't run anyway. Override with
  `--both` / `--cli-only` / `--gui-only`. On an older-glibc system it warns and
  points at the source-built light CLI (`cargo install --git … fwaun-tools-cli`).
  Prebuilt releases remain full-only; the portable *light* build is
  source-only (documented in the README).
- **Renamed the project to `fwaun-tools`.** Binaries are now `fwaun-tools`
  (was `fwaun-tagger`) and `fwaun-tools-gui`; workspace crates are
  `fwaun-tools-*`. The project config file is now `fwaun-tools.toml`
  (user-level `~/.config/fwaun-tools/config.toml`). The previous
  `fwaun-tagger.toml` / `fwaun-tagger/config.toml` names — and the older
  `anima-tagger` ones — are still read as a deprecated fallback with a
  warning; rename to the new name to silence it.
- **CLI subcommands are grouped under `dataset` and `model`.** Existing
  dataset verbs move from `fwaun-tagger <verb>` to
  `fwaun-tools dataset <verb>` (e.g. `fwaun-tools dataset tag <dir>`,
  `fwaun-tools dataset export <dir>`).

### Added

- **`fwaun-tools model` — diffusion-checkpoint tools.** `merge-diff`
  (task-vector merge), `extract-lora` (SVD LoRA extraction), and `quant-int8`
  (INT8 + ConvRot quantization) over safetensors files, merged in from the
  standalone `fwaun-model-tools` project so both toolsets share one binary.
  Pure-Rust/CPU with no ONNX Runtime dependency, so they ship in the portable
  light build as well as full.
- **CLI: `add-tag` / `remove-tag` subcommands.** Bulk-edit the manual tag
  layer across a directory. `add-tag <dir> --tags <TAG>[,...]` appends each
  tag verbatim to every image's sidecar (`foo` positive, `-foo`
  suppression marker), creating a sidecar where none exists and skipping
  entries already present. `remove-tag <dir> --tags <TAG>[,...]` deletes
  matching manual entries case-insensitively — including `-foo` markers
  when `-foo` is passed — without adding any suppression marker or touching
  auto/booru tags. Together they cover a directory-wide tag rename
  (`remove-tag old` + `add-tag new`). Both support `--dry-run`; pass a
  leading-`-` tag after `=` (`--tags=-foo`) so it isn't parsed as a flag.

## [0.3.0] — 2026-07-09

### Added

- **Curation-only organizational tags.** A positive manual tag starting
  with an underscore (`_foo`) is kept in the data and counted for
  tag-group classification, but is never written to the exported `.txt`.
  Lets you distinguish "not yet reviewed" from "reviewed, deliberately
  none of these" — add an organizational tag as a tag-group member to
  give the latter its own Kanban column. The GUI colours such chips
  distinctly. (Suppression markers remain `-foo`.)
- **Tag groups for mutually-exclusive tag classification.** Declare
  named groups in `fwaun-tools.toml` (e.g.
  `[tag_group.official_costumes] tags = ["official_school_uniform",
  "official_lounge_wear"]`); each image is bucketed as one of the
  group's tags, "unset", or "violation" (multiple group tags coexist —
  informational, not an error). Single-tag groups are also valid for
  "is tag X set?" curation passes.
- **CLI: `validate-tag-group` subcommand.** Reports each image's bucket
  for a chosen group as a text table or JSON; `--problems-only` hides
  cleanly-classified rows.
- **GUI: Kanban view with drag-and-drop.** A new View → Kanban: <name>
  selector switches the central panel into one column per tag (plus
  "unset" and "violation"); thumbnails are draggable between columns
  and each drop rewrites `manual_tags` (positive on the destination
  tag, `-X` suppression on previously-present sibling tags). The
  violation column is read-only — to create a multi-tag image
  intentionally, use the detail panel.
- **CLI: `mv` subcommand.** `fwaun-tools mv <dir> <dest> --tags
  <TAG>[,...]` moves every image whose effective tag set (the same set
  `validate-tag-group` uses) contains all requested tags, together with
  its `.ron` sidecar. Sub-paths are preserved under `dest`, existing
  destination files are never overwritten, cross-filesystem moves fall
  back to copy+remove, and `--dry-run` previews. Matches are collected
  before any move, so a `dest` nested inside `dir` is safe.
- **musubi-tuner caption output.** `metadata --format musubi` writes a
  caption-only `meta.jsonl` (`{"image_path","caption"}` per line) for
  kohya-ss/musubi-tuner's `image_jsonl_file`; `--format sd-scripts`
  (default) still emits the tags+captions `meta.json`.
- **Tag-driven caption prefixes and suffixes.**
  `[export.<p>.caption_prefixes]` / `caption_suffixes` map a curation tag
  to a literal string prepended / appended to the caption (case-
  insensitive match, leading organizational `_` ignored, key order).
  Lets tags like `realistic` or a Krea-2 trailing trigger word drive a
  deterministic head/tail token instead of relying on the captioner's
  prose. Empty maps are a no-op. Both surfaced in the GUI config editor.
- **CLI: `--promote-to-manual=always`.** New mode that copies the
  resolved prompt's caption into `manual_caption` unconditionally,
  overwriting an already-promoted caption (previously only `never` /
  `if-empty`).
- **HuggingFace `HF_ENDPOINT` / `HF_HOME` support.** Model downloads now
  honor these environment variables, so a mirror (e.g.
  `HF_ENDPOINT=https://hf-mirror.com`) or a relocated cache works.
  (#9)

### Changed

- **Project renamed `anima-tagger` → `fwaun-tools`.** All crates are now
  `fwaun-tools-*`, the binaries are `fwaun-tools` / `fwaun-tools-gui`,
  and the per-directory config file is `fwaun-tools.toml` (user config:
  `fwaun-tools/config.toml`). The old `anima-tagger.toml` /
  `anima-tagger/config.toml` are still read as a fallback and print a
  deprecation warning; rename yours to the new name, as the fallback will
  be removed in a future release. The `anima` export-profile name and
  ANIMA-model conventions are unrelated to the project name and unchanged.
- **GUI: structured settings editor.** The Config… modal is now a tabbed
  form (General / Tagger / Captioner / Prompts / Export / Tag groups)
  instead of a raw TOML text area, with in-place key renaming and
  save-time validation of duplicate / empty names.

### Fixed

- **OpenAI-compatible captioner retries transient failures.** HTTP 5xx
  and transport errors are retried with exponential backoff (1/2/4s,
  capped at 30s) up to a new `max_retries` profile field (default 3);
  4xx client errors are not retried. Previously a single 500 aborted the
  whole run.

## [0.2.1] — 2026-05-07

### Fixed

- **ONNX captioner crashed on the first decode step.** `Tensor::from_array`
  in `ort` 2.0.0-rc.12 rejects any dim `< 1`, so the empty initial KV
  cache (`[1, 8, 0, 128]`) and the post-prefill empty vision input
  (`[0, 2560]`) both failed with `Invalid dimension #3; all dimensions
  must be >= 1 when creating a tensor from row data`. Zero-sized inputs
  now go through `Tensor::<f32>::new(&Allocator::default(), shape)`
  (CreateTensorAsOrtValue), which accepts zero-sized dims; non-empty
  KV cache still goes through `from_array`.

## [0.2.0] — 2026-05-07

### Added

- **Progress overlay for tagger / captioner / booru runs.** Long-
  running ops now happen on a background thread with an mpsc channel
  feeding per-image progress back to the UI; the GUI keeps repainting
  and shows a centered modal with the operation label, a progress bar,
  and an `N / total images` counter. Sidecar I/O still happens on the
  main thread (each per-image worker result is applied as it arrives),
  so model state stays consistent and there is no race against the
  user clicking around.
- **GUI: skip already-processed images on tagger / captioner / booru
  runs.** Selections are filtered against the in-memory sidecars before
  the worker spawns, mirroring the CLI's default behavior (no
  `--force`). Tagger and booru skip per image; captioner skips per
  `(image, prompt-key)` pair so a partially-captioned image still
  gets its missing prompts run. The status banner reports how many
  were skipped, or notes when nothing remains to do.
- **GUI: delete image action.** A red `Delete image…` /
  `Delete selected images…` button at the bottom of the detail panel
  opens a confirmation modal that lists the affected files. On
  confirm, the image and its `.ron` sidecar are removed from disk
  and the in-memory grid / selection / thumbnail / edit-buffer state
  is updated in lock-step.

### Changed

- **GUI rewritten from dioxus-desktop to egui / eframe.** The release
  binaries are now genuinely single-file: macOS, Linux, and Windows
  builds ship a single `anima-tagger-gui` binary with no `.app`,
  AppImage, or MSI wrapper. The Linux build keeps a glibc 2.39 floor
  (still inherited from the prebuilt onnxruntime), but no longer
  depends on webkit2gtk or other extra runtime installs.
- Bundled `NotoSansJP-Regular.otf` (~4.5 MB JP-only subset) for CJK
  rendering, since egui's default font set is latin-only. Total
  release-build size: ~40 MB.

### Removed

- `Dioxus.toml`, `packaging/macos/{Info.plist.template,build-app.sh}`,
  the `dx bundle` step in the release workflow, and the
  `libwebkit2gtk-4.1-dev` / `libxdo-dev` apt installs in CI.

## [0.1.0] — 2026-05-04

First public release. Everything below is the initial feature set; future
versions will list deltas from here.

### Added

#### Data model
- `<basename>.ron` per-image sidecar holding manual edits, raw tagger
  output, raw Danbooru output, captions per model, caption hint, and
  per-source provenance (model name, fetched-at timestamp).
- Negative-tag suppression: a `-foo` entry in `manual_tags` hides `foo`
  from export regardless of which auto/booru source produced it. The
  flag survives re-tagging with a different model.
- Per-image `caption_hint` reference info — passed to the captioner as
  context but never written into the exported `.txt`.
- Manual caption as final-output override (not an auto-prefix);
  promote-to-manual flow copies an auto caption into manual when the
  user wants to lock it in.
- Per-caption `skip` flag — keep a stored caption but exclude it from
  export.

#### Configuration (`anima-tagger.toml`)
- Per-folder TOML config with `[export.*]`, `[tagger.*]`, `[captioner.*]`
  profile maps and `default_*` selectors.
- Walk-up discovery: `anima-tagger.toml` is found in any ancestor
  directory, the way `git` finds `.git`.
- User-level config at `$XDG_CONFIG_HOME/anima-tagger/config.toml`
  merged underneath the project config.
- Shared `[captioner_prompts]` library — define a prompt once, reference
  it by name from any captioner profile.
- Hard-coded defaults for tagger and captioner so users with no config
  file get a working setup on first run.

#### Tagger (`anima-tagger-tagger`)
- WD14-family ONNX tagger via `ort` 2.0.0-rc.12. Single-model pipeline
  with square-pad → bicubic resize → BGR uint8→f32 NHWC preprocessing.

#### Captioner (`anima-tagger-captioner`)
- Qwen3-VL-4B ONNX captioner (3-session pipeline: vision → embedding →
  merged prefill/decode with KV cache, INT4-quantized decoder).
- OpenAI-compatible HTTP backend as an alternative — works against
  llama.cpp, koboldcpp, Ollama, LM Studio, vLLM, etc.
- Multiple named prompts per profile, stored in the sidecar as
  `{profile}.{prompt}` so they coexist without re-loading the model.
- `cli tokens` subcommand for pre-flight context-budget checks.

#### Booru (`anima-tagger-booru`)
- Danbooru md5-lookup tag fetcher (bit-identical match only).

#### CLI (`anima-tagger`)
- Subcommands: `tag`, `caption`, `booru`, `export`, `metadata`,
  `status`, `tokens`.
- Two output modes: per-image `<image>.txt` (DreamBooth/LoRA caption
  files) or single `meta.json` (sd-scripts fine-tune mode).
- Tag underscores replaced with spaces on export.

#### GUI (`anima-tagger-gui`)
- Single-window Dioxus desktop app: open folder, image grid, detail
  panel.
- Three tag sources rendered in one chip list, color-coded by
  provenance; user edits via type-to-add and `×`-to-remove without
  having to think about which source.
- Tag-substring filter, contain-fit thumbnails, status flags
  (`T`/`C`/`B`/`M`/`H`).
- Bulk-edit mode for multi-selection: caption hint apply-to-all,
  manual entry union, common-tag summary, per-model caption counts.
- **Config editor modal** (`Config…` button) — edits the project's
  `anima-tagger.toml` directly with parse-time validation; saving
  drops the cached tagger/captioner instances so the next run picks
  up the new profile.
- **Bilingual UI** — English and Japanese, defaults to host locale
  (via `sys-locale`). User selection persists at
  `~/.config/anima-tagger/gui-prefs.toml`.

#### Distribution
- GitHub Actions release workflow: `v*` tag pushes produce
  `macos-arm64`, `linux-x64`, `linux-arm64`, and `windows-x64` artifacts
  via `dx bundle` (macOS `.app` tar.gz, Linux AppImage, Windows MSI),
  plus a per-target CLI archive. SHA256SUMS is generated and attached.
- `install.sh` (Bash) and `install.ps1` (PowerShell) installers that
  resolve the latest GitHub release, verify checksums, and place the
  CLI / GUI in canonical user-local locations.
- Per-crate Cargo manifests fully populated for crates.io publishing
  (description, categories, license files, version pins on inter-crate
  path deps).
- Dual MIT / Apache-2.0 licensing with copies of both LICENSE files in
  every crate directory.

### Known limitations

- Captioner is INT4-only by default — only the int4 4B variant is
  published as a prebuilt ONNX. Fp16/fp32 require running upstream
  `builder.py` to re-export.
- No caption-side beam search (greedy decoding only).
- Tagger / captioner / booru runs block the Dioxus event loop — no
  progress bar, no cancel button.
- Single image at a time on the model side (no batching).
- Booru lookup matches bit-identical files only — re-encoded copies
  miss.
- Only Danbooru is wired up; Gelbooru / Konachan / Yande.re need
  adapters.
- No undo. Edits commit straight to disk on blur / click.
- No image preview pane; thumbnails are the only visual.
- Linux release binaries require glibc 2.39+ (Ubuntu 24.04, Fedora 40+,
  Arch). Older distros need to build from source.
- Windows builds are produced by CI but not regularly tested by the
  maintainer.

[Unreleased]: https://github.com/fwaunstp/fwaun-tools/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/fwaunstp/fwaun-tools/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/fwaunstp/fwaun-tools/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/fwaunstp/fwaun-tools/releases/tag/v0.2.1
[0.2.0]: https://github.com/fwaunstp/fwaun-tools/releases/tag/v0.2.0
[0.1.0]: https://github.com/fwaunstp/fwaun-tools/releases/tag/v0.1.0
