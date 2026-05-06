# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/fwaunstp/anima-tagger/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fwaunstp/anima-tagger/releases/tag/v0.2.0
[0.1.0]: https://github.com/fwaunstp/anima-tagger/releases/tag/v0.1.0
