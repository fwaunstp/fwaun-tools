# anima-tagger

Tag and caption editor for local Stable Diffusion LoRA datasets. Combines
manual edits, an automatic WD14-family tagger, Qwen3-VL captioning, and
Danbooru API fetches under one editing surface so the user doesn't have to
think about provenance while curating.

Built primarily for [ANIMA preview][anima] LoRA training, but the data
model and export profiles are not ANIMA-specific.

> [日本語版 README](README.ja.md)

[anima]: https://civitai.com/models/anima-preview

## Features

- **One chip list, three sources.** Manual tags, auto-tagger output, and
  Danbooru tags live in the same list — color-coded, but the user
  curates without thinking about provenance.
- **Negative-tag suppression that survives model swaps.** Mark `-foo`
  once; re-running the tagger preserves the suppression even with a
  different model.
- **Per-folder configuration via `anima-tagger.toml`.** Pick the tagger
  model, captioner, export profile and threshold per dataset.
- **Two output modes.** `export` writes one `<image>.txt` per image
  (sd-scripts DreamBooth/LoRA caption-file mode); `metadata` writes a
  single `meta.json` (sd-scripts fine-tune mode).
- **Bilingual GUI.** English / 日本語 toggle, defaults to host locale.
- **CLI for batch operations**, GUI for curation.

## Install

### macOS (Apple Silicon) / Linux (x64 or arm64)

```sh
curl -fsSL https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.sh | sh
```

### Windows (x64)

```powershell
irm https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.ps1 | iex
```

Both installers download the latest GitHub release, verify SHA256, and
drop both binaries side-by-side:

| Platform | CLI                                        | GUI                                                |
| -------- | ------------------------------------------ | -------------------------------------------------- |
| macOS    | `~/.local/bin/anima-tagger`                | `~/.local/bin/anima-tagger-gui`                    |
| Linux    | `~/.local/bin/anima-tagger`                | `~/.local/bin/anima-tagger-gui`                    |
| Windows  | `%USERPROFILE%\bin\anima-tagger.exe`       | `%USERPROFILE%\bin\anima-tagger-gui.exe`           |

Pin a specific version with `--version v0.1.0` (or `-Version v0.1.0` on
PowerShell).

The GUI is a single self-contained binary (built with [egui][egui]) —
no `.app`, no AppImage, no MSI. On Linux it depends on the standard
X11 / Wayland system libraries that ship with every desktop
distribution, but no extra runtime install is required.

On macOS the binary is **not notarized**. The installer clears the
`com.apple.quarantine` attribute, but if Gatekeeper still blocks it
when launched from Finder, run it from Terminal once
(`~/.local/bin/anima-tagger-gui`).

[egui]: https://github.com/emilk/egui

### Linux glibc requirement

The Linux release binaries link against the glibc shipped on
**Ubuntu 24.04 (glibc 2.39)**. They will not run on Ubuntu 22.04, Debian
12, or earlier — the prebuilt ONNX Runtime that the tagger / captioner
depend on references `__isoc23_*` symbols introduced in glibc 2.38.
Build from source on older distros, or upgrade.

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
git clone https://github.com/fwaunstp/anima-tagger
cd anima-tagger
cargo build --release -p anima-tagger-cli
cargo build --release -p anima-tagger-gui
```

## Quick start

1. Launch `anima-tagger-gui` (or run the CLI directly — see below).
2. **Open folder…** → pick a directory of images.
3. (Optional) **Config…** → write `anima-tagger.toml` for the dataset.
   Sensible defaults apply if you skip this.
4. Select images, then click **Run tagger** / **Run captioner** /
   **Fetch booru**. The first run downloads the relevant ONNX models
   into the HuggingFace cache (`~/.cache/huggingface/hub`).
5. Curate: add manual tags, suppress unwanted auto/booru tags
   (`×` strikes them through), edit captions.
6. Export to disk:

   ```sh
   anima-tagger export <dir>          # one .txt per image
   anima-tagger metadata <dir>        # single meta.json
   ```

## Configuration overview

`anima-tagger.toml` lives in the dataset directory. Everything is
optional — without it, defaults kick in. See
[`crates/core/anima-tagger.toml.example`](crates/core/anima-tagger.toml.example) for the
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

```
anima-tagger tag <dir>      [--model NAME] [--threshold X] [--force]
anima-tagger caption <dir>  [--model NAME] [--force]
anima-tagger booru <dir>    [--source danbooru] [--force]
anima-tagger export <dir>   [--profile NAME] [--threshold X]
anima-tagger metadata <dir> [--profile NAME] [--threshold X] [--output PATH]
anima-tagger status <dir>
anima-tagger tokens <dir>
```

## Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** — architecture, crate layout,
  ONNX session shapes, ort version notes. Read this before
  contributing.
- **[crates/core/anima-tagger.toml.example](crates/core/anima-tagger.toml.example)** —
  annotated configuration example.

## License

Dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
