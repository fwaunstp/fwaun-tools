# GUI assets

## Icon

`icon.png` is the window/taskbar icon. [`main.rs`](../src/main.rs) loads
it at compile time via `include_bytes!` and hands it to `eframe`'s window
icon (`with_icon`). The window icon path (winit `Icon::from_rgba`)
accepts raster RGBA only.

Requirements for `icon.png`:

- **RGBA** PNG (transparent background)
- **square**, 8-bit/channel, non-interlaced
- **512x512 or larger** so it can double as the bundle source below
  (the current icon is 1024x1024)

## Bundling

`dx bundle` derives every platform-specific icon (macOS `.icns`, Windows
`.ico`, Linux PNGs) from the single `icon.png` source, so keep it large
and square (512x512 or 1024x1024 downsamples best).
