//! Full-size image preview.
//!
//! The grid only ever holds `THUMB_SIZE` textures — enough to pick an image
//! out of a folder, nowhere near enough to check whether a tag actually
//! describes it. This decodes the source again at (at most) screen resolution
//! and shows it over the app, with the arrow keys walking the same list the
//! grid was showing so a verification pass doesn't mean closing and reopening
//! for every image.
//!
//! Exactly one decoded image is held at a time. A 4K RGBA texture is ~34 MB of
//! VRAM, so keeping a history of them to make Left/Right instant would cost
//! far more than the decode it saves.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::thread;

use eframe::egui;
use egui::{ColorImage, Key, Modifiers, TextureHandle};
use image::imageops::FilterType;

use crate::i18n::T;

/// Hard ceiling on the decoded preview's longest edge, in pixels. The display
/// is fit-to-window, so anything past the screen's own resolution is detail
/// nobody can see; this only bounds the pathological case of a very large
/// screen or scale factor.
const MAX_EDGE: u32 = 4096;
/// Floor for the same, so a preview opened while the window is tiny still
/// decodes something worth looking at.
const MIN_EDGE: u32 = 512;

/// What the caller has to do after a frame of the preview. The OS-handoff
/// variants are routed back out rather than handled here so all the
/// error reporting for them lives in one place.
pub enum PreviewAction {
    /// Nothing to do — keep it open.
    Keep,
    /// Dismissed (Esc, the close button, or a click on the backdrop).
    Close,
    /// Hand the shown image to the OS default viewer.
    OpenExternal,
    /// Reveal the shown image in the file manager.
    Reveal,
}

enum DecodeMsg {
    Ready { req: u64, texture: TextureHandle },
    Failed { req: u64, error: String },
}

pub struct Preview {
    /// The images Left/Right walk, in the order the view listed them.
    /// Captured when the preview opens: re-deriving it every frame would let a
    /// tag edit that changes what the active filter matches silently teleport
    /// the user somewhere else mid-pass.
    order: Vec<PathBuf>,
    index: usize,
    /// Id of the in-flight decode. Results carrying an older id are dropped —
    /// holding an arrow key down queues decodes faster than they finish, and
    /// only the last one requested is worth showing.
    req: u64,
    /// The last image that finished decoding. Deliberately *not* cleared when
    /// navigating: keeping it on screen under the spinner stops the overlay
    /// from collapsing and re-expanding on every step.
    texture: Option<TextureHandle>,
    error: Option<String>,
    rx: Option<Receiver<DecodeMsg>>,
}

impl Preview {
    /// Open the preview on `path`, which must be one of `order`.
    pub fn open(ctx: &egui::Context, order: Vec<PathBuf>, path: &Path) -> Option<Self> {
        let index = order.iter().position(|p| p == path)?;
        // A text field still holding keyboard focus from before the overlay
        // opened would eat the arrow keys before the modal ever sees them.
        ctx.memory_mut(|m| m.stop_text_input());
        let mut preview = Self {
            order,
            index,
            req: 0,
            texture: None,
            error: None,
            rx: None,
        };
        preview.request(ctx);
        Some(preview)
    }

    pub fn path(&self) -> &Path {
        &self.order[self.index]
    }

    /// Start decoding the current image, superseding any decode still running.
    fn request(&mut self, ctx: &egui::Context) {
        self.req += 1;
        self.error = None;
        let req = self.req;
        let path = self.path().to_path_buf();
        let max_edge = max_edge(ctx);
        let ctx = ctx.clone();
        let (tx, rx) = channel();
        thread::spawn(move || {
            let msg = match decode(&path, max_edge, &ctx) {
                Ok(texture) => DecodeMsg::Ready { req, texture },
                Err(error) => DecodeMsg::Failed { req, error },
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
        self.rx = Some(rx);
    }

    /// Move `delta` images along the captured list.
    fn step(&mut self, ctx: &egui::Context, delta: isize) {
        let next = wrap_index(self.index, self.order.len(), delta);
        if next == self.index {
            return;
        }
        self.index = next;
        self.request(ctx);
    }

    fn poll(&mut self) {
        let Some(rx) = self.rx.as_ref() else {
            return;
        };
        let msg = match rx.try_recv() {
            Ok(msg) => msg,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            // The decode thread died without answering. Nothing is coming, so
            // stop showing a spinner that would never resolve.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.rx = None;
                return;
            }
        };
        self.rx = None;
        match msg {
            DecodeMsg::Ready { req, texture } if req == self.req => {
                self.texture = Some(texture);
                self.error = None;
            }
            DecodeMsg::Failed { req, error } if req == self.req => {
                self.texture = None;
                self.error = Some(error);
            }
            // Superseded by a later navigation — drop it.
            _ => {}
        }
    }

    /// Draw one frame of the overlay.
    pub fn show(&mut self, ctx: &egui::Context, t: T) -> PreviewAction {
        self.poll();
        let loading = self.rx.is_some();
        let filename = self
            .path()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let position = t.preview_position(self.index + 1, self.order.len());

        // Room for the header row and the frame's own padding, and past that
        // for a strip of backdrop on every side — the backdrop is a dismiss
        // target, so it has to stay reachable with the pointer.
        let screen = ctx.screen_rect().size();
        let bounds = egui::vec2((screen.x - 96.0).max(160.0), (screen.y - 128.0).max(120.0));

        let mut action = PreviewAction::Keep;
        let mut nav = 0isize;
        let modal = egui::Modal::new(egui::Id::new("image_preview")).show(ctx, |ui| {
            // The modal auto-sizes to its content, and the content is an image
            // that was decoded to fit `bounds` — but a long filename in the
            // header would happily push it past the screen edge.
            ui.set_max_width(bounds.x);
            ui.vertical(|ui| {
                // Everything left-to-right rather than pinning the buttons to
                // the right edge: a right-aligned layout claims the full
                // available width, which here is the whole screen, and the
                // overlay would stop hugging the image.
                ui.horizontal(|ui| {
                    if ui.button("◀").on_hover_text(t.preview_prev()).clicked() {
                        nav = -1;
                    }
                    if ui.button("▶").on_hover_text(t.preview_next()).clicked() {
                        nav = 1;
                    }
                    ui.label(egui::RichText::new(&filename).monospace());
                    ui.label(egui::RichText::new(position).weak().small());
                    if loading {
                        ui.spinner();
                    }
                    ui.separator();
                    if ui.button(t.open_external()).clicked() {
                        action = PreviewAction::OpenExternal;
                    }
                    if ui.button(t.reveal_in_folder()).clicked() {
                        action = PreviewAction::Reveal;
                    }
                    if ui.button(t.close()).clicked() {
                        action = PreviewAction::Close;
                    }
                });
                ui.separator();
                match (&self.texture, &self.error) {
                    (Some(texture), _) => {
                        let size = fit(texture.size_vec2() / ctx.pixels_per_point(), bounds);
                        ui.add(egui::Image::new(texture).fit_to_exact_size(size));
                    }
                    (None, Some(err)) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 180, 180),
                            t.preview_decode_failed(err),
                        );
                    }
                    // First open, nothing decoded yet. Every later navigation
                    // keeps the previous image up, so this is the only frame
                    // shape that has to reserve space of its own.
                    (None, None) => {
                        ui.allocate_ui(egui::vec2(320.0, 180.0), |ui| {
                            ui.centered_and_justified(|ui| ui.spinner());
                        });
                    }
                }
            });
        });

        if modal.should_close() {
            return PreviewAction::Close;
        }
        // Only the topmost modal gets the arrow keys, and `consume_key` takes
        // them out of the queue so nothing underneath acts on them too.
        if modal.is_top_modal {
            ctx.input_mut(|i| {
                if i.consume_key(Modifiers::NONE, Key::ArrowLeft) {
                    nav = -1;
                }
                if i.consume_key(Modifiers::NONE, Key::ArrowRight) {
                    nav = 1;
                }
            });
        }
        if nav != 0 {
            self.step(ctx, nav);
        }
        action
    }
}

/// `index` moved by `delta` within `len`, wrapping at both ends so the last
/// image's Right lands back on the first. A dataset filtered down to a single
/// image stays put rather than re-decoding itself forever.
fn wrap_index(index: usize, len: usize, delta: isize) -> usize {
    if len < 2 {
        return index;
    }
    let len_i = len as isize;
    (((index as isize + delta) % len_i + len_i) % len_i) as usize
}

/// Longest edge to decode to for the current window, in pixels.
fn max_edge(ctx: &egui::Context) -> u32 {
    let screen = ctx.screen_rect().size();
    let longest = screen.x.max(screen.y) * ctx.pixels_per_point();
    (longest.round().max(0.0) as u32).clamp(MIN_EDGE, MAX_EDGE)
}

/// Largest size with the image's aspect ratio that fits inside `bounds`.
/// Never scales past 1:1 — an upscaled blur says nothing new about the image,
/// which is the whole reason the preview exists.
fn fit(image: egui::Vec2, bounds: egui::Vec2) -> egui::Vec2 {
    if image.x <= 0.0 || image.y <= 0.0 {
        return egui::Vec2::ZERO;
    }
    let scale = (bounds.x / image.x).min(bounds.y / image.y).min(1.0);
    image * scale
}

/// Decode `path` and upload it as a texture, shrinking it to `max_edge` first.
///
/// Runs on a worker thread: a 4K source is tens of milliseconds of decode at
/// best, and `Context::load_texture` is internally synchronized, so the upload
/// can come along and keep the whole thing off the UI thread.
fn decode(path: &Path, max_edge: u32, ctx: &egui::Context) -> Result<TextureHandle, String> {
    let image = image::open(path).map_err(|e| e.to_string())?;
    // Only ever shrink. `resize` alone would happily blow a 256×256 source up
    // to screen size, spending VRAM to add nothing.
    let image = if image.width() > max_edge || image.height() > max_edge {
        // Triangle rather than `thumbnail`'s fast path: at preview size the
        // resampling artifacts are visible, and this runs at most once per
        // image the user actually looks at.
        image.resize(max_edge, max_edge, FilterType::Triangle)
    } else {
        image
    };
    let rgba = image.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let color = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    Ok(ctx.load_texture(
        format!("preview::{}", path.display()),
        color,
        egui::TextureOptions::LINEAR,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32) -> egui::Vec2 {
        egui::vec2(x, y)
    }

    #[test]
    fn fit_shrinks_to_the_tighter_axis() {
        // Wide image in a square box: width is what binds.
        assert_eq!(fit(v(4000.0, 1000.0), v(1000.0, 1000.0)), v(1000.0, 250.0));
        // Tall image in the same box: height binds.
        assert_eq!(fit(v(1000.0, 4000.0), v(1000.0, 1000.0)), v(250.0, 1000.0));
    }

    #[test]
    fn fit_never_upscales() {
        assert_eq!(fit(v(300.0, 200.0), v(1920.0, 1080.0)), v(300.0, 200.0));
    }

    #[test]
    fn fit_tolerates_a_degenerate_image() {
        assert_eq!(fit(v(0.0, 0.0), v(100.0, 100.0)), egui::Vec2::ZERO);
    }

    #[test]
    fn stepping_wraps_at_both_ends() {
        assert_eq!(wrap_index(0, 3, 1), 1);
        assert_eq!(wrap_index(2, 3, 1), 0);
        assert_eq!(wrap_index(0, 3, -1), 2);
    }

    #[test]
    fn stepping_a_single_image_stays_put() {
        assert_eq!(wrap_index(0, 1, 1), 0);
        assert_eq!(wrap_index(0, 1, -1), 0);
        assert_eq!(wrap_index(0, 0, 1), 0);
    }
}
