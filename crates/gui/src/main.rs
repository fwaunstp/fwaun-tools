#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod config_ui;
mod i18n;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::thread;

use anima_tagger_booru::{BooruClient, BooruError};
use anima_tagger_captioner::Captioner;
use anima_tagger_core::config::{CONFIG_FILE, ProjectConfig, TagGroup};
use anima_tagger_core::sidecar::{AutoTag, BooruInfo, BooruTag, Sidecar, TaggerInfo, sidecar_path_for};
use anima_tagger_core::tag_group::{self, Classification, DropTarget};
use anima_tagger_core::walk::iter_images;
use anima_tagger_tagger::Tagger;
use chrono::{DateTime, Utc};
use eframe::egui;
use egui::{ColorImage, Key, TextureHandle};

use crate::config_ui::{ConfigAction, ConfigDraft, ConfigTab, show_config_modal};
use crate::i18n::{Lang, T, load_pref_or_detect, save_pref};

/// Bundled CJK font so Japanese labels render out of the box without a
/// system font fallback. Subset OTF, ~4.5 MB. If a third script
/// (Korean / Chinese / etc.) is ever requested, switch this to a
/// probe-path lookup against the OS font dirs (macOS:
/// `/System/Library/Fonts/Supplemental/HiraginoSans-W3.ttc`, Windows:
/// `C:\Windows\Fonts\YuGothM.ttc` / `meiryo.ttc`, Linux:
/// `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` with
/// `FontData.index = 1` for the JP face). For Japanese-only the bundle
/// cost is acceptable.
const JP_FONT: &[u8] = include_bytes!("../assets/NotoSansJP-Regular.otf");
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");
const THUMB_SIZE: u32 = 256;
const THUMB_DRAW_PX: f32 = 160.0;

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("anima-tagger")
        .with_inner_size([1200.0, 800.0]);
    if let Ok(icon) = eframe::icon_data::from_png_bytes(ICON_PNG) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "anima-tagger",
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(AnimaTaggerApp::new()))
        }),
    )
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("noto-jp".into(), egui::FontData::from_static(JP_FONT).into());
    // Append, not prepend — keep latin glyph fidelity for the default
    // proportional font, fall through to Noto JP for codepoints the
    // primary face doesn't cover.
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("noto-jp".into());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("noto-jp".into());
    ctx.set_fonts(fonts);
}

#[derive(Clone)]
struct ImageItem {
    path: PathBuf,
    sidecar: Sidecar,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    All,
    Untagged,
    AutoTagged,
    NoManual,
    NoCaption,
    NoHint,
    NoBooru,
}

impl Filter {
    fn matches(self, item: &ImageItem) -> bool {
        match self {
            Self::All => true,
            Self::Untagged => !item.sidecar.is_auto_tagged() && item.sidecar.manual_tags.is_empty(),
            Self::AutoTagged => item.sidecar.is_auto_tagged(),
            Self::NoManual => item.sidecar.manual_tags.is_empty(),
            Self::NoCaption => !item.sidecar.is_captioned(),
            Self::NoHint => item.sidecar.caption_hint.is_none(),
            Self::NoBooru => !item.sidecar.has_booru(),
        }
    }
    fn label(self, t: T) -> &'static str {
        match self {
            Self::All => t.filter_all(),
            Self::Untagged => t.filter_untagged(),
            Self::AutoTagged => t.filter_auto_tagged(),
            Self::NoManual => t.filter_no_manual(),
            Self::NoCaption => t.filter_no_caption(),
            Self::NoHint => t.filter_no_hint(),
            Self::NoBooru => t.filter_no_booru(),
        }
    }
    const ALL: [Filter; 7] = [
        Self::All,
        Self::Untagged,
        Self::AutoTagged,
        Self::NoManual,
        Self::NoCaption,
        Self::NoHint,
        Self::NoBooru,
    ];
}

// ───────── Worker types ─────────
//
// Long-running ops (tagger / captioner / booru) run on a background
// thread so the GUI keeps repainting and the user sees a progress
// modal. Communication is mpsc: the worker streams `Progress` updates
// and per-image `*Result` messages, ending with a single `Done` that
// hands the (possibly newly-loaded) model back to the main thread.

#[derive(Clone, Copy, PartialEq)]
enum WorkerOp {
    Tagger,
    Captioner,
    Booru,
}

#[derive(Clone)]
struct Progress {
    op: WorkerOp,
    current: usize,
    total: usize,
}

enum DoneKind {
    Tagger(Option<Tagger>),
    Captioner(Option<Captioner>),
    Booru,
}

enum WorkerMsg {
    Progress(Progress),
    TaggerResult {
        path: PathBuf,
        tags: Vec<AutoTag>,
        model: String,
        ts: DateTime<Utc>,
    },
    CaptionerResult {
        path: PathBuf,
        entries: Vec<(String, String)>,
    },
    BooruResult {
        path: PathBuf,
        tags: Vec<BooruTag>,
        info: BooruInfo,
    },
    Error(String),
    Done(DoneKind),
}

struct AnimaTaggerApp {
    folder: Option<PathBuf>,
    images: Vec<ImageItem>,
    selected: HashSet<PathBuf>,
    filter: Filter,
    tag_filter: String,
    tag_input: String,
    loading: bool,
    error_msg: Option<String>,
    tagger: Option<Tagger>,
    captioner: Option<Captioner>,

    // Modal: config editor
    config_open: bool,
    config_draft: Option<ConfigDraft>,
    config_tab: ConfigTab,
    config_error: Option<String>,
    // Resolved target for the config modal: an ancestor's
    // `anima-tagger.toml` if one exists, otherwise the path where a new
    // file would be created in the current folder. `None` while the
    // modal is closed or no folder is loaded.
    config_path: Option<PathBuf>,

    // Localization
    lang: Lang,

    // Per-image text-edit buffers, persisted across frames so the user's
    // typing isn't clobbered every redraw. Re-initialized from the
    // sidecar when the selected image changes.
    manual_caption_buf: HashMap<PathBuf, String>,
    caption_hint_buf: HashMap<PathBuf, String>,
    last_single: Option<PathBuf>,

    // Bulk edit state — re-initialized when the selection signature
    // changes.
    bulk_hint_buf: String,
    bulk_signature: u64,

    // GPU texture handles for thumbnails.
    thumbnails: HashMap<PathBuf, TextureHandle>,

    // Background-worker progress feed. `worker_rx.is_some()` is the
    // single source of truth for "an op is in flight"; once Done lands
    // it goes back to None and the action buttons re-enable.
    progress: Option<Progress>,
    worker_rx: Option<Receiver<WorkerMsg>>,

    // When Some, the next `update()` shows a confirmation modal before
    // removing these paths' image+sidecar files from disk.
    pending_delete: Option<Vec<PathBuf>>,

    // Cached effective ProjectConfig for the current folder. Loaded by
    // `load_folder` so the Kanban view can read `tag_groups` without
    // re-parsing TOML each frame. `None` when no folder is loaded or
    // the config failed to load (treated as empty).
    project_config: Option<ProjectConfig>,
    // Current main-area view mode.
    view_mode: ViewMode,
    // Active drag in the Kanban view, if any. The payload carries one or
    // more image paths (multi-select drag carries the whole selection).
    kanban_drag: Option<KanbanDrag>,
}

/// Main-area view mode. `Grid` is the existing thumbnail grid; `Kanban`
/// classifies images into one column per tag of the named tag_group plus
/// "unset" and "violation" columns. The group name is owned to keep
/// borrow lifetimes simple.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ViewMode {
    Grid,
    Kanban { group: String },
}

/// In-flight Kanban drag. Captured the first frame `Response::dragged()`
/// fires on a thumbnail; cleared on drop or when the drag is cancelled
/// (pointer released without hitting a column).
#[derive(Debug, Clone)]
struct KanbanDrag {
    paths: Vec<PathBuf>,
}

impl AnimaTaggerApp {
    fn new() -> Self {
        Self {
            folder: None,
            images: Vec::new(),
            selected: HashSet::new(),
            filter: Filter::All,
            tag_filter: String::new(),
            tag_input: String::new(),
            loading: false,
            error_msg: None,
            tagger: None,
            captioner: None,
            config_open: false,
            config_draft: None,
            config_tab: ConfigTab::default(),
            config_error: None,
            config_path: None,
            lang: load_pref_or_detect(),
            manual_caption_buf: HashMap::new(),
            caption_hint_buf: HashMap::new(),
            last_single: None,
            bulk_hint_buf: String::new(),
            bulk_signature: 0,
            thumbnails: HashMap::new(),
            progress: None,
            worker_rx: None,
            pending_delete: None,
            project_config: None,
            view_mode: ViewMode::Grid,
            kanban_drag: None,
        }
    }

    fn t(&self) -> T {
        T::new(self.lang)
    }

    fn load_folder(&mut self, ctx: &egui::Context, dir: &Path) {
        self.folder = Some(dir.to_path_buf());
        self.images.clear();
        self.thumbnails.clear();
        self.selected.clear();
        self.tagger = None;
        self.captioner = None;
        self.manual_caption_buf.clear();
        self.caption_hint_buf.clear();
        self.last_single = None;
        self.bulk_hint_buf.clear();
        self.bulk_signature = 0;
        self.kanban_drag = None;

        // Best-effort config load. A broken TOML is reported in the
        // banner; the app still functions in Grid mode without groups.
        match ProjectConfig::load_or_default(dir) {
            Ok(cfg) => self.project_config = Some(cfg),
            Err(e) => {
                self.project_config = None;
                self.error_msg = Some(format!("config load failed: {e}"));
            }
        }
        // Drop a stale Kanban view if its group no longer exists in the
        // newly-loaded config.
        if let ViewMode::Kanban { group } = &self.view_mode {
            let still_exists = self
                .project_config
                .as_ref()
                .map(|c| c.tag_groups.contains_key(group))
                .unwrap_or(false);
            if !still_exists {
                self.view_mode = ViewMode::Grid;
            }
        }

        for path in iter_images(dir) {
            let sidecar = Sidecar::load_or_default(&path).unwrap_or_default();
            if let Some(tex) = make_thumbnail_texture(&path, THUMB_SIZE, ctx) {
                self.thumbnails.insert(path.clone(), tex);
            }
            self.images.push(ImageItem { path, sidecar });
        }
    }
}

impl eframe::App for AnimaTaggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker();
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| self.ui_toolbar(ui, ctx));
        if let Some(err) = self.error_msg.clone() {
            egui::TopBottomPanel::top("error_banner").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(255, 180, 180), format!("⚠ {err}"));
                    if ui.small_button("×").clicked() {
                        self.error_msg = None;
                    }
                });
            });
        }
        egui::SidePanel::right("detail")
            .resizable(true)
            .default_width(380.0)
            .min_width(300.0)
            // Without a max_width, a multiline TextEdit with
            // desired_width(f32::INFINITY) inside the panel feeds an
            // infinite content width back into the panel's auto-size
            // logic and the panel keeps growing on every frame, eating
            // the thumbnail grid. Cap it to a sensible maximum.
            .max_width(600.0)
            .show(ctx, |ui| {
                // Reserve the add-input row at the bottom first so the
                // ScrollArea inside ui_detail doesn't claim every
                // pixel of vertical space and hide it.
                egui::TopBottomPanel::bottom("add_input_panel")
                    .resizable(false)
                    .show_inside(ui, |ui| {
                        ui.add_space(4.0);
                        self.ui_add_input(ui);
                        ui.add_space(4.0);
                    });
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    self.ui_detail(ui);
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.view_mode.clone() {
                ViewMode::Grid => self.ui_grid(ui),
                ViewMode::Kanban { group } => self.ui_kanban(ui, &group),
            }
        });
        if self.config_open {
            self.ui_config_modal(ctx);
        }
        if self.pending_delete.is_some() {
            self.ui_delete_modal(ctx);
        }
        if self.progress.is_some() {
            self.ui_progress_overlay(ctx);
        }
    }
}

// ───────── Toolbar ─────────

impl AnimaTaggerApp {
    fn ui_toolbar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let t = self.t();
        ui.horizontal_wrapped(|ui| {
            if ui.button(t.open_folder()).clicked() {
                if let Some(picked) = rfd::FileDialog::new().pick_folder() {
                    self.load_folder(ctx, &picked);
                }
            }
            let cfg_btn = ui
                .button(t.config_button())
                .on_hover_text(t.config_button_title());
            if cfg_btn.clicked() {
                // Walk up from the loaded folder to find the nearest
                // existing `anima-tagger.toml` so a config kept at the
                // dataset root is edited in place rather than getting
                // shadowed by a new sibling file in a subdirectory.
                // Falls back to the current folder when nothing exists
                // up the tree (new file will be created on save).
                let (path, draft, load_err) = match self.folder.as_ref() {
                    Some(p) => {
                        let resolved = ProjectConfig::find_project_config(p)
                            .unwrap_or_else(|| p.join(CONFIG_FILE));
                        let (cfg, err) = if resolved.exists() {
                            match fs::read_to_string(&resolved)
                                .map_err(|e| e.to_string())
                                .and_then(|s| {
                                    toml::from_str::<ProjectConfig>(&s)
                                        .map_err(|e| e.to_string())
                                }) {
                                Ok(c) => (c, None),
                                Err(e) => (ProjectConfig::default(), Some(e)),
                            }
                        } else {
                            (ProjectConfig::default(), None)
                        };
                        (Some(resolved), ConfigDraft::from_config(cfg), err)
                    }
                    None => (
                        None,
                        ConfigDraft::from_config(ProjectConfig::default()),
                        None,
                    ),
                };
                self.config_path = path;
                self.config_draft = Some(draft);
                self.config_tab = ConfigTab::default();
                self.config_error = load_err.map(|e| t.cfg_err_load(&e));
                self.config_open = true;
            }

            // Folder name
            let folder_label = match self.folder.as_ref() {
                Some(p) => p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(String::from)
                    .unwrap_or_else(|| p.display().to_string()),
                None => t.no_folder().to_string(),
            };
            ui.label(folder_label);

            ui.separator();

            // Filter dropdown
            egui::ComboBox::from_id_salt("filter_combo")
                .selected_text(self.filter.label(t))
                .show_ui(ui, |ui| {
                    for f in Filter::ALL {
                        ui.selectable_value(&mut self.filter, f, f.label(t));
                    }
                });

            // View dropdown — Grid plus one entry per [tag_group.<name>].
            let group_names: Vec<String> = self
                .project_config
                .as_ref()
                .map(|c| c.tag_groups.keys().cloned().collect())
                .unwrap_or_default();
            let view_label = match &self.view_mode {
                ViewMode::Grid => t.view_grid().to_string(),
                ViewMode::Kanban { group } => {
                    format!("{}{group}", t.view_kanban_prefix())
                }
            };
            let view_resp = egui::ComboBox::from_id_salt("view_combo")
                .selected_text(view_label)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(self.view_mode, ViewMode::Grid),
                            t.view_grid(),
                        )
                        .clicked()
                    {
                        self.view_mode = ViewMode::Grid;
                    }
                    for name in &group_names {
                        let active = matches!(
                            &self.view_mode,
                            ViewMode::Kanban { group } if group == name
                        );
                        let label = format!("{}{name}", t.view_kanban_prefix());
                        if ui.selectable_label(active, label).clicked() {
                            self.view_mode = ViewMode::Kanban {
                                group: name.clone(),
                            };
                        }
                    }
                })
                .response;
            if group_names.is_empty() {
                view_resp.on_hover_text(t.kanban_no_groups_hint());
            }

            // Tag filter input
            ui.add(
                egui::TextEdit::singleline(&mut self.tag_filter)
                    .hint_text(t.tag_filter_placeholder())
                    .desired_width(160.0),
            );

            let folder_set = self.folder.is_some();
            let has_sel = !self.selected.is_empty();

            if ui
                .add_enabled(folder_set, egui::Button::new(t.select_visible()))
                .clicked()
            {
                let visible: HashSet<PathBuf> = self
                    .images
                    .iter()
                    .filter(|i| self.filter.matches(i))
                    .filter(|i| {
                        self.tag_filter.trim().is_empty()
                            || matches_tag_query(i, &self.tag_filter.trim().to_lowercase())
                    })
                    .map(|i| i.path.clone())
                    .collect();
                self.selected = visible;
            }
            if ui
                .add_enabled(has_sel, egui::Button::new(t.clear_selection()))
                .clicked()
            {
                self.selected.clear();
            }

            ui.separator();

            let can_run = has_sel && !self.loading;
            if ui
                .add_enabled(can_run, egui::Button::new(t.run_tagger()))
                .clicked()
            {
                self.run_tagger(ctx);
            }
            if ui
                .add_enabled(can_run, egui::Button::new(t.run_captioner()))
                .clicked()
            {
                self.run_captioner(ctx);
            }
            if ui
                .add_enabled(can_run, egui::Button::new(t.fetch_booru()))
                .clicked()
            {
                self.run_booru(ctx);
            }

            ui.separator();

            // Language selector
            let lang_label = match self.lang {
                Lang::En => "English",
                Lang::Ja => "日本語",
            };
            egui::ComboBox::from_id_salt("lang_combo")
                .selected_text(lang_label)
                .width(96.0)
                .show_ui(ui, |ui| {
                    let mut new_lang = self.lang;
                    ui.selectable_value(&mut new_lang, Lang::En, "English");
                    ui.selectable_value(&mut new_lang, Lang::Ja, "日本語");
                    if new_lang != self.lang {
                        self.lang = new_lang;
                        save_pref(new_lang);
                    }
                });

            if self.loading {
                ui.label(t.working());
            }
            ui.label(
                t.images_selected_summary(self.images.len(), self.selected.len()),
            );
        });
    }
}

// ───────── Grid ─────────

impl AnimaTaggerApp {
    fn ui_grid(&mut self, ui: &mut egui::Ui) {
        let t = self.t();
        let visible: Vec<PathBuf> = self
            .images
            .iter()
            .filter(|i| self.filter.matches(i))
            .filter(|i| {
                self.tag_filter.trim().is_empty()
                    || matches_tag_query(i, &self.tag_filter.trim().to_lowercase())
            })
            .map(|i| i.path.clone())
            .collect();

        if visible.is_empty() {
            ui.centered_and_justified(|ui| ui.label(t.no_images()));
            return;
        }

        let modifiers = ui.input(|i| i.modifiers);

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                let cell = THUMB_DRAW_PX + 12.0;
                let cols = ((ui.available_width() / cell).floor() as usize).max(1);
                egui::Grid::new("thumb_grid")
                    .spacing([6.0, 6.0])
                    .show(ui, |ui| {
                        for (i, path) in visible.iter().enumerate() {
                            self.ui_thumb(ui, path, modifiers);
                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
            });
    }

    fn ui_thumb(&mut self, ui: &mut egui::Ui, path: &Path, mods: egui::Modifiers) {
        let texture = match self.thumbnails.get(path) {
            Some(t) => t.clone(),
            None => return,
        };
        let item = match self.images.iter().find(|i| i.path == path) {
            Some(it) => it,
            None => return,
        };
        let is_selected = self.selected.contains(path);

        let frame = egui::Frame::group(ui.style())
            .inner_margin(2.0)
            .stroke(if is_selected {
                egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
            } else {
                egui::Stroke::new(2.0, egui::Color32::TRANSPARENT)
            });

        let response = frame
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    let img = egui::Image::new(&texture)
                        .fit_to_exact_size(egui::vec2(THUMB_DRAW_PX, THUMB_DRAW_PX));
                    ui.add(img);
                    ui.label(
                        egui::RichText::new(status_flags(&item.sidecar))
                            .size(10.0)
                            .monospace(),
                    )
                    .on_hover_text(self.t().thumb_status_title());
                });
            })
            .response
            .interact(egui::Sense::click());

        if response.clicked() {
            let multi = mods.command || mods.shift || mods.ctrl;
            if multi {
                if is_selected {
                    self.selected.remove(path);
                } else {
                    self.selected.insert(path.to_path_buf());
                }
            } else {
                self.selected.clear();
                self.selected.insert(path.to_path_buf());
            }
        }
    }

    /// Render the Kanban view for `group_name`. One column per tag in
    /// the group, plus an "unset" column and a "violation" column.
    /// Thumbnails are draggable; dropping into a tag column or "unset"
    /// rewrites `manual_tags` via [`tag_group::apply_drop`]. The
    /// "violation" column is intentionally not a drop target.
    fn ui_kanban(&mut self, ui: &mut egui::Ui, group_name: &str) {
        let t = self.t();

        // Resolve the group from the cached config. If it disappeared
        // (TOML edited live, group renamed, etc.), fall back to Grid.
        let group = match self
            .project_config
            .as_ref()
            .and_then(|c| c.tag_groups.get(group_name).cloned())
        {
            Some(g) => g,
            None => {
                self.view_mode = ViewMode::Grid;
                ui.centered_and_justified(|ui| ui.label(t.kanban_no_groups_hint()));
                return;
            }
        };

        // Bucket images by classification. Each bucket holds the image
        // paths in load order. Same `Filter` and tag-search filtering as
        // the grid view, so the toolbar controls keep their meaning.
        let mut by_tag: Vec<(String, Vec<PathBuf>)> =
            group.tags.iter().map(|t| (t.clone(), Vec::new())).collect();
        let mut unset: Vec<PathBuf> = Vec::new();
        let mut violation: Vec<PathBuf> = Vec::new();
        let tag_filter = self.tag_filter.trim().to_lowercase();

        for item in &self.images {
            if !self.filter.matches(item) {
                continue;
            }
            if !tag_filter.is_empty() && !matches_tag_query(item, &tag_filter) {
                continue;
            }
            match tag_group::classify(&item.sidecar, &group) {
                Classification::Tag(tag) => {
                    if let Some(slot) = by_tag.iter_mut().find(|(name, _)| *name == tag) {
                        slot.1.push(item.path.clone());
                    } else {
                        violation.push(item.path.clone());
                    }
                }
                Classification::Unset => unset.push(item.path.clone()),
                Classification::Violation(_) => violation.push(item.path.clone()),
            }
        }

        let mods = ui.input(|i| i.modifiers);
        let column_count = by_tag.len() + 2;
        let column_w =
            ((ui.available_width() - 12.0 * column_count as f32) / column_count as f32).max(160.0);

        // First column whose rect contained the pointer at the moment
        // the drag was released — that's where the drop lands. None
        // means either no drop happened this frame, or the release
        // happened over a non-drop-target column / outside the panel.
        let mut drop_target: Option<DropTarget> = None;

        egui::ScrollArea::horizontal()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    for (name, paths) in &by_tag {
                        if let Some(t) = self.ui_kanban_column(
                            ui,
                            name,
                            name,
                            paths,
                            column_w,
                            mods,
                            Some(DropTarget::Tag(name.clone())),
                        ) {
                            drop_target = Some(t);
                        }
                    }
                    if let Some(t) = self.ui_kanban_column(
                        ui,
                        "__unset__",
                        t.kanban_unset_column(),
                        &unset,
                        column_w,
                        mods,
                        Some(DropTarget::Unset),
                    ) {
                        drop_target = Some(t);
                    }
                    // Violation column is read-only — drop target = None.
                    self.ui_kanban_column(
                        ui,
                        "__violation__",
                        t.kanban_violation_column(),
                        &violation,
                        column_w,
                        mods,
                        None,
                    );
                });
            });

        // Resolve the drag at end-of-frame so all columns have rendered
        // and we know which one (if any) the pointer was over on release.
        if let Some(target) = drop_target
            && let Some(drag) = self.kanban_drag.take()
        {
            self.apply_kanban_drop(&group, &target, &drag.paths);
        } else if ui.input(|i| i.pointer.any_released()) {
            // Released without hitting a drop target — abandon the drag.
            self.kanban_drag = None;
        }
    }

    /// Render one Kanban column. `id` is a stable per-column identifier
    /// used for egui scroll-area / drop-target IDs; `heading` is the
    /// human label shown at the top. `drop_target` is `None` for
    /// read-only columns (the violation bucket); columns with `Some`
    /// participate in drag-and-drop and return that target if the
    /// pointer was over the column when the drag was released this
    /// frame.
    #[allow(clippy::too_many_arguments)]
    fn ui_kanban_column(
        &mut self,
        ui: &mut egui::Ui,
        id: &str,
        heading: &str,
        paths: &[PathBuf],
        width: f32,
        mods: egui::Modifiers,
        drop_target: Option<DropTarget>,
    ) -> Option<DropTarget> {
        let dragging = self.kanban_drag.is_some() && drop_target.is_some();
        let column_response = ui
            .allocate_ui(egui::vec2(width, ui.available_height()), |ui| {
                let frame = egui::Frame::group(ui.style())
                    .inner_margin(6.0)
                    .stroke(if dragging && ui.rect_contains_pointer(ui.max_rect()) {
                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                    } else {
                        ui.visuals().widgets.noninteractive.bg_stroke
                    });
                frame
                    .show(ui, |ui| {
                        ui.set_min_width(width);
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{heading} ({})", paths.len()))
                                    .strong(),
                            );
                            ui.separator();
                            egui::ScrollArea::vertical()
                                .id_salt(("kanban_col", id))
                                .auto_shrink([false; 2])
                                .show(ui, |ui| {
                                    for path in paths {
                                        if drop_target.is_some() {
                                            self.ui_kanban_thumb(ui, path, mods);
                                        } else {
                                            // Violation column: clickable
                                            // for selection but not draggable.
                                            self.ui_thumb(ui, path, mods);
                                        }
                                    }
                                });
                        });
                    })
                    .response
            })
            .response;

        if let Some(target) = drop_target
            && self.kanban_drag.is_some()
            && column_response.rect.contains(
                ui.input(|i| i.pointer.hover_pos())
                    .unwrap_or(egui::Pos2::ZERO),
            )
            && ui.input(|i| i.pointer.any_released())
        {
            return Some(target);
        }
        None
    }

    /// Like `ui_thumb`, but with click-and-drag sensing so dragging a
    /// thumbnail starts a Kanban move. Click behavior (selection) is
    /// preserved — egui distinguishes click from drag automatically.
    fn ui_kanban_thumb(&mut self, ui: &mut egui::Ui, path: &Path, mods: egui::Modifiers) {
        let texture = match self.thumbnails.get(path) {
            Some(t) => t.clone(),
            None => return,
        };
        let item = match self.images.iter().find(|i| i.path == path) {
            Some(it) => it,
            None => return,
        };
        let is_selected = self.selected.contains(path);

        let frame = egui::Frame::group(ui.style())
            .inner_margin(2.0)
            .stroke(if is_selected {
                egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
            } else {
                egui::Stroke::new(2.0, egui::Color32::TRANSPARENT)
            });

        let response = frame
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    let img = egui::Image::new(&texture)
                        .fit_to_exact_size(egui::vec2(THUMB_DRAW_PX, THUMB_DRAW_PX));
                    ui.add(img);
                    ui.label(
                        egui::RichText::new(status_flags(&item.sidecar))
                            .size(10.0)
                            .monospace(),
                    )
                    .on_hover_text(self.t().thumb_status_title());
                });
            })
            .response
            .interact(egui::Sense::click_and_drag());

        if response.clicked() {
            let multi = mods.command || mods.shift || mods.ctrl;
            if multi {
                if is_selected {
                    self.selected.remove(path);
                } else {
                    self.selected.insert(path.to_path_buf());
                }
            } else {
                self.selected.clear();
                self.selected.insert(path.to_path_buf());
            }
        }
        if response.drag_started() {
            // If the dragged thumb is one of several selected, carry
            // the whole selection. Otherwise carry just this one (and
            // leave the prior selection intact).
            let paths: Vec<PathBuf> = if is_selected && self.selected.len() > 1 {
                self.selected.iter().cloned().collect()
            } else {
                vec![path.to_path_buf()]
            };
            self.kanban_drag = Some(KanbanDrag { paths });
        }
    }

    /// Apply a Kanban drop: mutate each path's in-memory sidecar via
    /// [`tag_group::apply_drop`] and persist it to disk. Errors surface
    /// in the top error banner; remaining paths are still attempted so
    /// a single broken file doesn't abort the whole drop.
    fn apply_kanban_drop(
        &mut self,
        group: &TagGroup,
        target: &DropTarget,
        paths: &[PathBuf],
    ) {
        let t = self.t();
        for path in paths {
            let Some(item) = self.images.iter_mut().find(|i| i.path == *path) else {
                continue;
            };
            tag_group::apply_drop(&mut item.sidecar, group, target);
            let save_result = item.sidecar.save(&item.path);
            let path_str = item.path.display().to_string();
            if let Err(e) = save_result {
                self.error_msg = Some(t.kanban_drop_failed(&path_str, &e.to_string()));
            }
        }
    }
}

fn status_flags(s: &Sidecar) -> String {
    let t = if s.is_auto_tagged() { 'T' } else { ' ' };
    let c = if s.is_captioned() { 'C' } else { ' ' };
    let b = if s.has_booru() { 'B' } else { ' ' };
    let m = if !s.manual_tags.is_empty() { 'M' } else { ' ' };
    let h = if s.caption_hint.is_some() { 'H' } else { ' ' };
    format!("{t}{c}{b}{m}{h}")
}

// ───────── Detail panel ─────────

impl AnimaTaggerApp {
    fn ui_detail(&mut self, ui: &mut egui::Ui) {
        let t = self.t();
        let sel: Vec<PathBuf> = self.selected.iter().cloned().collect();
        let n = sel.len();

        if n == 0 {
            self.last_single = None;
            ui.label(t.select_to_edit());
            ui.add_space(6.0);
            ui.label(egui::RichText::new(t.tip_suppress()).small().weak());
            return;
        }

        if n == 1 {
            let path = sel[0].clone();
            self.refresh_single_buffers_if_needed(&path);
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.ui_single_detail(ui, &path);
                });
        } else {
            self.last_single = None;
            let signature = bulk_signature(&sel);
            if self.bulk_signature != signature {
                self.bulk_signature = signature;
                self.bulk_hint_buf = canonical_bulk_hint(&self.images, &sel);
            }
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    self.ui_bulk_detail(ui, &sel);
                });
        }
    }

    fn refresh_single_buffers_if_needed(&mut self, path: &Path) {
        if self.last_single.as_deref() != Some(path) {
            self.last_single = Some(path.to_path_buf());
            if let Some(item) = self.images.iter().find(|i| i.path == path) {
                self.manual_caption_buf.insert(
                    path.to_path_buf(),
                    item.sidecar.manual_caption.clone().unwrap_or_default(),
                );
                self.caption_hint_buf.insert(
                    path.to_path_buf(),
                    item.sidecar.caption_hint.clone().unwrap_or_default(),
                );
            }
        }
    }

    fn ui_add_input(&mut self, ui: &mut egui::Ui) {
        let t = self.t();
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.tag_input)
                    .hint_text(t.add_input_placeholder())
                    .desired_width(ui.available_width() - 60.0),
            );
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            let click = ui.button(t.add_button()).clicked();
            if enter || click {
                let v = std::mem::take(&mut self.tag_input);
                let trimmed = v.trim().to_string();
                if !trimmed.is_empty() {
                    self.add_manual_tag_to_selected(&trimmed);
                }
                if enter {
                    r.request_focus();
                }
            }
        });
    }

    fn add_manual_tag_to_selected(&mut self, tag: &str) {
        let sel = self.selected.clone();
        for img in self.images.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            if img.sidecar.add_manual_tag(tag.to_string()) {
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
}

// ───────── Single-image detail ─────────

impl AnimaTaggerApp {
    fn ui_single_detail(&mut self, ui: &mut egui::Ui, path: &Path) {
        let t = self.t();
        let item = match self.images.iter().find(|i| i.path == path) {
            Some(it) => it.clone(),
            None => return,
        };

        let filename = item
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        ui.label(egui::RichText::new(filename).monospace().weak());

        ui.add_space(6.0);
        section_title(ui, t.section_tags());
        let manual_positives: Vec<String> = item
            .sidecar
            .manual_positive_tags()
            .map(|s| s.to_string())
            .collect();
        if manual_positives.is_empty()
            && item.sidecar.auto_tags.is_empty()
            && item.sidecar.booru_tags.is_empty()
        {
            ui.weak(t.empty_tags());
        } else {
            let mut to_remove_manual: Vec<String> = Vec::new();
            let mut to_toggle_suppression: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for tag in &manual_positives {
                    if chip(ui, tag, ChipKind::Manual, false) {
                        to_remove_manual.push(tag.clone());
                    }
                }
                for at in &item.sidecar.auto_tags {
                    let suppressed = item.sidecar.is_suppressed(&at.tag);
                    if chip(
                        ui,
                        &format!("{} ({:.2})", at.tag, at.score),
                        ChipKind::Auto,
                        suppressed,
                    ) {
                        to_toggle_suppression.push(at.tag.clone());
                    }
                }
                for bt in &item.sidecar.booru_tags {
                    let suppressed = item.sidecar.is_suppressed(&bt.tag);
                    if chip(
                        ui,
                        &format!("{} [B]", bt.tag),
                        ChipKind::Booru,
                        suppressed,
                    ) {
                        to_toggle_suppression.push(bt.tag.clone());
                    }
                }
            });
            for tag in to_remove_manual {
                self.remove_manual_at(path, &tag);
            }
            for tag in to_toggle_suppression {
                self.toggle_suppression_at(path, &tag);
            }
        }

        ui.add_space(6.0);
        section_title(ui, t.section_caption_hint());
        let path_owned = path.to_path_buf();
        let avail = ui.available_width();
        let buf = self
            .caption_hint_buf
            .entry(path_owned.clone())
            .or_default();
        let r = ui.add(
            egui::TextEdit::multiline(buf)
                .desired_width(avail)
                .desired_rows(3)
                .hint_text(t.caption_hint_placeholder()),
        );
        if r.lost_focus() {
            let new_text = buf.clone();
            self.save_caption_hint(path, &new_text);
        }

        ui.add_space(6.0);
        section_title(ui, t.section_manual_caption());
        let avail = ui.available_width();
        let buf = self
            .manual_caption_buf
            .entry(path_owned.clone())
            .or_default();
        let r = ui.add(
            egui::TextEdit::multiline(buf)
                .desired_width(avail)
                .desired_rows(3)
                .hint_text(t.manual_caption_placeholder()),
        );
        if r.lost_focus() {
            let new_text = buf.clone();
            self.save_manual_caption(path, &new_text);
        }

        ui.add_space(6.0);
        section_title(ui, t.section_auto_captions());
        if item.sidecar.captions.is_empty() {
            ui.weak(t.empty_auto_captions());
        } else {
            let mut to_promote: Vec<(String, String)> = Vec::new();
            let mut to_toggle_skip: Vec<String> = Vec::new();
            let mut to_remove_caption: Vec<String> = Vec::new();
            for (model, entry) in item.sidecar.captions.iter() {
                let frame = egui::Frame::group(ui.style())
                    .inner_margin(egui::Margin::same(6))
                    .stroke(if entry.skip {
                        egui::Stroke::new(1.0, egui::Color32::DARK_GRAY)
                    } else {
                        egui::Stroke::new(1.0, egui::Color32::from_gray(60))
                    });
                frame.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(model).monospace().small().weak());
                        if ui
                            .small_button(t.promote_to_manual())
                            .on_hover_text(t.promote_to_manual_title())
                            .clicked()
                        {
                            to_promote.push((model.clone(), entry.caption.clone()));
                        }
                        let skip_label = if entry.skip { t.unskip() } else { t.skip() };
                        let skip_title = if entry.skip {
                            t.unskip_title()
                        } else {
                            t.skip_title()
                        };
                        if ui
                            .small_button(skip_label)
                            .on_hover_text(skip_title)
                            .clicked()
                        {
                            to_toggle_skip.push(model.clone());
                        }
                        if ui
                            .small_button("×")
                            .on_hover_text(t.remove_caption_title())
                            .clicked()
                        {
                            to_remove_caption.push(model.clone());
                        }
                    });
                    let caption_text = if entry.skip {
                        egui::RichText::new(&entry.caption).strikethrough().weak()
                    } else {
                        egui::RichText::new(&entry.caption)
                    };
                    ui.label(caption_text);
                });
            }
            for (model, text) in to_promote {
                self.copy_caption_to_manual(path, &text);
                let _ = model;
            }
            for model in to_toggle_skip {
                self.toggle_caption_skip_at(path, &model);
            }
            for model in to_remove_caption {
                self.remove_caption_at(path, &model);
            }
        }

        if let Some(b) = item.sidecar.booru.as_ref() {
            ui.add_space(6.0);
            section_title(ui, t.section_booru());
            let label = if let Some(id) = b.post_id {
                format!("{}: #{id}", b.source)
            } else {
                b.source.clone()
            };
            ui.weak(label);
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .button(egui::RichText::new(t.delete_image()).color(egui::Color32::from_rgb(220, 90, 90)))
            .on_hover_text(t.delete_image_title())
            .clicked()
        {
            self.pending_delete = Some(vec![path.to_path_buf()]);
        }
    }
}

// ───────── Bulk detail ─────────

impl AnimaTaggerApp {
    fn ui_bulk_detail(&mut self, ui: &mut egui::Ui, sel: &[PathBuf]) {
        let t = self.t();
        let n = sel.len();
        let selected_items: Vec<ImageItem> = self
            .images
            .iter()
            .filter(|i| sel.contains(&i.path))
            .cloned()
            .collect();

        ui.weak(t.n_selected_bulk(n));

        ui.add_space(6.0);
        section_title(ui, t.section_bulk_caption_hint());
        let hint_values: Vec<&str> = selected_items
            .iter()
            .map(|i| i.sidecar.caption_hint.as_deref().unwrap_or(""))
            .collect();
        let hints_uniform = hint_values.iter().all(|v| *v == hint_values[0]);
        if !hints_uniform {
            ui.add(egui::Label::new(
                egui::RichText::new(t.bulk_hints_differ()).small().weak(),
            ));
        }
        let avail = ui.available_width();
        ui.add(
            egui::TextEdit::multiline(&mut self.bulk_hint_buf)
                .desired_width(avail)
                .desired_rows(3)
                .hint_text(t.bulk_hint_placeholder()),
        );
        ui.horizontal(|ui| {
            if ui.button(t.bulk_hint_apply()).clicked() {
                let text = self.bulk_hint_buf.clone();
                self.bulk_set_caption_hint(sel, &text);
            }
            if ui.button(t.bulk_hint_clear()).clicked() {
                self.bulk_hint_buf.clear();
                self.bulk_set_caption_hint(sel, "");
            }
        });

        ui.add_space(6.0);
        section_title(ui, t.section_manual_entries());
        let mut manual_order: Vec<String> = Vec::new();
        let mut manual_counts: HashMap<String, usize> = HashMap::new();
        for item in &selected_items {
            for tag in &item.sidecar.manual_tags {
                if !manual_counts.contains_key(tag) {
                    manual_order.push(tag.clone());
                }
                *manual_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        if manual_order.is_empty() {
            ui.weak(t.empty_simple());
        } else {
            let mut to_remove: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for tag in &manual_order {
                    let count = manual_counts[tag];
                    let label = if count < n {
                        format!("{tag} ({count}/{n})")
                    } else {
                        tag.clone()
                    };
                    let kind = if tag.starts_with('-') {
                        ChipKind::Negative
                    } else {
                        ChipKind::Manual
                    };
                    if chip(ui, &label, kind, false) {
                        to_remove.push(tag.clone());
                    }
                }
            });
            for tag in to_remove {
                self.bulk_remove_manual(sel, &tag);
            }
        }

        ui.add_space(6.0);
        section_title(ui, t.section_common_tags());
        let common = compute_common_tags(&selected_items);
        if common.is_empty() {
            ui.add(egui::Label::new(
                egui::RichText::new(t.empty_simple()).small().weak(),
            ));
        } else {
            ui.horizontal_wrapped(|ui| {
                for (tag, count) in &common {
                    // Common-tag readout — read-only summary, not
                    // actionable.
                    let _ = chip(ui, &format!("{tag} ({count}/{n})"), ChipKind::Auto, false);
                }
            });
        }

        ui.add_space(6.0);
        section_title(ui, t.section_bulk_manual_caption());
        if ui
            .button(t.bulk_clear_manual())
            .on_hover_text(t.bulk_clear_manual_title())
            .clicked()
        {
            self.bulk_clear_manual_caption(sel);
        }

        ui.add_space(6.0);
        section_title(ui, t.section_bulk_auto_captions());
        let mut caption_models: Vec<String> = Vec::new();
        let mut caption_counts: HashMap<String, usize> = HashMap::new();
        for item in &selected_items {
            for model in item.sidecar.captions.keys() {
                if !caption_counts.contains_key(model) {
                    caption_models.push(model.clone());
                }
                *caption_counts.entry(model.clone()).or_insert(0) += 1;
            }
        }
        caption_models.sort();
        if caption_models.is_empty() {
            ui.add(egui::Label::new(
                egui::RichText::new(t.empty_simple()).small().weak(),
            ));
        } else {
            let mut to_promote: Vec<String> = Vec::new();
            let mut to_remove: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for model in &caption_models {
                    let count = caption_counts[model];
                    let label = format!("{model} ({count}/{n})");
                    ui.group(|ui| {
                        ui.label(label);
                        if ui
                            .small_button(t.promote_to_manual())
                            .on_hover_text(t.bulk_promote_title())
                            .clicked()
                        {
                            to_promote.push(model.clone());
                        }
                        if ui
                            .small_button("×")
                            .on_hover_text(t.bulk_remove_caption_title())
                            .clicked()
                        {
                            to_remove.push(model.clone());
                        }
                    });
                }
            });
            for model in to_promote {
                self.bulk_promote_to_manual(sel, &model);
            }
            for model in to_remove {
                self.bulk_remove_caption(sel, &model);
            }
        }

        ui.add_space(6.0);
        ui.add(egui::Label::new(
            egui::RichText::new(t.switch_to_single_hint()).small().weak(),
        ));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .button(egui::RichText::new(t.delete_images()).color(egui::Color32::from_rgb(220, 90, 90)))
            .on_hover_text(t.delete_image_title())
            .clicked()
        {
            self.pending_delete = Some(sel.to_vec());
        }
    }
}

// ───────── Config modal ─────────

impl AnimaTaggerApp {
    fn ui_config_modal(&mut self, ctx: &egui::Context) {
        let t = self.t();
        let target_label = match self.config_path.as_ref() {
            Some(p) => p.display().to_string(),
            None => t.no_folder().to_string(),
        };
        let Some(draft) = self.config_draft.as_mut() else {
            // Defensive: modal should only be open with a draft present.
            self.config_open = false;
            return;
        };
        let action = show_config_modal(
            ctx,
            t,
            &target_label,
            draft,
            &mut self.config_tab,
            &mut self.config_error,
        );
        match action {
            ConfigAction::None => {}
            ConfigAction::Cancel => {
                self.config_open = false;
                self.config_error = None;
                self.config_draft = None;
                self.config_path = None;
            }
            ConfigAction::Save => {
                let Some(draft) = self.config_draft.as_ref() else {
                    return;
                };
                let cfg = match draft.to_config(t) {
                    Ok(c) => c,
                    Err(e) => {
                        self.config_error = Some(e);
                        return;
                    }
                };
                let toml_text = match toml::to_string_pretty(&cfg) {
                    Ok(s) => s,
                    Err(e) => {
                        self.config_error = Some(format!("serialize: {e}"));
                        return;
                    }
                };
                let Some(target) = self.config_path.clone() else {
                    self.error_msg = Some(t.err_open_folder_first());
                    return;
                };
                if let Some(parent) = target.parent()
                    && let Err(e) = fs::create_dir_all(parent)
                {
                    self.config_error = Some(format!("create {}: {e}", parent.display()));
                    return;
                }
                if let Err(e) = fs::write(&target, toml_text.as_bytes()) {
                    self.config_error = Some(format!("write {}: {e}", target.display()));
                    return;
                }
                // Drop cached models so the next run resolves against
                // the new profile, and refresh the in-memory effective
                // config so Kanban / view dropdowns pick up tag-group
                // changes immediately.
                self.tagger = None;
                self.captioner = None;
                if let Some(folder) = self.folder.clone() {
                    match ProjectConfig::load_or_default(&folder) {
                        Ok(c) => self.project_config = Some(c),
                        Err(e) => self.error_msg = Some(format!("config reload: {e}")),
                    }
                }
                self.config_error = None;
                self.config_open = false;
                self.config_draft = None;
                self.config_path = None;
            }
        }
    }
}

// ───────── Delete confirmation ─────────

impl AnimaTaggerApp {
    fn ui_delete_modal(&mut self, ctx: &egui::Context) {
        let t = self.t();
        let Some(paths) = self.pending_delete.clone() else {
            return;
        };
        let n = paths.len();
        let mut do_delete = false;
        let mut do_cancel = false;
        let mut open = true;
        egui::Window::new(t.delete_confirm_title())
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_min_width(360.0);
                ui.label(t.delete_confirm_body(n));
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .max_height(160.0)
                    .show(ui, |ui| {
                        for p in &paths {
                            let name = p
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or_default();
                            ui.label(egui::RichText::new(name).monospace().small().weak());
                        }
                    });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button(t.delete_confirm_cancel()).clicked() {
                        do_cancel = true;
                    }
                    if ui
                        .button(
                            egui::RichText::new(t.delete_confirm_ok())
                                .color(egui::Color32::from_rgb(220, 90, 90)),
                        )
                        .clicked()
                    {
                        do_delete = true;
                    }
                });
            });
        if !open || do_cancel {
            self.pending_delete = None;
            return;
        }
        if do_delete {
            self.delete_paths(&paths);
            self.pending_delete = None;
        }
    }

    fn delete_paths(&mut self, paths: &[PathBuf]) {
        let mut errors: Vec<String> = Vec::new();
        for p in paths {
            let sidecar = sidecar_path_for(p);
            if let Err(e) = fs::remove_file(p) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    errors.push(self.t().err_delete_failed(&p.display().to_string(), &e.to_string()));
                    continue;
                }
            }
            if sidecar.exists() {
                if let Err(e) = fs::remove_file(&sidecar) {
                    errors.push(
                        self.t()
                            .err_delete_failed(&sidecar.display().to_string(), &e.to_string()),
                    );
                }
            }
        }
        let to_remove: HashSet<PathBuf> = paths.iter().cloned().collect();
        self.images.retain(|i| !to_remove.contains(&i.path));
        self.selected.retain(|p| !to_remove.contains(p));
        for p in &to_remove {
            self.thumbnails.remove(p);
            self.manual_caption_buf.remove(p);
            self.caption_hint_buf.remove(p);
        }
        if let Some(last) = self.last_single.as_ref()
            && to_remove.contains(last)
        {
            self.last_single = None;
        }
        // Force bulk-edit buffers to re-derive against the new selection.
        self.bulk_signature = 0;
        if !errors.is_empty() {
            self.error_msg = Some(errors.join("; "));
        }
    }
}

// ───────── Long-running operations (background thread) ─────────
//
// Each run_* spawns a worker thread, ships any pre-loaded model into
// it, and stores the receiver. The UI keeps repainting via
// ctx.request_repaint() calls inside the worker, and `update()` polls
// the channel each frame (`poll_worker`). When the worker emits Done,
// the (possibly-new) model handle comes back through the channel and
// gets re-cached.

impl AnimaTaggerApp {
    fn run_tagger(&mut self, ctx: &egui::Context) {
        let t = self.t();
        let Some(folder) = self.folder.clone() else {
            self.error_msg = Some(t.err_open_folder_first());
            return;
        };
        let cfg = match ProjectConfig::load_or_default(&folder) {
            Ok(c) => c,
            Err(e) => {
                self.error_msg = Some(e.to_string());
                return;
            }
        };
        let (model_name, profile) = cfg.resolve_tagger(None);
        let sel_all: Vec<PathBuf> = self.selected.iter().cloned().collect();
        if sel_all.is_empty() {
            return;
        }
        // Skip images that already have auto-tags — matches the CLI's
        // default behavior (`anima-tagger tag` without --force).
        let already: std::collections::HashSet<PathBuf> = self
            .images
            .iter()
            .filter(|i| sel_all.contains(&i.path) && i.sidecar.is_auto_tagged())
            .map(|i| i.path.clone())
            .collect();
        let sel: Vec<PathBuf> = sel_all
            .iter()
            .filter(|p| !already.contains(*p))
            .cloned()
            .collect();
        let skipped = already.len();
        if sel.is_empty() {
            self.error_msg = Some(t.info_all_already_tagged().to_string());
            return;
        }
        if skipped > 0 {
            self.error_msg = Some(t.info_skipped_already_tagged(skipped));
        }
        let total = sel.len();
        let mut tagger = self.tagger.take();
        let storage_threshold = profile.storage_threshold;
        let profile_for_load = profile.clone();
        let ctx_clone = ctx.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            if tagger.is_none() {
                match Tagger::from_profile(&profile_for_load) {
                    Ok(t) => tagger = Some(t),
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Error(format!("tagger load: {e}")));
                        let _ = tx.send(WorkerMsg::Done(DoneKind::Tagger(None)));
                        ctx_clone.request_repaint();
                        return;
                    }
                }
            }
            let tagger_inst = tagger.as_mut().expect("loaded above");
            let now = Utc::now();
            for (i, path) in sel.iter().enumerate() {
                let _ = tx.send(WorkerMsg::Progress(Progress {
                    op: WorkerOp::Tagger,
                    current: i,
                    total,
                }));
                ctx_clone.request_repaint();
                match tagger_inst.tag_image(path, storage_threshold) {
                    Ok(tags) => {
                        let _ = tx.send(WorkerMsg::TaggerResult {
                            path: path.clone(),
                            tags,
                            model: model_name.clone(),
                            ts: now,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Error(format!("{}: {e}", path.display())));
                    }
                }
                ctx_clone.request_repaint();
            }
            let _ = tx.send(WorkerMsg::Progress(Progress {
                op: WorkerOp::Tagger,
                current: total,
                total,
            }));
            let _ = tx.send(WorkerMsg::Done(DoneKind::Tagger(tagger)));
            ctx_clone.request_repaint();
        });

        self.worker_rx = Some(rx);
        self.loading = true;
        self.progress = Some(Progress {
            op: WorkerOp::Tagger,
            current: 0,
            total,
        });
    }

    fn run_captioner(&mut self, ctx: &egui::Context) {
        let t = self.t();
        let Some(folder) = self.folder.clone() else {
            self.error_msg = Some(t.err_open_folder_first());
            return;
        };
        let cfg = match ProjectConfig::load_or_default(&folder) {
            Ok(c) => c,
            Err(e) => {
                self.error_msg = Some(e.to_string());
                return;
            }
        };
        let (model_name, profile) = cfg.resolve_captioner(None);
        let library = cfg.prompt_library();
        let prompts = match profile.resolved_prompts(&library) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(e.to_string());
                return;
            }
        };
        let sel_all: Vec<PathBuf> = self.selected.iter().cloned().collect();
        if sel_all.is_empty() {
            return;
        }
        // Skip per (image, prompt-key) pair: only ship images that have
        // at least one prompt-key not already present in the sidecar.
        // Mirrors the CLI's default behavior (`anima-tagger caption`
        // without --force).
        let prompt_keys: Vec<String> =
            prompts.iter().map(|(n, _)| format!("{model_name}.{n}")).collect();
        let existing_keys: HashMap<PathBuf, HashSet<String>> = self
            .images
            .iter()
            .filter(|i| sel_all.contains(&i.path))
            .map(|i| {
                (
                    i.path.clone(),
                    i.sidecar.captions.keys().cloned().collect::<HashSet<_>>(),
                )
            })
            .collect();
        let sel: Vec<PathBuf> = sel_all
            .iter()
            .filter(|p| {
                let have = existing_keys.get(*p);
                prompt_keys
                    .iter()
                    .any(|k| have.is_none_or(|s| !s.contains(k)))
            })
            .cloned()
            .collect();
        let skipped = sel_all.len() - sel.len();
        if sel.is_empty() {
            self.error_msg = Some(t.info_all_already_tagged().to_string());
            return;
        }
        if skipped > 0 {
            self.error_msg = Some(t.info_skipped_already_tagged(skipped));
        }
        let total = sel.len();
        let hints: HashMap<PathBuf, Option<String>> = self
            .images
            .iter()
            .filter(|i| sel.contains(&i.path))
            .map(|i| (i.path.clone(), i.sidecar.caption_hint.clone()))
            .collect();

        let mut captioner = self.captioner.take();
        let profile_for_load = profile.clone();
        let ctx_clone = ctx.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            if captioner.is_none() {
                match Captioner::from_profile(&profile_for_load) {
                    Ok(c) => captioner = Some(c),
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Error(format!("captioner load: {e}")));
                        let _ = tx.send(WorkerMsg::Done(DoneKind::Captioner(None)));
                        ctx_clone.request_repaint();
                        return;
                    }
                }
            }
            let captioner_inst = captioner.as_mut().expect("loaded above");
            for (i, path) in sel.iter().enumerate() {
                let _ = tx.send(WorkerMsg::Progress(Progress {
                    op: WorkerOp::Captioner,
                    current: i,
                    total,
                }));
                ctx_clone.request_repaint();
                let hint = hints.get(path).cloned().flatten();
                let have = existing_keys.get(path);
                let mut entries: Vec<(String, String)> = Vec::new();
                for (pname, ptext) in &prompts {
                    let key = format!("{model_name}.{pname}");
                    if have.is_some_and(|s| s.contains(&key)) {
                        continue;
                    }
                    match captioner_inst.caption_image(path, ptext, hint.as_deref()) {
                        Ok(caption) => entries.push((key, caption)),
                        Err(e) => {
                            let _ = tx.send(WorkerMsg::Error(format!(
                                "{} [{pname}]: {e}",
                                path.display()
                            )));
                        }
                    }
                }
                if !entries.is_empty() {
                    let _ = tx.send(WorkerMsg::CaptionerResult {
                        path: path.clone(),
                        entries,
                    });
                }
                ctx_clone.request_repaint();
            }
            let _ = tx.send(WorkerMsg::Progress(Progress {
                op: WorkerOp::Captioner,
                current: total,
                total,
            }));
            let _ = tx.send(WorkerMsg::Done(DoneKind::Captioner(captioner)));
            ctx_clone.request_repaint();
        });

        self.worker_rx = Some(rx);
        self.loading = true;
        self.progress = Some(Progress {
            op: WorkerOp::Captioner,
            current: 0,
            total,
        });
    }

    fn run_booru(&mut self, ctx: &egui::Context) {
        let t = self.t();
        let sel_all: Vec<PathBuf> = self.selected.iter().cloned().collect();
        if sel_all.is_empty() {
            return;
        }
        // Skip images that already have booru data — matches the CLI's
        // default behavior (`anima-tagger booru` without --force).
        let already: HashSet<PathBuf> = self
            .images
            .iter()
            .filter(|i| sel_all.contains(&i.path) && i.sidecar.has_booru())
            .map(|i| i.path.clone())
            .collect();
        let sel: Vec<PathBuf> = sel_all
            .iter()
            .filter(|p| !already.contains(*p))
            .cloned()
            .collect();
        let skipped = already.len();
        if sel.is_empty() {
            self.error_msg = Some(t.info_all_already_tagged().to_string());
            return;
        }
        if skipped > 0 {
            self.error_msg = Some(t.info_skipped_already_tagged(skipped));
        }
        let total = sel.len();
        let ctx_clone = ctx.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            let client = BooruClient::danbooru();
            for (i, path) in sel.iter().enumerate() {
                let _ = tx.send(WorkerMsg::Progress(Progress {
                    op: WorkerOp::Booru,
                    current: i,
                    total,
                }));
                ctx_clone.request_repaint();
                match client.fetch_for_image(path) {
                    Ok((tags, info)) => {
                        let _ = tx.send(WorkerMsg::BooruResult {
                            path: path.clone(),
                            tags,
                            info,
                        });
                    }
                    Err(BooruError::NotFound(_)) => {}
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Error(format!("{}: {e}", path.display())));
                    }
                }
                ctx_clone.request_repaint();
            }
            let _ = tx.send(WorkerMsg::Progress(Progress {
                op: WorkerOp::Booru,
                current: total,
                total,
            }));
            let _ = tx.send(WorkerMsg::Done(DoneKind::Booru));
            ctx_clone.request_repaint();
        });

        self.worker_rx = Some(rx);
        self.loading = true;
        self.progress = Some(Progress {
            op: WorkerOp::Booru,
            current: 0,
            total,
        });
    }

    fn poll_worker(&mut self) {
        if self.worker_rx.is_none() {
            return;
        }
        // Drain everything currently buffered. We can't hold a borrow
        // of self.worker_rx across the apply_worker_msg call (which
        // mutably borrows self), so each iteration grabs the receiver
        // briefly to try_recv, drops the borrow, then dispatches.
        loop {
            let recv = match self.worker_rx.as_ref() {
                Some(rx) => rx.try_recv(),
                None => break,
            };
            match recv {
                Ok(msg) => self.apply_worker_msg(msg),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Worker dropped without sending Done — clean up
                    // anyway so the UI doesn't get stuck on the
                    // progress overlay.
                    self.worker_rx = None;
                    self.progress = None;
                    self.loading = false;
                    break;
                }
            }
        }
    }

    fn apply_worker_msg(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Progress(p) => self.progress = Some(p),
            WorkerMsg::TaggerResult {
                path,
                tags,
                model,
                ts,
            } => {
                if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
                    img.sidecar.auto_tags = tags;
                    img.sidecar.tagger = Some(TaggerInfo {
                        model,
                        tagged_at: ts,
                    });
                    let _ = img.sidecar.save(&img.path);
                }
            }
            WorkerMsg::CaptionerResult { path, entries } => {
                if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
                    for (key, caption) in entries {
                        img.sidecar.set_caption(key, caption);
                    }
                    let _ = img.sidecar.save(&img.path);
                }
            }
            WorkerMsg::BooruResult { path, tags, info } => {
                if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
                    img.sidecar.booru_tags = tags;
                    img.sidecar.booru = Some(info);
                    let _ = img.sidecar.save(&img.path);
                }
            }
            WorkerMsg::Error(e) => {
                self.error_msg = Some(e);
            }
            WorkerMsg::Done(kind) => {
                match kind {
                    DoneKind::Tagger(t) => self.tagger = t,
                    DoneKind::Captioner(c) => self.captioner = c,
                    DoneKind::Booru => {}
                }
                self.progress = None;
                self.loading = false;
                self.worker_rx = None;
            }
        }
    }

    fn ui_progress_overlay(&self, ctx: &egui::Context) {
        let Some(p) = self.progress.clone() else {
            return;
        };
        let t = self.t();
        let label = match p.op {
            WorkerOp::Tagger => t.op_tagging(),
            WorkerOp::Captioner => t.op_captioning(),
            WorkerOp::Booru => t.op_fetching_booru(),
        };
        let frac = if p.total == 0 {
            0.0
        } else {
            (p.current as f32) / (p.total as f32)
        };
        egui::Window::new("anima-tagger-progress")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .title_bar(false)
            .show(ctx, |ui| {
                ui.set_min_width(280.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(4.0);
                    ui.heading(label);
                    ui.add_space(8.0);
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .desired_width(260.0)
                            .show_percentage(),
                    );
                    ui.add_space(4.0);
                    ui.label(t.progress_count(p.current, p.total));
                    ui.add_space(4.0);
                });
            });
    }
}

// ───────── Sidecar mutators ─────────

impl AnimaTaggerApp {
    fn save_manual_caption(&mut self, path: &Path, text: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
            img.sidecar.set_manual_caption(text);
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn save_caption_hint(&mut self, path: &Path, text: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
            img.sidecar.set_caption_hint(text);
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn copy_caption_to_manual(&mut self, path: &Path, text: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
            img.sidecar.set_manual_caption(text);
            let _ = img.sidecar.save(&img.path);
            self.manual_caption_buf
                .insert(path.to_path_buf(), text.to_string());
        }
    }
    fn remove_caption_at(&mut self, path: &Path, model: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path)
            && img.sidecar.remove_caption(model)
        {
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn toggle_caption_skip_at(&mut self, path: &Path, model: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path)
            && img.sidecar.toggle_caption_skip(model).is_some()
        {
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn remove_manual_at(&mut self, path: &Path, tag: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path)
            && img.sidecar.remove_manual_tag(tag)
        {
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn toggle_suppression_at(&mut self, path: &Path, tag: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path) {
            let changed = if img.sidecar.is_suppressed(tag) {
                img.sidecar.unsuppress(tag)
            } else {
                img.sidecar.suppress(tag)
            };
            if changed {
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
    fn bulk_remove_manual(&mut self, paths: &[PathBuf], tag: &str) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            if img.sidecar.remove_manual_tag(tag) {
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
    fn bulk_remove_caption(&mut self, paths: &[PathBuf], model: &str) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            if img.sidecar.remove_caption(model) {
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
    fn bulk_promote_to_manual(&mut self, paths: &[PathBuf], model: &str) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            let manual_empty = img
                .sidecar
                .manual_caption
                .as_deref()
                .map(str::trim)
                .map(|s| s.is_empty())
                .unwrap_or(true);
            if !manual_empty {
                continue;
            }
            let Some(entry) = img.sidecar.captions.get(model) else {
                continue;
            };
            let text = entry.caption.clone();
            img.sidecar.set_manual_caption(&text);
            let _ = img.sidecar.save(&img.path);
        }
    }
    fn bulk_clear_manual_caption(&mut self, paths: &[PathBuf]) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            if img.sidecar.manual_caption.is_some() {
                img.sidecar.set_manual_caption("");
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
    fn bulk_set_caption_hint(&mut self, paths: &[PathBuf], text: &str) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            img.sidecar.set_caption_hint(text);
            let _ = img.sidecar.save(&img.path);
        }
    }
}

// ───────── Helpers ─────────

#[derive(Clone, Copy)]
enum ChipKind {
    Manual,
    Negative,
    Auto,
    Booru,
}

impl ChipKind {
    fn fill(self) -> egui::Color32 {
        match self {
            Self::Manual => egui::Color32::from_rgb(45, 74, 110),
            Self::Negative => egui::Color32::from_rgb(90, 45, 45),
            Self::Auto => egui::Color32::from_rgb(58, 58, 58),
            Self::Booru => egui::Color32::from_rgb(45, 90, 58),
        }
    }
    fn fg(self) -> egui::Color32 {
        match self {
            Self::Manual => egui::Color32::from_rgb(207, 227, 255),
            Self::Negative => egui::Color32::from_rgb(255, 208, 208),
            Self::Auto => egui::Color32::from_rgb(204, 204, 204),
            Self::Booru => egui::Color32::from_rgb(207, 229, 208),
        }
    }
}

/// Render a tag chip. Returns `true` when the user clicked it
/// (interpretation is up to the caller — usually "remove" or "toggle
/// suppression").
///
/// Implemented as a single `egui::Button` instead of a Frame + inner
/// horizontal layout because nested layouts inside `horizontal_wrapped`
/// suppress wrap-on-overflow — egui's wrap engine measures each child
/// after placement, and a Frame's inner sublayout can over-allocate
/// width and push subsequent chips off-screen.
fn chip(ui: &mut egui::Ui, label: &str, kind: ChipKind, suppressed: bool) -> bool {
    let mut text = egui::RichText::new(format!("{label}  ×"))
        .color(kind.fg())
        .size(12.0);
    if suppressed {
        text = text.strikethrough();
    }
    ui.add(
        egui::Button::new(text)
            .fill(kind.fill())
            .corner_radius(egui::CornerRadius::same(8))
            .stroke(egui::Stroke::NONE),
    )
    .clicked()
}

fn section_title(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .small()
            .weak()
            .strong(),
    );
}

fn make_thumbnail_texture(path: &Path, max_size: u32, ctx: &egui::Context) -> Option<TextureHandle> {
    let img = image::open(path).ok()?;
    let thumb = img.thumbnail(max_size, max_size).to_rgba8();
    let size = [thumb.width() as usize, thumb.height() as usize];
    let pixels: Vec<u8> = thumb.into_raw();
    let color_image = ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(
        format!("thumb::{}", path.display()),
        color_image,
        egui::TextureOptions::LINEAR,
    ))
}

fn matches_tag_query(item: &ImageItem, needle_lower: &str) -> bool {
    if item
        .sidecar
        .manual_tags
        .iter()
        .any(|t| t.to_lowercase().contains(needle_lower))
    {
        return true;
    }
    if item
        .sidecar
        .auto_tags
        .iter()
        .any(|at| at.tag.to_lowercase().contains(needle_lower))
    {
        return true;
    }
    item.sidecar
        .booru_tags
        .iter()
        .any(|bt| bt.tag.to_lowercase().contains(needle_lower))
}

fn compute_common_tags(items: &[ImageItem]) -> Vec<(String, usize)> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut display: HashMap<String, String> = HashMap::new();
    for item in items {
        let mut seen: HashSet<String> = HashSet::new();
        let auto = item.sidecar.auto_tags.iter().map(|at| at.tag.as_str());
        let booru = item.sidecar.booru_tags.iter().map(|bt| bt.tag.as_str());
        for tag in auto.chain(booru) {
            let key = tag.to_lowercase();
            if key.is_empty() || !seen.insert(key.clone()) {
                continue;
            }
            if !counts.contains_key(&key) {
                order.push(key.clone());
                display.insert(key.clone(), tag.to_string());
            }
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut out: Vec<(String, usize)> = order
        .into_iter()
        .filter_map(|k| {
            let c = counts[&k];
            if c >= 2 {
                Some((display.remove(&k).unwrap_or(k), c))
            } else {
                None
            }
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn bulk_signature(paths: &[PathBuf]) -> u64 {
    let mut h: u64 = 0;
    for p in paths {
        for b in p.display().to_string().bytes() {
            h = h.wrapping_mul(31).wrapping_add(b as u64);
        }
        h ^= 0x9E37_79B9_7F4A_7C15;
    }
    h
}

fn canonical_bulk_hint(images: &[ImageItem], sel: &[PathBuf]) -> String {
    let values: Vec<&str> = sel
        .iter()
        .filter_map(|p| images.iter().find(|i| &i.path == p))
        .map(|i| i.sidecar.caption_hint.as_deref().unwrap_or(""))
        .collect();
    if values.is_empty() {
        return String::new();
    }
    if values.iter().all(|v| *v == values[0]) {
        values[0].to_string()
    } else {
        String::new()
    }
}
