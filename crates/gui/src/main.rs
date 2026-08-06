#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

mod config_ui;
mod i18n;
mod model_ui;
mod prefs;
mod preview;
mod thumb_cache;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

use chrono::{DateTime, Utc};
use eframe::egui;
use egui::{ColorImage, Key, TextureHandle};
use fwaun_tools_booru::{BooruClient, BooruError};
use fwaun_tools_captioner::Captioner;
use fwaun_tools_core::common_tags::{self, CommonTags};
use fwaun_tools_core::config::{CONFIG_FILE, ProjectConfig, TagGroup};
use fwaun_tools_core::sidecar::{
    AutoTag, BooruInfo, BooruTag, Sidecar, TaggerInfo, is_organizational, sidecar_path_for,
};
use fwaun_tools_core::tag_group::{self, Classification, DropTarget};
use fwaun_tools_core::walk::iter_images;
use fwaun_tools_tagger::Tagger;

use crate::config_ui::{AppSettings, ConfigAction, ConfigDraft, ConfigTab, show_config_modal};
use crate::i18n::{Lang, T};
use crate::model_ui::ModelApp;
use crate::prefs::GuiPrefs;
use crate::preview::{Preview, PreviewAction};
use crate::thumb_cache::{CacheKey, FileStamp, ThumbCache};

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
        .with_title("fwaun-tools")
        .with_inner_size([1200.0, 800.0]);
    if let Ok(icon) = eframe::icon_data::from_png_bytes(ICON_PNG) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "fwaun-tools",
        options,
        Box::new(|cc| {
            install_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(App::new()))
        }),
    )
}

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "noto-jp".into(),
        egui::FontData::from_static(JP_FONT).into(),
    );
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
            Self::NoHint => item.sidecar.caption_hints.is_empty(),
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
    LoadFolder,
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
    LoadFolder,
    Tagger(Option<Box<Tagger>>),
    Captioner(Option<Box<Captioner>>),
    Booru,
}

enum WorkerMsg {
    Progress(Progress),
    /// One image finished loading during a folder scan: its sidecar plus a
    /// pre-built thumbnail texture (built on the worker thread — egui's
    /// `Context::load_texture` is internally synchronized, so this keeps the
    /// decode + upload off the UI thread). The stamp travels with it so the
    /// next differential scan knows which source version this texture is of.
    ImageLoaded {
        path: PathBuf,
        sidecar: Box<Sidecar>,
        thumbnail: Option<TextureHandle>,
        stamp: Option<FileStamp>,
    },
    /// An already-loaded image whose bytes are unchanged: only its sidecar is
    /// re-read, since another process (the CLI, an editor) may have rewritten
    /// it. Parsing one small RON file is nothing next to decoding an image, so
    /// a differential reload does this for every surviving path.
    SidecarReloaded {
        path: PathBuf,
        sidecar: Box<Sidecar>,
    },
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
    // Boxed: both models own ort sessions and are large; keeping them behind a
    // Box shrinks the App struct and the `DoneKind` message that hands them back
    // from the worker (clippy's `large_enum_variant`).
    tagger: Option<Box<Tagger>>,
    captioner: Option<Box<Captioner>>,

    // Modal: config editor
    config_open: bool,
    config_draft: Option<ConfigDraft>,
    config_tab: ConfigTab,
    config_error: Option<String>,
    // Resolved target for the config modal: an ancestor's
    // `fwaun-tools.toml` if one exists, otherwise the path where a new
    // file would be created in the current folder. `None` while the
    // modal is closed or no folder is loaded.
    config_path: Option<PathBuf>,

    // Localization
    lang: Lang,

    // Machine-local preferences (`gui-prefs.toml`): language plus the
    // thumbnail-cache settings. Held in full so a write of one field doesn't
    // drop the others.
    prefs: GuiPrefs,
    // Live cache handle, `None` when disabled in prefs or the platform has no
    // cache directory. Rebuilt by `apply_cache_prefs` whenever prefs change.
    thumb_cache: Option<ThumbCache>,
    // Cache size for the settings readout, measured on demand (it stats every
    // entry, so not per-frame).
    cache_size: Option<u64>,

    // Per-image text-edit buffers, persisted across frames so the user's
    // typing isn't clobbered every redraw. Re-initialized from the
    // sidecar when the selected image changes.
    manual_caption_buf: HashMap<PathBuf, String>,
    last_single: Option<PathBuf>,

    // Shared "add caption hint" input, used by both the single- and
    // bulk-detail panes (each adds to the current selection, like tags).
    hint_input: String,

    // GPU texture handles for thumbnails.
    thumbnails: HashMap<PathBuf, TextureHandle>,
    // Source mtime+size each live thumbnail was built from. A differential
    // reload regenerates only the paths whose stamp moved (or that have no
    // texture yet); everything else keeps the texture it already has.
    stamps: HashMap<PathBuf, FileStamp>,
    // Scan order of the in-flight folder scan. `images` is rebuilt in this
    // order when the scan finishes, so files added since the last scan land
    // in their natural position rather than appended at the end.
    scan_order: Option<Vec<PathBuf>>,
    // True while a *full* (re)load is in flight. Full loads start from an
    // empty `images`, so each ImageLoaded is unconditionally a push; a
    // differential scan has to look for an existing entry first.
    scan_full: bool,

    // Background-worker progress feed. `worker_rx.is_some()` is the
    // single source of truth for "an op is in flight"; once Done lands
    // it goes back to None and the action buttons re-enable.
    progress: Option<Progress>,
    worker_rx: Option<Receiver<WorkerMsg>>,
    // Cooperative cancel for the in-flight op. Set by the progress
    // overlay's Cancel button; the worker checks it at the top of each
    // per-item iteration and stops early (results already sent are kept).
    // Cleared alongside `worker_rx` when the op finishes.
    cancel_flag: Option<Arc<AtomicBool>>,

    // When Some, the next `update()` shows a confirmation modal before
    // removing these paths' image+sidecar files from disk.
    pending_delete: Option<Vec<PathBuf>>,

    // Cached effective ProjectConfig for the current folder. Loaded by
    // `load_folder` so the Kanban view can read `tag_groups` without
    // re-parsing TOML each frame. `None` when no folder is loaded or
    // the config failed to load (treated as empty).
    project_config: Option<ProjectConfig>,
    // Dataset-wide tag layer from the effective config's `common_tags`,
    // resolved alongside `project_config`. Applied to every image without
    // touching sidecars, so it has to be passed to every core call that
    // reads an image's tags.
    common_tags: CommonTags,
    // The config file the loaded folder owns directly, when it owns one —
    // i.e. the folder is the dataset root. `Some` enables the "edit the
    // shared layer instead of every sidecar" path for whole-dataset bulk
    // tag edits.
    root_config_path: Option<PathBuf>,
    // Current main-area view mode.
    view_mode: ViewMode,
    // Active drag in the Kanban view, if any. The payload carries one or
    // more image paths (multi-select drag carries the whole selection).
    kanban_drag: Option<KanbanDrag>,

    // Full-size preview overlay, `Some` while it's up. Owns its own decode
    // thread and texture — see [`preview`].
    preview: Option<Preview>,
    // Failures from the OS handoff (`Open in default app` / `Show in
    // folder`). Those calls run on a throwaway thread — `opener::reveal`
    // blocks on a COM round-trip that has no business freezing the UI — so
    // their errors need a way back to the error banner.
    external_err: (Sender<String>, Receiver<String>),
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
        let prefs = prefs::load();
        let lang = prefs.lang();
        let thumb_cache = cache_for(&prefs);
        // Enforce the budget once per launch, off the UI thread: `prune`
        // stats every entry, and at startup nothing is waiting on it.
        if let Some(cache) = thumb_cache.clone() {
            let limit = prefs.thumb_cache.limit_mb.saturating_mul(1024 * 1024);
            let max_age = age_limit(&prefs);
            thread::spawn(move || cache.prune(limit, max_age));
        }
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
            lang,
            prefs,
            thumb_cache,
            cache_size: None,
            manual_caption_buf: HashMap::new(),
            last_single: None,
            hint_input: String::new(),
            thumbnails: HashMap::new(),
            stamps: HashMap::new(),
            scan_order: None,
            scan_full: false,
            progress: None,
            worker_rx: None,
            cancel_flag: None,
            pending_delete: None,
            project_config: None,
            common_tags: CommonTags::default(),
            root_config_path: None,
            view_mode: ViewMode::Grid,
            kanban_drag: None,
            preview: None,
            external_err: channel(),
        }
    }

    fn t(&self) -> T {
        T::new(self.lang)
    }

    /// Open a folder from scratch: drop everything the previous folder owned
    /// (including the model cache — the new folder's config may point at a
    /// different model) and scan it in full.
    fn load_folder(&mut self, ctx: &egui::Context, dir: &Path) {
        self.folder = Some(dir.to_path_buf());
        self.images.clear();
        self.thumbnails.clear();
        self.stamps.clear();
        self.selected.clear();
        self.tagger = None;
        self.captioner = None;
        self.manual_caption_buf.clear();
        self.last_single = None;
        self.hint_input.clear();
        self.kanban_drag = None;
        self.preview = None;
        self.scan_folder(ctx, dir, true);
    }

    /// Re-scan the folder already open, keeping everything that hasn't
    /// changed on disk.
    ///
    /// The point is to make "I added images / ran the CLI over this dataset"
    /// cheap: the thumbnail textures the app already holds are reused, so only
    /// genuinely new or rewritten files pay for a decode. Selection, scroll
    /// position, view mode and the loaded models all survive, which a
    /// close-and-reopen loses.
    fn reload_folder(&mut self, ctx: &egui::Context) {
        let Some(dir) = self.folder.clone() else {
            self.error_msg = Some(self.t().err_open_folder_first());
            return;
        };
        // A drag can't survive the images under it being re-shuffled.
        self.kanban_drag = None;
        // Sidecars are about to be re-read, and the whole point of a reload is
        // that they may have changed underneath us. Drop the manual-caption
        // edit buffers so they re-seed from what's now on disk instead of
        // writing a pre-reload copy back over an external edit; `last_single`
        // is what triggers that re-seed.
        self.manual_caption_buf.clear();
        self.last_single = None;
        self.scan_folder(ctx, &dir, false);
    }

    /// Shared body of [`Self::load_folder`] and [`Self::reload_folder`].
    ///
    /// `full` is only a statement about what the caller already cleared —
    /// the per-file decisions below would reach the same conclusion either
    /// way, it just skips the bookkeeping when nothing is retained.
    fn scan_folder(&mut self, ctx: &egui::Context, dir: &Path, full: bool) {
        // Best-effort config load. A broken TOML is reported in the
        // banner; the app still functions in Grid mode without groups.
        match ProjectConfig::load_or_default(dir) {
            Ok(cfg) => {
                self.common_tags = cfg.resolve_common_tags();
                self.project_config = Some(cfg);
            }
            Err(e) => {
                self.project_config = None;
                self.common_tags = CommonTags::default();
                self.error_msg = Some(format!("config load failed: {e}"));
            }
        }
        self.root_config_path = ProjectConfig::dataset_root_config(dir);
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

        // Decoding every image and uploading a thumbnail texture is the slow
        // part of opening a folder — on a large dataset it froze the UI with no
        // feedback. Stream it from a background thread instead: the progress
        // overlay shows how far along we are, images appear as they load, and
        // the Cancel button can stop a huge folder mid-scan.
        let paths: Vec<PathBuf> = iter_images(dir).collect();

        if !full {
            // Files that vanished from disk. Same bookkeeping as a delete,
            // minus the `fs::remove_file` — the removal already happened
            // outside the app.
            let scanned: HashSet<&PathBuf> = paths.iter().collect();
            let gone: Vec<PathBuf> = self
                .images
                .iter()
                .map(|i| i.path.clone())
                .filter(|p| !scanned.contains(p))
                .collect();
            self.forget_paths(&gone);
        }

        // Split the scan into "needs a new thumbnail" and "sidecar only".
        let loaded: HashSet<&PathBuf> = self.images.iter().map(|i| &i.path).collect();
        let jobs: Vec<ScanJob> = paths
            .iter()
            .map(|path| {
                let stamp = FileStamp::of(path);
                let needs_thumb = needs_thumbnail(
                    full,
                    loaded.contains(path),
                    self.thumbnails.contains_key(path),
                    self.stamps.get(path),
                    stamp.as_ref(),
                );
                ScanJob {
                    path: path.clone(),
                    stamp,
                    needs_thumb,
                }
            })
            .collect();

        let total = jobs.len();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let ctx_clone = ctx.clone();
        let cache = self.thumb_cache.clone();
        let cache_limit = self.prefs.thumb_cache.limit_mb.saturating_mul(1024 * 1024);
        let cache_age = age_limit(&self.prefs);
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            for (i, job) in jobs.iter().enumerate() {
                if cancel_worker.load(Ordering::Relaxed) {
                    break;
                }
                let _ = tx.send(WorkerMsg::Progress(Progress {
                    op: WorkerOp::LoadFolder,
                    current: i,
                    total,
                }));
                let sidecar = Box::new(Sidecar::load_or_default(&job.path).unwrap_or_default());
                let msg = if job.needs_thumb {
                    let thumbnail = make_thumbnail_texture(
                        &job.path,
                        job.stamp,
                        THUMB_SIZE,
                        &ctx_clone,
                        cache.as_ref(),
                    );
                    WorkerMsg::ImageLoaded {
                        path: job.path.clone(),
                        sidecar,
                        thumbnail,
                        stamp: job.stamp,
                    }
                } else {
                    WorkerMsg::SidecarReloaded {
                        path: job.path.clone(),
                        sidecar,
                    }
                };
                let _ = tx.send(msg);
                ctx_clone.request_repaint();
            }
            let _ = tx.send(WorkerMsg::Progress(Progress {
                op: WorkerOp::LoadFolder,
                current: total,
                total,
            }));
            let _ = tx.send(WorkerMsg::Done(DoneKind::LoadFolder));
            ctx_clone.request_repaint();
            // Entries this scan just wrote may have pushed the cache over
            // budget. Trim after the UI is already unblocked.
            if let Some(cache) = cache {
                cache.prune(cache_limit, cache_age);
            }
        });

        self.scan_order = Some(paths);
        self.scan_full = full;
        self.worker_rx = Some(rx);
        self.cancel_flag = Some(cancel);
        self.loading = true;
        self.progress = Some(Progress {
            op: WorkerOp::LoadFolder,
            current: 0,
            total,
        });
    }

    /// Drop everything the app holds for these paths, without touching disk.
    /// Shared by the delete flow and by a reload that finds files gone.
    fn forget_paths(&mut self, paths: &[PathBuf]) {
        if paths.is_empty() {
            return;
        }
        let to_remove: HashSet<PathBuf> = paths.iter().cloned().collect();
        self.images.retain(|i| !to_remove.contains(&i.path));
        self.selected.retain(|p| !to_remove.contains(p));
        for p in &to_remove {
            self.thumbnails.remove(p);
            self.stamps.remove(p);
            self.manual_caption_buf.remove(p);
        }
        if let Some(last) = self.last_single.as_ref()
            && to_remove.contains(last)
        {
            self.last_single = None;
        }
        // The preview holds a path list captured when it opened, so a file
        // that just went away would sit there as a decode error the user
        // can't do anything about.
        if let Some(preview) = self.preview.as_ref()
            && to_remove.contains(preview.path())
        {
            self.preview = None;
        }
    }

    /// Rebuild the cache handle after a prefs change. The measured size is
    /// left alone: toggling the setting doesn't move a byte on disk.
    fn apply_cache_prefs(&mut self) {
        self.thumb_cache = cache_for(&self.prefs);
    }

    /// The images the current filter and tag query let through, in `images`
    /// order. Both grid cells and preview navigation walk this, so "the next
    /// image" means the same thing in each.
    fn visible_paths(&self) -> Vec<PathBuf> {
        let needle = self.tag_filter.trim().to_lowercase();
        self.images
            .iter()
            .filter(|i| self.filter.matches(i))
            .filter(|i| needle.is_empty() || matches_tag_query(i, &needle))
            .map(|i| i.path.clone())
            .collect()
    }

    fn measure_cache(&mut self) {
        // Measure through a handle built from the *path*, not the enabled
        // flag: after turning the cache off the user still wants to see what
        // is sitting on disk and be able to clear it.
        self.cache_size = ThumbCache::open().map(|c| c.total_size());
    }
}

/// One entry in a folder scan, resolved on the UI thread (a `stat` each) so
/// the worker only has to execute the decisions.
struct ScanJob {
    path: PathBuf,
    stamp: Option<FileStamp>,
    needs_thumb: bool,
}

/// Whether a scanned path has to be decoded again, or can keep the texture
/// the app already holds.
///
/// The expensive answer is `true`, so every case that isn't provably
/// unchanged has to land there: a file the app doesn't know yet, one whose
/// texture never materialized (a cancelled or failed earlier scan), one whose
/// mtime/size moved, and one where either stamp is missing — an unreadable
/// `stat` gives nothing to compare, and reusing a texture on that basis would
/// silently pin a stale thumbnail with no way to ever refresh it.
fn needs_thumbnail(
    full: bool,
    is_loaded: bool,
    has_texture: bool,
    known_stamp: Option<&FileStamp>,
    current_stamp: Option<&FileStamp>,
) -> bool {
    if full || !is_loaded || !has_texture {
        return true;
    }
    match (known_stamp, current_stamp) {
        (Some(known), Some(current)) => known != current,
        _ => true,
    }
}

/// Put `images` back into the order the scan walked the directory in.
///
/// A differential scan appends each newly-discovered file as it finishes
/// loading, which would otherwise leave everything added since the last scan
/// clustered at the end of the grid instead of sitting next to its
/// neighbours. Anything the scan didn't cover — a run the user cancelled —
/// keeps its relative order at the end rather than being dropped.
fn reorder_to_scan_order(images: &mut [ImageItem], order: &[PathBuf]) {
    let rank: HashMap<&PathBuf, usize> = order.iter().enumerate().map(|(i, p)| (p, i)).collect();
    images.sort_by_key(|item| rank.get(&item.path).copied().unwrap_or(usize::MAX));
}

fn cache_for(prefs: &GuiPrefs) -> Option<ThumbCache> {
    if prefs.thumb_cache.enabled {
        ThumbCache::open()
    } else {
        None
    }
}

/// Prefs' `max_age_days` as a duration, or `None` for "no expiry".
fn age_limit(prefs: &GuiPrefs) -> Option<Duration> {
    match prefs.thumb_cache.max_age_days {
        0 => None,
        days => Some(Duration::from_secs(u64::from(days) * 24 * 60 * 60)),
    }
}

/// Which top-level screen is showing. The two are fully independent: the
/// dataset editor and the model-checkpoint tools share no state.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Dataset,
    Model,
}

/// Top-level app: a mode tab bar (plus the shared language toggle) over two
/// otherwise-independent screens. The language preference lives on the
/// dataset app (its historical owner) and is passed down to the model tab.
struct App {
    mode: AppMode,
    dataset: AnimaTaggerApp,
    model: ModelApp,
}

impl App {
    fn new() -> Self {
        Self {
            mode: AppMode::Dataset,
            dataset: AnimaTaggerApp::new(),
            model: ModelApp::new(),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let t = T::new(self.dataset.lang);
        egui::TopBottomPanel::top("mode_bar").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, AppMode::Dataset, t.mode_dataset());
                ui.selectable_value(&mut self.mode, AppMode::Model, t.mode_model());

                // Language selector, pinned to the right so it's reachable
                // from either mode.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let lang_label = match self.dataset.lang {
                        Lang::En => "English",
                        Lang::Ja => "日本語",
                    };
                    egui::ComboBox::from_id_salt("lang_combo")
                        .selected_text(lang_label)
                        .width(96.0)
                        .show_ui(ui, |ui| {
                            let mut new_lang = self.dataset.lang;
                            ui.selectable_value(&mut new_lang, Lang::En, "English");
                            ui.selectable_value(&mut new_lang, Lang::Ja, "日本語");
                            if new_lang != self.dataset.lang {
                                self.dataset.lang = new_lang;
                                self.dataset.prefs.set_lang(new_lang);
                                prefs::save(&self.dataset.prefs);
                            }
                        });
                });
            });
            ui.add_space(2.0);
        });

        match self.mode {
            AppMode::Dataset => self.dataset.ui(ctx),
            AppMode::Model => self.model.ui(ctx, self.dataset.lang),
        }
    }
}

impl AnimaTaggerApp {
    fn ui(&mut self, ctx: &egui::Context) {
        self.poll_worker();
        self.poll_external_errors();
        // Drawn first, though it paints on top: the overlay claims the arrow
        // keys with `consume_key`, and whoever runs first in the frame wins
        // them. Everything below would otherwise scroll or move a caret while
        // the user is stepping through images.
        self.ui_preview(ctx);
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
        egui::CentralPanel::default().show(ctx, |ui| match self.view_mode.clone() {
            ViewMode::Grid => self.ui_grid(ui),
            ViewMode::Kanban { group } => self.ui_kanban(ui, &group),
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
            // Disabled mid-op: opening a new folder would orphan the running
            // worker and swap the data out from under it.
            if ui
                .add_enabled(!self.loading, egui::Button::new(t.open_folder()))
                .clicked()
                && let Some(picked) = rfd::FileDialog::new().pick_folder()
            {
                self.load_folder(ctx, &picked);
            }
            // Manual, not automatic: a filesystem watcher (`notify`) would be
            // the obvious next step, but re-scanning under the user mid-edit
            // is its own hazard, so the trigger stays explicit for now.
            if ui
                .add_enabled(
                    self.folder.is_some() && !self.loading,
                    egui::Button::new(t.reload_folder()),
                )
                .on_hover_text(t.reload_folder_title())
                .clicked()
            {
                self.reload_folder(ctx);
            }
            let cfg_btn = ui
                .button(t.config_button())
                .on_hover_text(t.config_button_title());
            if cfg_btn.clicked() {
                // Walk up from the loaded folder to find the nearest
                // existing `fwaun-tools.toml` so a config kept at the
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
                                    toml::from_str::<ProjectConfig>(&s).map_err(|e| e.to_string())
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

            // View dropdown — Grid plus one entry per *exclusive*
            // [tag_group.<name>]. Non-exclusive groups exist for caption
            // steering (their tags are meant to co-occur), so they don't map
            // onto the Kanban's one-column-per-tag / violation model.
            let group_names: Vec<String> = self
                .project_config
                .as_ref()
                .map(|c| {
                    c.tag_groups
                        .iter()
                        .filter(|(_, g)| g.exclusive)
                        .map(|(name, _)| name.clone())
                        .collect()
                })
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
                        .selectable_label(matches!(self.view_mode, ViewMode::Grid), t.view_grid())
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
                self.selected = self.visible_paths().into_iter().collect();
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

            if self.loading {
                ui.label(t.working());
            }
            ui.label(t.images_selected_summary(self.images.len(), self.selected.len()));
        });
    }
}

// ───────── Grid ─────────

/// Something a thumbnail's context menu (or a double click) asked for. Recorded
/// inside the menu closure and applied after it, since the closure only has a
/// borrowed `Response` to work with.
enum ThumbAction {
    Preview,
    OpenExternal,
    Reveal,
}

impl AnimaTaggerApp {
    fn ui_grid(&mut self, ui: &mut egui::Ui) {
        let t = self.t();
        let visible = self.visible_paths();

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
        let Some(response) = self.thumb_card(ui, path, egui::Sense::click()) else {
            return;
        };
        self.handle_thumb_interaction(&response, path, mods);
    }

    /// Draw one thumbnail card — image, status flags, selection outline — and
    /// return its response. `None` when the path has no texture yet (a scan
    /// still in flight) or has dropped out of `images`.
    ///
    /// `sense` is the only thing the grid and the Kanban disagree on: the
    /// latter needs drag sensing so a thumbnail can be thrown at a column.
    fn thumb_card(
        &self,
        ui: &mut egui::Ui,
        path: &Path,
        sense: egui::Sense,
    ) -> Option<egui::Response> {
        let texture = self.thumbnails.get(path)?.clone();
        let item = self.images.iter().find(|i| i.path == path)?;
        let is_selected = self.selected.contains(path);

        let frame = egui::Frame::group(ui.style())
            .inner_margin(2.0)
            .stroke(if is_selected {
                egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
            } else {
                egui::Stroke::new(2.0, egui::Color32::TRANSPARENT)
            });

        Some(
            frame
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
                .interact(sense),
        )
    }

    /// The interactions every thumbnail shares: click selects, double click
    /// opens the preview, right click opens the context menu. The Kanban
    /// layers its drag handling on top of this.
    fn handle_thumb_interaction(
        &mut self,
        response: &egui::Response,
        path: &Path,
        mods: egui::Modifiers,
    ) {
        let is_selected = self.selected.contains(path);
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
        // Right-clicking something outside the selection makes it the
        // selection first, so the menu that pops up is unambiguously about
        // the thumbnail under the cursor. Right-clicking *inside* a
        // multi-selection leaves it alone.
        if response.secondary_clicked() && !is_selected {
            self.selected.clear();
            self.selected.insert(path.to_path_buf());
        }

        // The menu closure can't touch `self` — `response.context_menu` is
        // called on a borrow of it — so it records an intent and we act on it
        // once the closure is done.
        let t = self.t();
        let mut action = if response.double_clicked() {
            Some(ThumbAction::Preview)
        } else {
            None
        };
        response.context_menu(|ui| {
            if ui
                .button(t.open_preview())
                .on_hover_text(t.open_preview_title())
                .clicked()
            {
                action = Some(ThumbAction::Preview);
                ui.close();
            }
            if ui.button(t.open_external()).clicked() {
                action = Some(ThumbAction::OpenExternal);
                ui.close();
            }
            if ui.button(t.reveal_in_folder()).clicked() {
                action = Some(ThumbAction::Reveal);
                ui.close();
            }
        });
        match action {
            Some(ThumbAction::Preview) => self.open_preview(&response.ctx, path),
            Some(ThumbAction::OpenExternal) => self.open_external(path, false),
            Some(ThumbAction::Reveal) => self.open_external(path, true),
            None => {}
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
            match tag_group::classify(&item.sidecar, &group, &self.common_tags) {
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
                let frame = egui::Frame::group(ui.style()).inner_margin(6.0).stroke(
                    if dragging && ui.rect_contains_pointer(ui.max_rect()) {
                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill)
                    } else {
                        ui.visuals().widgets.noninteractive.bg_stroke
                    },
                );
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
        let Some(response) = self.thumb_card(ui, path, egui::Sense::click_and_drag()) else {
            return;
        };
        let is_selected = self.selected.contains(path);
        self.handle_thumb_interaction(&response, path, mods);
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
    fn apply_kanban_drop(&mut self, group: &TagGroup, target: &DropTarget, paths: &[PathBuf]) {
        let t = self.t();
        let common = self.common_tags.clone();
        for path in paths {
            let Some(item) = self.images.iter_mut().find(|i| i.path == *path) else {
                continue;
            };
            tag_group::apply_drop(&mut item.sidecar, group, target, &common);
            let save_result = item.sidecar.save(&item.path);
            let path_str = item.path.display().to_string();
            if let Err(e) = save_result {
                self.error_msg = Some(t.kanban_drop_failed(&path_str, &e.to_string()));
            }
        }
    }
}

// ───────── Preview / OS handoff ─────────

impl AnimaTaggerApp {
    /// Open the full-size preview on `path`, with the currently visible images
    /// as the list its arrow keys walk.
    ///
    /// The Kanban view groups those images into columns rather than listing
    /// them in this order, so there the arrow keys step through the filtered
    /// dataset rather than along the column under the cursor. Both orders are
    /// defensible; this one keeps a verification pass covering everything
    /// exactly once.
    fn open_preview(&mut self, ctx: &egui::Context, path: &Path) {
        let mut order = self.visible_paths();
        // The detail panel can still be showing an image the active filter no
        // longer matches — tagging it is exactly what stops it matching
        // "Untagged". Preview it alone rather than refusing to open.
        if !order.iter().any(|p| p == path) {
            order = vec![path.to_path_buf()];
        }
        self.preview = Preview::open(ctx, order, path);
    }

    /// Hand `path` to the OS: the default viewer, or the file manager with the
    /// file selected when `reveal`.
    ///
    /// Off the UI thread, because `opener::reveal` blocks on a COM round-trip
    /// that can take a noticeable moment when the file manager isn't running
    /// yet. Failures come back through `external_err` and land in the error
    /// banner — there's no second thing worth trying automatically.
    fn open_external(&mut self, path: &Path, reveal: bool) {
        let path = path.to_path_buf();
        let tx = self.external_err.0.clone();
        let t = self.t();
        thread::spawn(move || {
            let result = if reveal {
                opener::reveal(&path)
            } else {
                opener::open(&path)
            };
            if let Err(e) = result {
                let _ = tx.send(t.err_open_external(&path.display().to_string(), &e.to_string()));
            }
        });
    }

    /// Drain whatever the OS-handoff threads reported since the last frame.
    fn poll_external_errors(&mut self) {
        while let Ok(err) = self.external_err.1.try_recv() {
            self.error_msg = Some(err);
        }
    }

    /// Draw the preview overlay and act on whatever the user did in it.
    fn ui_preview(&mut self, ctx: &egui::Context) {
        let t = self.t();
        // The borrow of `self.preview` has to end before the actions below can
        // touch the rest of `self`, hence collecting both results first.
        let Some((action, path)) = self
            .preview
            .as_mut()
            .map(|p| (p.show(ctx, t), p.path().to_path_buf()))
        else {
            return;
        };
        match action {
            PreviewAction::Keep => {}
            PreviewAction::Close => self.preview = None,
            PreviewAction::OpenExternal => self.open_external(&path, false),
            PreviewAction::Reveal => self.open_external(&path, true),
        }
    }
}

fn status_flags(s: &Sidecar) -> String {
    let t = if s.is_auto_tagged() { 'T' } else { ' ' };
    let c = if s.is_captioned() { 'C' } else { ' ' };
    let b = if s.has_booru() { 'B' } else { ' ' };
    let m = if !s.manual_tags.is_empty() { 'M' } else { ' ' };
    let h = if !s.caption_hints.is_empty() {
        'H'
    } else {
        ' '
    };
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

    /// True when the current selection *is* the whole dataset: the open
    /// folder owns the config (so it's the dataset root, not a subdirectory)
    /// and every loaded image is selected.
    ///
    /// This is the condition under which a bulk tag edit is a statement
    /// about the dataset rather than about a set of images, so it belongs in
    /// `common_tags` — see [`common_tags`](fwaun_tools_core::common_tags).
    /// Deselecting a single image is the escape hatch back to per-sidecar
    /// edits.
    fn is_whole_dataset_selection(&self) -> bool {
        self.root_config_path.is_some()
            && !self.images.is_empty()
            && self.selected.len() == self.images.len()
    }

    /// Add or remove `tag` in the dataset root config's `common_tags`, then
    /// reload the layer so the UI reflects it immediately. Returns `false`
    /// (having set the error banner) if the edit couldn't be written.
    fn edit_common_tag(&mut self, tag: &str, remove: bool) -> bool {
        let Some(path) = self.root_config_path.clone() else {
            return false;
        };
        let tags = [tag.to_string()];
        let result = if remove {
            common_tags::remove_from_config_file(&path, &tags, false)
        } else {
            common_tags::add_to_config_file(&path, &tags, false)
        };
        match result {
            Ok(_) => {
                self.reload_project_config();
                true
            }
            Err(e) => {
                self.error_msg = Some(
                    self.t()
                        .common_tag_write_failed(&path.display().to_string(), &e.to_string()),
                );
                false
            }
        }
    }

    /// Re-read the effective config for the open folder. Called after the
    /// app itself rewrites `common_tags`, so the in-memory layer, the Kanban
    /// buckets and the chip lists all pick the change up.
    fn reload_project_config(&mut self) {
        let Some(dir) = self.folder.clone() else {
            return;
        };
        match ProjectConfig::load_or_default(&dir) {
            Ok(cfg) => {
                self.common_tags = cfg.resolve_common_tags();
                self.project_config = Some(cfg);
            }
            Err(e) => self.error_msg = Some(format!("config load failed: {e}")),
        }
    }

    fn add_manual_tag_to_selected(&mut self, tag: &str) {
        // A tag added to every image in the dataset is a property of the
        // dataset: write it once to `common_tags` rather than copying it
        // into every sidecar, so images added later inherit it too.
        if self.is_whole_dataset_selection() && self.edit_common_tag(tag, false) {
            return;
        }
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

    /// The "add a caption hint" input, shared by the single- and bulk-detail
    /// panes. Like the tag input, it appends to every selected image, so a
    /// shared reference fact can be applied across a whole character in one go.
    fn ui_caption_hint_add_input(&mut self, ui: &mut egui::Ui) {
        let t = self.t();
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.hint_input)
                    .hint_text(t.add_hint_placeholder())
                    .desired_width(ui.available_width() - 60.0),
            );
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
            let click = ui.button(t.add_button()).clicked();
            if enter || click {
                let v = std::mem::take(&mut self.hint_input);
                let trimmed = v.trim().to_string();
                if !trimmed.is_empty() {
                    self.add_caption_hint_to_selected(&trimmed);
                }
                if enter {
                    r.request_focus();
                }
            }
        });
    }

    fn add_caption_hint_to_selected(&mut self, hint: &str) {
        let sel = self.selected.clone();
        for img in self.images.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            if img.sidecar.add_caption_hint(hint.to_string()) {
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

        let filename = item.path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        ui.label(egui::RichText::new(filename).monospace().weak());
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button(t.open_preview())
                .on_hover_text(t.open_preview_title())
                .clicked()
            {
                self.open_preview(ui.ctx(), path);
            }
            if ui.small_button(t.open_external()).clicked() {
                self.open_external(path, false);
            }
            if ui.small_button(t.reveal_in_folder()).clicked() {
                self.open_external(path, true);
            }
        });

        ui.add_space(6.0);
        section_title(ui, t.section_tags());
        let manual_positives: Vec<String> = item
            .sidecar
            .manual_positive_tags()
            .map(|s| s.to_string())
            .collect();
        // Entries from the dataset-wide layer this image doesn't override.
        // They aren't in the sidecar, so they'd otherwise be invisible here
        // while silently steering the export.
        let inherited = self.inherited_common_entries(&item.sidecar);
        // Suppression state has to consider both layers, or a tag hidden by
        // a `common_tags` `-foo` would still render un-struck.
        let merged = self.common_tags.merged_manual_tags(&item.sidecar);
        let suppressed_stems = fwaun_tools_core::sidecar::suppressed_stems(&merged);
        let is_suppressed = |tag: &str| suppressed_stems.contains(&tag.trim().to_lowercase());

        if manual_positives.is_empty()
            && inherited.is_empty()
            && item.sidecar.auto_tags.is_empty()
            && item.sidecar.booru_tags.is_empty()
        {
            ui.weak(t.empty_tags());
        } else {
            let mut to_remove_manual: Vec<String> = Vec::new();
            let mut to_toggle_suppression: Vec<String> = Vec::new();
            let mut to_override_common: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for entry in &inherited {
                    if chip(ui, &format!("{entry} [C]"), ChipKind::Common, false) {
                        to_override_common.push(entry.clone());
                    }
                }
                for tag in &manual_positives {
                    if chip(ui, tag, manual_chip_kind(tag), false) {
                        to_remove_manual.push(tag.clone());
                    }
                }
                for at in &item.sidecar.auto_tags {
                    if chip(
                        ui,
                        &format!("{} ({:.2})", at.tag, at.score),
                        ChipKind::Auto,
                        is_suppressed(&at.tag),
                    ) {
                        to_toggle_suppression.push(at.tag.clone());
                    }
                }
                for bt in &item.sidecar.booru_tags {
                    if chip(
                        ui,
                        &format!("{} [B]", bt.tag),
                        ChipKind::Booru,
                        is_suppressed(&bt.tag),
                    ) {
                        to_toggle_suppression.push(bt.tag.clone());
                    }
                }
            });
            if !inherited.is_empty() {
                ui.add(egui::Label::new(
                    egui::RichText::new(t.dataset_tags_override_hint())
                        .small()
                        .weak(),
                ));
            }
            for entry in to_override_common {
                self.override_common_entry_at(path, &entry);
            }
            for tag in to_remove_manual {
                self.remove_manual_at(path, &tag);
            }
            for tag in to_toggle_suppression {
                self.toggle_suppression_at(path, &tag);
            }
        }

        ui.add_space(6.0);
        section_title(ui, t.section_caption_hint());
        if item.sidecar.caption_hints.is_empty() {
            ui.weak(t.empty_hints());
        } else {
            let mut to_remove: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for hint in &item.sidecar.caption_hints {
                    if chip(ui, hint, ChipKind::Manual, false) {
                        to_remove.push(hint.clone());
                    }
                }
            });
            for hint in to_remove {
                self.remove_caption_hint_at(path, &hint);
            }
        }
        self.ui_caption_hint_add_input(ui);

        ui.add_space(6.0);
        section_title(ui, t.section_manual_caption());
        let path_owned = path.to_path_buf();
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
            .button(
                egui::RichText::new(t.delete_image()).color(egui::Color32::from_rgb(220, 90, 90)),
            )
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
        let mut hint_order: Vec<String> = Vec::new();
        let mut hint_counts: HashMap<String, usize> = HashMap::new();
        for item in &selected_items {
            for hint in &item.sidecar.caption_hints {
                if !hint_counts.contains_key(hint) {
                    hint_order.push(hint.clone());
                }
                *hint_counts.entry(hint.clone()).or_insert(0) += 1;
            }
        }
        if hint_order.is_empty() {
            ui.weak(t.empty_hints());
        } else {
            let mut to_remove: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for hint in &hint_order {
                    let count = hint_counts[hint];
                    let label = if count < n {
                        format!("{hint} ({count}/{n})")
                    } else {
                        hint.clone()
                    };
                    if chip(ui, &label, ChipKind::Manual, false) {
                        to_remove.push(hint.clone());
                    }
                }
            });
            for hint in to_remove {
                self.bulk_remove_caption_hint(sel, &hint);
            }
        }
        self.ui_caption_hint_add_input(ui);

        // The dataset-wide layer, listed before the per-image entries so the
        // "these apply to everything" tags read first. Only entries no
        // selected image overrides are shown — an overridden one isn't in
        // effect across the selection.
        let inherited: Vec<String> = {
            let mut common_entries: Option<Vec<String>> = None;
            for item in &selected_items {
                let entries = self.inherited_common_entries(&item.sidecar);
                common_entries = Some(match common_entries {
                    None => entries,
                    Some(prev) => prev.into_iter().filter(|e| entries.contains(e)).collect(),
                });
            }
            common_entries.unwrap_or_default()
        };
        if !inherited.is_empty() {
            ui.add_space(6.0);
            section_title(ui, t.section_dataset_tags());
            let whole_dataset = self.is_whole_dataset_selection();
            let mut clicked: Vec<String> = Vec::new();
            ui.horizontal_wrapped(|ui| {
                for entry in &inherited {
                    if chip(ui, entry, ChipKind::Common, false) {
                        clicked.push(entry.clone());
                    }
                }
            });
            ui.add(egui::Label::new(
                egui::RichText::new(if whole_dataset {
                    t.dataset_tags_root_hint()
                } else {
                    t.dataset_tags_override_hint()
                })
                .small()
                .weak(),
            ));
            for entry in clicked {
                self.bulk_common_entry_action(sel, &entry);
            }
        }

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
                    let kind = manual_chip_kind(tag);
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
        section_title(ui, t.section_shared_tags());
        let common = compute_shared_tags(&selected_items);
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
            egui::RichText::new(t.switch_to_single_hint())
                .small()
                .weak(),
        ));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .button(
                egui::RichText::new(t.delete_images()).color(egui::Color32::from_rgb(220, 90, 90)),
            )
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
        if self.config_draft.is_none() {
            // Defensive: modal should only be open with a draft present.
            self.config_open = false;
            return;
        }
        // The App tab edits `prefs` in place; anything it changed is persisted
        // right after the frame, so machine-local settings don't ride on the
        // dataset config's Save button.
        let prefs_before = self.prefs.clone();
        let cache_root = ThumbCache::open();
        let action = {
            let mut app = AppSettings {
                prefs: &mut self.prefs,
                cache_root: cache_root.as_ref().map(|c| c.root()),
                cache_size: self.cache_size,
            };
            let draft = self.config_draft.as_mut().expect("checked above");
            show_config_modal(
                ctx,
                t,
                &target_label,
                draft,
                &mut app,
                &mut self.config_tab,
                &mut self.config_error,
            )
        };
        if self.prefs != prefs_before {
            prefs::save(&self.prefs);
            self.apply_cache_prefs();
        }
        // Measure once on landing on the App tab rather than on every modal
        // open: it stats every cache entry, and most visits here are for the
        // dataset tabs.
        if self.config_tab == ConfigTab::App && self.cache_size.is_none() {
            self.measure_cache();
        }
        match action {
            ConfigAction::None => {}
            ConfigAction::MeasureCache => self.measure_cache(),
            ConfigAction::ClearCache => {
                if let Some(Err(e)) = cache_root.as_ref().map(|c| c.clear()) {
                    self.config_error = Some(t.cfg_thumb_cache_clear_failed(&e.to_string()));
                }
                self.measure_cache();
            }
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
                            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
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
                    errors.push(
                        self.t()
                            .err_delete_failed(&p.display().to_string(), &e.to_string()),
                    );
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
        self.forget_paths(paths);
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
        // default behavior (`fwaun-tools dataset tag` without --force).
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
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            if tagger.is_none() {
                match Tagger::from_profile(&profile_for_load) {
                    Ok(t) => tagger = Some(Box::new(t)),
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
                if cancel_worker.load(Ordering::Relaxed) {
                    break;
                }
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
        self.cancel_flag = Some(cancel);
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
        // Mirrors the CLI's default behavior (`fwaun-tools dataset caption`
        // without --force).
        let prompt_keys: Vec<String> = prompts
            .iter()
            .map(|(n, _)| format!("{model_name}.{n}"))
            .collect();
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
        // Tag groups drive both the caption hints (facts fed as context when
        // a tag combination is present) and the caption prefix, which is
        // embedded in the prompt so the model's output continues from what
        // export will prepend.
        let tag_groups = self
            .project_config
            .as_ref()
            .map(|c| c.tag_groups.clone())
            .unwrap_or_default();
        let common = self.common_tags.clone();
        let hints: HashMap<PathBuf, Option<String>> = self
            .images
            .iter()
            .filter(|i| sel.contains(&i.path))
            .map(|i| {
                let extra = tag_group::resolved_caption_hints(&i.sidecar, &tag_groups, &common);
                (i.path.clone(), i.sidecar.caption_hint_prompt_with(&extra))
            })
            .collect();
        let prefixes: HashMap<PathBuf, Option<String>> = self
            .images
            .iter()
            .filter(|i| sel.contains(&i.path))
            .map(|i| {
                let p = tag_group::resolved_caption_prefix(&i.sidecar, &tag_groups, &common);
                (i.path.clone(), (!p.is_empty()).then_some(p))
            })
            .collect();

        let mut captioner = self.captioner.take();
        let profile_for_load = profile.clone();
        let ctx_clone = ctx.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            if captioner.is_none() {
                match Captioner::from_profile(&profile_for_load) {
                    Ok(c) => captioner = Some(Box::new(c)),
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
                if cancel_worker.load(Ordering::Relaxed) {
                    break;
                }
                let _ = tx.send(WorkerMsg::Progress(Progress {
                    op: WorkerOp::Captioner,
                    current: i,
                    total,
                }));
                ctx_clone.request_repaint();
                let hint = hints.get(path).cloned().flatten();
                let prefix = prefixes.get(path).cloned().flatten();
                let have = existing_keys.get(path);
                let mut entries: Vec<(String, String)> = Vec::new();
                for (pname, ptext) in &prompts {
                    let key = format!("{model_name}.{pname}");
                    if have.is_some_and(|s| s.contains(&key)) {
                        continue;
                    }
                    match captioner_inst.caption_image(
                        path,
                        ptext,
                        hint.as_deref(),
                        prefix.as_deref(),
                    ) {
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
        self.cancel_flag = Some(cancel);
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
        // default behavior (`fwaun-tools dataset booru` without --force).
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
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = channel::<WorkerMsg>();

        thread::spawn(move || {
            let client = BooruClient::danbooru();
            for (i, path) in sel.iter().enumerate() {
                if cancel_worker.load(Ordering::Relaxed) {
                    break;
                }
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
        self.cancel_flag = Some(cancel);
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
        // Drain everything currently buffered. We can't hold a borrow of
        // self.worker_rx across the apply_worker_msg call (which mutably
        // borrows self), so try_recv runs inside the match scrutinee — the
        // receiver borrow ends there, before we dispatch the message.
        loop {
            let msg = match self.worker_rx.as_ref().map(|rx| rx.try_recv()) {
                Some(Ok(msg)) => msg,
                None | Some(Err(std::sync::mpsc::TryRecvError::Empty)) => break,
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    // Worker dropped without sending Done — clean up anyway so
                    // the UI doesn't get stuck on the progress overlay.
                    self.worker_rx = None;
                    self.progress = None;
                    self.loading = false;
                    self.cancel_flag = None;
                    self.scan_order = None;
                    self.scan_full = false;
                    break;
                }
            };
            self.apply_worker_msg(msg);
        }
    }

    fn apply_worker_msg(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Progress(p) => self.progress = Some(p),
            WorkerMsg::ImageLoaded {
                path,
                sidecar,
                thumbnail,
                stamp,
            } => {
                if let Some(tex) = thumbnail {
                    self.thumbnails.insert(path.clone(), tex);
                }
                match stamp {
                    // No stamp means the `stat` failed, so there's nothing to
                    // compare against next time — forget any older one rather
                    // than let it claim the new texture is current.
                    Some(s) => self.stamps.insert(path.clone(), s),
                    None => self.stamps.remove(&path),
                };
                // A full load starts from an empty `images`, so skip the
                // linear search — over a few thousand files it would make
                // opening a folder quadratic for no possible hit.
                if self.scan_full {
                    self.images.push(ImageItem {
                        path,
                        sidecar: *sidecar,
                    });
                } else if let Some(existing) = self.images.iter_mut().find(|i| i.path == path) {
                    existing.sidecar = *sidecar;
                } else {
                    self.images.push(ImageItem {
                        path,
                        sidecar: *sidecar,
                    });
                }
            }
            WorkerMsg::SidecarReloaded { path, sidecar } => {
                if let Some(existing) = self.images.iter_mut().find(|i| i.path == path) {
                    existing.sidecar = *sidecar;
                }
            }
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
                    DoneKind::LoadFolder => {
                        // A differential scan appends new files as they load,
                        // which would leave them clustered at the end of the
                        // grid. Put `images` back into scan order — the order
                        // the user sees in their file manager. Anything the
                        // scan didn't cover (a cancelled run) keeps its
                        // relative position at the end.
                        if let Some(order) = self.scan_order.take() {
                            reorder_to_scan_order(&mut self.images, &order);
                        }
                        self.scan_full = false;
                    }
                    DoneKind::Tagger(t) => self.tagger = t,
                    DoneKind::Captioner(c) => self.captioner = c,
                    DoneKind::Booru => {}
                }
                self.progress = None;
                self.loading = false;
                self.worker_rx = None;
                self.cancel_flag = None;
            }
        }
    }

    fn ui_progress_overlay(&mut self, ctx: &egui::Context) {
        let Some(p) = self.progress.clone() else {
            return;
        };
        let t = self.t();
        let label = match p.op {
            WorkerOp::LoadFolder => t.op_loading_folder(),
            WorkerOp::Tagger => t.op_tagging(),
            WorkerOp::Captioner => t.op_captioning(),
            WorkerOp::Booru => t.op_fetching_booru(),
        };
        let frac = if p.total == 0 {
            0.0
        } else {
            (p.current as f32) / (p.total as f32)
        };
        // Already asked to stop? Show a spinner + "cancelling…" and disable the
        // button so a second click can't do anything — the worker stops at its
        // next iteration boundary.
        let cancelling = self
            .cancel_flag
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false);
        let mut request_cancel = false;
        egui::Window::new("fwaun-tools-progress")
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
                    ui.add_space(8.0);
                    if cancelling {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(t.cancelling());
                        });
                    } else if ui.button(t.cancel()).clicked() {
                        request_cancel = true;
                    }
                    ui.add_space(4.0);
                });
            });
        if request_cancel && let Some(flag) = &self.cancel_flag {
            flag.store(true, Ordering::Relaxed);
        }
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
    fn remove_caption_hint_at(&mut self, path: &Path, hint: &str) {
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path)
            && img.sidecar.remove_caption_hint(hint)
        {
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
    /// Toggle whether `tag` (an auto/booru entry) is suppressed on one image,
    /// accounting for the dataset-wide layer.
    ///
    /// A `common_tags` `-foo` can't be undone by dropping a marker the
    /// sidecar doesn't have, so un-suppressing against the shared layer
    /// writes a positive per-image entry instead — the same override the
    /// core merge rule looks for.
    fn toggle_suppression_at(&mut self, path: &Path, tag: &str) {
        let common_suppresses = self
            .common_tags
            .entry_for(tag)
            .is_some_and(|e| e.trim_start().starts_with('-'));
        let Some(img) = self.images.iter_mut().find(|i| i.path == path) else {
            return;
        };
        let merged = self.common_tags.merged_manual_tags(&img.sidecar);
        let effective_suppressed = fwaun_tools_core::sidecar::suppressed_stems(&merged)
            .contains(&tag.trim().to_lowercase());

        let mut changed = false;
        if effective_suppressed {
            changed |= img.sidecar.unsuppress(tag);
            if common_suppresses {
                changed |= img.sidecar.add_manual_tag(tag);
            }
        } else {
            changed |= img.sidecar.remove_manual_tag_ci(tag) > 0;
            changed |= img.sidecar.suppress(tag);
        }
        if changed {
            let _ = img.sidecar.save(&img.path);
        }
    }

    /// The dataset-wide entries in effect for `sc` — those it doesn't
    /// already override with an entry of its own.
    fn inherited_common_entries(&self, sc: &Sidecar) -> Vec<String> {
        if self.common_tags.is_empty() {
            return Vec::new();
        }
        let owned: HashSet<String> = sc
            .manual_tags
            .iter()
            .map(|t| common_tags::tag_stem(t))
            .collect();
        self.common_tags
            .entries()
            .iter()
            .filter(|e| !owned.contains(&common_tags::tag_stem(e)))
            .cloned()
            .collect()
    }

    /// Cancel a dataset-wide entry for `path` alone by writing the opposite
    /// form into its sidecar.
    fn override_common_entry_at(&mut self, path: &Path, entry: &str) {
        let override_entry = common_override_entry(entry);
        if let Some(img) = self.images.iter_mut().find(|i| i.path == path)
            && img.sidecar.add_manual_tag(override_entry)
        {
            let _ = img.sidecar.save(&img.path);
        }
    }

    /// Handle a click on a dataset-tag chip in the bulk panel: remove it
    /// from the config when the selection is the whole dataset (the edit is
    /// about the dataset), otherwise override it across the selection.
    fn bulk_common_entry_action(&mut self, paths: &[PathBuf], entry: &str) {
        if self.is_whole_dataset_selection() {
            self.edit_common_tag(entry, true);
            return;
        }
        let override_entry = common_override_entry(entry);
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            if img.sidecar.add_manual_tag(override_entry.clone()) {
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
    fn bulk_remove_caption_hint(&mut self, paths: &[PathBuf], hint: &str) {
        for img in self.images.iter_mut() {
            if !paths.contains(&img.path) {
                continue;
            }
            if img.sidecar.remove_caption_hint(hint) {
                let _ = img.sidecar.save(&img.path);
            }
        }
    }
}

// ───────── Helpers ─────────

#[derive(Clone, Copy)]
enum ChipKind {
    Manual,
    Negative,
    /// Curation-only `_foo` manual entry: kept in the data and counted for
    /// tag-group classification, but never exported. Coloured distinctly so
    /// the user can tell at a glance it won't reach the training caption.
    Organizational,
    Auto,
    Booru,
    /// Entry from the dataset-wide `common_tags` layer in
    /// `fwaun-tools.toml`. Not stored in this image's sidecar, so it gets
    /// its own colour — clicking it either overrides it for the selection
    /// or (at the dataset root, everything selected) edits the config.
    Common,
}

impl ChipKind {
    fn fill(self) -> egui::Color32 {
        match self {
            Self::Manual => egui::Color32::from_rgb(45, 74, 110),
            Self::Negative => egui::Color32::from_rgb(90, 45, 45),
            Self::Organizational => egui::Color32::from_rgb(74, 58, 100),
            Self::Auto => egui::Color32::from_rgb(58, 58, 58),
            Self::Booru => egui::Color32::from_rgb(45, 90, 58),
            Self::Common => egui::Color32::from_rgb(96, 74, 40),
        }
    }
    fn fg(self) -> egui::Color32 {
        match self {
            Self::Manual => egui::Color32::from_rgb(207, 227, 255),
            Self::Negative => egui::Color32::from_rgb(255, 208, 208),
            Self::Organizational => egui::Color32::from_rgb(226, 213, 255),
            Self::Auto => egui::Color32::from_rgb(204, 204, 204),
            Self::Booru => egui::Color32::from_rgb(207, 229, 208),
            Self::Common => egui::Color32::from_rgb(255, 226, 184),
        }
    }
}

/// The per-image manual entry that cancels a dataset-wide `entry` for one
/// image: the opposite form of the same tag, which the core merge rule
/// treats as an override.
fn common_override_entry(entry: &str) -> String {
    let t = entry.trim();
    match t.strip_prefix('-') {
        Some(positive) => positive.trim().to_string(),
        None => format!("-{}", t.trim_start_matches('_')),
    }
}

/// Pick the chip colour for a raw manual entry: `-foo` suppression markers,
/// `_foo` curation-only organizational tags, or plain positive tags.
fn manual_chip_kind(tag: &str) -> ChipKind {
    if tag.trim_start().starts_with('-') {
        ChipKind::Negative
    } else if is_organizational(tag) {
        ChipKind::Organizational
    } else {
        ChipKind::Manual
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

/// Produce the thumbnail image for `path`, from the on-disk cache when it has
/// one and by decoding the source when it doesn't.
///
/// A missing stamp (the `stat` failed) bypasses the cache in both directions
/// rather than key an entry that could never be invalidated.
fn make_thumbnail_image(
    path: &Path,
    stamp: Option<FileStamp>,
    max_size: u32,
    cache: Option<&ThumbCache>,
) -> Option<ColorImage> {
    let slot = cache.zip(stamp.map(|s| CacheKey::new(path, s, max_size)));
    if let Some(cached) = slot.as_ref().and_then(|(c, k)| c.load(k)) {
        return Some(cached);
    }
    let img = image::open(path).ok()?;
    let thumb = img.thumbnail(max_size, max_size).to_rgba8();
    let size = [thumb.width() as usize, thumb.height() as usize];
    let pixels: Vec<u8> = thumb.into_raw();
    if let Some((c, k)) = slot.as_ref() {
        c.store(k, size, &pixels);
    }
    Some(ColorImage::from_rgba_unmultiplied(size, &pixels))
}

/// [`make_thumbnail_image`] plus the texture upload. Called from the scan
/// worker: `Context::load_texture` is internally synchronized, so the whole
/// decode-and-upload stays off the UI thread.
fn make_thumbnail_texture(
    path: &Path,
    stamp: Option<FileStamp>,
    max_size: u32,
    ctx: &egui::Context,
    cache: Option<&ThumbCache>,
) -> Option<TextureHandle> {
    let color_image = make_thumbnail_image(path, stamp, max_size, cache)?;
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

/// Tags that at least two of the selected images share, most-frequent first.
/// A read-only curation aid — distinct from the dataset-wide `common_tags`
/// layer, which comes from the config and actually affects the export.
fn compute_shared_tags(items: &[ImageItem]) -> Vec<(String, usize)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(mtime_ns: u128, size: u64) -> FileStamp {
        FileStamp { mtime_ns, size }
    }

    fn item(path: &str) -> ImageItem {
        ImageItem {
            path: PathBuf::from(path),
            sidecar: Sidecar::default(),
        }
    }

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn full_scan_always_regenerates() {
        let s = stamp(1, 1);
        assert!(needs_thumbnail(true, true, true, Some(&s), Some(&s)));
    }

    #[test]
    fn unchanged_file_keeps_its_texture() {
        let s = stamp(1, 1);
        assert!(!needs_thumbnail(false, true, true, Some(&s), Some(&s)));
    }

    #[test]
    fn new_or_untextured_file_regenerates() {
        let s = stamp(1, 1);
        // Not in `images` yet — a file added since the last scan.
        assert!(needs_thumbnail(false, false, false, None, Some(&s)));
        // Known, but an earlier scan was cancelled before its texture landed.
        assert!(needs_thumbnail(false, true, false, Some(&s), Some(&s)));
    }

    #[test]
    fn moved_stamp_regenerates() {
        let old = stamp(1, 100);
        // Rewritten in place: same size, newer mtime.
        assert!(needs_thumbnail(
            false,
            true,
            true,
            Some(&old),
            Some(&stamp(2, 100))
        ));
        // Same mtime, different size — a copy that preserved timestamps.
        assert!(needs_thumbnail(
            false,
            true,
            true,
            Some(&old),
            Some(&stamp(1, 101))
        ));
    }

    #[test]
    fn missing_stamp_regenerates() {
        let s = stamp(1, 1);
        // `stat` failed now, or never succeeded before: nothing to compare, so
        // don't let an unreadable file pin a stale thumbnail forever.
        assert!(needs_thumbnail(false, true, true, Some(&s), None));
        assert!(needs_thumbnail(false, true, true, None, Some(&s)));
        assert!(needs_thumbnail(false, true, true, None, None));
    }

    #[test]
    fn reorder_places_new_files_among_their_neighbours() {
        // b.png appeared between a and c since the last scan, so it loaded
        // last and got appended.
        let mut images = vec![item("a.png"), item("c.png"), item("b.png")];
        reorder_to_scan_order(&mut images, &paths(&["a.png", "b.png", "c.png"]));
        let got: Vec<_> = images.iter().map(|i| i.path.to_str().unwrap()).collect();
        assert_eq!(got, ["a.png", "b.png", "c.png"]);
    }

    #[test]
    fn reorder_keeps_unscanned_items_at_the_end() {
        // A cancelled scan never reached c.png; it keeps its entry rather
        // than vanishing from the grid.
        let mut images = vec![item("c.png"), item("b.png"), item("a.png")];
        reorder_to_scan_order(&mut images, &paths(&["a.png", "b.png"]));
        let got: Vec<_> = images.iter().map(|i| i.path.to_str().unwrap()).collect();
        assert_eq!(got, ["a.png", "b.png", "c.png"]);
    }

    #[test]
    fn reorder_of_an_empty_order_is_a_no_op() {
        let mut images = vec![item("b.png"), item("a.png")];
        reorder_to_scan_order(&mut images, &[]);
        let got: Vec<_> = images.iter().map(|i| i.path.to_str().unwrap()).collect();
        assert_eq!(got, ["b.png", "a.png"]);
    }

    #[test]
    fn age_limit_maps_zero_to_no_expiry() {
        let mut prefs = GuiPrefs::default();
        prefs.thumb_cache.max_age_days = 0;
        assert_eq!(age_limit(&prefs), None);
        prefs.thumb_cache.max_age_days = 2;
        assert_eq!(age_limit(&prefs), Some(Duration::from_secs(2 * 86_400)));
    }

    #[test]
    fn cache_handle_follows_the_enabled_flag() {
        let mut prefs = GuiPrefs::default();
        prefs.thumb_cache.enabled = false;
        assert!(cache_for(&prefs).is_none());
    }

    /// End-to-end over the path a folder scan actually takes: decode a real
    /// file, populate the cache, then serve the same request from it. The
    /// second call must not touch the source, which the test enforces by
    /// deleting it in between.
    #[test]
    fn thumbnail_falls_back_to_the_cache_when_the_source_is_gone() {
        let dir = std::env::temp_dir().join(format!("fwaun-thumb-e2e-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let src = dir.join("img.png");
        image::RgbaImage::from_pixel(512, 256, image::Rgba([10, 20, 30, 255]))
            .save(&src)
            .unwrap();

        let cache = ThumbCache::at(dir.join("cache"));
        let stamp = FileStamp::of(&src);
        assert!(stamp.is_some());

        let first = make_thumbnail_image(&src, stamp, THUMB_SIZE, Some(&cache));
        assert!(first.is_some());
        assert!(cache.total_size() > 0, "the miss should have populated it");

        fs::remove_file(&src).unwrap();
        let second = make_thumbnail_image(&src, stamp, THUMB_SIZE, Some(&cache));
        let second = second.expect("served from cache, source not needed");
        // 512x256 shrunk to fit THUMB_SIZE, aspect preserved — the cached
        // entry has to round-trip those dimensions, not the source's.
        assert_eq!(second.size, [256, 128]);
        assert_eq!(second.pixels, first.unwrap().pixels);

        // Without a cache the same call has nothing to fall back on.
        assert!(make_thumbnail_image(&src, stamp, THUMB_SIZE, None).is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
