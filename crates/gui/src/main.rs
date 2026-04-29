use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anima_tagger_booru::{BooruClient, BooruError};
use anima_tagger_captioner::Captioner;
use anima_tagger_core::config::ProjectConfig;
use anima_tagger_core::sidecar::{Sidecar, TaggerInfo};
use anima_tagger_core::walk::iter_images;
use anima_tagger_tagger::Tagger;
use base64::Engine;
use chrono::Utc;
use dioxus::prelude::*;
use image::ImageFormat;

const THUMB_SIZE: u32 = 256;

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, PartialEq)]
struct ImageItem {
    path: PathBuf,
    thumbnail: String,
    sidecar: Sidecar,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    All,
    Untagged,
    AutoTagged,
    NoManual,
    NoCaption,
    NoBooru,
}

impl Filter {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Untagged => "Untagged",
            Self::AutoTagged => "Auto-tagged",
            Self::NoManual => "No manual tags",
            Self::NoCaption => "No caption",
            Self::NoBooru => "No booru",
        }
    }
    fn matches(self, item: &ImageItem) -> bool {
        match self {
            Self::All => true,
            Self::Untagged => !item.sidecar.is_auto_tagged() && item.sidecar.manual_tags.is_empty(),
            Self::AutoTagged => item.sidecar.is_auto_tagged(),
            Self::NoManual => item.sidecar.manual_tags.is_empty(),
            Self::NoCaption => !item.sidecar.is_captioned(),
            Self::NoBooru => !item.sidecar.has_booru(),
        }
    }
}

#[component]
fn App() -> Element {
    let folder = use_signal(|| None::<PathBuf>);
    let images = use_signal(Vec::<ImageItem>::new);
    let selected = use_signal(HashSet::<PathBuf>::new);
    let filter = use_signal(|| Filter::All);
    let tag_filter = use_signal(String::new);
    let loading = use_signal(|| false);
    let tag_input = use_signal(String::new);
    let error_msg = use_signal(|| None::<String>);
    let tagger_state: Signal<Option<Tagger>> = use_signal(|| None);
    let captioner_state: Signal<Option<Captioner>> = use_signal(|| None);

    let tag_query = tag_filter.read().trim().to_lowercase();
    let visible: Vec<ImageItem> = images
        .read()
        .iter()
        .filter(|i| filter.read().matches(i))
        .filter(|i| tag_query.is_empty() || matches_tag_query(i, &tag_query))
        .cloned()
        .collect();

    rsx! {
        style { {APP_CSS} }
        div { class: "app",
            Toolbar {
                folder, images, selected, filter, tag_filter, loading,
                error_msg, tagger_state, captioner_state,
            }
            ErrorBanner { error_msg }
            div { class: "workspace",
                Grid { items: visible, selected }
                DetailPanel { images, selected, tag_input }
            }
        }
    }
}

/// Substring match (case-insensitive) against any of an image's manual /
/// auto / booru tag stems. Used by the toolbar tag-filter so users can
/// narrow to e.g. all images with `1girl` and bulk-edit them — say,
/// suppress `-1girl` and add a manual replacement like `woman`.
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

#[component]
fn ErrorBanner(mut error_msg: Signal<Option<String>>) -> Element {
    let msg = error_msg.read().clone();
    match msg {
        None => rsx! {},
        Some(m) => rsx! {
            div { class: "error-banner",
                span { "{m}" }
                button {
                    class: "dismiss",
                    onclick: move |_| error_msg.set(None),
                    "×"
                }
            }
        },
    }
}

#[component]
fn Toolbar(
    folder: Signal<Option<PathBuf>>,
    mut images: Signal<Vec<ImageItem>>,
    mut selected: Signal<HashSet<PathBuf>>,
    mut filter: Signal<Filter>,
    mut tag_filter: Signal<String>,
    mut loading: Signal<bool>,
    error_msg: Signal<Option<String>>,
    tagger_state: Signal<Option<Tagger>>,
    captioner_state: Signal<Option<Captioner>>,
) -> Element {
    let on_open = move |_| {
        let Some(picked) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        loading.set(true);
        let loaded = load_folder(&picked);
        let mut f = folder;
        f.set(Some(picked));
        images.set(loaded);
        selected.set(HashSet::new());
        // Folder change invalidates lazily-loaded models (different config may
        // point at different model paths).
        let mut t = tagger_state;
        t.set(None);
        let mut c = captioner_state;
        c.set(None);
        loading.set(false);
    };

    let select_all_visible = move |_| {
        let imgs = images.read();
        let cur_filter = *filter.read();
        let tag_query = tag_filter.read().trim().to_lowercase();
        let new_sel: HashSet<PathBuf> = imgs
            .iter()
            .filter(|i| cur_filter.matches(i))
            .filter(|i| tag_query.is_empty() || matches_tag_query(i, &tag_query))
            .map(|i| i.path.clone())
            .collect();
        selected.set(new_sel);
    };

    let clear_selection = move |_| selected.set(HashSet::new());

    let folder_label = match folder.read().as_ref() {
        Some(p) => p
            .file_name()
            .and_then(|s| s.to_str())
            .map(String::from)
            .unwrap_or_else(|| p.display().to_string()),
        None => "(no folder)".to_string(),
    };
    let count = images.read().len();
    let sel_count = selected.read().len();
    let has_sel = sel_count > 0;
    let folder_set = folder.read().is_some();

    rsx! {
        div { class: "toolbar",
            button { onclick: on_open, "Open folder…" }
            span { class: "folder-name", "{folder_label}" }
            select {
                value: "{filter.read().label()}",
                onchange: move |evt| {
                    let f = match evt.value().as_str() {
                        "Untagged" => Filter::Untagged,
                        "Auto-tagged" => Filter::AutoTagged,
                        "No manual tags" => Filter::NoManual,
                        "No caption" => Filter::NoCaption,
                        "No booru" => Filter::NoBooru,
                        _ => Filter::All,
                    };
                    filter.set(f);
                },
                option { value: "All", "All" }
                option { value: "Untagged", "Untagged" }
                option { value: "Auto-tagged", "Auto-tagged" }
                option { value: "No manual tags", "No manual tags" }
                option { value: "No caption", "No caption" }
                option { value: "No booru", "No booru" }
            }
            input {
                class: "tag-filter",
                placeholder: "filter by tag…",
                value: "{tag_filter}",
                oninput: move |evt| tag_filter.set(evt.value()),
            }
            button {
                class: "secondary",
                onclick: select_all_visible,
                disabled: !folder_set,
                "Select visible"
            }
            button {
                class: "secondary",
                onclick: clear_selection,
                disabled: !has_sel,
                "Clear sel."
            }
            span { class: "spacer" }
            button {
                onclick: move |_| run_tagger(folder, images, selected, tagger_state, error_msg, loading),
                disabled: !has_sel || *loading.read(),
                "Run tagger"
            }
            button {
                onclick: move |_| run_captioner(folder, images, selected, captioner_state, error_msg, loading),
                disabled: !has_sel || *loading.read(),
                "Run captioner"
            }
            button {
                onclick: move |_| run_booru(images, selected, error_msg, loading),
                disabled: !has_sel || *loading.read(),
                "Fetch booru"
            }
            if *loading.read() { span { class: "muted", "Working…" } }
            span { class: "muted", "{count} images · {sel_count} selected" }
        }
    }
}

#[component]
fn Grid(items: Vec<ImageItem>, mut selected: Signal<HashSet<PathBuf>>) -> Element {
    if items.is_empty() {
        return rsx! { div { class: "grid empty", p { class: "muted", "No images." } } };
    }
    rsx! {
        div { class: "grid",
            for item in items.iter().cloned() {
                Thumb { item: item.clone(), selected }
            }
        }
    }
}

#[component]
fn Thumb(item: ImageItem, mut selected: Signal<HashSet<PathBuf>>) -> Element {
    let is_selected = selected.read().contains(&item.path);
    let class = if is_selected { "thumb selected" } else { "thumb" };
    let auto_flag = if item.sidecar.is_auto_tagged() { "T" } else { " " };
    let cap_flag = if item.sidecar.is_captioned() { "C" } else { " " };
    let booru_flag = if item.sidecar.has_booru() { "B" } else { " " };
    let manual_flag = if !item.sidecar.manual_tags.is_empty() {
        "M"
    } else {
        " "
    };
    let path_for_click = item.path.clone();

    rsx! {
        div {
            class: "{class}",
            onclick: move |evt| {
                let mods = evt.modifiers();
                let mut sel = selected.write();
                let multi = mods.ctrl() || mods.meta() || mods.shift();
                if multi {
                    if sel.contains(&path_for_click) {
                        sel.remove(&path_for_click);
                    } else {
                        sel.insert(path_for_click.clone());
                    }
                } else {
                    sel.clear();
                    sel.insert(path_for_click.clone());
                }
            },
            img { src: "{item.thumbnail}" }
            span { class: "thumb-status", "{auto_flag}{cap_flag}{booru_flag}{manual_flag}" }
        }
    }
}

#[component]
fn DetailPanel(
    mut images: Signal<Vec<ImageItem>>,
    selected: Signal<HashSet<PathBuf>>,
    mut tag_input: Signal<String>,
) -> Element {
    let sel_paths: Vec<PathBuf> = selected.read().iter().cloned().collect();
    let n = sel_paths.len();

    if n == 0 {
        return rsx! {
            aside { class: "detail",
                p { class: "muted", "Select one or more images to edit tags." }
                p { class: "muted small",
                    "Tip: type "
                    code { "-tag" }
                    " in the input to suppress an auto/booru tag (it stays in the data but is hidden from export)."
                }
            }
        };
    }

    let imgs_snapshot = images.read().clone();

    let mut do_add = move |raw: String| {
        let tag = raw.trim().to_string();
        if tag.is_empty() {
            return;
        }
        let sel = selected.read().clone();
        let mut imgs = images.write();
        for img in imgs.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            if img.sidecar.add_manual_tag(tag.clone()) {
                let _ = img.sidecar.save(&img.path);
            }
        }
    };

    rsx! {
        aside { class: "detail",
            if n == 1 {
                if let Some(item) = imgs_snapshot.iter().find(|i| i.path == sel_paths[0]) {
                    SingleDetail {
                        key: "{item.path.display()}",
                        item: item.clone(),
                        images, selected, tag_input,
                    }
                }
            } else {
                BulkDetail {
                    items: imgs_snapshot.clone(),
                    selected_paths: sel_paths.clone(),
                    images, tag_input,
                }
            }

            div { class: "input-row",
                input {
                    placeholder: "tag, or -tag to suppress",
                    value: "{tag_input}",
                    oninput: move |evt| tag_input.set(evt.value()),
                    onkeydown: move |evt: KeyboardEvent| {
                        if evt.key() == Key::Enter {
                            let v = tag_input.read().clone();
                            do_add(v);
                            tag_input.set(String::new());
                        }
                    },
                }
                button {
                    onclick: move |_| {
                        let v = tag_input.read().clone();
                        do_add(v);
                        tag_input.set(String::new());
                    },
                    "Add"
                }
            }
        }
    }
}

#[component]
fn SingleDetail(
    item: ImageItem,
    images: Signal<Vec<ImageItem>>,
    selected: Signal<HashSet<PathBuf>>,
    tag_input: Signal<String>,
) -> Element {
    let _ = selected;
    let _ = tag_input;
    let path = item.path.clone();
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let manual_positives: Vec<String> = item
        .sidecar
        .manual_positive_tags()
        .map(|s| s.to_string())
        .collect();

    rsx! {
        p { class: "filename", "{filename}" }

        div { class: "section-title", "Tags" }
        if manual_positives.is_empty() && item.sidecar.auto_tags.is_empty() && item.sidecar.booru_tags.is_empty() {
            p { class: "muted", "(none yet — add manual or run tagger/booru)" }
        } else {
            div { class: "tag-list",
                for tag in manual_positives.iter().cloned() {
                    {
                        let path_for = path.clone();
                        let tag_for = tag.clone();
                        rsx! {
                            span { class: "chip manual",
                                "{tag}"
                                span {
                                    class: "chip-x",
                                    onclick: move |_| remove_manual_at(images, path_for.clone(), tag_for.clone()),
                                    "×"
                                }
                            }
                        }
                    }
                }
                for at in item.sidecar.auto_tags.iter().cloned() {
                    {
                        let suppressed = item.sidecar.is_suppressed(&at.tag);
                        let cls = if suppressed { "chip auto suppressed" } else { "chip auto" };
                        let path_for = path.clone();
                        let tag_for = at.tag.clone();
                        rsx! {
                            span { class: "{cls}",
                                "{at.tag}"
                                span { class: "score", "{at.score:.2}" }
                                span {
                                    class: "chip-x",
                                    onclick: move |_| toggle_suppression_at(images, path_for.clone(), tag_for.clone()),
                                    "×"
                                }
                            }
                        }
                    }
                }
                for bt in item.sidecar.booru_tags.iter().cloned() {
                    {
                        let suppressed = item.sidecar.is_suppressed(&bt.tag);
                        let cls = if suppressed { "chip booru suppressed" } else { "chip booru" };
                        let path_for = path.clone();
                        let tag_for = bt.tag.clone();
                        rsx! {
                            span { class: "{cls}",
                                "{bt.tag}"
                                span { class: "src", "B" }
                                span {
                                    class: "chip-x",
                                    onclick: move |_| toggle_suppression_at(images, path_for.clone(), tag_for.clone()),
                                    "×"
                                }
                            }
                        }
                    }
                }
            }
        }

        div { class: "section-title", "Caption (manual — exported)" }
        {
            let manual_text = item.sidecar.manual_caption.clone().unwrap_or_default();
            let editor_key = format!("{}::{}", path.display(), manual_text);
            rsx! {
                ManualCaptionEditor {
                    key: "{editor_key}",
                    path: path.clone(),
                    images,
                    initial: manual_text,
                }
            }
        }

        div { class: "section-title", "Auto captions" }
        if item.sidecar.captions.is_empty() {
            p { class: "muted small", "(none — run captioner)" }
        } else {
            for (model, entry) in item.sidecar.captions.iter() {
                {
                    let model_name = model.clone();
                    let model_for_skip = model.clone();
                    let model_for_remove = model.clone();
                    let text = entry.caption.clone();
                    let text_for_copy = text.clone();
                    let path_for_copy = path.clone();
                    let path_for_skip = path.clone();
                    let path_for_remove = path.clone();
                    let skipped = entry.skip;
                    let block_class = if skipped { "auto-caption skipped" } else { "auto-caption" };
                    let skip_label = if skipped { "unskip" } else { "skip" };
                    let skip_title = if skipped {
                        "Re-enable this caption for export"
                    } else {
                        "Keep this caption stored but exclude from export"
                    };
                    rsx! {
                        div { class: "{block_class}",
                            div { class: "auto-caption-head",
                                span { class: "model-name", "{model_name}" }
                                button {
                                    class: "tiny",
                                    title: "Copy this caption into the manual caption field",
                                    onclick: move |_| copy_caption_to_manual(images, path_for_copy.clone(), text_for_copy.clone()),
                                    "→ manual"
                                }
                                button {
                                    class: "tiny secondary",
                                    title: "{skip_title}",
                                    onclick: move |_| toggle_caption_skip_at(images, path_for_skip.clone(), model_for_skip.clone()),
                                    "{skip_label}"
                                }
                                button {
                                    class: "tiny secondary",
                                    title: "Remove this auto caption",
                                    onclick: move |_| remove_caption_at(images, path_for_remove.clone(), model_for_remove.clone()),
                                    "×"
                                }
                            }
                            p { class: "caption", "{text}" }
                        }
                    }
                }
            }
        }
        if let Some(b) = item.sidecar.booru.as_ref() {
            div { class: "section-title", "Booru" }
            p { class: "muted small",
                "{b.source}"
                if let Some(id) = b.post_id { ": #{id}" }
            }
        }
    }
}

#[component]
fn ManualCaptionEditor(
    path: PathBuf,
    images: Signal<Vec<ImageItem>>,
    initial: String,
) -> Element {
    let mut buf = use_signal(|| initial.clone());
    let path_for_change = path.clone();
    rsx! {
        textarea {
            class: "manual-caption",
            placeholder: "Manual caption (e.g. \"Left girl is Alice. Right girl is Bob.\") — prepended to auto caption on export. Click outside to save.",
            value: "{buf}",
            rows: "3",
            oninput: move |evt| buf.set(evt.value()),
            onchange: move |evt| save_manual_caption(images, path_for_change.clone(), evt.value()),
        }
    }
}

fn save_manual_caption(mut images: Signal<Vec<ImageItem>>, path: PathBuf, text: String) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path) {
        img.sidecar.set_manual_caption(&text);
        let _ = img.sidecar.save(&img.path);
    }
}

fn copy_caption_to_manual(mut images: Signal<Vec<ImageItem>>, path: PathBuf, text: String) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path) {
        img.sidecar.set_manual_caption(&text);
        let _ = img.sidecar.save(&img.path);
    }
}

fn remove_caption_at(mut images: Signal<Vec<ImageItem>>, path: PathBuf, model: String) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path)
        && img.sidecar.remove_caption(&model)
    {
        let _ = img.sidecar.save(&img.path);
    }
}

fn toggle_caption_skip_at(
    mut images: Signal<Vec<ImageItem>>,
    path: PathBuf,
    model: String,
) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path)
        && img.sidecar.toggle_caption_skip(&model).is_some()
    {
        let _ = img.sidecar.save(&img.path);
    }
}

fn remove_manual_at(mut images: Signal<Vec<ImageItem>>, path: PathBuf, tag: String) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path)
        && img.sidecar.remove_manual_tag(&tag)
    {
        let _ = img.sidecar.save(&img.path);
    }
}

fn toggle_suppression_at(mut images: Signal<Vec<ImageItem>>, path: PathBuf, tag: String) {
    let mut imgs = images.write();
    if let Some(img) = imgs.iter_mut().find(|i| i.path == path) {
        let changed = if img.sidecar.is_suppressed(&tag) {
            img.sidecar.unsuppress(&tag)
        } else {
            img.sidecar.suppress(&tag)
        };
        if changed {
            let _ = img.sidecar.save(&img.path);
        }
    }
}

fn bulk_remove_manual(mut images: Signal<Vec<ImageItem>>, paths: Vec<PathBuf>, tag: String) {
    let mut imgs = images.write();
    for img in imgs.iter_mut() {
        if !paths.contains(&img.path) {
            continue;
        }
        if img.sidecar.remove_manual_tag(&tag) {
            let _ = img.sidecar.save(&img.path);
        }
    }
}

#[component]
fn BulkDetail(
    items: Vec<ImageItem>,
    selected_paths: Vec<PathBuf>,
    mut images: Signal<Vec<ImageItem>>,
    tag_input: Signal<String>,
) -> Element {
    let _ = tag_input;
    let n = selected_paths.len();
    let selected_items: Vec<&ImageItem> = items
        .iter()
        .filter(|i| selected_paths.contains(&i.path))
        .collect();

    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for item in &selected_items {
        for tag in &item.sidecar.manual_tags {
            if !counts.contains_key(tag) {
                order.push(tag.clone());
            }
            *counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    rsx! {
        p { class: "muted", "{n} images selected — bulk edit" }
        div { class: "section-title", "Manual entries (union)" }
        if order.is_empty() {
            p { class: "muted", "(none)" }
        } else {
            div { class: "tag-list",
                for tag in order.into_iter() {
                    {
                        let count = counts[&tag];
                        let label = if count < n {
                            format!("{tag} ({count}/{n})")
                        } else {
                            tag.clone()
                        };
                        let cls = if tag.starts_with('-') { "chip negative" } else { "chip manual" };
                        let paths_for = selected_paths.clone();
                        let tag_for = tag.clone();
                        rsx! {
                            span { class: "{cls}",
                                "{label}"
                                span {
                                    class: "chip-x",
                                    onclick: move |_| bulk_remove_manual(images, paths_for.clone(), tag_for.clone()),
                                    "×"
                                }
                            }
                        }
                    }
                }
            }
        }
        p { class: "muted small",
            "Auto/booru tags hidden in bulk view; switch to single selection to edit per-tag."
        }
    }
}

// ───────── Long-running operations ─────────

fn run_tagger(
    folder: Signal<Option<PathBuf>>,
    mut images: Signal<Vec<ImageItem>>,
    selected: Signal<HashSet<PathBuf>>,
    mut tagger_state: Signal<Option<Tagger>>,
    mut error_msg: Signal<Option<String>>,
    mut loading: Signal<bool>,
) {
    let Some(folder_path) = folder.read().clone() else {
        error_msg.set(Some("Open a folder first.".into()));
        return;
    };
    let cfg = match ProjectConfig::load_or_default(&folder_path) {
        Ok(c) => c,
        Err(e) => {
            error_msg.set(Some(e.to_string()));
            return;
        }
    };
    let (model_name, profile) = cfg.resolve_tagger(None);

    let sel: Vec<PathBuf> = selected.read().iter().cloned().collect();
    if sel.is_empty() {
        return;
    }

    loading.set(true);

    {
        let mut t = tagger_state.write();
        if t.is_none() {
            match Tagger::from_profile(&profile) {
                Ok(loaded) => *t = Some(loaded),
                Err(e) => {
                    error_msg.set(Some(format!("tagger load: {e}")));
                    loading.set(false);
                    return;
                }
            }
        }
    }

    {
        let mut t = tagger_state.write();
        let tagger_inst = t.as_mut().unwrap();
        let mut imgs = images.write();
        let now = Utc::now();
        for img in imgs.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            match tagger_inst.tag_image(&img.path, profile.storage_threshold) {
                Ok(tags) => {
                    img.sidecar.auto_tags = tags;
                    img.sidecar.tagger = Some(TaggerInfo {
                        model: model_name.clone(),
                        tagged_at: now,
                    });
                    let _ = img.sidecar.save(&img.path);
                }
                Err(e) => {
                    error_msg.set(Some(format!("{}: {e}", img.path.display())));
                }
            }
        }
    }
    loading.set(false);
}

fn run_captioner(
    folder: Signal<Option<PathBuf>>,
    mut images: Signal<Vec<ImageItem>>,
    selected: Signal<HashSet<PathBuf>>,
    mut captioner_state: Signal<Option<Captioner>>,
    mut error_msg: Signal<Option<String>>,
    mut loading: Signal<bool>,
) {
    let Some(folder_path) = folder.read().clone() else {
        error_msg.set(Some("Open a folder first.".into()));
        return;
    };
    let cfg = match ProjectConfig::load_or_default(&folder_path) {
        Ok(c) => c,
        Err(e) => {
            error_msg.set(Some(e.to_string()));
            return;
        }
    };
    let (model_name, profile) = cfg.resolve_captioner(None);

    let sel: Vec<PathBuf> = selected.read().iter().cloned().collect();
    if sel.is_empty() {
        return;
    }

    loading.set(true);

    {
        let mut c = captioner_state.write();
        if c.is_none() {
            match Captioner::from_profile(&profile) {
                Ok(loaded) => *c = Some(loaded),
                Err(e) => {
                    error_msg.set(Some(format!("captioner load: {e}")));
                    loading.set(false);
                    return;
                }
            }
        }
    }

    {
        let mut c = captioner_state.write();
        let captioner_inst = c.as_mut().unwrap();
        let mut imgs = images.write();
        for img in imgs.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            match captioner_inst.caption_image(&img.path) {
                Ok(caption) => {
                    img.sidecar.set_caption(model_name.clone(), caption);
                    let _ = img.sidecar.save(&img.path);
                }
                Err(e) => {
                    error_msg.set(Some(format!("{}: {e}", img.path.display())));
                }
            }
        }
    }
    loading.set(false);
}

fn run_booru(
    mut images: Signal<Vec<ImageItem>>,
    selected: Signal<HashSet<PathBuf>>,
    mut error_msg: Signal<Option<String>>,
    mut loading: Signal<bool>,
) {
    let sel: Vec<PathBuf> = selected.read().iter().cloned().collect();
    if sel.is_empty() {
        return;
    }
    loading.set(true);

    let client = BooruClient::danbooru();
    {
        let mut imgs = images.write();
        for img in imgs.iter_mut() {
            if !sel.contains(&img.path) {
                continue;
            }
            match client.fetch_for_image(&img.path) {
                Ok((tags, info)) => {
                    img.sidecar.booru_tags = tags;
                    img.sidecar.booru = Some(info);
                    let _ = img.sidecar.save(&img.path);
                }
                Err(BooruError::NotFound(_)) => {
                    // not on booru — silent skip, no error banner
                }
                Err(e) => {
                    error_msg.set(Some(format!("{}: {e}", img.path.display())));
                }
            }
        }
    }
    loading.set(false);
}

// ───────── I/O helpers ─────────

fn load_folder(dir: &Path) -> Vec<ImageItem> {
    let mut out = Vec::new();
    for path in iter_images(dir) {
        let sidecar = Sidecar::load_or_default(&path).unwrap_or_default();
        let thumbnail = make_thumbnail(&path, THUMB_SIZE).unwrap_or_default();
        out.push(ImageItem {
            path,
            thumbnail,
            sidecar,
        });
    }
    out
}

fn make_thumbnail(path: &Path, max_size: u32) -> anyhow::Result<String> {
    let img = image::open(path)?;
    let thumb = img.thumbnail(max_size, max_size).to_rgb8();
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(thumb).write_to(&mut Cursor::new(&mut buf), ImageFormat::Jpeg)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

const APP_CSS: &str = r#"
* { box-sizing: border-box; }
html, body, #main { margin: 0; height: 100%; }
body {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    background: #1e1e1e;
    color: #e6e6e6;
    font-size: 13px;
}
.app { display: flex; flex-direction: column; height: 100vh; }
.toolbar {
    padding: 8px 12px;
    border-bottom: 1px solid #333;
    background: #252526;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}
.toolbar .spacer { flex: 1; }
.toolbar .folder-name {
    color: #ccc; font-size: 12px;
    max-width: 240px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.toolbar button, .input-row button {
    background: #4a9eff; color: white; border: none;
    padding: 5px 12px; border-radius: 4px;
    cursor: pointer; font-size: 12px;
}
.toolbar button:hover:not(:disabled), .input-row button:hover { background: #5fa8ff; }
.toolbar button:disabled { background: #333; color: #666; cursor: not-allowed; }
.toolbar button.secondary { background: #3a3a3a; color: #e6e6e6; }
.toolbar button.secondary:hover:not(:disabled) { background: #4a4a4a; }
.toolbar select, .input-row input, .toolbar input.tag-filter {
    background: #2a2a2a; border: 1px solid #444; color: #e6e6e6;
    padding: 4px 8px; border-radius: 4px; font-size: 12px;
}
.toolbar input.tag-filter { width: 160px; }
.toolbar input.tag-filter:focus { outline: 1px solid #4a9eff; border-color: #4a9eff; }
.error-banner {
    background: #5a1f1f; color: #ffd0d0;
    padding: 6px 12px; border-bottom: 1px solid #732;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 12px;
}
.error-banner .dismiss {
    background: transparent; color: #ffd0d0; border: none;
    cursor: pointer; font-size: 16px; padding: 0 6px;
}
.workspace { display: flex; flex: 1; overflow: hidden; }
.grid {
    flex: 1; overflow-y: auto; padding: 12px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px;
    align-content: start;
}
.grid.empty { display: flex; align-items: center; justify-content: center; }
.thumb {
    aspect-ratio: 1;
    border: 2px solid transparent; border-radius: 4px;
    overflow: hidden; cursor: pointer;
    background: #2a2a2a; position: relative; user-select: none;
}
.thumb img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: contain; display: block; pointer-events: none;
}
.thumb.selected { border-color: #4a9eff; }
.thumb-status {
    position: absolute; top: 4px; right: 4px;
    font-size: 10px;
    background: rgba(0,0,0,0.65); color: #fff;
    padding: 2px 5px; border-radius: 2px;
    font-family: ui-monospace, monospace; white-space: pre;
}
.detail {
    width: 340px;
    border-left: 1px solid #333;
    overflow-y: auto;
    padding: 12px;
    background: #252526;
    display: flex; flex-direction: column;
}
.filename { font-family: ui-monospace, monospace; font-size: 11px; color: #aaa; margin: 0 0 4px; }
.section-title {
    font-size: 11px; text-transform: uppercase; color: #999;
    margin-top: 12px; margin-bottom: 4px; letter-spacing: 0.04em;
}
.tag-list { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
    padding: 3px 7px; border-radius: 12px;
    font-size: 12px;
    display: inline-flex; align-items: center; gap: 3px;
    line-height: 1.4;
}
.chip.manual { background: #2d4a6e; color: #cfe3ff; }
.chip.negative { background: #5a2d2d; color: #ffd0d0; }
.chip.auto { background: #3a3a3a; color: #ccc; }
.chip.booru { background: #2d5a3a; color: #cfe5d0; }
.chip.suppressed {
    text-decoration: line-through;
    opacity: 0.55;
}
.chip-x { cursor: pointer; opacity: 0.55; padding: 0 2px; font-weight: bold; }
.chip-x:hover { opacity: 1; }
.score { color: #888; font-size: 10px; margin-left: 2px; }
.src { color: #888; font-size: 10px; margin-left: 2px; }
.input-row { display: flex; gap: 6px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #333; }
.input-row input { flex: 1; }
.muted { color: #999; font-size: 12px; margin: 0; }
.muted.small { font-size: 11px; }
.caption { color: #ddd; font-size: 12px; line-height: 1.4; margin: 4px 0; }
.manual-caption {
    width: 100%;
    background: #1e2a3a;
    border: 1px solid #2d4a6e;
    color: #cfe3ff;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-family: inherit;
    resize: vertical;
    min-height: 60px;
}
.manual-caption:focus { outline: 1px solid #4a9eff; border-color: #4a9eff; }
.auto-caption {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 8px;
    margin-bottom: 6px;
}
.auto-caption.skipped {
    opacity: 0.55;
    border-style: dashed;
}
.auto-caption.skipped .caption { text-decoration: line-through; }
.auto-caption-head {
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 4px;
}
.model-name {
    flex: 1;
    color: #aaa;
    font-family: ui-monospace, monospace;
    font-size: 11px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
button.tiny {
    background: #2d4a6e; color: #cfe3ff; border: none;
    padding: 2px 6px; border-radius: 3px;
    font-size: 11px; cursor: pointer;
}
button.tiny:hover { background: #3a5a85; }
button.tiny.secondary { background: #3a3a3a; color: #ccc; }
button.tiny.secondary:hover { background: #4a4a4a; }
code {
    background: #2a2a2a; padding: 1px 4px; border-radius: 3px;
    font-family: ui-monospace, monospace; font-size: 11px;
}
"#;
