//! Form-based editor for `anima-tagger.toml`.
//!
//! Mirrors `ProjectConfig` as a `ConfigDraft` so the modal can mutate
//! it in-place across frames. Maps become ordered `Vec<(String, T)>`
//! to make key renames trivial and to surface duplicate names as a
//! save-time error rather than silently coalescing entries.

use std::collections::{BTreeMap, BTreeSet};

use anima_tagger_core::config::{
    BUILT_IN_CAPTIONER_REPO, BUILT_IN_CAPTIONER_SUBDIR, BUILT_IN_TAGGER_REPO, CaptionerProfile,
    ExportProfile, OnnxCaptionerProfile, OpenAiCaptionerProfile, ProjectConfig, TagGroup,
    TaggerProfile,
};
use eframe::egui;

use crate::i18n::T;

#[derive(Debug, Clone)]
pub struct ConfigDraft {
    pub default_profile: Option<String>,
    pub default_tagger: Option<String>,
    pub default_captioner: Option<String>,
    pub export: Vec<(String, ExportProfileDraft)>,
    pub tagger: Vec<(String, TaggerProfile)>,
    pub captioner: Vec<(String, CaptionerProfile)>,
    pub captioner_prompts: Vec<(String, String)>,
    pub tag_groups: Vec<(String, TagGroup)>,
}

#[derive(Debug, Clone)]
pub struct ExportProfileDraft {
    pub threshold: f32,
    pub shuffle: bool,
    pub exclude_categories: Vec<String>,
    pub category_prefixes: Vec<(String, String)>,
    pub caption_prefixes: Vec<(String, String)>,
    pub caption_suffixes: Vec<(String, String)>,
}

impl From<ExportProfile> for ExportProfileDraft {
    fn from(p: ExportProfile) -> Self {
        Self {
            threshold: p.threshold,
            shuffle: p.shuffle,
            exclude_categories: p.exclude_categories,
            category_prefixes: p.category_prefixes.into_iter().collect(),
            caption_prefixes: p.caption_prefixes.into_iter().collect(),
            caption_suffixes: p.caption_suffixes.into_iter().collect(),
        }
    }
}

impl ConfigDraft {
    pub fn from_config(cfg: ProjectConfig) -> Self {
        Self {
            default_profile: cfg.default_profile,
            default_tagger: cfg.default_tagger,
            default_captioner: cfg.default_captioner,
            export: cfg
                .export
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            tagger: cfg.tagger.into_iter().collect(),
            captioner: cfg.captioner.into_iter().collect(),
            captioner_prompts: cfg.captioner_prompts.into_iter().collect(),
            tag_groups: cfg.tag_groups.into_iter().collect(),
        }
    }

    /// Validate names + rebuild a `ProjectConfig`. Borrows so a save
    /// failure (duplicate names etc.) leaves the draft intact for the
    /// user to fix and retry. Errors are localized via `t`.
    pub fn to_config(&self, t: T) -> Result<ProjectConfig, String> {
        fn collect<V: Clone>(
            entries: &[(String, V)],
            section: &str,
            t: super::i18n::T,
        ) -> Result<BTreeMap<String, V>, String> {
            let mut out = BTreeMap::new();
            let mut seen = BTreeSet::new();
            for (raw, v) in entries {
                let name = raw.trim().to_string();
                if name.is_empty() {
                    return Err(t.cfg_err_empty_name(section));
                }
                if !seen.insert(name.clone()) {
                    return Err(t.cfg_err_duplicate_name(section, &name));
                }
                out.insert(name, v.clone());
            }
            Ok(out)
        }

        let mut export = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for (raw, draft) in &self.export {
            let name = raw.trim().to_string();
            if name.is_empty() {
                return Err(t.cfg_err_empty_name("export"));
            }
            if !seen.insert(name.clone()) {
                return Err(t.cfg_err_duplicate_name("export", &name));
            }
            // Validate prefix keys: empty key is meaningless, and duplicate
            // keys would silently merge in the BTreeMap. Same rule for the
            // category- and caption-prefix tables.
            let collect_prefixes = |entries: &[(String, String)], section: &str| {
                let mut prefixes = BTreeMap::new();
                let mut prefix_seen = BTreeSet::new();
                for (k_raw, v) in entries {
                    let k = k_raw.trim().to_string();
                    if k.is_empty() {
                        return Err(t.cfg_err_empty_name(&format!("export.{name}.{section}")));
                    }
                    if !prefix_seen.insert(k.clone()) {
                        return Err(
                            t.cfg_err_duplicate_name(&format!("export.{name}.{section}"), &k)
                        );
                    }
                    prefixes.insert(k, v.clone());
                }
                Ok(prefixes)
            };
            let category_prefixes = collect_prefixes(&draft.category_prefixes, "category_prefix")?;
            let caption_prefixes = collect_prefixes(&draft.caption_prefixes, "caption_prefix")?;
            let caption_suffixes = collect_prefixes(&draft.caption_suffixes, "caption_suffix")?;
            export.insert(
                name,
                ExportProfile {
                    threshold: draft.threshold,
                    shuffle: draft.shuffle,
                    exclude_categories: draft.exclude_categories.clone(),
                    category_prefixes,
                    caption_prefixes,
                    caption_suffixes,
                },
            );
        }

        let tagger = collect(&self.tagger, "tagger", t)?;
        let captioner = collect(&self.captioner, "captioner", t)?;
        let captioner_prompts = collect(&self.captioner_prompts, "captioner_prompts", t)?;
        let tag_groups = collect(&self.tag_groups, "tag_group", t)?;

        let cfg = ProjectConfig {
            default_profile: self
                .default_profile
                .clone()
                .filter(|s| !s.trim().is_empty()),
            default_tagger: self
                .default_tagger
                .clone()
                .filter(|s| !s.trim().is_empty()),
            default_captioner: self
                .default_captioner
                .clone()
                .filter(|s| !s.trim().is_empty()),
            export,
            tagger,
            captioner,
            captioner_prompts,
            tag_groups,
        };
        cfg.validate_tag_groups().map_err(|e| e.to_string())?;
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigTab {
    General,
    Tagger,
    Captioner,
    Prompts,
    Export,
    TagGroups,
}

impl Default for ConfigTab {
    fn default() -> Self {
        Self::General
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigAction {
    None,
    Save,
    Cancel,
}

pub fn show_config_modal(
    ctx: &egui::Context,
    t: T,
    target_label: &str,
    draft: &mut ConfigDraft,
    tab: &mut ConfigTab,
    error: &mut Option<String>,
) -> ConfigAction {
    let mut action = ConfigAction::None;
    let mut window_open = true;
    egui::Window::new(t.cfg_window_title())
        .open(&mut window_open)
        .collapsible(false)
        .resizable(true)
        .default_size([780.0, 600.0])
        .min_width(560.0)
        .show(ctx, |ui| {
            ui.label(egui::RichText::new(target_label).monospace().weak());
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                tab_button(ui, tab, ConfigTab::General, t.cfg_tab_general());
                tab_button(ui, tab, ConfigTab::Tagger, t.cfg_tab_tagger());
                tab_button(ui, tab, ConfigTab::Captioner, t.cfg_tab_captioner());
                tab_button(ui, tab, ConfigTab::Prompts, t.cfg_tab_prompts());
                tab_button(ui, tab, ConfigTab::Export, t.cfg_tab_export());
                tab_button(ui, tab, ConfigTab::TagGroups, t.cfg_tab_tag_groups());
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(440.0)
                .show(ui, |ui| match *tab {
                    ConfigTab::General => ui_general(ui, t, draft),
                    ConfigTab::Tagger => ui_tagger(ui, t, draft),
                    ConfigTab::Captioner => ui_captioner(ui, t, draft),
                    ConfigTab::Prompts => ui_prompts(ui, t, draft),
                    ConfigTab::Export => ui_export(ui, t, draft),
                    ConfigTab::TagGroups => ui_tag_groups(ui, t, draft),
                });
            if let Some(err) = error.clone() {
                ui.add_space(4.0);
                ui.colored_label(egui::Color32::from_rgb(255, 180, 180), err);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button(t.config_save()).clicked() {
                    action = ConfigAction::Save;
                }
                if ui.button(t.config_cancel()).clicked() {
                    action = ConfigAction::Cancel;
                }
            });
        });
    if !window_open && action == ConfigAction::None {
        action = ConfigAction::Cancel;
    }
    action
}

fn tab_button(ui: &mut egui::Ui, current: &mut ConfigTab, value: ConfigTab, label: &str) {
    if ui.selectable_label(*current == value, label).clicked() {
        *current = value;
    }
}

// ───────── tabs ─────────

fn ui_general(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    egui::Grid::new("cfg_general_grid")
        .num_columns(2)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label(t.cfg_default_profile());
            optional_combo(
                ui,
                "default_profile",
                &mut draft.default_profile,
                draft.export.iter().map(|(k, _)| k.as_str()),
                t.cfg_none(),
            );
            ui.end_row();

            ui.label(t.cfg_default_tagger());
            optional_combo(
                ui,
                "default_tagger",
                &mut draft.default_tagger,
                draft.tagger.iter().map(|(k, _)| k.as_str()),
                t.cfg_none(),
            );
            ui.end_row();

            ui.label(t.cfg_default_captioner());
            optional_combo(
                ui,
                "default_captioner",
                &mut draft.default_captioner,
                draft.captioner.iter().map(|(k, _)| k.as_str()),
                t.cfg_none(),
            );
            ui.end_row();
        });
    ui.add_space(8.0);
    ui.label(egui::RichText::new(t.cfg_general_note()).weak());
}

fn ui_tagger(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    let mut remove: Option<usize> = None;
    for (idx, (name, profile)) in draft.tagger.iter_mut().enumerate() {
        let header = if name.is_empty() {
            format!("({})", t.cfg_unnamed())
        } else {
            name.clone()
        };
        egui::CollapsingHeader::new(header)
            .id_salt(("tagger_entry", idx))
            .default_open(true)
            .show(ui, |ui| {
                if ui.button(t.cfg_remove()).clicked() {
                    remove = Some(idx);
                }
                egui::Grid::new(("tagger_grid", idx))
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(t.cfg_name());
                        ui.text_edit_singleline(name);
                        ui.end_row();

                        ui.label(t.cfg_repo());
                        ui.text_edit_singleline(&mut profile.repo);
                        ui.end_row();

                        ui.label(t.cfg_revision());
                        optional_text(ui, ("tagger_rev", idx), &mut profile.revision);
                        ui.end_row();

                        ui.label(t.cfg_input_size());
                        ui.add(
                            egui::DragValue::new(&mut profile.input_size)
                                .range(32..=4096)
                                .speed(1.0),
                        );
                        ui.end_row();

                        ui.label(t.cfg_storage_threshold());
                        ui.add(
                            egui::DragValue::new(&mut profile.storage_threshold)
                                .range(0.0..=1.0)
                                .speed(0.005),
                        );
                        ui.end_row();
                    });
            });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        draft.tagger.remove(i);
    }
    if ui.button(t.cfg_add_tagger()).clicked() {
        draft.tagger.push((
            unique_name("tagger", &draft.tagger),
            TaggerProfile {
                repo: BUILT_IN_TAGGER_REPO.to_string(),
                revision: None,
                input_size: 448,
                storage_threshold: 0.10,
            },
        ));
    }
}

fn ui_captioner(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    let mut remove: Option<usize> = None;
    for (idx, (name, profile)) in draft.captioner.iter_mut().enumerate() {
        let header = if name.is_empty() {
            format!("({})", t.cfg_unnamed())
        } else {
            name.clone()
        };
        egui::CollapsingHeader::new(header)
            .id_salt(("captioner_entry", idx))
            .default_open(true)
            .show(ui, |ui| {
                if ui.button(t.cfg_remove()).clicked() {
                    remove = Some(idx);
                }
                egui::Grid::new(("captioner_grid", idx))
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(t.cfg_name());
                        ui.text_edit_singleline(name);
                        ui.end_row();

                        ui.label(t.cfg_kind());
                        captioner_kind_combo(ui, idx, profile);
                        ui.end_row();
                    });
                match profile {
                    CaptionerProfile::Onnx(p) => ui_captioner_onnx(ui, t, idx, p),
                    CaptionerProfile::Openai(p) => ui_captioner_openai(ui, t, idx, p),
                }
            });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        draft.captioner.remove(i);
    }
    ui.horizontal(|ui| {
        if ui.button(t.cfg_add_captioner_onnx()).clicked() {
            draft.captioner.push((
                unique_name("onnx", &draft.captioner),
                CaptionerProfile::Onnx(OnnxCaptionerProfile {
                    repo: BUILT_IN_CAPTIONER_REPO.to_string(),
                    revision: None,
                    subdir: Some(BUILT_IN_CAPTIONER_SUBDIR.to_string()),
                    prompts: vec!["default".to_string()],
                    max_pixels: 589_824,
                    max_new_tokens: 512,
                }),
            ));
        }
        if ui.button(t.cfg_add_captioner_openai()).clicked() {
            draft.captioner.push((
                unique_name("openai", &draft.captioner),
                CaptionerProfile::Openai(OpenAiCaptionerProfile {
                    endpoint: "http://localhost:8080/v1".to_string(),
                    model: Some("local".to_string()),
                    api_key: None,
                    prompts: vec!["default".to_string()],
                    max_tokens: 512,
                    temperature: None,
                    max_edge: 1024,
                    jpeg_quality: 90,
                    timeout_secs: 600,
                max_retries: 3,
                }),
            ));
        }
    });
}

fn captioner_kind_combo(ui: &mut egui::Ui, idx: usize, profile: &mut CaptionerProfile) {
    let current = match profile {
        CaptionerProfile::Onnx(_) => "onnx",
        CaptionerProfile::Openai(_) => "openai",
    };
    let mut next = current;
    egui::ComboBox::from_id_salt(("captioner_kind", idx))
        .selected_text(current)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut next, "onnx", "ONNX");
            ui.selectable_value(&mut next, "openai", "OpenAI");
        });
    if next != current {
        // Switching kind discards variant-specific fields; shared
        // fields (`prompts`) are preserved by reading them off the old
        // value before replacing.
        let prompts = match &profile {
            CaptionerProfile::Onnx(p) => p.prompts.clone(),
            CaptionerProfile::Openai(p) => p.prompts.clone(),
        };
        *profile = match next {
            "onnx" => CaptionerProfile::Onnx(OnnxCaptionerProfile {
                repo: BUILT_IN_CAPTIONER_REPO.to_string(),
                revision: None,
                subdir: Some(BUILT_IN_CAPTIONER_SUBDIR.to_string()),
                prompts,
                max_pixels: 589_824,
                max_new_tokens: 512,
            }),
            _ => CaptionerProfile::Openai(OpenAiCaptionerProfile {
                endpoint: "http://localhost:8080/v1".to_string(),
                model: Some("local".to_string()),
                api_key: None,
                prompts,
                max_tokens: 512,
                temperature: None,
                max_edge: 1024,
                jpeg_quality: 90,
                timeout_secs: 600,
            max_retries: 3,
            }),
        };
    }
}

fn ui_captioner_onnx(ui: &mut egui::Ui, t: T, idx: usize, p: &mut OnnxCaptionerProfile) {
    egui::Grid::new(("cap_onnx_grid", idx))
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label(t.cfg_repo());
            ui.text_edit_singleline(&mut p.repo);
            ui.end_row();

            ui.label(t.cfg_revision());
            optional_text(ui, ("cap_onnx_rev", idx), &mut p.revision);
            ui.end_row();

            ui.label(t.cfg_subdir());
            optional_text(ui, ("cap_onnx_sub", idx), &mut p.subdir);
            ui.end_row();

            ui.label(t.cfg_max_pixels());
            ui.add(
                egui::DragValue::new(&mut p.max_pixels)
                    .range(1024..=4_194_304)
                    .speed(1024.0),
            );
            ui.end_row();

            ui.label(t.cfg_max_new_tokens());
            ui.add(
                egui::DragValue::new(&mut p.max_new_tokens)
                    .range(1..=8192)
                    .speed(1.0),
            );
            ui.end_row();
        });
    ui.add_space(4.0);
    ui.label(t.cfg_prompts());
    string_list_editor(ui, ("cap_onnx_prompts", idx), &mut p.prompts, t);
}

fn ui_captioner_openai(ui: &mut egui::Ui, t: T, idx: usize, p: &mut OpenAiCaptionerProfile) {
    egui::Grid::new(("cap_openai_grid", idx))
        .num_columns(2)
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            ui.label(t.cfg_endpoint());
            ui.text_edit_singleline(&mut p.endpoint);
            ui.end_row();

            ui.label(t.cfg_model());
            optional_text(ui, ("cap_openai_model", idx), &mut p.model);
            ui.end_row();

            ui.label(t.cfg_api_key());
            optional_password(ui, ("cap_openai_key", idx), &mut p.api_key);
            ui.end_row();

            ui.label(t.cfg_max_tokens());
            ui.add(
                egui::DragValue::new(&mut p.max_tokens)
                    .range(1..=32768)
                    .speed(1.0),
            );
            ui.end_row();

            ui.label(t.cfg_temperature());
            optional_dragvalue_f32(
                ui,
                ("cap_openai_temp", idx),
                &mut p.temperature,
                0.0..=2.0,
                0.05,
                0.7,
            );
            ui.end_row();

            ui.label(t.cfg_max_edge());
            ui.add(
                egui::DragValue::new(&mut p.max_edge)
                    .range(0..=8192)
                    .speed(8.0),
            );
            ui.end_row();

            ui.label(t.cfg_jpeg_quality());
            ui.add(
                egui::DragValue::new(&mut p.jpeg_quality)
                    .range(1..=100)
                    .speed(1.0),
            );
            ui.end_row();

            ui.label(t.cfg_timeout_secs());
            ui.add(
                egui::DragValue::new(&mut p.timeout_secs)
                    .range(1..=86_400)
                    .speed(1.0),
            );
            ui.end_row();

            ui.label(t.cfg_max_retries());
            ui.add(
                egui::DragValue::new(&mut p.max_retries)
                    .range(0..=10)
                    .speed(1.0),
            );
            ui.end_row();
        });
    ui.add_space(4.0);
    ui.label(t.cfg_prompts());
    string_list_editor(ui, ("cap_openai_prompts", idx), &mut p.prompts, t);
}

fn ui_prompts(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    let mut remove: Option<usize> = None;
    for (idx, (name, body)) in draft.captioner_prompts.iter_mut().enumerate() {
        let header = if name.is_empty() {
            format!("({})", t.cfg_unnamed())
        } else {
            name.clone()
        };
        egui::CollapsingHeader::new(header)
            .id_salt(("prompt_entry", idx))
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(t.cfg_name());
                    ui.text_edit_singleline(name);
                    if ui.button(t.cfg_remove()).clicked() {
                        remove = Some(idx);
                    }
                });
                ui.add(
                    egui::TextEdit::multiline(body)
                        .desired_width(f32::INFINITY)
                        .desired_rows(4),
                );
            });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        draft.captioner_prompts.remove(i);
    }
    if ui.button(t.cfg_add_prompt()).clicked() {
        draft
            .captioner_prompts
            .push((unique_name("prompt", &draft.captioner_prompts), String::new()));
    }
    ui.add_space(6.0);
    ui.label(egui::RichText::new(t.cfg_prompts_note()).weak());
}

fn ui_export(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    let mut remove: Option<usize> = None;
    for (idx, (name, p)) in draft.export.iter_mut().enumerate() {
        let header = if name.is_empty() {
            format!("({})", t.cfg_unnamed())
        } else {
            name.clone()
        };
        egui::CollapsingHeader::new(header)
            .id_salt(("export_entry", idx))
            .default_open(true)
            .show(ui, |ui| {
                if ui.button(t.cfg_remove()).clicked() {
                    remove = Some(idx);
                }
                egui::Grid::new(("export_grid", idx))
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(t.cfg_name());
                        ui.text_edit_singleline(name);
                        ui.end_row();

                        ui.label(t.cfg_threshold());
                        ui.add(
                            egui::DragValue::new(&mut p.threshold)
                                .range(0.0..=1.0)
                                .speed(0.005),
                        );
                        ui.end_row();

                        ui.label(t.cfg_shuffle());
                        ui.checkbox(&mut p.shuffle, "");
                        ui.end_row();
                    });
                ui.add_space(4.0);
                ui.label(t.cfg_exclude_categories());
                string_list_editor(
                    ui,
                    ("export_excl", idx),
                    &mut p.exclude_categories,
                    t,
                );
                ui.add_space(4.0);
                ui.label(t.cfg_category_prefixes());
                kv_list_editor(
                    ui,
                    ("export_pref", idx),
                    &mut p.category_prefixes,
                    t.cfg_category(),
                    t.cfg_prefix(),
                    t,
                );
                ui.add_space(4.0);
                ui.label(t.cfg_caption_prefixes());
                kv_list_editor(
                    ui,
                    ("export_cap_pref", idx),
                    &mut p.caption_prefixes,
                    t.cfg_tag(),
                    t.cfg_prefix(),
                    t,
                );
                ui.add_space(4.0);
                ui.label(t.cfg_caption_suffixes());
                kv_list_editor(
                    ui,
                    ("export_cap_suf", idx),
                    &mut p.caption_suffixes,
                    t.cfg_tag(),
                    t.cfg_suffix(),
                    t,
                );
            });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        draft.export.remove(i);
    }
    if ui.button(t.cfg_add_export()).clicked() {
        draft.export.push((
            unique_name("export", &draft.export),
            ExportProfileDraft {
                threshold: 0.35,
                shuffle: false,
                exclude_categories: Vec::new(),
                category_prefixes: Vec::new(),
                caption_prefixes: Vec::new(),
                caption_suffixes: Vec::new(),
            },
        ));
    }
}

fn ui_tag_groups(ui: &mut egui::Ui, t: T, draft: &mut ConfigDraft) {
    let mut remove: Option<usize> = None;
    for (idx, (name, group)) in draft.tag_groups.iter_mut().enumerate() {
        let header = if name.is_empty() {
            format!("({})", t.cfg_unnamed())
        } else {
            name.clone()
        };
        egui::CollapsingHeader::new(header)
            .id_salt(("group_entry", idx))
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(t.cfg_name());
                    ui.text_edit_singleline(name);
                    if ui.button(t.cfg_remove()).clicked() {
                        remove = Some(idx);
                    }
                });
                ui.label(t.cfg_tags());
                string_list_editor(ui, ("group_tags", idx), &mut group.tags, t);
            });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        draft.tag_groups.remove(i);
    }
    if ui.button(t.cfg_add_tag_group()).clicked() {
        draft.tag_groups.push((
            unique_name("group", &draft.tag_groups),
            TagGroup { tags: Vec::new() },
        ));
    }
    ui.add_space(6.0);
    ui.label(egui::RichText::new(t.cfg_tag_groups_note()).weak());
}

// ───────── shared widgets ─────────

fn optional_combo<'a>(
    ui: &mut egui::Ui,
    salt: &'static str,
    value: &mut Option<String>,
    options: impl IntoIterator<Item = &'a str>,
    none_label: &str,
) {
    let current = value.clone();
    let display = current.clone().unwrap_or_else(|| none_label.to_string());
    egui::ComboBox::from_id_salt(salt)
        .selected_text(display)
        .show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), none_label).clicked() {
                *value = None;
            }
            for opt in options {
                let active = current.as_deref() == Some(opt);
                if ui.selectable_label(active, opt).clicked() {
                    *value = Some(opt.to_string());
                }
            }
        });
}

fn optional_text(ui: &mut egui::Ui, salt: impl std::hash::Hash, value: &mut Option<String>) {
    ui.push_id(salt, |ui| {
        let mut buf = value.clone().unwrap_or_default();
        let resp = ui.text_edit_singleline(&mut buf);
        if resp.changed() {
            *value = if buf.trim().is_empty() { None } else { Some(buf) };
        }
    });
}

fn optional_password(ui: &mut egui::Ui, salt: impl std::hash::Hash, value: &mut Option<String>) {
    ui.push_id(salt, |ui| {
        let mut buf = value.clone().unwrap_or_default();
        let resp = ui.add(egui::TextEdit::singleline(&mut buf).password(true));
        if resp.changed() {
            *value = if buf.is_empty() { None } else { Some(buf) };
        }
    });
}

fn optional_dragvalue_f32(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    value: &mut Option<f32>,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
    enable_default: f32,
) {
    ui.push_id(salt, |ui| {
        ui.horizontal(|ui| {
            let mut enabled = value.is_some();
            if ui.checkbox(&mut enabled, "").changed() {
                *value = if enabled { Some(enable_default) } else { None };
            }
            if let Some(v) = value.as_mut() {
                ui.add(
                    egui::DragValue::new(v)
                        .range(range)
                        .speed(speed),
                );
            }
        });
    });
}

fn string_list_editor(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    items: &mut Vec<String>,
    t: T,
) {
    ui.push_id(salt, |ui| {
        let mut remove: Option<usize> = None;
        for (i, item) in items.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.text_edit_singleline(item);
                if ui.small_button("×").on_hover_text(t.cfg_remove()).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            items.remove(i);
        }
        if ui.small_button(t.cfg_add()).clicked() {
            items.push(String::new());
        }
    });
}

fn kv_list_editor(
    ui: &mut egui::Ui,
    salt: impl std::hash::Hash,
    items: &mut Vec<(String, String)>,
    key_label: &str,
    value_label: &str,
    t: T,
) {
    ui.push_id(salt, |ui| {
        let mut remove: Option<usize> = None;
        egui::Grid::new("kv_grid")
            .num_columns(3)
            .spacing([6.0, 4.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new(key_label).weak());
                ui.label(egui::RichText::new(value_label).weak());
                ui.label("");
                ui.end_row();
                for (i, (k, v)) in items.iter_mut().enumerate() {
                    ui.text_edit_singleline(k);
                    ui.text_edit_singleline(v);
                    if ui.small_button("×").on_hover_text(t.cfg_remove()).clicked() {
                        remove = Some(i);
                    }
                    ui.end_row();
                }
            });
        if let Some(i) = remove {
            items.remove(i);
        }
        if ui.small_button(t.cfg_add()).clicked() {
            items.push((String::new(), String::new()));
        }
    });
}

fn unique_name<T>(stem: &str, entries: &[(String, T)]) -> String {
    let used: BTreeSet<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    if !used.contains(stem) {
        return stem.to_string();
    }
    for n in 2.. {
        let candidate = format!("{stem}_{n}");
        if !used.contains(candidate.as_str()) {
            return candidate;
        }
    }
    unreachable!("u64 exhausted naming entries");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i18n::Lang;

    #[test]
    fn draft_round_trips_through_project_config() {
        let mut original = ProjectConfig::default();
        original.default_profile = Some("anima".into());
        original.tagger.insert(
            "wd".into(),
            TaggerProfile {
                repo: "r".into(),
                revision: Some("main".into()),
                input_size: 448,
                storage_threshold: 0.1,
            },
        );
        original.tag_groups.insert(
            "costumes".into(),
            TagGroup {
                tags: vec!["a".into(), "b".into()],
            },
        );

        let draft = ConfigDraft::from_config(original.clone());
        let rebuilt = draft.to_config(T::new(Lang::En)).expect("round-trip ok");

        assert_eq!(rebuilt.default_profile, original.default_profile);
        assert_eq!(rebuilt.tagger.len(), original.tagger.len());
        assert_eq!(
            rebuilt.tag_groups.get("costumes").map(|g| &g.tags),
            original.tag_groups.get("costumes").map(|g| &g.tags),
        );
    }

    #[test]
    fn duplicate_names_rejected_at_save_time() {
        let mut draft = ConfigDraft::from_config(ProjectConfig::default());
        draft.tagger.push(("dup".into(), TaggerProfile::built_in()));
        draft.tagger.push(("dup".into(), TaggerProfile::built_in()));
        let err = draft
            .to_config(T::new(Lang::En))
            .expect_err("duplicate names should fail validation");
        assert!(err.contains("dup"), "error should name the dupe: {err}");
    }

    #[test]
    fn empty_names_rejected_at_save_time() {
        let mut draft = ConfigDraft::from_config(ProjectConfig::default());
        draft.tag_groups.push((
            "  ".into(),
            TagGroup {
                tags: vec!["a".into()],
            },
        ));
        draft
            .to_config(T::new(Lang::En))
            .expect_err("whitespace-only names should fail validation");
    }
}
