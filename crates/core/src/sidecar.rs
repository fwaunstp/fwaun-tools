use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod category {
    pub const GENERAL: &str = "general";
    pub const ARTIST: &str = "artist";
    pub const COPYRIGHT: &str = "copyright";
    pub const CHARACTER: &str = "character";
    pub const META: &str = "meta";
    pub const RATING: &str = "rating";
}

pub const SIDECAR_EXTENSION: &str = "ron";

/// Manual entries beginning with this character are treated as suppression
/// markers (e.g. `-watermark` removes any auto/booru tag with stem `watermark`
/// from the export, regardless of which tagger produced it).
pub const NEGATIVE_PREFIX: char = '-';

/// Positive manual entries beginning with this character are *organizational*
/// tags: curation-only labels kept out of the exported training `.txt` but
/// still part of the effective tag set used by tag-group classification /
/// the Kanban view. Lets the user distinguish "not yet reviewed" (no group
/// tag at all) from "reviewed, deliberately none of these" (an organizational
/// tag like `_no_character` placed in the group) without polluting the
/// training captions. Booru-style tags use `_` mid-word but never lead with
/// it, so a leading `_` is an unambiguous marker.
pub const ORGANIZATIONAL_PREFIX: char = '_';

/// True if `tag` is an organizational (curation-only, never-exported) entry.
pub fn is_organizational(tag: &str) -> bool {
    tag.trim().starts_with(ORGANIZATIONAL_PREFIX)
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "SidecarOnDisk")]
pub struct Sidecar {
    /// Manual entries. `foo` = positive (always exported); `-foo` = suppression
    /// marker (removes matching auto/booru tag from export). Negative entries
    /// are never themselves emitted to the training `.txt` file.
    #[serde(default)]
    pub manual_tags: Vec<String>,
    #[serde(default)]
    pub auto_tags: Vec<AutoTag>,
    #[serde(default)]
    pub booru_tags: Vec<BooruTag>,
    /// Captions from automatic captioners, keyed by the resolved profile name
    /// (e.g. `qwen3-vl-4b`). Each captioner run overwrites only its own entry,
    /// so users can compare outputs across models — important for NSFW where
    /// some models refuse. Export uses `manual_caption` only; the GUI exposes
    /// a "copy" action to seed `manual_caption` from any of these.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub captions: BTreeMap<String, CaptionEntry>,
    /// Hand-edited final caption. When set, fully overrides any auto
    /// captions on export. Seed it via the GUI's copy-from-auto button,
    /// or type it by hand. For per-image reference info that should
    /// influence generation (character names, positions, …), use
    /// `caption_hints` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_caption: Option<String>,
    /// Reference facts handed to the captioner as a system turn at generation
    /// time (e.g. character names + positions). Each entry is one fact; they
    /// are formatted as a markdown bullet list by [`caption_hint_prompt`] so
    /// the model can weave them into its own prose rather than the caller
    /// prepending them verbatim. Edited like tags (add/remove/bulk-add) so a
    /// shared fact — "blue hair girl is Laundry Dragonmaid" — can be applied
    /// across every image that shows that character. Never appears in the
    /// exported `.txt` directly — it's pure input to the captioner.
    ///
    /// [`caption_hint_prompt`]: Self::caption_hint_prompt
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub caption_hints: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tagger: Option<TaggerInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub booru: Option<BooruInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoTag {
    pub tag: String,
    pub score: f32,
    pub category: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooruTag {
    pub tag: String,
    pub category: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionEntry {
    pub caption: String,
    pub captioned_at: DateTime<Utc>,
    /// Mark a generated caption as not-for-export without deleting it
    /// (e.g. the model refused, or the wording is off). Skipped entries
    /// stay visible in the GUI for reference.
    #[serde(default, skip_serializing_if = "is_false")]
    pub skip: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaggerInfo {
    pub model: String,
    pub tagged_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooruInfo {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_url: Option<String>,
    pub fetched_at: DateTime<Utc>,
}

/// On-disk schema. Holds the legacy single-caption fields so older sidecars
/// keep loading; `From<SidecarOnDisk>` folds them into `captions`.
#[derive(Debug, Clone, Default, Deserialize)]
struct SidecarOnDisk {
    #[serde(default)]
    manual_tags: Vec<String>,
    #[serde(default)]
    auto_tags: Vec<AutoTag>,
    #[serde(default)]
    booru_tags: Vec<BooruTag>,
    #[serde(default)]
    captions: BTreeMap<String, CaptionEntry>,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    captioner: Option<LegacyCaptionerInfo>,
    #[serde(default)]
    manual_caption: Option<String>,
    /// Current form: a list of reference facts.
    #[serde(default)]
    caption_hints: Vec<String>,
    /// Legacy single free-text hint (possibly multi-line). Folded into
    /// `caption_hints` on load — split on newlines so an old multi-line hint
    /// becomes one bullet per line.
    #[serde(default)]
    caption_hint: Option<String>,
    #[serde(default)]
    tagger: Option<TaggerInfo>,
    #[serde(default)]
    booru: Option<BooruInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct LegacyCaptionerInfo {
    model: String,
    captioned_at: DateTime<Utc>,
}

impl From<SidecarOnDisk> for Sidecar {
    fn from(d: SidecarOnDisk) -> Self {
        let mut captions = d.captions;
        if let Some(text) = d.caption.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            let (model, captioned_at) = match d.captioner {
                Some(info) => (info.model, info.captioned_at),
                None => ("legacy".to_string(), Utc::now()),
            };
            captions.entry(model).or_insert(CaptionEntry {
                caption: text,
                captioned_at,
                skip: false,
            });
        }
        // Fold the legacy single hint into the list: split on newlines (an old
        // multi-line hint becomes one bullet per line), trim, drop blanks, and
        // dedup against anything already in `caption_hints`.
        let mut caption_hints = d.caption_hints;
        if let Some(legacy) = d.caption_hint {
            for line in legacy.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !caption_hints.iter().any(|x| x == trimmed) {
                    caption_hints.push(trimmed.to_string());
                }
            }
        }
        Self {
            manual_tags: d.manual_tags,
            auto_tags: d.auto_tags,
            booru_tags: d.booru_tags,
            captions,
            manual_caption: d.manual_caption,
            caption_hints,
            tagger: d.tagger,
            booru: d.booru,
        }
    }
}

#[derive(Debug, Error)]
pub enum SidecarError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ron parse error on {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: ron::de::SpannedError,
    },
    #[error("ron serialize error: {0}")]
    Serialize(#[from] ron::Error),
}

pub fn sidecar_path_for(image: &Path) -> PathBuf {
    image.with_extension(SIDECAR_EXTENSION)
}

/// Collapse embedded newlines / tabs / runs of whitespace into a single
/// space. Used at export time so a caption like
/// `"foo\nbar"` doesn't accidentally become two variation lines when
/// joined with `\n` for sd-scripts.
fn flatten_caption(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn pretty_config() -> PrettyConfig {
    PrettyConfig::default()
        .struct_names(false)
        .indentor("  ".to_string())
}

impl Sidecar {
    pub fn load(image: &Path) -> Result<Option<Self>, SidecarError> {
        let path = sidecar_path_for(image);
        if !path.exists() {
            return Ok(None);
        }
        let s = fs::read_to_string(&path).map_err(|source| SidecarError::Io {
            path: path.clone(),
            source,
        })?;
        let parsed = ron::de::from_str(&s).map_err(|source| SidecarError::Parse {
            path: path.clone(),
            source,
        })?;
        Ok(Some(parsed))
    }

    pub fn load_or_default(image: &Path) -> Result<Self, SidecarError> {
        Ok(Self::load(image)?.unwrap_or_default())
    }

    pub fn save(&self, image: &Path) -> Result<(), SidecarError> {
        let path = sidecar_path_for(image);
        let body = ron::ser::to_string_pretty(self, pretty_config())?;
        let mut tmp_os = path.as_os_str().to_owned();
        tmp_os.push(".tmp");
        let tmp = PathBuf::from(tmp_os);
        fs::write(&tmp, body).map_err(|source| SidecarError::Io {
            path: tmp.clone(),
            source,
        })?;
        fs::rename(&tmp, &path).map_err(|source| SidecarError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(())
    }

    pub fn is_auto_tagged(&self) -> bool {
        self.tagger.is_some()
    }

    pub fn is_captioned(&self) -> bool {
        !self.captions.is_empty()
    }

    pub fn set_caption(&mut self, model: impl Into<String>, text: impl Into<String>) {
        self.captions.insert(
            model.into(),
            CaptionEntry {
                caption: text.into(),
                captioned_at: Utc::now(),
                skip: false,
            },
        );
    }

    pub fn remove_caption(&mut self, model: &str) -> bool {
        self.captions.remove(model).is_some()
    }

    /// Returns the new skip state, or `None` if the model isn't present.
    pub fn toggle_caption_skip(&mut self, model: &str) -> Option<bool> {
        self.captions.get_mut(model).map(|e| {
            e.skip = !e.skip;
            e.skip
        })
    }

    /// Caption text written on export.
    ///
    /// `manual_caption`, when set, fully overrides any auto captions —
    /// it's a hand-edited final version. Otherwise each active (non-skip)
    /// auto caption is flattened to a single line and the lines are
    /// joined with `\n` so sd-scripts' multi-caption shuffler can pick
    /// a variation per step.
    ///
    /// Embedded newlines in any auto caption are collapsed to spaces so
    /// the `\n` separator can never split a single caption mid-sentence.
    /// Per-image reference info (character names, positions, etc.) is
    /// carried by `caption_hints` and consumed at caption-generation time;
    /// it never appears in the exported text.
    pub fn export_caption(&self) -> Option<String> {
        let manual = self
            .manual_caption
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        if let Some(text) = manual {
            return Some(text.to_string());
        }

        let active: Vec<String> = self
            .captions
            .values()
            .filter(|e| !e.skip)
            .map(|e| flatten_caption(&e.caption))
            .filter(|s| !s.is_empty())
            .collect();

        if active.is_empty() {
            None
        } else {
            Some(active.join("\n"))
        }
    }

    pub fn set_manual_caption(&mut self, text: &str) {
        let trimmed = text.trim();
        self.manual_caption = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
    }

    /// Add one reference fact. Trimmed; a blank or exact-duplicate entry is a
    /// no-op. Returns `true` if newly added.
    pub fn add_caption_hint(&mut self, hint: impl Into<String>) -> bool {
        let h = hint.into();
        let trimmed = h.trim();
        if trimmed.is_empty() || self.caption_hints.iter().any(|x| x == trimmed) {
            return false;
        }
        self.caption_hints.push(trimmed.to_string());
        true
    }

    /// Remove a reference fact by exact match. Returns `true` if one was removed.
    pub fn remove_caption_hint(&mut self, hint: &str) -> bool {
        let before = self.caption_hints.len();
        self.caption_hints.retain(|x| x != hint);
        before != self.caption_hints.len()
    }

    /// The reference block passed to the captioner as a system turn, or `None`
    /// when there are no (non-blank) hints. Each fact becomes one markdown
    /// bullet so the model reads them as a list rather than one run-on line.
    pub fn caption_hint_prompt(&self) -> Option<String> {
        let lines: Vec<String> = self
            .caption_hints
            .iter()
            .map(|h| h.trim())
            .filter(|h| !h.is_empty())
            .map(|h| format!("- {h}"))
            .collect();
        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    }

    pub fn has_booru(&self) -> bool {
        self.booru.is_some()
    }

    /// Iterates positive manual entries (skipping suppression markers).
    pub fn manual_positive_tags(&self) -> impl Iterator<Item = &str> {
        self.manual_tags
            .iter()
            .filter(|t| !t.trim().starts_with(NEGATIVE_PREFIX))
            .map(|t| t.as_str())
    }

    /// Returns lowercase stems suppressed by `-foo` manual entries.
    pub fn suppressed_set(&self) -> HashSet<String> {
        self.manual_tags
            .iter()
            .filter_map(|t| {
                t.trim()
                    .strip_prefix(NEGATIVE_PREFIX)
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
            })
            .collect()
    }

    pub fn is_suppressed(&self, tag: &str) -> bool {
        let key = tag.trim().to_lowercase();
        if key.is_empty() {
            return false;
        }
        self.manual_tags.iter().any(|m| {
            m.trim()
                .strip_prefix(NEGATIVE_PREFIX)
                .map(|s| s.trim().to_lowercase() == key)
                .unwrap_or(false)
        })
    }

    /// Append a manual entry verbatim (positive or `-foo` suppression). Returns
    /// `true` if newly added, `false` if it was already present or empty.
    pub fn add_manual_tag(&mut self, tag: impl Into<String>) -> bool {
        let t = tag.into();
        let trimmed = t.trim();
        if trimmed.is_empty() || self.manual_tags.iter().any(|x| x == trimmed) {
            return false;
        }
        self.manual_tags.push(trimmed.to_string());
        true
    }

    pub fn remove_manual_tag(&mut self, tag: &str) -> bool {
        let before = self.manual_tags.len();
        self.manual_tags.retain(|x| x != tag);
        before != self.manual_tags.len()
    }

    /// Remove every manual entry whose trimmed, lowercased form equals
    /// `tag` (also trimmed+lowercased) — the case-insensitive counterpart
    /// to [`remove_manual_tag`], used by the CLI `remove-tag` bulk pass.
    ///
    /// This deletes the existing manual slot only. It never adds a `-foo`
    /// suppression marker (that's [`suppress`](Self::suppress)) and never
    /// touches auto/booru tags. The leading `-` of a suppression marker is
    /// part of the compared string, so removing `foo` never removes
    /// `-foo`; to drop a suppression marker, pass `-foo` explicitly.
    /// Returns the number of entries removed.
    pub fn remove_manual_tag_ci(&mut self, tag: &str) -> usize {
        let key = tag.trim().to_lowercase();
        if key.is_empty() {
            return 0;
        }
        let before = self.manual_tags.len();
        self.manual_tags.retain(|m| m.trim().to_lowercase() != key);
        before - self.manual_tags.len()
    }

    /// Add `-tag` as a suppression marker if not already present.
    pub fn suppress(&mut self, tag: &str) -> bool {
        let trimmed = tag.trim();
        if trimmed.is_empty() {
            return false;
        }
        let neg = format!("-{trimmed}");
        if self.manual_tags.iter().any(|x| x == &neg) {
            return false;
        }
        self.manual_tags.push(neg);
        true
    }

    /// Remove the `-tag` suppression marker if present.
    pub fn unsuppress(&mut self, tag: &str) -> bool {
        let neg = format!("-{}", tag.trim());
        let before = self.manual_tags.len();
        self.manual_tags.retain(|x| x != &neg);
        before != self.manual_tags.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_caption_migrates_into_captions_map() {
        let ron_text = r#"(
            manual_tags: [],
            auto_tags: [],
            booru_tags: [],
            caption: Some("a girl"),
            captioner: Some((model: "qwen3-vl-4b", captioned_at: "2025-01-01T00:00:00Z")),
        )"#;
        let sc: Sidecar = ron::de::from_str(ron_text).unwrap();
        assert_eq!(sc.captions.len(), 1);
        let entry = sc.captions.get("qwen3-vl-4b").unwrap();
        assert_eq!(entry.caption, "a girl");
    }

    #[test]
    fn export_caption_no_captions_at_all_is_none() {
        let sc = Sidecar::default();
        assert_eq!(sc.export_caption(), None);
    }

    #[test]
    fn export_caption_only_manual_returns_it_verbatim() {
        let mut sc = Sidecar::default();
        sc.set_manual_caption("manual text");
        assert_eq!(sc.export_caption().as_deref(), Some("manual text"));
    }

    #[test]
    fn export_caption_only_auto_no_manual_returns_auto_lines() {
        let mut sc = Sidecar::default();
        sc.set_caption("a", "first caption");
        sc.set_caption("b", "second caption");
        // BTreeMap iterates in key order: a, b.
        assert_eq!(
            sc.export_caption().as_deref(),
            Some("first caption\nsecond caption")
        );
    }

    #[test]
    fn export_caption_manual_overrides_auto() {
        let mut sc = Sidecar::default();
        sc.set_manual_caption("hand-edited final caption");
        sc.set_caption("a", "a girl smiling");
        sc.set_caption("b", "young woman, smile");
        assert_eq!(
            sc.export_caption().as_deref(),
            Some("hand-edited final caption")
        );
    }

    #[test]
    fn export_caption_skipped_entries_are_excluded() {
        let mut sc = Sidecar::default();
        sc.set_caption("refused", "I can't describe this image");
        sc.set_caption("good", "a girl in a garden");
        assert_eq!(sc.toggle_caption_skip("refused"), Some(true));
        assert_eq!(sc.export_caption().as_deref(), Some("a girl in a garden"));
    }

    #[test]
    fn export_caption_all_skipped_falls_back_to_manual_only() {
        let mut sc = Sidecar::default();
        sc.set_manual_caption("manual fallback");
        sc.set_caption("a", "auto1");
        sc.set_caption("b", "auto2");
        sc.toggle_caption_skip("a");
        sc.toggle_caption_skip("b");
        assert_eq!(sc.export_caption().as_deref(), Some("manual fallback"));
    }

    #[test]
    fn export_caption_flattens_embedded_newlines() {
        let mut sc = Sidecar::default();
        sc.set_caption("a", "line1\n\nline2  with    spaces");
        sc.set_caption("b", "another\tline");
        assert_eq!(
            sc.export_caption().as_deref(),
            Some("line1 line2 with spaces\nanother line")
        );
    }

    #[test]
    fn export_caption_skips_empty_after_flatten() {
        let mut sc = Sidecar::default();
        sc.set_caption("blank", "   \n\n  ");
        sc.set_caption("real", "actual caption");
        assert_eq!(sc.export_caption().as_deref(), Some("actual caption"));
    }

    #[test]
    fn caption_hints_round_trip_and_omit_when_empty() {
        let mut sc = Sidecar::default();
        assert!(sc.add_caption_hint("blue hair girl is Laundry Dragonmaid"));
        assert!(sc.add_caption_hint("red hair girl is Kitchen Dragonmaid"));
        // Blank and exact-duplicate entries are no-ops.
        assert!(!sc.add_caption_hint("   "));
        assert!(!sc.add_caption_hint("blue hair girl is Laundry Dragonmaid"));

        let serialized = ron::ser::to_string_pretty(&sc, pretty_config()).unwrap();
        assert!(
            serialized.contains("caption_hints"),
            "non-empty hints must serialize: {serialized}"
        );
        let mut round: Sidecar = ron::de::from_str(&serialized).unwrap();
        assert_eq!(
            round.caption_hints,
            vec![
                "blue hair girl is Laundry Dragonmaid".to_string(),
                "red hair girl is Kitchen Dragonmaid".to_string(),
            ]
        );

        // Formatted for the captioner as a markdown bullet list.
        assert_eq!(
            round.caption_hint_prompt().as_deref(),
            Some("- blue hair girl is Laundry Dragonmaid\n- red hair girl is Kitchen Dragonmaid")
        );

        assert!(round.remove_caption_hint("blue hair girl is Laundry Dragonmaid"));
        assert!(round.remove_caption_hint("red hair girl is Kitchen Dragonmaid"));
        assert!(round.caption_hints.is_empty());
        assert_eq!(round.caption_hint_prompt(), None);
        let serialized = ron::ser::to_string_pretty(&round, pretty_config()).unwrap();
        assert!(
            !serialized.contains("caption_hints"),
            "empty hints must not appear in serialized form: {serialized}"
        );
    }

    #[test]
    fn legacy_single_caption_hint_migrates_into_list() {
        // Old sidecars stored one free-text (possibly multi-line) hint; it
        // folds into the list, one bullet per line, deduped.
        let ron_text = r#"(
            manual_tags: [],
            auto_tags: [],
            booru_tags: [],
            caption_hint: Some("Left girl is Alice.\nRight boy is Bob."),
        )"#;
        let sc: Sidecar = ron::de::from_str(ron_text).unwrap();
        assert_eq!(
            sc.caption_hints,
            vec![
                "Left girl is Alice.".to_string(),
                "Right boy is Bob.".to_string(),
            ]
        );
    }

    #[test]
    fn old_sidecar_without_caption_hint_loads_cleanly() {
        let ron_text = r#"(
            manual_tags: ["foo"],
            auto_tags: [],
            booru_tags: [],
        )"#;
        let sc: Sidecar = ron::de::from_str(ron_text).unwrap();
        assert!(sc.caption_hints.is_empty());
        assert_eq!(sc.manual_tags, vec!["foo".to_string()]);
    }

    #[test]
    fn remove_manual_tag_ci_matches_case_insensitively() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("Short_Hair".into());
        sc.manual_tags.push("keep".into());
        assert_eq!(sc.remove_manual_tag_ci("short_hair"), 1);
        assert_eq!(sc.manual_tags, vec!["keep".to_string()]);
    }

    #[test]
    fn remove_manual_tag_ci_removes_suppression_marker_but_not_positive() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("-watermark".into());
        sc.manual_tags.push("watermark".into());
        // Removing the positive form leaves the `-` marker intact …
        assert_eq!(sc.remove_manual_tag_ci("watermark"), 1);
        assert_eq!(sc.manual_tags, vec!["-watermark".to_string()]);
        // … and the marker is only removed when passed with its `-`.
        assert_eq!(sc.remove_manual_tag_ci("-watermark"), 1);
        assert!(sc.manual_tags.is_empty());
    }

    #[test]
    fn remove_manual_tag_ci_removes_all_duplicates_and_counts_them() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("dup".into());
        sc.manual_tags.push("DUP".into());
        sc.manual_tags.push("other".into());
        assert_eq!(sc.remove_manual_tag_ci("dup"), 2);
        assert_eq!(sc.manual_tags, vec!["other".to_string()]);
    }

    #[test]
    fn remove_manual_tag_ci_empty_key_is_noop() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("x".into());
        assert_eq!(sc.remove_manual_tag_ci("   "), 0);
        assert_eq!(sc.manual_tags, vec!["x".to_string()]);
    }

    #[test]
    fn set_caption_overwrites_same_model_only() {
        let mut sc = Sidecar::default();
        sc.set_caption("a", "first");
        sc.set_caption("b", "second");
        sc.set_caption("a", "first-v2");
        assert_eq!(sc.captions.get("a").unwrap().caption, "first-v2");
        assert_eq!(sc.captions.get("b").unwrap().caption, "second");
    }
}
