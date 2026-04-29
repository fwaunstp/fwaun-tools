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
    /// Free-text caption written verbatim on export. Seed it via the GUI's
    /// copy-from-auto button, or type it by hand.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_caption: Option<String>,
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
            });
        }
        Self {
            manual_tags: d.manual_tags,
            auto_tags: d.auto_tags,
            booru_tags: d.booru_tags,
            captions,
            manual_caption: d.manual_caption,
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
            },
        );
    }

    pub fn remove_caption(&mut self, model: &str) -> bool {
        self.captions.remove(model).is_some()
    }

    /// The caption written to training metadata on export. Prefers the
    /// manual caption; falls back to the most recently generated auto
    /// caption when the manual field is empty/unset, so an unreviewed
    /// dataset still exports something useful instead of dropping the
    /// caption entirely.
    pub fn export_caption(&self) -> Option<String> {
        if let Some(text) = self
            .manual_caption
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Some(text.to_string());
        }
        self.captions
            .values()
            .max_by_key(|e| e.captioned_at)
            .map(|e| e.caption.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn set_manual_caption(&mut self, text: &str) {
        let trimmed = text.trim();
        self.manual_caption = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
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
    fn export_caption_prefers_manual_then_falls_back_to_latest_auto() {
        let mut sc = Sidecar::default();
        // No captions at all → None.
        assert_eq!(sc.export_caption(), None);

        // Only an auto caption → fall back to it.
        sc.set_caption("modelA", "auto text");
        assert_eq!(sc.export_caption().as_deref(), Some("auto text"));

        // A second auto caption with a later timestamp wins the fallback.
        std::thread::sleep(std::time::Duration::from_millis(2));
        sc.set_caption("modelB", "newer auto text");
        assert_eq!(sc.export_caption().as_deref(), Some("newer auto text"));

        // Manual caption overrides any auto.
        sc.set_manual_caption("manual text");
        assert_eq!(sc.export_caption().as_deref(), Some("manual text"));

        // Empty/whitespace manual falls back again.
        sc.set_manual_caption("   ");
        assert_eq!(sc.export_caption().as_deref(), Some("newer auto text"));
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
