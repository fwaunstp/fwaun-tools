use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CONFIG_FILE: &str = "anima-tagger.toml";
/// Per-user config file relative to `$XDG_CONFIG_HOME` (defaulting to
/// `~/.config`). Provides shared defaults for `[captioner.*]` /
/// `[tagger.*]` / `[export.*]` profiles so each dataset directory
/// doesn't need its own copy. Per-directory `anima-tagger.toml` still
/// wins on key collision.
pub const USER_CONFIG_RELATIVE: &str = "anima-tagger/config.toml";
pub const DEFAULT_PROFILE_NAME: &str = "anima";

/// Built-in tagger profile name + repo, used when nothing is configured.
pub const BUILT_IN_TAGGER_NAME: &str = "wd-eva02-large-v3";
pub const BUILT_IN_TAGGER_REPO: &str = "SmilingWolf/wd-eva02-large-tagger-v3";

/// Built-in captioner profile name + repo, used when nothing is configured.
pub const BUILT_IN_CAPTIONER_NAME: &str = "qwen3-vl-4b";
pub const BUILT_IN_CAPTIONER_REPO: &str = "onnx-community/Qwen3-4B-VL-ONNX";
/// onnx-community packs multiple variants (2B/4B/8B, different precision
/// combos) into the same repo under variant-named subdirectories. The default
/// is the 4B vision-fp32 / text-int4 build, the only prebuilt 4B variant
/// published.
pub const BUILT_IN_CAPTIONER_SUBDIR: &str = "qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub default_tagger: Option<String>,
    #[serde(default)]
    pub default_captioner: Option<String>,
    #[serde(default)]
    pub export: BTreeMap<String, ExportProfile>,
    #[serde(default)]
    pub tagger: BTreeMap<String, TaggerProfile>,
    #[serde(default)]
    pub captioner: BTreeMap<String, CaptionerProfile>,
}

/// HuggingFace-hosted WD14-family tagger profile. Models are downloaded into
/// the shared hf-hub cache on first use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggerProfile {
    /// HuggingFace repo id, e.g. `"SmilingWolf/wd-eva02-large-tagger-v3"`.
    pub repo: String,
    /// Optional git revision/branch/tag to pin (defaults to the repo's `main`).
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default = "default_input_size")]
    pub input_size: u32,
    #[serde(default = "default_storage_threshold")]
    pub storage_threshold: f32,
}

fn default_input_size() -> u32 {
    448
}

fn default_storage_threshold() -> f32 {
    0.10
}

/// Captioner profile. Tagged on `kind` so users can mix backends in one
/// config: a local ONNX run for cheap shots, plus an OpenAI-compatible
/// HTTP backend (llama.cpp / koboldcpp / Ollama / LM Studio / vLLM) for
/// larger or NSFW-uncensored models that have no ONNX export.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CaptionerProfile {
    Onnx(OnnxCaptionerProfile),
    Openai(OpenAiCaptionerProfile),
}

/// HuggingFace-hosted Qwen3-VL-family ONNX captioner. Dynamic-resolution
/// pipeline (32-pixel patch grid, smart-resized at runtime), so instead of
/// a fixed `input_size` we cap the area via `max_pixels`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxCaptionerProfile {
    /// HuggingFace repo id, e.g. `"onnx-community/Qwen3-4B-VL-ONNX"`.
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
    /// Subdirectory inside the repo holding the ONNX files
    /// (`qwen3vl-vision.onnx`, `qwen3vl-embedding.onnx`, `model.onnx` +
    /// `model.onnx.data`, `tokenizer.json`). onnx-community ships multiple
    /// variants per repo under separate subdirs; for forks that put files at
    /// the repo root, leave this empty / `""`.
    #[serde(default)]
    pub subdir: Option<String>,
    #[serde(default = "default_caption_prompt")]
    pub prompt: String,
    /// Upper bound on (resized_h * resized_w) during smart_resize. Larger
    /// values give richer captions but quadratically more vision tokens.
    #[serde(default = "default_max_pixels")]
    pub max_pixels: u32,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

/// OpenAI-compatible chat-completions captioner. Works against any server
/// that implements `/chat/completions` with vision (`image_url` content
/// parts): llama.cpp `llama-server`, koboldcpp, Ollama, LM Studio, vLLM,
/// TGI, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCaptionerProfile {
    /// Base URL up to and including `/v1` (we append `/chat/completions`).
    /// e.g. `"http://localhost:8080/v1"` for llama-server's default.
    pub endpoint: String,
    /// Model name to send. Many local servers ignore it but a few require
    /// a non-empty value; defaults to `"local"` if unset.
    #[serde(default)]
    pub model: Option<String>,
    /// Bearer token. Empty/None = no `Authorization` header.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_caption_prompt")]
    pub prompt: String,
    #[serde(default = "default_max_new_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Resize the image so the longest edge is at most this many pixels
    /// before sending. 0 = send the file bytes verbatim. Default 1024
    /// keeps payloads sane against vision encoders that internally cap
    /// pixel counts.
    #[serde(default = "default_openai_max_edge")]
    pub max_edge: u32,
    /// JPEG quality (1-100) when `max_edge > 0`.
    #[serde(default = "default_openai_jpeg_quality")]
    pub jpeg_quality: u8,
    /// Per-request timeout in seconds. Long-running CPU servers can need
    /// several minutes for a single caption.
    #[serde(default = "default_openai_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_caption_prompt() -> String {
    "Describe this image in detail.".to_string()
}

fn default_max_pixels() -> u32 {
    // 768*768. Smart-resize will round down to the nearest 28-multiple and
    // produce ~196 vision tokens for a square image — a workable balance
    // between detail and decode time on CPU.
    589_824
}

fn default_max_new_tokens() -> usize {
    1024
}

fn default_openai_max_edge() -> u32 {
    1024
}

fn default_openai_jpeg_quality() -> u8 {
    90
}

fn default_openai_timeout_secs() -> u64 {
    600
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProfile {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
    #[serde(default)]
    pub exclude_categories: Vec<String>,
    /// Map of category -> string prefix to apply to auto tags of that category
    /// (e.g. ANIMA: `{ "artist" = "@" }`).
    #[serde(default)]
    pub category_prefixes: BTreeMap<String, String>,
}

fn default_threshold() -> f32 {
    0.35
}

fn default_shuffle() -> bool {
    // sd-scripts and most modern LoRA trainers shuffle tags themselves at
    // training time, so don't shuffle on export by default.
    false
}

impl Default for ExportProfile {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            shuffle: default_shuffle(),
            exclude_categories: Vec::new(),
            category_prefixes: BTreeMap::new(),
        }
    }
}

impl ExportProfile {
    pub fn anima() -> Self {
        let mut category_prefixes = BTreeMap::new();
        category_prefixes.insert("artist".to_string(), "@".to_string());
        Self {
            threshold: default_threshold(),
            shuffle: default_shuffle(),
            exclude_categories: Vec::new(),
            category_prefixes,
        }
    }

    pub fn category_prefix(&self, category: &str) -> Option<&str> {
        self.category_prefixes.get(category).map(String::as_str)
    }

    pub fn all_prefixes(&self) -> impl Iterator<Item = &str> {
        self.category_prefixes.values().map(String::as_str)
    }
}

impl TaggerProfile {
    pub fn built_in() -> Self {
        Self {
            repo: BUILT_IN_TAGGER_REPO.to_string(),
            revision: None,
            input_size: default_input_size(),
            storage_threshold: default_storage_threshold(),
        }
    }
}

impl CaptionerProfile {
    pub fn built_in() -> Self {
        Self::Onnx(OnnxCaptionerProfile {
            repo: BUILT_IN_CAPTIONER_REPO.to_string(),
            revision: None,
            subdir: Some(BUILT_IN_CAPTIONER_SUBDIR.to_string()),
            prompt: default_caption_prompt(),
            max_pixels: default_max_pixels(),
            max_new_tokens: default_max_new_tokens(),
        })
    }

    /// Short human-readable description (HF repo, or HTTP endpoint).
    pub fn source_label(&self) -> String {
        match self {
            Self::Onnx(p) => p.repo.clone(),
            Self::Openai(p) => p.endpoint.clone(),
        }
    }
}

impl Default for ProjectConfig {
    fn default() -> Self {
        let mut export = BTreeMap::new();
        export.insert("anima".to_string(), ExportProfile::anima());
        export.insert("plain".to_string(), ExportProfile::default());
        Self {
            default_profile: Some(DEFAULT_PROFILE_NAME.to_string()),
            default_tagger: None,
            default_captioner: None,
            export,
            tagger: BTreeMap::new(),
            captioner: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("io error on {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse error on {path}: {source}")]
    Parse {
        path: std::path::PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

impl ProjectConfig {
    /// Resolve `$XDG_CONFIG_HOME/anima-tagger/config.toml`, falling back to
    /// `$HOME/.config/anima-tagger/config.toml`. Returns `None` if neither
    /// env var is set (no usable home).
    pub fn user_config_path() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
            return Some(PathBuf::from(xdg).join(USER_CONFIG_RELATIVE));
        }
        std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(|home| PathBuf::from(home).join(".config").join(USER_CONFIG_RELATIVE))
    }

    fn load_path(path: &Path) -> Result<Option<Self>, ConfigError> {
        if !path.exists() {
            return Ok(None);
        }
        let s = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg = toml::from_str(&s).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Some(cfg))
    }

    /// Load only the per-directory `anima-tagger.toml`, ignoring the user
    /// config. Kept for callers that need to inspect what a single
    /// dataset declared.
    pub fn load(dir: &Path) -> Result<Option<Self>, ConfigError> {
        Self::load_path(&dir.join(CONFIG_FILE))
    }

    /// User-level config (no merge with project).
    pub fn load_user() -> Result<Option<Self>, ConfigError> {
        match Self::user_config_path() {
            Some(p) => Self::load_path(&p),
            None => Ok(None),
        }
    }

    /// Load the merged effective config: defaults ← user config ← project
    /// config. Project entries override user entries with the same key,
    /// and user entries override hard-coded defaults. Missing files are
    /// not errors — only parse/IO failures bubble up.
    pub fn load_or_default(dir: &Path) -> Result<Self, ConfigError> {
        let mut cfg = Self::default();
        if let Some(user) = Self::load_user()? {
            cfg.merge_from(user);
        }
        if let Some(project) = Self::load(dir)? {
            cfg.merge_from(project);
        }
        Ok(cfg)
    }

    /// Overlay `other` onto `self`. `other`'s scalars overwrite `self`'s
    /// when set; map entries union, with `other` winning on key collision.
    fn merge_from(&mut self, other: ProjectConfig) {
        if other.default_profile.is_some() {
            self.default_profile = other.default_profile;
        }
        if other.default_tagger.is_some() {
            self.default_tagger = other.default_tagger;
        }
        if other.default_captioner.is_some() {
            self.default_captioner = other.default_captioner;
        }
        for (k, v) in other.export {
            self.export.insert(k, v);
        }
        for (k, v) in other.tagger {
            self.tagger.insert(k, v);
        }
        for (k, v) in other.captioner {
            self.captioner.insert(k, v);
        }
    }

    pub fn resolve_profile(&self, name: Option<&str>) -> ExportProfile {
        let key = name
            .map(str::to_string)
            .or_else(|| self.default_profile.clone());
        if let Some(k) = key.as_deref()
            && let Some(p) = self.export.get(k)
        {
            return p.clone();
        }
        ExportProfile::default()
    }

    /// Resolve a tagger profile. Order: explicit `name`, then `default_tagger`,
    /// then the built-in profile. Always succeeds — auto-download means a
    /// configured profile is no longer required.
    pub fn resolve_tagger(&self, name: Option<&str>) -> (String, TaggerProfile) {
        let key = name
            .map(str::to_string)
            .or_else(|| self.default_tagger.clone());
        if let Some(k) = key
            && let Some(profile) = self.tagger.get(&k)
        {
            return (k, profile.clone());
        }
        (BUILT_IN_TAGGER_NAME.to_string(), TaggerProfile::built_in())
    }

    /// Resolve a captioner profile, falling back to the built-in if nothing
    /// matches. Same logic as `resolve_tagger`.
    pub fn resolve_captioner(&self, name: Option<&str>) -> (String, CaptionerProfile) {
        let key = name
            .map(str::to_string)
            .or_else(|| self.default_captioner.clone());
        if let Some(k) = key
            && let Some(profile) = self.captioner.get(&k)
        {
            return (k, profile.clone());
        }
        (
            BUILT_IN_CAPTIONER_NAME.to_string(),
            CaptionerProfile::built_in(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_project_overrides_user() {
        let mut user = ProjectConfig::default();
        user.default_captioner = Some("user-cap".into());
        user.captioner.insert(
            "shared".into(),
            CaptionerProfile::Openai(OpenAiCaptionerProfile {
                endpoint: "http://user".into(),
                model: None,
                api_key: None,
                prompt: default_caption_prompt(),
                max_tokens: 100,
                temperature: None,
                max_edge: 1024,
                jpeg_quality: 90,
                timeout_secs: 600,
            }),
        );

        let mut project = ProjectConfig::default();
        project.captioner.insert(
            "shared".into(),
            CaptionerProfile::Openai(OpenAiCaptionerProfile {
                endpoint: "http://project".into(),
                model: None,
                api_key: None,
                prompt: default_caption_prompt(),
                max_tokens: 200,
                temperature: None,
                max_edge: 1024,
                jpeg_quality: 90,
                timeout_secs: 600,
            }),
        );

        let mut merged = ProjectConfig::default();
        merged.merge_from(user);
        merged.merge_from(project);

        assert_eq!(merged.default_captioner.as_deref(), Some("user-cap"));
        match merged.captioner.get("shared").unwrap() {
            CaptionerProfile::Openai(p) => assert_eq!(p.endpoint, "http://project"),
            _ => panic!("expected openai variant"),
        }
    }

    #[test]
    fn project_only_keys_survive_merge() {
        let user = ProjectConfig::default();
        let mut project = ProjectConfig::default();
        project.tagger.insert(
            "wd-tagger".into(),
            TaggerProfile::built_in(),
        );

        let mut merged = ProjectConfig::default();
        merged.merge_from(user);
        merged.merge_from(project);

        assert!(merged.tagger.contains_key("wd-tagger"));
    }
}
