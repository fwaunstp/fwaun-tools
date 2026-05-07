use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CONFIG_FILE: &str = "anima-tagger.toml";

/// Annotated TOML template covering every supported profile field.
/// Shipped alongside the crate so consumers (e.g. the GUI's "Config…"
/// modal) can show users a starting point without having to maintain
/// a separate copy.
pub const CONFIG_EXAMPLE: &str = include_str!("../anima-tagger.toml.example");
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
    /// Shared prompt library — define each prompt once here and reference
    /// it by name from any captioner profile's `prompts = [...]`. The
    /// built-in `default` is always available; redefining `default` here
    /// overrides it.
    #[serde(default)]
    pub captioner_prompts: BTreeMap<String, String>,
    /// Named groups of tags that should be mutually exclusive on each
    /// image (e.g. costume variants, pose categories). Used by the CLI's
    /// `validate-tag-group` command and by the GUI Kanban view to bucket
    /// images into one column per tag, plus an "unset" and "violation"
    /// column. Single-tag groups are valid — handy for "is tag X set or
    /// not?" curation passes.
    #[serde(default, rename = "tag_group")]
    pub tag_groups: BTreeMap<String, TagGroup>,
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
    /// Names of prompts to run against the same loaded model. Looked up
    /// in `[captioner_prompts]` (or the built-in library). Sidecar
    /// entries are keyed `{profile_name}.{prompt_name}` so multiple
    /// prompts coexist without re-loading the model. Defaults to
    /// `["default"]`.
    #[serde(default = "default_profile_prompts")]
    pub prompts: Vec<String>,
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
    /// Names of prompts to run. See `OnnxCaptionerProfile::prompts`.
    #[serde(default = "default_profile_prompts")]
    pub prompts: Vec<String>,
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

/// The built-in `default` prompt text. Sized to fit comfortably inside
/// ANIMA's 512-token training ceiling. Users can override `default` (or
/// add more named prompts) via `[captioner_prompts]` in their config.
pub const BUILT_IN_DEFAULT_PROMPT: &str =
    "Describe this image in detail in 3-5 sentences (under 200 words).";

/// Built-in prompt library: a single `default` entry. Merged with the
/// user's `[captioner_prompts]` table at resolution time, with user
/// entries taking precedence on key collision.
pub fn default_prompt_library() -> BTreeMap<String, String> {
    BTreeMap::from([(
        "default".to_string(),
        BUILT_IN_DEFAULT_PROMPT.to_string(),
    )])
}

fn default_profile_prompts() -> Vec<String> {
    vec!["default".to_string()]
}

fn default_max_pixels() -> u32 {
    // 768*768. Smart-resize will round down to the nearest 28-multiple and
    // produce ~196 vision tokens for a square image — a workable balance
    // between detail and decode time on CPU.
    589_824
}

fn default_max_new_tokens() -> usize {
    // Matches ANIMA's training-time qwen3 / t5 max_token_length default
    // (512). Going higher wastes decode time on tokens the base model
    // wouldn't have seen during training. Bump per-profile if you're
    // captioning for a different downstream model.
    512
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

/// Named group of tags. Currently always treated as mutually exclusive
/// on each image — i.e. at most one of `tags` is expected to be present
/// in the effective tag set (manual positive ∪ auto ∪ booru, minus
/// `-foo` suppressions). Two or more co-occurring is a "violation" —
/// flagged but not an error, since edge cases like character setting
/// sheets legitimately show multiple costumes in one frame.
///
/// Single-tag groups are valid and useful for a "set / unset" split on
/// one tag (e.g. `[tag_group.solo_check] tags = ["solo"]`).
///
/// A future `exclusive: bool` field can be added with
/// `#[serde(default = "...")]` (defaulting to `true`) without breaking
/// existing configs, if non-exclusive grouping ever becomes useful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagGroup {
    /// Tags that participate in this group.
    pub tags: Vec<String>,
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
            prompts: default_profile_prompts(),
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

    fn prompt_names(&self) -> &[String] {
        match self {
            Self::Onnx(p) => &p.prompts,
            Self::Openai(p) => &p.prompts,
        }
    }

    /// Replace this profile's `prompts` field. Used by the CLI to apply
    /// a `--prompts` override at runtime without editing the config file.
    pub fn set_prompt_names(&mut self, names: Vec<String>) {
        match self {
            Self::Onnx(p) => p.prompts = names,
            Self::Openai(p) => p.prompts = names,
        }
    }

    /// Resolve this profile's prompt names against `library`, returning
    /// (name, text) pairs in the order the profile listed them.
    /// Duplicates are collapsed. An empty `prompts` list resolves to
    /// `["default"]` (so a profile that omits the field still captions).
    pub fn resolved_prompts(
        &self,
        library: &BTreeMap<String, String>,
    ) -> Result<Vec<(String, String)>, ConfigError> {
        let names = self.prompt_names();
        let fallback = default_profile_prompts();
        let names: &[String] = if names.is_empty() { &fallback } else { names };

        let mut out: Vec<(String, String)> = Vec::with_capacity(names.len());
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in names {
            if !seen.insert(name.as_str()) {
                continue;
            }
            let text = library
                .get(name)
                .ok_or_else(|| ConfigError::UnknownPrompt(name.clone()))?;
            out.push((name.clone(), text.clone()));
        }
        Ok(out)
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
            captioner_prompts: BTreeMap::new(),
            tag_groups: BTreeMap::new(),
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
    #[error("unknown prompt name `{0}` — define it in [captioner_prompts] or pick an existing one")]
    UnknownPrompt(String),
    #[error("tag_group `{0}` has no tags — every group must list at least one tag")]
    EmptyTagGroup(String),
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

    /// Walk up from `start` looking for an `anima-tagger.toml`. Returns the
    /// first matching file (analogous to how `git` finds `.git`). This lets
    /// users keep a single config at the dataset root while operating on
    /// subdirectories.
    pub fn find_project_config(start: &Path) -> Option<PathBuf> {
        let abs = start
            .canonicalize()
            .unwrap_or_else(|_| start.to_path_buf());
        abs.ancestors()
            .map(|d| d.join(CONFIG_FILE))
            .find(|p| p.is_file())
    }

    /// Load the project `anima-tagger.toml`, searching `dir` and its
    /// ancestors. Ignores the user config. Returns `None` if no project
    /// config exists anywhere up the tree.
    pub fn load(dir: &Path) -> Result<Option<Self>, ConfigError> {
        match Self::find_project_config(dir) {
            Some(p) => Self::load_path(&p),
            None => Ok(None),
        }
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
        cfg.validate_tag_groups()?;
        Ok(cfg)
    }

    /// Reject obviously broken `[tag_group.*]` entries. Called from
    /// `load_or_default` after the merge so an error here represents the
    /// user's effective config (a user-level entry can be repaired by
    /// project-level override and vice versa). Single-tag groups,
    /// cross-group overlap, and tags absent from every image are all
    /// allowed — they show up as informational signals in the CLI's
    /// validate output and the GUI Kanban view rather than hard errors.
    pub fn validate_tag_groups(&self) -> Result<(), ConfigError> {
        for (name, group) in &self.tag_groups {
            if group.tags.is_empty() {
                return Err(ConfigError::EmptyTagGroup(name.clone()));
            }
        }
        Ok(())
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
        for (k, v) in other.captioner_prompts {
            self.captioner_prompts.insert(k, v);
        }
        for (k, v) in other.tag_groups {
            self.tag_groups.insert(k, v);
        }
    }

    /// Effective prompt library: built-in defaults overlaid with the
    /// user's `[captioner_prompts]` (user entries win). Pass to
    /// `CaptionerProfile::resolved_prompts` when invoking captions.
    pub fn prompt_library(&self) -> BTreeMap<String, String> {
        let mut lib = default_prompt_library();
        for (k, v) in &self.captioner_prompts {
            lib.insert(k.clone(), v.clone());
        }
        lib
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

    /// Drops a fresh subdirectory under `temp_dir()` on Drop so tests that
    /// touch the real filesystem don't leak files between runs.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "anima-tagger-test-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn find_project_config_walks_up_to_parent() {
        let root = TempDir::new("walkup");
        let nested = root.path().join("a").join("b").join("c");
        fs::create_dir_all(&nested).unwrap();
        let cfg_path = root.path().join(CONFIG_FILE);
        fs::write(&cfg_path, "default_profile = \"plain\"\n").unwrap();

        let found = ProjectConfig::find_project_config(&nested)
            .expect("should walk up to root config");
        assert_eq!(found.canonicalize().unwrap(), cfg_path.canonicalize().unwrap());

        let cfg = ProjectConfig::load(&nested)
            .expect("load ok")
            .expect("config present");
        assert_eq!(cfg.default_profile.as_deref(), Some("plain"));
    }

    #[test]
    fn find_project_config_prefers_nearest_ancestor() {
        let root = TempDir::new("nearest");
        let mid = root.path().join("mid");
        let leaf = mid.join("leaf");
        fs::create_dir_all(&leaf).unwrap();
        fs::write(root.path().join(CONFIG_FILE), "default_profile = \"root\"\n").unwrap();
        fs::write(mid.join(CONFIG_FILE), "default_profile = \"mid\"\n").unwrap();

        let cfg = ProjectConfig::load(&leaf).unwrap().unwrap();
        assert_eq!(cfg.default_profile.as_deref(), Some("mid"));
    }

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
                prompts: default_profile_prompts(),
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
                prompts: default_profile_prompts(),
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
    fn resolved_prompts_falls_back_to_default_when_unset() {
        let cfg = CaptionerProfile::built_in();
        let library = ProjectConfig::default().prompt_library();
        let prompts = cfg.resolved_prompts(&library).unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].0, "default");
        assert_eq!(prompts[0].1, BUILT_IN_DEFAULT_PROMPT);
    }

    #[test]
    fn resolved_prompts_returns_names_in_listed_order() {
        let cfg = CaptionerProfile::Onnx(OnnxCaptionerProfile {
            repo: "r".into(),
            revision: None,
            subdir: None,
            prompts: vec!["character".into(), "default".into()],
            max_pixels: default_max_pixels(),
            max_new_tokens: default_max_new_tokens(),
        });
        let mut config = ProjectConfig::default();
        config
            .captioner_prompts
            .insert("character".into(), "Describe characters.".into());
        let library = config.prompt_library();
        let prompts = cfg.resolved_prompts(&library).unwrap();
        assert_eq!(
            prompts.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["character", "default"]
        );
    }

    #[test]
    fn resolved_prompts_unknown_name_errors() {
        let cfg = CaptionerProfile::Onnx(OnnxCaptionerProfile {
            repo: "r".into(),
            revision: None,
            subdir: None,
            prompts: vec!["nonexistent".into()],
            max_pixels: default_max_pixels(),
            max_new_tokens: default_max_new_tokens(),
        });
        let library = ProjectConfig::default().prompt_library();
        let err = cfg.resolved_prompts(&library).unwrap_err();
        match err {
            ConfigError::UnknownPrompt(name) => assert_eq!(name, "nonexistent"),
            other => panic!("expected UnknownPrompt, got {other:?}"),
        }
    }

    #[test]
    fn captioner_prompts_user_override_wins_over_built_in_default() {
        let mut config = ProjectConfig::default();
        config
            .captioner_prompts
            .insert("default".into(), "Describe briefly.".into());
        let library = config.prompt_library();
        assert_eq!(library.get("default").map(String::as_str), Some("Describe briefly."));
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

    /// Guard: when a field is added to any of the profile structs, the
    /// shipped example (`crates/core/anima-tagger.toml.example`) has to grow alongside
    /// it. The test serializes a fully-populated synthetic instance of each
    /// profile, then asserts that at least one profile of the matching kind
    /// in the example covers every produced key.
    ///
    /// Legacy / deprecated fields are left as `None` so they don't have to
    /// appear in the example (they round-trip via `skip_serializing_if`).
    #[test]
    fn example_config_documents_every_supported_field() {
        use std::collections::BTreeSet;

        let example_str = CONFIG_EXAMPLE;
        let cfg: ProjectConfig = toml::from_str(example_str)
            .expect("anima-tagger.toml.example must parse as ProjectConfig");
        let raw: toml::Value = toml::from_str(example_str)
            .expect("anima-tagger.toml.example must parse as toml::Value");
        let raw_table = raw.as_table().expect("example must be a top-level table");

        for k in [
            "default_profile",
            "default_tagger",
            "default_captioner",
            "captioner_prompts",
        ] {
            assert!(
                raw_table.contains_key(k),
                "example missing top-level key `{k}`"
            );
        }

        if let Some(p) = &cfg.default_profile {
            assert!(
                cfg.export.contains_key(p),
                "default_profile = {p:?} but no matching [export.{p}] in the example"
            );
        }
        if let Some(t) = &cfg.default_tagger {
            assert!(
                cfg.tagger.contains_key(t),
                "default_tagger = {t:?} but no matching [tagger.{t}] in the example"
            );
        }
        if let Some(c) = &cfg.default_captioner {
            assert!(
                cfg.captioner.contains_key(c),
                "default_captioner = {c:?} but no matching [captioner.{c}] in the example"
            );
        }

        fn struct_keys<T: serde::Serialize>(v: T) -> BTreeSet<String> {
            #[derive(serde::Serialize)]
            struct Wrap<T: serde::Serialize> {
                inner: T,
            }
            let s = toml::to_string(&Wrap { inner: v }).expect("serialize wrapped value");
            let parsed: toml::Value = toml::from_str(&s).expect("re-parse wrapped value");
            parsed
                .get("inner")
                .and_then(|v| v.as_table())
                .expect("wrapped value must serialize to a table")
                .keys()
                .cloned()
                .collect()
        }

        fn missing_from_best_match(
            section: Option<&toml::Value>,
            expected: &BTreeSet<String>,
            filter: impl Fn(&toml::Table) -> bool,
        ) -> Result<(), BTreeSet<String>> {
            let Some(table) = section.and_then(|v| v.as_table()) else {
                return Err(expected.clone());
            };
            let mut best: Option<BTreeSet<String>> = None;
            for profile in table.values() {
                let Some(pt) = profile.as_table() else {
                    continue;
                };
                if !filter(pt) {
                    continue;
                }
                let actual: BTreeSet<String> = pt.keys().cloned().collect();
                let missing: BTreeSet<String> =
                    expected.difference(&actual).cloned().collect();
                if missing.is_empty() {
                    return Ok(());
                }
                if best.as_ref().is_none_or(|b| missing.len() < b.len()) {
                    best = Some(missing);
                }
            }
            Err(best.unwrap_or_else(|| expected.clone()))
        }

        let full_export = ExportProfile {
            threshold: 0.35,
            shuffle: true,
            exclude_categories: vec!["meta".into()],
            category_prefixes: BTreeMap::from([("artist".into(), "@".into())]),
        };
        let full_tagger = TaggerProfile {
            repo: "r".into(),
            revision: Some("main".into()),
            input_size: 448,
            storage_threshold: 0.10,
        };
        let full_onnx = CaptionerProfile::Onnx(OnnxCaptionerProfile {
            repo: "r".into(),
            revision: Some("main".into()),
            subdir: Some("d".into()),
            prompts: vec!["default".into()],
            max_pixels: default_max_pixels(),
            max_new_tokens: default_max_new_tokens(),
        });
        let full_openai = CaptionerProfile::Openai(OpenAiCaptionerProfile {
            endpoint: "http://x".into(),
            model: Some("m".into()),
            api_key: Some("k".into()),
            prompts: vec!["default".into()],
            max_tokens: default_max_new_tokens(),
            temperature: Some(0.7),
            max_edge: default_openai_max_edge(),
            jpeg_quality: default_openai_jpeg_quality(),
            timeout_secs: default_openai_timeout_secs(),
        });
        let full_tag_group = TagGroup {
            tags: vec!["x".into()],
        };

        let expected_export = struct_keys(full_export);
        let expected_tagger = struct_keys(full_tagger);
        let expected_onnx = struct_keys(full_onnx);
        let expected_openai = struct_keys(full_openai);
        let expected_tag_group = struct_keys(full_tag_group);

        if let Err(missing) =
            missing_from_best_match(raw_table.get("export"), &expected_export, |_| true)
        {
            panic!(
                "no [export.*] profile in crates/core/anima-tagger.toml.example covers every \
                 ExportProfile field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("tagger"), &expected_tagger, |_| true)
        {
            panic!(
                "no [tagger.*] profile in crates/core/anima-tagger.toml.example covers every \
                 TaggerProfile field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) = missing_from_best_match(
            raw_table.get("captioner"),
            &expected_onnx,
            |t| t.get("kind").and_then(|v| v.as_str()) == Some("onnx"),
        ) {
            panic!(
                "no [captioner.*] profile with `kind = \"onnx\"` in \
                 crates/core/anima-tagger.toml.example covers every OnnxCaptionerProfile field; \
                 closest match is missing {missing:?}"
            );
        }
        if let Err(missing) = missing_from_best_match(
            raw_table.get("captioner"),
            &expected_openai,
            |t| t.get("kind").and_then(|v| v.as_str()) == Some("openai"),
        ) {
            panic!(
                "no [captioner.*] profile with `kind = \"openai\"` in \
                 crates/core/anima-tagger.toml.example covers every OpenAiCaptionerProfile field; \
                 closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("tag_group"), &expected_tag_group, |_| true)
        {
            panic!(
                "no [tag_group.*] entry in crates/core/anima-tagger.toml.example covers every \
                 TagGroup field; closest match is missing {missing:?}"
            );
        }
    }

    #[test]
    fn tag_group_round_trips_through_toml() {
        let mut cfg = ProjectConfig::default();
        cfg.tag_groups.insert(
            "official_costumes".into(),
            TagGroup {
                tags: vec!["a".into(), "b".into()],
            },
        );
        let s = toml::to_string(&cfg).expect("serialize");
        let parsed: ProjectConfig = toml::from_str(&s).expect("re-parse");
        let group = parsed
            .tag_groups
            .get("official_costumes")
            .expect("group survives round-trip");
        assert_eq!(group.tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn validate_tag_groups_rejects_empty_tags() {
        let mut cfg = ProjectConfig::default();
        cfg.tag_groups
            .insert("foo".into(), TagGroup { tags: Vec::new() });
        match cfg.validate_tag_groups() {
            Err(ConfigError::EmptyTagGroup(name)) => assert_eq!(name, "foo"),
            other => panic!("expected EmptyTagGroup, got {other:?}"),
        }
    }

    #[test]
    fn validate_tag_groups_accepts_single_tag_group() {
        let mut cfg = ProjectConfig::default();
        cfg.tag_groups.insert(
            "solo_check".into(),
            TagGroup {
                tags: vec!["solo".into()],
            },
        );
        cfg.validate_tag_groups().expect("single-tag group is valid");
    }
}
