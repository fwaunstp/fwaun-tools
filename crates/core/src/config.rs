use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CONFIG_FILE: &str = "fwaun-tools.toml";

/// Former per-directory config filenames, newest-first, from earlier project
/// names (`fwaun-tagger`, then `anima-tagger`). Each is still read (with a
/// deprecation warning) when no [`CONFIG_FILE`] is present, so existing
/// datasets keep working across renames. Support for these fallbacks will be
/// removed in a future release.
pub const LEGACY_CONFIG_FILES: &[&str] = &["fwaun-tagger.toml", "anima-tagger.toml"];

/// Annotated TOML template covering every supported profile field.
/// Shipped alongside the crate so consumers (e.g. the GUI's "Config…"
/// modal) can show users a starting point without having to maintain
/// a separate copy.
pub const CONFIG_EXAMPLE: &str = include_str!("../fwaun-tools.toml.example");
/// Per-user config file relative to `$XDG_CONFIG_HOME` (defaulting to
/// `~/.config`). Provides shared defaults for `[captioner.*]` /
/// `[tagger.*]` / `[export.*]` profiles so each dataset directory
/// doesn't need its own copy. Per-directory `fwaun-tools.toml` still
/// wins on key collision.
pub const USER_CONFIG_RELATIVE: &str = "fwaun-tools/config.toml";

/// Former per-user config paths, newest-first, from earlier project names.
/// Each is read (with a deprecation warning) only when
/// [`USER_CONFIG_RELATIVE`] is absent.
pub const LEGACY_USER_CONFIG_RELATIVES: &[&str] =
    &["fwaun-tagger/config.toml", "anima-tagger/config.toml"];

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

/// Warn (once per resolved file) that a config is being loaded from a
/// deprecated location left over from an earlier project name. The fallback
/// keeps existing datasets working through the renames but will be removed
/// later.
fn warn_legacy_config(legacy: &Path) {
    eprintln!(
        "warning: loaded deprecated config `{}`. The project has been renamed \
         to `fwaun-tools`; rename this file to `{CONFIG_FILE}` (or the \
         user-level `{USER_CONFIG_RELATIVE}`) — the legacy fallback will be \
         removed in a future release.",
        legacy.display(),
    );
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub default_tagger: Option<String>,
    #[serde(default)]
    pub default_captioner: Option<String>,
    #[serde(default)]
    pub default_upscaler: Option<String>,
    #[serde(default)]
    pub default_editor: Option<String>,
    /// Manual entries applied to every image in the dataset without being
    /// written into any sidecar — the trigger word for a character LoRA plus
    /// the `-foo` suppressions that fold its traits (hair colour, eye colour,
    /// …) into that trigger. Uses the manual-tag syntax verbatim: `foo`
    /// positive, `-foo` suppression, `_foo` curation-only.
    ///
    /// An image opts out of any entry by naming the same tag itself in any
    /// form, so a common `-red_hair` is cancelled by a per-image `red_hair`.
    /// Resolve with [`resolve_common_tags`](Self::resolve_common_tags); see
    /// [`crate::common_tags`] for the full semantics.
    ///
    /// Unlike the profile tables this does not union across config levels: a
    /// non-empty project list replaces the user-level one outright, since the
    /// set is dataset-specific and a half-inherited list would be worse than
    /// either.
    #[serde(default)]
    pub common_tags: Vec<String>,
    #[serde(default)]
    pub export: BTreeMap<String, ExportProfile>,
    #[serde(default)]
    pub tagger: BTreeMap<String, TaggerProfile>,
    #[serde(default)]
    pub captioner: BTreeMap<String, CaptionerProfile>,
    #[serde(default)]
    pub upscaler: BTreeMap<String, UpscalerProfile>,
    #[serde(default)]
    pub editor: BTreeMap<String, EditorProfile>,
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
    /// How many times to re-run generation when the model returns an empty
    /// caption (trimmed to nothing). An empty result is otherwise saved
    /// verbatim and marks the `{model}.{prompt}` key as done, so it never
    /// regenerates without `--force`. After the retries are exhausted the
    /// caption is reported as an error instead of being stored. 0 = never
    /// retry. Default 2. (Greedy ONNX decode is deterministic, so a retry
    /// reproduces the same empty output — the value here still governs how
    /// many attempts are made before the empty result surfaces as an error.)
    #[serde(default = "default_empty_retries")]
    pub empty_retries: u32,
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
    /// How many times to retry a request that fails with a transient error
    /// (HTTP 5xx or a transport/network failure). Some local servers — gpt-oss
    /// harmony parsing in particular — intermittently 500 on a request that
    /// succeeds on a fresh attempt. 0 = never retry. Default 3.
    #[serde(default = "default_openai_max_retries")]
    pub max_retries: u32,
    /// How many times to re-run generation when the server returns an empty
    /// caption (trimmed to nothing) — distinct from `max_retries`, which
    /// covers transient HTTP/transport failures. An empty result is otherwise
    /// saved verbatim and marks the `{model}.{prompt}` key as done, so it
    /// never regenerates without `--force`; after the retries are exhausted it
    /// is reported as an error instead of being stored. 0 = never retry.
    /// Default 2.
    #[serde(default = "default_empty_retries")]
    pub empty_retries: u32,
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
    BTreeMap::from([("default".to_string(), BUILT_IN_DEFAULT_PROMPT.to_string())])
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

fn default_openai_max_retries() -> u32 {
    3
}

fn default_empty_retries() -> u32 {
    2
}

/// Default ComfyUI server root — the stock local install listens here.
pub const DEFAULT_COMFYUI_BASE_URL: &str = "http://127.0.0.1:8188";

/// ComfyUI-backed upscaler profile. Each image is sent to an existing ComfyUI
/// server over its HTTP API (`/upload/image` → `/prompt` → `/history` →
/// `/view`) and the upscaled result pulled back — no per-image manual
/// upload/download. Configure a model filename for the built-in ESRGAN-style
/// workflow, or point at your own API-format workflow export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscalerProfile {
    /// ComfyUI server root, e.g. `"http://127.0.0.1:8188"`.
    #[serde(default = "default_comfyui_base_url")]
    pub base_url: String,
    /// Upscale-model filename as it appears in ComfyUI's
    /// `models/upscale_models/` dir, e.g. `"RealESRGAN_x4plus.pth"`. Used by
    /// the built-in workflow; run `dataset upscale-models` to list what the
    /// server offers. Ignored when `workflow_template` is set.
    #[serde(default)]
    pub upscale_model: Option<String>,
    /// Path to an API-format workflow JSON exported from ComfyUI
    /// (*Save (API Format)*). When set, this graph is used instead of the
    /// built-in one; the single `LoadImage` node is fed each dataset image and
    /// the `SaveImage` node's output is retrieved. Enables diffusion-based
    /// upscalers (Ultimate SD Upscale, tiled img2img, …).
    #[serde(default)]
    pub workflow_template: Option<PathBuf>,
    /// After upscaling, cap the longest edge to this many pixels (Lanczos3
    /// downscale when exceeded). Model upscalers emit a fixed multiplier
    /// (usually ×4); this keeps a dataset's resolutions bounded. `0` = keep the
    /// model's native output size.
    #[serde(default = "default_upscaler_max_edge")]
    pub max_edge: u32,
    /// Per-request / whole-job timeout in seconds. Diffusion-based workflows on
    /// a busy server can take a while per image.
    #[serde(default = "default_upscaler_timeout_secs")]
    pub timeout_secs: u64,
    /// Pause between `/history` status polls, in milliseconds.
    #[serde(default = "default_upscaler_poll_ms")]
    pub poll_interval_ms: u64,
}

fn default_comfyui_base_url() -> String {
    DEFAULT_COMFYUI_BASE_URL.to_string()
}

fn default_upscaler_max_edge() -> u32 {
    2048
}

fn default_upscaler_timeout_secs() -> u64 {
    600
}

fn default_upscaler_poll_ms() -> u64 {
    750
}

impl Default for UpscalerProfile {
    fn default() -> Self {
        Self {
            base_url: default_comfyui_base_url(),
            upscale_model: None,
            workflow_template: None,
            max_edge: default_upscaler_max_edge(),
            timeout_secs: default_upscaler_timeout_secs(),
            poll_interval_ms: default_upscaler_poll_ms(),
        }
    }
}

/// Default `model` widget value for the built-in edit workflow — Google's Nano
/// Banana 2. The string is the exact combo option ComfyUI's Gemini image node
/// exposes; `dataset edit-models` prints what a given server offers (the other
/// current option is `gemini-3-pro-image-preview`, i.e. Nano Banana Pro).
pub const DEFAULT_EDIT_MODEL: &str = "Nano Banana 2 (Gemini 3.1 Flash Image)";

/// ComfyUI-backed image-editor profile: one text instruction applied to every
/// image in a dataset through the server's Gemini image (Nano Banana) API node.
///
/// The motivating pass is "keep the drawn character, re-render the background
/// photographically" over an illustrated set, so the resulting LoRA learns the
/// mixed drawn-subject/real-scene look. Because the instruction is the same for
/// the whole dataset it belongs here rather than on the command line.
///
/// The node bills per image, so `dataset edit` defaults to skipping images that
/// already have an output and offers `--dry-run` / `--limit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorProfile {
    /// ComfyUI server root, e.g. `"http://127.0.0.1:8188"`.
    #[serde(default = "default_comfyui_base_url")]
    pub base_url: String,
    /// comfy.org account API key, generated at <https://platform.comfy.org>.
    /// The Gemini node is a paid API node and rejects a run without one
    /// ("Please login first to use this node") — being signed into the web UI
    /// does not authenticate requests made over the HTTP API.
    ///
    /// Prefer leaving this unset and exporting `COMFY_API_KEY` instead: a
    /// dataset-local `fwaun-tools.toml` usually travels with the dataset, and a
    /// key committed alongside it is a key to somebody else's billing.
    #[serde(default)]
    pub api_key: Option<String>,
    /// The edit instruction sent with every image. Required unless a
    /// `workflow_template` bakes its own prompt in; `--prompt` overrides it.
    #[serde(default)]
    pub prompt: Option<String>,
    /// `model` widget value, e.g. `"Nano Banana 2 (Gemini 3.1 Flash Image)"`.
    /// Run `dataset edit-models` to list what the server offers. Ignored when
    /// `workflow_template` is set.
    #[serde(default = "default_edit_model")]
    pub model: String,
    /// `"auto"` matches each input image's aspect ratio — the right choice for
    /// a dataset pass, since a fixed ratio re-crops the subject.
    #[serde(default = "default_edit_aspect_ratio")]
    pub aspect_ratio: String,
    /// Output resolution tier: `"1K"`, `"2K"`, or `"4K"`. Higher tiers cost
    /// more per image and are billed by the API node.
    #[serde(default = "default_edit_resolution")]
    pub resolution: String,
    /// Passed through to the node. The API only makes a best effort at
    /// reproducing a seed, so this bounds run-to-run drift rather than
    /// eliminating it.
    #[serde(default = "default_edit_seed")]
    pub seed: u64,
    /// Replace the node's built-in system prompt. Leave unset to keep it.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Path to an API-format workflow JSON exported from ComfyUI
    /// (*Save (API Format)*). When set, this graph is used instead of the
    /// built-in one: the single `LoadImage` node is fed each dataset image, the
    /// prompt is injected (see `prompt_node`), and the `SaveImage` node's output
    /// is retrieved. Use it for graphs the built-in one can't express — a
    /// reference background wired in through `Batch Images`, a mask, a
    /// post-process chain.
    #[serde(default)]
    pub workflow_template: Option<PathBuf>,
    /// Node id in that template to write `prompt` into. Leave unset to
    /// auto-detect the single node with a string `prompt` widget; set it when
    /// the template has several.
    #[serde(default)]
    pub prompt_node: Option<String>,
    /// Cap the result's longest edge to this many pixels (Lanczos3 downscale
    /// when exceeded). `0` = keep the model's native output, which is what an
    /// edit pass usually wants — the resolution tier already decides the size.
    #[serde(default = "default_editor_max_edge")]
    pub max_edge: u32,
    /// Per-request / whole-job timeout in seconds. A hosted API round trip is
    /// slower than a local upscale, and `thinking` models slower still.
    #[serde(default = "default_upscaler_timeout_secs")]
    pub timeout_secs: u64,
    /// Pause between `/history` status polls, in milliseconds.
    #[serde(default = "default_upscaler_poll_ms")]
    pub poll_interval_ms: u64,
}

fn default_edit_model() -> String {
    DEFAULT_EDIT_MODEL.to_string()
}

fn default_edit_aspect_ratio() -> String {
    "auto".to_string()
}

fn default_edit_resolution() -> String {
    "1K".to_string()
}

fn default_edit_seed() -> u64 {
    42
}

fn default_editor_max_edge() -> u32 {
    0
}

impl Default for EditorProfile {
    fn default() -> Self {
        Self {
            base_url: default_comfyui_base_url(),
            api_key: None,
            prompt: None,
            model: default_edit_model(),
            aspect_ratio: default_edit_aspect_ratio(),
            resolution: default_edit_resolution(),
            seed: default_edit_seed(),
            system_prompt: None,
            workflow_template: None,
            prompt_node: None,
            max_edge: default_editor_max_edge(),
            timeout_secs: default_upscaler_timeout_secs(),
            poll_interval_ms: default_upscaler_poll_ms(),
        }
    }
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
    /// Map of tag name -> caption prefix string. On caption export, for each
    /// entry whose tag is present among the image's positive manual tags
    /// (matched case-insensitively, ignoring a leading organizational `_`),
    /// the prefix string is prepended verbatim to the caption. Matched
    /// prefixes are emitted in key order; include any separator (e.g. a
    /// trailing `", "`) in the value yourself.
    ///
    /// Folds curation tags (e.g. `realistic`, `super_deformed`) into a
    /// deterministic caption prefix for caption-only training
    /// (musubi-tuner), instead of relying on the captioner to weave a
    /// trigger word into its prose. Defaults to empty (no prefixing).
    #[serde(default)]
    pub caption_prefixes: BTreeMap<String, String>,
    /// Like `caption_prefixes`, but appended to the end of the caption
    /// instead of the front. Same matching rules. Use this for trainers /
    /// inference templates that expect the trigger word as a trailing tag
    /// — e.g. ComfyUI's Krea-2 template concatenates the LoRA trigger word
    /// onto the end of the prompt (`{prompt}, {trigger}`), so a caption
    /// trained the same way (`{description}, {trigger}`) lines up. Include
    /// your own leading separator (e.g. a `", "`) in the value. Defaults to
    /// empty (no suffixing).
    #[serde(default)]
    pub caption_suffixes: BTreeMap<String, String>,
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
            caption_prefixes: BTreeMap::new(),
            caption_suffixes: BTreeMap::new(),
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
/// A group can also carry caption steering — `caption_hint` /
/// `caption_prefix` / `caption_suffix` — applied when *all* of the group's
/// `tags` are present on an image (logical AND). This lets a tag
/// combination (e.g. count/gender × concept, like `["1girl",
/// "breaking_through_fourth_wall"]`) inject the right phrasing without
/// per-image editing. Groups used this way are typically `exclusive =
/// false`, since their member tags are meant to co-occur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagGroup {
    /// Tags that participate in this group.
    pub tags: Vec<String>,
    /// Whether the group's tags are mutually exclusive on each image.
    /// `true` (default): at most one tag is expected; two or more
    /// co-occurring is a "violation" flagged by `validate-tag-group` and
    /// the Kanban view. `false`: the tags are meant to co-occur (e.g. a
    /// count/gender tag plus a concept trigger), so co-occurrence is not
    /// flagged.
    #[serde(default = "default_exclusive")]
    pub exclusive: bool,
    /// Reference fact fed to the captioner as generation `context` when all
    /// of the group's `tags` are present. Auto-collected into the
    /// caption-hint block alongside the sidecar's per-image
    /// `caption_hints`, so the model can weave it into a natural,
    /// position-aware description. Never exported to the training `.txt`.
    #[serde(default)]
    pub caption_hint: Option<String>,
    /// Caption prefix folded in when all of the group's `tags` are present:
    /// prepended at export, and used as the assistant-turn prefill seed at
    /// generation so the generated body continues from it naturally. When
    /// several groups match, their prefixes concatenate in ascending
    /// `priority`; groups that tie are ordered per-image by
    /// [`AffixSeed`](crate::tag_group::AffixSeed) rather than by name, so
    /// the order varies across the dataset without varying between runs.
    /// Include your own trailing separator (e.g. `". "`) in `content`.
    #[serde(default)]
    pub caption_prefix: Option<CaptionAffix>,
    /// Like `caption_prefix`, but appended after the caption body instead of
    /// prepended. Same match + ordering rules; include your own leading
    /// separator.
    #[serde(default)]
    pub caption_suffix: Option<CaptionAffix>,
}

fn default_exclusive() -> bool {
    true
}

impl Default for TagGroup {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            exclusive: default_exclusive(),
            caption_hint: None,
            caption_prefix: None,
            caption_suffix: None,
        }
    }
}

/// A caption affix (prefix or suffix) contributed by a matching [`TagGroup`].
/// `priority` orders concatenation when several groups match on one image:
/// ascending, so a lower number sits closer to the front (for prefixes) or
/// nearer the body (for suffixes). Equal priorities are shuffled per image —
/// see [`AffixSeed`](crate::tag_group::AffixSeed).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaptionAffix {
    /// The literal affix text. Include any separator yourself (a trailing
    /// `", "` for a prefix, a leading `", "` for a suffix).
    pub content: String,
    /// Ascending sort key for concatenation order. Defaults to 0.
    #[serde(default)]
    pub priority: i64,
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
            caption_prefixes: BTreeMap::new(),
            caption_suffixes: BTreeMap::new(),
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
            empty_retries: default_empty_retries(),
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
            default_upscaler: None,
            default_editor: None,
            common_tags: Vec::new(),
            export,
            tagger: BTreeMap::new(),
            captioner: BTreeMap::new(),
            upscaler: BTreeMap::new(),
            editor: BTreeMap::new(),
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
        // Boxed: `toml::de::Error` is large enough that an unboxed variant made
        // every `Result<_, ConfigError>` trip clippy's `result_large_err`.
        #[source]
        source: Box<toml::de::Error>,
    },
    #[error("unknown prompt name `{0}` — define it in [captioner_prompts] or pick an existing one")]
    UnknownPrompt(String),
    #[error("tag_group `{0}` has no tags — every group must list at least one tag")]
    EmptyTagGroup(String),
}

impl ProjectConfig {
    /// Base config directory: `$XDG_CONFIG_HOME`, falling back to
    /// `$HOME/.config`. `None` if neither env var is set (no usable home).
    fn user_config_base() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
            return Some(PathBuf::from(xdg));
        }
        std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(|home| PathBuf::from(home).join(".config"))
    }

    /// Resolve `$XDG_CONFIG_HOME/fwaun-tools/config.toml`, falling back to
    /// `$HOME/.config/fwaun-tools/config.toml`. Returns `None` if neither
    /// env var is set (no usable home). This always points at the current
    /// location — it's the path the GUI writes and displays.
    pub fn user_config_path() -> Option<PathBuf> {
        Self::user_config_base().map(|base| base.join(USER_CONFIG_RELATIVE))
    }

    /// Path to load the user config from: the current location if it exists,
    /// otherwise the newest legacy path that exists (with a deprecation
    /// warning), otherwise the current location (which simply won't exist).
    fn user_config_load_path() -> Option<PathBuf> {
        let base = Self::user_config_base()?;
        let primary = base.join(USER_CONFIG_RELATIVE);
        if primary.exists() {
            return Some(primary);
        }
        for rel in LEGACY_USER_CONFIG_RELATIVES {
            let legacy = base.join(rel);
            if legacy.exists() {
                warn_legacy_config(&legacy);
                return Some(legacy);
            }
        }
        Some(primary)
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
            source: Box::new(source),
        })?;
        Ok(Some(cfg))
    }

    /// Walk up from `start` looking for a project config. Returns the first
    /// matching file (analogous to how `git` finds `.git`). This lets users
    /// keep a single config at the dataset root while operating on
    /// subdirectories. At each level the current [`CONFIG_FILE`] name wins;
    /// a legacy name from [`LEGACY_CONFIG_FILES`] is accepted as a fallback
    /// and triggers a deprecation warning.
    pub fn find_project_config(start: &Path) -> Option<PathBuf> {
        let abs = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
        for dir in abs.ancestors() {
            let primary = dir.join(CONFIG_FILE);
            if primary.is_file() {
                return Some(primary);
            }
            for name in LEGACY_CONFIG_FILES {
                let legacy = dir.join(name);
                if legacy.is_file() {
                    warn_legacy_config(&legacy);
                    return Some(legacy);
                }
            }
        }
        None
    }

    /// Directory of the config found by
    /// [`find_project_config`](Self::find_project_config) — the dataset
    /// root, as far as anything that needs a stable per-image identity is
    /// concerned. Canonicalized (inherited from `find_project_config`), so
    /// it strips cleanly off a canonicalized image path; that's what
    /// [`AffixSeed::for_image`] wants, and why this isn't the right thing
    /// to show a user.
    ///
    /// [`AffixSeed::for_image`]: crate::tag_group::AffixSeed::for_image
    pub fn project_root(dir: &Path) -> Option<PathBuf> {
        Self::find_project_config(dir)?
            .parent()
            .map(Path::to_path_buf)
    }

    /// The config file `dir` owns *directly*, if any — i.e. `dir` is the
    /// dataset root rather than a subdirectory covered by an ancestor's
    /// config. Accepts the legacy filenames on the same terms as
    /// [`find_project_config`](Self::find_project_config).
    ///
    /// This is the test for "does a dataset-wide edit belong in the config
    /// file?": a bulk tag operation spanning the whole dataset writes to
    /// `common_tags` here instead of to every sidecar. A subdirectory
    /// deliberately fails the test — an edit there covers only part of the
    /// dataset, so it has to live in the individual sidecars.
    /// Deliberately does not canonicalize: the returned path is echoed back
    /// to the user, and on Windows `canonicalize` yields a `\\?\` verbatim
    /// path. A plain join is enough to test `dir` itself.
    pub fn dataset_root_config(dir: &Path) -> Option<PathBuf> {
        let primary = dir.join(CONFIG_FILE);
        if primary.is_file() {
            return Some(primary);
        }
        LEGACY_CONFIG_FILES
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.is_file())
    }

    /// True when `dir` is a dataset root (see
    /// [`dataset_root_config`](Self::dataset_root_config)).
    pub fn is_dataset_root(dir: &Path) -> bool {
        Self::dataset_root_config(dir).is_some()
    }

    /// Load the project `fwaun-tools.toml`, searching `dir` and its
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
        match Self::user_config_load_path() {
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
        if other.default_upscaler.is_some() {
            self.default_upscaler = other.default_upscaler;
        }
        if other.default_editor.is_some() {
            self.default_editor = other.default_editor;
        }
        // Replace rather than union: see the field docs on `common_tags`.
        if !other.common_tags.is_empty() {
            self.common_tags = other.common_tags;
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
        for (k, v) in other.upscaler {
            self.upscaler.insert(k, v);
        }
        for (k, v) in other.editor {
            self.editor.insert(k, v);
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

    /// Resolve the dataset-wide manual tag layer declared by `common_tags`.
    /// Cheap enough to call once per command / folder load; pass the result
    /// down to `export::*` and `tag_group::*`.
    pub fn resolve_common_tags(&self) -> crate::common_tags::CommonTags {
        crate::common_tags::CommonTags::new(&self.common_tags)
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

    /// Resolve an upscaler profile. Order: explicit `name`, then
    /// `default_upscaler`, then a built-in default pointing at the stock local
    /// ComfyUI (`http://127.0.0.1:8188`) with no model preselected. The caller
    /// still needs to supply an `upscale_model` or `workflow_template` (via the
    /// profile or a CLI flag) before a run can start.
    pub fn resolve_upscaler(&self, name: Option<&str>) -> (String, UpscalerProfile) {
        let key = name
            .map(str::to_string)
            .or_else(|| self.default_upscaler.clone());
        if let Some(k) = key
            && let Some(profile) = self.upscaler.get(&k)
        {
            return (k, profile.clone());
        }
        ("comfyui".to_string(), UpscalerProfile::default())
    }

    /// Resolve an image-editor profile. Order: explicit `name`, then
    /// `default_editor`, then a built-in default pointing at the stock local
    /// ComfyUI with Nano Banana 2 selected and no prompt. The caller still needs
    /// to supply a `prompt` (via the profile or `--prompt`) before a run can
    /// start — an edit pass with no instruction has nothing to do.
    pub fn resolve_editor(&self, name: Option<&str>) -> (String, EditorProfile) {
        let key = name
            .map(str::to_string)
            .or_else(|| self.default_editor.clone());
        if let Some(k) = key
            && let Some(profile) = self.editor.get(&k)
        {
            return (k, profile.clone());
        }
        ("comfyui".to_string(), EditorProfile::default())
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
                "fwaun-tools-test-{tag}-{}-{}",
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

        let found =
            ProjectConfig::find_project_config(&nested).expect("should walk up to root config");
        assert_eq!(
            found.canonicalize().unwrap(),
            cfg_path.canonicalize().unwrap()
        );

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
        fs::write(
            root.path().join(CONFIG_FILE),
            "default_profile = \"root\"\n",
        )
        .unwrap();
        fs::write(mid.join(CONFIG_FILE), "default_profile = \"mid\"\n").unwrap();

        let cfg = ProjectConfig::load(&leaf).unwrap().unwrap();
        assert_eq!(cfg.default_profile.as_deref(), Some("mid"));
    }

    #[test]
    fn find_project_config_falls_back_to_legacy_name() {
        let root = TempDir::new("legacy");
        let nested = root.path().join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        let legacy_path = root.path().join(LEGACY_CONFIG_FILES[0]);
        fs::write(&legacy_path, "default_profile = \"legacy\"\n").unwrap();

        let found = ProjectConfig::find_project_config(&nested)
            .expect("should fall back to the legacy config name");
        assert_eq!(
            found.canonicalize().unwrap(),
            legacy_path.canonicalize().unwrap()
        );

        let cfg = ProjectConfig::load(&nested).unwrap().unwrap();
        assert_eq!(cfg.default_profile.as_deref(), Some("legacy"));
    }

    #[test]
    fn current_config_name_wins_over_legacy_in_same_dir() {
        let root = TempDir::new("both-names");
        fs::write(root.path().join(CONFIG_FILE), "default_profile = \"new\"\n").unwrap();
        fs::write(
            root.path().join(LEGACY_CONFIG_FILES[0]),
            "default_profile = \"old\"\n",
        )
        .unwrap();

        let cfg = ProjectConfig::load(root.path()).unwrap().unwrap();
        assert_eq!(cfg.default_profile.as_deref(), Some("new"));
    }

    #[test]
    fn merge_project_overrides_user() {
        let mut user = ProjectConfig {
            default_captioner: Some("user-cap".into()),
            ..Default::default()
        };
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
                max_retries: 3,
                empty_retries: 2,
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
                max_retries: 3,
                empty_retries: 2,
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
            empty_retries: default_empty_retries(),
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
            empty_retries: default_empty_retries(),
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
        assert_eq!(
            library.get("default").map(String::as_str),
            Some("Describe briefly.")
        );
    }

    #[test]
    fn project_only_keys_survive_merge() {
        let user = ProjectConfig::default();
        let mut project = ProjectConfig::default();
        project
            .tagger
            .insert("wd-tagger".into(), TaggerProfile::built_in());

        let mut merged = ProjectConfig::default();
        merged.merge_from(user);
        merged.merge_from(project);

        assert!(merged.tagger.contains_key("wd-tagger"));
    }

    /// Guard: when a field is added to any of the profile structs, the
    /// shipped example (`crates/core/fwaun-tools.toml.example`) has to grow alongside
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
            .expect("fwaun-tools.toml.example must parse as ProjectConfig");
        let raw: toml::Value = toml::from_str(example_str)
            .expect("fwaun-tools.toml.example must parse as toml::Value");
        let raw_table = raw.as_table().expect("example must be a top-level table");

        for k in [
            "default_profile",
            "default_tagger",
            "default_captioner",
            "common_tags",
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
        if let Some(u) = &cfg.default_upscaler {
            assert!(
                cfg.upscaler.contains_key(u),
                "default_upscaler = {u:?} but no matching [upscaler.{u}] in the example"
            );
        }
        if let Some(e) = &cfg.default_editor {
            assert!(
                cfg.editor.contains_key(e),
                "default_editor = {e:?} but no matching [editor.{e}] in the example"
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
                let missing: BTreeSet<String> = expected.difference(&actual).cloned().collect();
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
            caption_prefixes: BTreeMap::from([(
                "realistic".into(),
                "realistic proportions, ".into(),
            )]),
            caption_suffixes: BTreeMap::from([(
                "realistic".into(),
                ", realistic proportions".into(),
            )]),
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
            empty_retries: default_empty_retries(),
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
            max_retries: default_openai_max_retries(),
            empty_retries: default_empty_retries(),
        });
        let full_tag_group = TagGroup {
            tags: vec!["x".into()],
            exclusive: false,
            caption_hint: Some("h".into()),
            caption_prefix: Some(CaptionAffix {
                content: "p".into(),
                priority: 1,
            }),
            caption_suffix: Some(CaptionAffix {
                content: "s".into(),
                priority: 1,
            }),
        };
        let full_upscaler = UpscalerProfile {
            base_url: "http://x".into(),
            upscale_model: Some("m.pth".into()),
            workflow_template: Some(PathBuf::from("wf.json")),
            max_edge: default_upscaler_max_edge(),
            timeout_secs: default_upscaler_timeout_secs(),
            poll_interval_ms: default_upscaler_poll_ms(),
        };

        let full_editor = EditorProfile {
            base_url: "http://x".into(),
            api_key: Some("k".into()),
            prompt: Some("p".into()),
            model: default_edit_model(),
            aspect_ratio: default_edit_aspect_ratio(),
            resolution: default_edit_resolution(),
            seed: default_edit_seed(),
            system_prompt: Some("s".into()),
            workflow_template: Some(PathBuf::from("wf.json")),
            prompt_node: Some("4".into()),
            max_edge: default_editor_max_edge(),
            timeout_secs: default_upscaler_timeout_secs(),
            poll_interval_ms: default_upscaler_poll_ms(),
        };

        let expected_export = struct_keys(full_export);
        let expected_tagger = struct_keys(full_tagger);
        let expected_onnx = struct_keys(full_onnx);
        let expected_openai = struct_keys(full_openai);
        let expected_tag_group = struct_keys(full_tag_group);
        let expected_upscaler = struct_keys(full_upscaler);
        let expected_editor = struct_keys(full_editor);

        if let Err(missing) =
            missing_from_best_match(raw_table.get("export"), &expected_export, |_| true)
        {
            panic!(
                "no [export.*] profile in crates/core/fwaun-tools.toml.example covers every \
                 ExportProfile field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("tagger"), &expected_tagger, |_| true)
        {
            panic!(
                "no [tagger.*] profile in crates/core/fwaun-tools.toml.example covers every \
                 TaggerProfile field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("captioner"), &expected_onnx, |t| {
                t.get("kind").and_then(|v| v.as_str()) == Some("onnx")
            })
        {
            panic!(
                "no [captioner.*] profile with `kind = \"onnx\"` in \
                 crates/core/fwaun-tools.toml.example covers every OnnxCaptionerProfile field; \
                 closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("captioner"), &expected_openai, |t| {
                t.get("kind").and_then(|v| v.as_str()) == Some("openai")
            })
        {
            panic!(
                "no [captioner.*] profile with `kind = \"openai\"` in \
                 crates/core/fwaun-tools.toml.example covers every OpenAiCaptionerProfile field; \
                 closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("tag_group"), &expected_tag_group, |_| true)
        {
            panic!(
                "no [tag_group.*] entry in crates/core/fwaun-tools.toml.example covers every \
                 TagGroup field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("upscaler"), &expected_upscaler, |_| true)
        {
            panic!(
                "no [upscaler.*] profile in crates/core/fwaun-tools.toml.example covers every \
                 UpscalerProfile field; closest match is missing {missing:?}"
            );
        }
        if let Err(missing) =
            missing_from_best_match(raw_table.get("editor"), &expected_editor, |_| true)
        {
            panic!(
                "no [editor.*] profile in crates/core/fwaun-tools.toml.example covers every \
                 EditorProfile field; closest match is missing {missing:?}"
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
                ..Default::default()
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
    fn common_tags_project_list_replaces_user_list() {
        let user = ProjectConfig {
            common_tags: vec!["user_trigger".into(), "-blue_hair".into()],
            ..Default::default()
        };
        let project = ProjectConfig {
            common_tags: vec!["himeko".into()],
            ..Default::default()
        };
        let mut merged = ProjectConfig::default();
        merged.merge_from(user);
        merged.merge_from(project);
        assert_eq!(merged.common_tags, vec!["himeko".to_string()]);
    }

    #[test]
    fn common_tags_empty_project_list_keeps_user_list() {
        let user = ProjectConfig {
            common_tags: vec!["user_trigger".into()],
            ..Default::default()
        };
        let mut merged = ProjectConfig::default();
        merged.merge_from(user);
        merged.merge_from(ProjectConfig::default());
        assert_eq!(merged.common_tags, vec!["user_trigger".to_string()]);
    }

    #[test]
    fn validate_tag_groups_rejects_empty_tags() {
        let mut cfg = ProjectConfig::default();
        cfg.tag_groups.insert(
            "foo".into(),
            TagGroup {
                tags: Vec::new(),
                ..Default::default()
            },
        );
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
                ..Default::default()
            },
        );
        cfg.validate_tag_groups()
            .expect("single-tag group is valid");
    }
}
