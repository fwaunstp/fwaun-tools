//! Captioner facade. Two backends, picked by `kind` in the captioner
//! profile: a local Qwen3-VL ONNX runtime ([`onnx`]) and an
//! OpenAI-compatible HTTP client ([`openai`]) that talks to llama.cpp,
//! koboldcpp, Ollama, LM Studio, vLLM, and friends.

#[cfg(feature = "onnx")]
mod onnx;
mod openai;

use std::path::Path;

use fwaun_tools_core::config::CaptionerProfile;
use fwaun_tools_core::hub;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CaptionerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ort: {0}")]
    Ort(String),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("hub: {0}")]
    Hub(#[from] hub::HubError),
    #[error("model output shape unexpected: {0}")]
    Shape(String),
    #[error("tokenized chat template did not contain exactly one <|image_pad|> token; got {0}")]
    ImagePadCount(usize),
    #[error("http: {0}")]
    Http(String),
    #[error("api response: {0}")]
    Api(String),
    #[error(
        "this is a light build without the local ONNX captioner; configure an OpenAI-compatible captioner, or install the full build for local Qwen3-VL"
    )]
    Unsupported,
}

#[cfg(feature = "onnx")]
impl<F> From<ort::Error<F>> for CaptionerError {
    fn from(e: ort::Error<F>) -> Self {
        CaptionerError::Ort(e.to_string())
    }
}

pub enum Captioner {
    #[cfg(feature = "onnx")]
    Onnx(onnx::OnnxCaptioner),
    OpenAi(openai::OpenAiCaptioner),
}

impl Captioner {
    pub fn from_profile(profile: &CaptionerProfile) -> Result<Self, CaptionerError> {
        match profile {
            #[cfg(feature = "onnx")]
            CaptionerProfile::Onnx(p) => {
                Ok(Self::Onnx(onnx::OnnxCaptioner::from_profile(p)?))
            }
            #[cfg(not(feature = "onnx"))]
            CaptionerProfile::Onnx(_) => Err(CaptionerError::Unsupported),
            CaptionerProfile::Openai(p) => {
                Ok(Self::OpenAi(openai::OpenAiCaptioner::from_profile(p)?))
            }
        }
    }

    /// Generate a caption for `image_path` using `prompt`. Callers iterate
    /// over `CaptionerProfile::resolved_prompts()` to drive multiple
    /// prompts against the same loaded model (sidecar keys are
    /// `{model}.{prompt_name}`).
    ///
    /// `context` is optional reference info (e.g. character names + screen
    /// positions) embedded inside the user turn alongside the image so the
    /// model treats it as image-specific facts rather than global persona
    /// guidance. `None` / empty passes the prompt through unchanged.
    ///
    /// `prefix` is the optional caption prefix the export step will prepend.
    /// When set, it is embedded *in the user prompt* (not as an assistant-turn
    /// prefill) with an instruction to continue from it, so the generated body
    /// flows naturally after the prefix. It is deliberately NOT sent as a
    /// trailing assistant message: doing so makes servers treat the turn as a
    /// raw continuation and a reasoning model's chain-of-thought spills into
    /// the caption instead of staying in its reasoning channel. The model is
    /// asked to emit only the continuation; as a safeguard an echoed copy of
    /// the prefix is stripped from the front of the result, so callers store
    /// the continuation and let export re-prepend the prefix (no doubling).
    pub fn caption_image(
        &mut self,
        image_path: &Path,
        prompt: &str,
        context: Option<&str>,
        prefix: Option<&str>,
    ) -> Result<String, CaptionerError> {
        let context = context.map(str::trim).filter(|s| !s.is_empty());
        let prefix = prefix.map(str::trim).filter(|s| !s.is_empty());
        let raw = match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(c) => c.caption_image(image_path, prompt, context, prefix)?,
            Self::OpenAi(c) => c.caption_image(image_path, prompt, context, prefix)?,
        };
        Ok(strip_echoed_prefix(&raw, prefix).trim().to_string())
    }
}

/// Normalize a caption to the *continuation*: a model asked to continue from
/// a prefix sometimes restates the prefix first despite the instruction not
/// to. Dropping a leading echo keeps the stored body free of the prefix so
/// export re-prepends it exactly once.
fn strip_echoed_prefix<'a>(text: &'a str, prefix: Option<&str>) -> &'a str {
    match prefix {
        Some(pfx) => {
            let pfx = pfx.trim();
            if pfx.is_empty() {
                text
            } else {
                text.trim_start().strip_prefix(pfx).unwrap_or(text)
            }
        }
        None => text,
    }
}

/// Build the user-turn text for a caption request: the optional reference
/// context (character names, positions, …), the actual prompt, and — when a
/// caption `prefix` applies — an instruction to continue from it.
///
/// Bare "Context: …" gets ignored too easily — the model treats it as
/// background and falls back to generic descriptions ("the girl on the
/// left" instead of the provided name). The phrasing here explicitly
/// instructs the model to *use* the names/details, while limiting it to
/// description (so the prompt's actual task still drives the output).
///
/// The `prefix` is embedded here in the user turn rather than sent as a
/// trailing assistant message: a partial assistant turn is interpreted as a
/// raw continuation by local servers, which makes a reasoning model's
/// chain-of-thought leak into the caption. Embedding it in the prompt keeps
/// the normal user→assistant structure (so thinking stays in its own
/// channel) while still steering the opening.
pub(crate) fn build_user_text(
    prompt: &str,
    context: Option<&str>,
    prefix: Option<&str>,
) -> String {
    let mut out = String::new();
    if let Some(ctx) = context {
        out.push_str("Use the following names and details when describing the image:\n");
        out.push_str(ctx);
        out.push_str("\n\n");
    }
    out.push_str(prompt);
    if let Some(pfx) = prefix {
        out.push_str(
            "\n\nThe caption has already been started with the exact text below. Write only \
             what comes next, as one continuous description that flows on naturally from it. \
             Do not repeat or rephrase this opening, and do not add any preamble before your \
             continuation:\n\n",
        );
        out.push_str(pfx);
    }
    out
}
