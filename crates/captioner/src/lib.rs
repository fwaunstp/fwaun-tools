//! Captioner facade. Two backends, picked by `kind` in the captioner
//! profile: a local Qwen3-VL ONNX runtime ([`onnx`]) and an
//! OpenAI-compatible HTTP client ([`openai`]) that talks to llama.cpp,
//! koboldcpp, Ollama, LM Studio, vLLM, and friends.

mod onnx;
mod openai;

use std::path::Path;

use fwaun_tagger_core::config::CaptionerProfile;
use fwaun_tagger_core::hub;
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
}

impl<F> From<ort::Error<F>> for CaptionerError {
    fn from(e: ort::Error<F>) -> Self {
        CaptionerError::Ort(e.to_string())
    }
}

pub enum Captioner {
    Onnx(onnx::OnnxCaptioner),
    OpenAi(openai::OpenAiCaptioner),
}

impl Captioner {
    pub fn from_profile(profile: &CaptionerProfile) -> Result<Self, CaptionerError> {
        match profile {
            CaptionerProfile::Onnx(p) => {
                Ok(Self::Onnx(onnx::OnnxCaptioner::from_profile(p)?))
            }
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
    pub fn caption_image(
        &mut self,
        image_path: &Path,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<String, CaptionerError> {
        let context = context.map(str::trim).filter(|s| !s.is_empty());
        match self {
            Self::Onnx(c) => c.caption_image(image_path, prompt, context),
            Self::OpenAi(c) => c.caption_image(image_path, prompt, context),
        }
    }
}

/// Build the user-turn text for a caption request: the optional reference
/// context (character names, positions, …) followed by the actual prompt.
///
/// Bare "Context: …" gets ignored too easily — the model treats it as
/// background and falls back to generic descriptions ("the girl on the
/// left" instead of the provided name). The phrasing here explicitly
/// instructs the model to *use* the names/details, while limiting it to
/// description (so the prompt's actual task still drives the output).
pub(crate) fn build_user_text(prompt: &str, context: Option<&str>) -> String {
    match context {
        Some(ctx) => format!(
            "Use the following names and details when describing the image:\n\
             {ctx}\n\n\
             {prompt}"
        ),
        None => prompt.to_string(),
    }
}
