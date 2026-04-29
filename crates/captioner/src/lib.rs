//! Captioner facade. Two backends, picked by `kind` in the captioner
//! profile: a local Qwen3-VL ONNX runtime ([`onnx`]) and an
//! OpenAI-compatible HTTP client ([`openai`]) that talks to llama.cpp,
//! koboldcpp, Ollama, LM Studio, vLLM, and friends.

mod onnx;
mod openai;

use std::path::Path;

use anima_tagger_core::config::CaptionerProfile;
use anima_tagger_core::hub;
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
    /// over `CaptionerProfile::resolved_prompts()` to drive multiple prompts
    /// (e.g. detail + brief) against the same loaded model.
    pub fn caption_image(
        &mut self,
        image_path: &Path,
        prompt: &str,
    ) -> Result<String, CaptionerError> {
        match self {
            Self::Onnx(c) => c.caption_image(image_path, prompt),
            Self::OpenAi(c) => c.caption_image(image_path, prompt),
        }
    }
}
