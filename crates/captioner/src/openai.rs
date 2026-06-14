//! OpenAI-compatible vision chat-completions client. Targets any local
//! server that implements `/v1/chat/completions` with `image_url`
//! content parts: llama.cpp `llama-server`, koboldcpp, Ollama (its
//! OpenAI-compat surface), LM Studio, vLLM, TGI, and similar.
//!
//! The image is JPEG-recompressed and sent as a `data:` URL — every
//! server tested handles base64 data URLs, but a few choke on remote
//! `http://` URLs because they refuse to fetch from the captioner host.

use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

use anima_tagger_core::config::OpenAiCaptionerProfile;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::CaptionerError;

pub struct OpenAiCaptioner {
    profile: OpenAiCaptionerProfile,
    agent: ureq::Agent,
}

impl OpenAiCaptioner {
    pub fn from_profile(profile: &OpenAiCaptionerProfile) -> Result<Self, CaptionerError> {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(profile.timeout_secs))
            .build();
        Ok(Self {
            profile: profile.clone(),
            agent,
        })
    }

    pub fn caption_image(
        &mut self,
        image_path: &Path,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<String, CaptionerError> {
        let data_url = encode_image_data_url(
            image_path,
            self.profile.max_edge,
            self.profile.jpeg_quality,
        )?;

        // Embed reference info in the same user turn as the image rather
        // than as a separate `system` message, so the model sees image +
        // context + instruction together. System messages tend to be
        // interpreted as global persona/style guidance, which is the wrong
        // framing for per-image facts like "left girl is Alice".
        let prompt_text = crate::build_user_text(prompt, context);

        let body = ChatRequest {
            model: self
                .profile
                .model
                .clone()
                .unwrap_or_else(|| "local".to_string()),
            max_tokens: self.profile.max_tokens,
            temperature: self.profile.temperature,
            stream: false,
            messages: vec![ChatMessage {
                role: "user".into(),
                content: vec![
                    ContentPart::ImageUrl {
                        image_url: ImageUrl { url: data_url },
                    },
                    ContentPart::Text {
                        text: prompt_text,
                    },
                ],
            }],
        };

        let url = format!("{}/chat/completions", self.profile.endpoint.trim_end_matches('/'));

        eprintln!(
            "[captioner:openai] POST {url} (model={}, max_tokens={})",
            body.model, body.max_tokens
        );

        // Some servers (gpt-oss harmony parsing in particular) intermittently
        // 500 on a request that succeeds on a fresh attempt. Retry transient
        // failures — HTTP 5xx and transport errors — but never 4xx, which are
        // deterministic client mistakes that won't fix themselves.
        let resp = {
            let mut attempt = 0u32;
            loop {
                // `send_json` consumes the request builder, so rebuild it each
                // attempt. The body is unchanged, so reuse the serialized JSON.
                let mut req = self
                    .agent
                    .post(&url)
                    .set("content-type", "application/json");
                if let Some(key) = self.profile.api_key.as_deref().filter(|s| !s.is_empty()) {
                    req = req.set("authorization", &format!("Bearer {key}"));
                }
                match req.send_json(&body) {
                    Ok(r) => break r,
                    Err(ureq::Error::Status(code, response)) => {
                        // llama-server / koboldcpp / Ollama all return a JSON
                        // error body on non-2xx — surface it so missing-mmproj
                        // and similar server-side misconfigurations are obvious
                        // from this side.
                        let err_body = response.into_string().unwrap_or_default();
                        let retryable = code >= 500;
                        if retryable && attempt < self.profile.max_retries {
                            attempt += 1;
                            let backoff = retry_backoff(attempt);
                            eprintln!(
                                "[captioner:openai] HTTP {code} (transient); retry \
                                 {attempt}/{} in {:.1}s",
                                self.profile.max_retries,
                                backoff.as_secs_f32()
                            );
                            std::thread::sleep(backoff);
                            continue;
                        }
                        return Err(CaptionerError::Api(format!(
                            "HTTP {code} from {url}: {err_body}"
                        )));
                    }
                    Err(ureq::Error::Transport(t)) => {
                        if attempt < self.profile.max_retries {
                            attempt += 1;
                            let backoff = retry_backoff(attempt);
                            eprintln!(
                                "[captioner:openai] transport error ({t}); retry \
                                 {attempt}/{} in {:.1}s",
                                self.profile.max_retries,
                                backoff.as_secs_f32()
                            );
                            std::thread::sleep(backoff);
                            continue;
                        }
                        return Err(CaptionerError::Http(t.to_string()));
                    }
                }
            }
        };
        let parsed: ChatResponse = resp
            .into_json()
            .map_err(|e| CaptionerError::Http(format!("decode body: {e}")))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| CaptionerError::Api("response had no choices".into()))?;
        let text = choice
            .message
            .content
            .ok_or_else(|| CaptionerError::Api("response message had no content".into()))?;

        Ok(text.trim().to_string())
    }
}

/// Exponential backoff between retry attempts: ~1s, 2s, 4s, … capped at 30s.
/// `attempt` is 1-based (the delay before the first retry uses `attempt == 1`).
fn retry_backoff(attempt: u32) -> Duration {
    let secs = 1u64.checked_shl(attempt.saturating_sub(1)).unwrap_or(u64::MAX);
    Duration::from_secs(secs.min(30))
}

fn encode_image_data_url(
    path: &Path,
    max_edge: u32,
    jpeg_quality: u8,
) -> Result<String, CaptionerError> {
    let mut buf = Vec::new();
    if max_edge == 0 {
        buf = std::fs::read(path)?;
        let mime = guess_mime_from_path(path);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
        return Ok(format!("data:{mime};base64,{b64}"));
    }
    let img = image::open(path)?;
    let (w, h) = (img.width(), img.height());
    let resized = if w.max(h) > max_edge {
        img.thumbnail(max_edge, max_edge)
    } else {
        img
    };
    let rgb = resized.to_rgb8();
    let quality = jpeg_quality.clamp(1, 100);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
        Cursor::new(&mut buf),
        quality,
    );
    encoder
        .encode(rgb.as_raw(), rgb.width(), rgb.height(), image::ColorType::Rgb8.into())
        .map_err(image::ImageError::from)?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

fn guess_mime_from_path(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: Vec<ContentPart>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: Option<String>,
}
