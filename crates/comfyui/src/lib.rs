//! Client for an **existing** ComfyUI server's HTTP API, plus batch helpers
//! built on top of it.
//!
//! [`Client`] wraps the handful of endpoints a headless workflow run needs —
//! `POST /upload/image`, `POST /prompt`, `GET /history/{id}`, `GET /view`, and
//! `GET /object_info` — so callers can push an image through any workflow
//! graph and pull the result back without touching the web UI.
//!
//! The [`upscale`] module is the first consumer: a batch image upscaler that
//! runs each dataset image through a model-based (ESRGAN-style) workflow, or a
//! user-supplied API-format workflow template. Future ComfyUI-backed batch
//! passes (img2img, background removal, …) can reuse the same [`Client`].

mod client;
pub mod upscale;

pub use client::{Client, ImageRef, UploadRef};

use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ComfyError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Transport failure or a non-2xx HTTP response. The message includes the
    /// response body when the server returned one (ComfyUI puts workflow
    /// validation detail there).
    #[error("comfyui http: {0}")]
    Http(String),
    /// `POST /prompt` accepted the request shape but rejected the graph — one
    /// or more nodes failed validation (bad model name, missing input, …).
    #[error("comfyui rejected the workflow: {0}")]
    NodeErrors(String),
    /// The ComfyUI run finished (or errored) without producing an output image.
    #[error("comfyui produced no output image for prompt {0}")]
    NoOutput(String),
    /// The run did not finish within the configured timeout.
    #[error("timed out after {0:?} waiting for ComfyUI to finish")]
    Timeout(Duration),
    /// A supplied workflow template could not be used (wrong format, missing
    /// `LoadImage`/`SaveImage`, …).
    #[error("workflow template: {0}")]
    Workflow(String),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ComfyError>;
