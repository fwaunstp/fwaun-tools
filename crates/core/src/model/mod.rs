//! Diffusion-checkpoint utilities over safetensors files: task-vector merge
//! ([`merge`]), low-rank LoRA extraction ([`lora`]), and INT8 + ConvRot
//! quantization ([`quant`]). Pure-Rust / CPU, streamed key-by-key so peak RAM
//! stays small — no ONNX Runtime, no glibc floor.
//!
//! Ported from the standalone `fwaun-model-tools` crate; exposed here so both
//! the CLI (`model` subcommands) and the GUI's "Model tools" tab share one
//! implementation. Gated behind the `model` cargo feature.
//!
//! Every `run` reports through a [`ProgressSink`] rather than printing
//! directly, so the CLI keeps its stderr/stdout lines while the GUI forwards
//! structured log + progress updates to its worker channel.

pub mod lora;
pub mod merge;
pub mod progress;
pub mod quant;
pub mod safetensors;

pub use progress::{ProgressSink, StreamProgress};
