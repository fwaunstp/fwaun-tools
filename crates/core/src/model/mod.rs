//! Diffusion-checkpoint utilities over safetensors files: task-vector merge
//! ([`merge`]), low-rank LoRA extraction ([`lora`]), and INT8 + ConvRot
//! quantization ([`quant`]). Pure-Rust / CPU, streamed key-by-key so peak RAM
//! stays small — no ONNX Runtime, no glibc floor.
//!
//! Ported from the standalone `fwaun-model-tools` crate; exposed here so both
//! the CLI (`model` subcommands) and, in future, a GUI tab share one
//! implementation. Gated behind the `model` cargo feature.

pub mod lora;
pub mod merge;
pub mod quant;
pub mod safetensors;
