//! `fwaun-tools model <verb>` — diffusion-checkpoint subcommands. Thin clap
//! layer over `fwaun_tools_core::model`; all the work happens in core.

use anyhow::Result;
use clap::{Args, Subcommand};

use fwaun_tools_core::model::StreamProgress;
use fwaun_tools_core::model::lora::{self, ExtractArgs};
use fwaun_tools_core::model::merge::{self, MergeArgs, ModelArch};
use fwaun_tools_core::model::quant::{self, QuantArgs};
use fwaun_tools_core::model::safetensors::Dtype;

/// Diffusion-checkpoint subcommands (`fwaun-tools model <verb>`).
#[derive(Subcommand)]
pub enum ModelCommand {
    /// Task-vector merge: output = target + multiplier * (tuned - base).
    ///
    /// Transfers a full fine-tune delta (tuned - base) onto another checkpoint.
    /// Supports Krea 2 and Anima key conventions. All math runs on CPU in f32,
    /// streaming key-by-key so peak RAM stays small.
    MergeDiff(MergeCommand),

    /// Quantize a bf16/fp16 checkpoint to int8 + ConvRot (comfy-kitchen layout).
    ///
    /// Auto-detects the per-token block linears (attention + FFN), rotates each
    /// with a block-Hadamard at the best power-of-4 group size, and stores int8
    /// weights + per-channel scales + a `comfy_quant` config. All math is CPU/f32
    /// and parallelized across cores. Run with `--dry-run` first on a new
    /// architecture to review the plan.
    QuantInt8(QuantCommand),

    /// Extract a low-rank LoRA from a full fine-tune: SVD of (tuned - base).
    ///
    /// For every shared 2D linear weight, factorizes the fine-tune delta into
    /// lora_up/lora_down at the requested rank and writes a kohya-ss/ComfyUI
    /// (`lora_unet_*`) LoRA. Reports the per-module energy captured so you can
    /// tell whether the rank is high enough to track the fine-tune. All math is
    /// CPU/f32 and parallelized across cores.
    ExtractLora(ExtractCommand),
}

#[derive(Args)]
pub struct ExtractCommand {
    /// Original model the fine-tune started from (bf16/fp16/fp32).
    #[arg(long)]
    base: std::path::PathBuf,

    /// Fine-tuned model (the full fine-tune output).
    #[arg(long)]
    tuned: std::path::PathBuf,

    /// Output LoRA safetensors path.
    #[arg(long, short)]
    output: std::path::PathBuf,

    /// LoRA rank (network dim). Higher = closer to the fine-tune, larger file.
    #[arg(long, default_value_t = 32)]
    rank: usize,

    /// Nominal alpha to store. Default: each module's own rank (multiplier 1
    /// reproduces the truncated delta exactly).
    #[arg(long)]
    alpha: Option<f32>,

    /// Output dtype for the LoRA weights (bf16, fp16, fp32).
    #[arg(long, default_value = "fp16")]
    save_dtype: String,

    /// Key-prefix convention: auto (default), krea2, or anima.
    #[arg(long, default_value = "auto")]
    model: String,

    /// Regex; only bare module paths matching this are extracted.
    #[arg(long)]
    include: Option<String>,

    /// Regex; matching bare module paths are skipped.
    #[arg(long)]
    exclude: Option<String>,

    /// Power iterations in the randomized SVD (more = more accurate, slower).
    #[arg(long, default_value_t = 2)]
    niter: usize,

    /// Oversampling added to the rank before projection (accuracy headroom).
    #[arg(long, default_value_t = 8)]
    oversample: usize,
}

#[derive(Args)]
pub struct QuantCommand {
    /// Source checkpoint (.safetensors, bf16/fp16/fp32; fp8_scaled is rejected).
    src: std::path::PathBuf,

    /// Output path. If omitted, derived from SRC (bf16/fp16/fp32 -> int8_convrot).
    dst: Option<std::path::PathBuf>,

    /// Report the plan and write nothing.
    #[arg(long)]
    dry_run: bool,

    /// Regex; matching layers are forced to passthrough.
    #[arg(long)]
    exclude: Option<String>,

    /// Regex; matching eligible layers are forced to quantize.
    #[arg(long)]
    include: Option<String>,

    /// Skip a layer if min(N,K) < this (0 disables). Small GEMMs never beat bf16.
    #[arg(long, default_value_t = 256)]
    min_gemm: usize,

    /// Downcast stray fp32 passthrough linears to the compute dtype.
    #[arg(long)]
    downcast_fp32: bool,

    /// Warn on any quantized layer whose relerr% exceeds this.
    #[arg(long, default_value_t = 2.0)]
    warn_thresh: f32,

    /// Write the full per-layer (relerr, cosine, gs) table to this path.
    #[arg(long)]
    verify_report: Option<std::path::PathBuf>,
}

#[derive(Args)]
pub struct MergeCommand {
    /// Original model the fine-tune started from (e.g. krea2_raw_bf16.safetensors).
    #[arg(long)]
    base: std::path::PathBuf,

    /// Fine-tuned model (the full fine-tune output).
    #[arg(long)]
    tuned: std::path::PathBuf,

    /// Model to receive the delta, bf16 (e.g. krea2_turbo_bf16.safetensors).
    #[arg(long)]
    target: std::path::PathBuf,

    /// Output safetensors path.
    #[arg(long, short)]
    output: std::path::PathBuf,

    /// Strength of the fine-tune delta (lower if it over-applies).
    #[arg(long, default_value_t = 1.0)]
    multiplier: f32,

    /// Override output dtype for merged keys (bf16, fp16, fp32). Default: keep target's dtype.
    #[arg(long)]
    save_dtype: Option<String>,

    /// Key-prefix convention: auto (default), krea2, or anima.
    #[arg(long, default_value = "auto")]
    model: String,
}

pub fn run(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::MergeDiff(cmd) => {
            let save_dtype = cmd.save_dtype.as_deref().map(Dtype::parse_save_dtype).transpose()?;
            let arch = ModelArch::parse(&cmd.model)?;
            merge::run(
                MergeArgs {
                    base: cmd.base,
                    tuned: cmd.tuned,
                    target: cmd.target,
                    output: cmd.output,
                    multiplier: cmd.multiplier,
                    save_dtype,
                    arch,
                },
                &mut StreamProgress::stderr(),
            )
        }
        ModelCommand::QuantInt8(cmd) => quant::run(
            QuantArgs {
                src: cmd.src,
                dst: cmd.dst,
                dry_run: cmd.dry_run,
                exclude: cmd.exclude,
                include: cmd.include,
                min_gemm: cmd.min_gemm,
                downcast_fp32: cmd.downcast_fp32,
                warn_thresh: cmd.warn_thresh,
                verify_report: cmd.verify_report,
            },
            &mut StreamProgress::stdout(),
        ),
        ModelCommand::ExtractLora(cmd) => {
            let save_dtype = Dtype::parse_save_dtype(&cmd.save_dtype)?;
            let arch = ModelArch::parse(&cmd.model)?;
            lora::run(
                ExtractArgs {
                    base: cmd.base,
                    tuned: cmd.tuned,
                    output: cmd.output,
                    rank: cmd.rank,
                    alpha: cmd.alpha,
                    save_dtype,
                    arch,
                    include: cmd.include,
                    exclude: cmd.exclude,
                    niter: cmd.niter,
                    oversample: cmd.oversample,
                },
                &mut StreamProgress::stderr(),
            )
        }
    }
}
