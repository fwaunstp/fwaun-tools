//! `fwaun-tools model <verb>` — diffusion-checkpoint subcommands. Thin clap
//! layer over `fwaun_tools_core::model`; all the work happens in core.

use anyhow::Result;
use clap::{Args, Subcommand};

use fwaun_tools_core::model::StreamProgress;
use fwaun_tools_core::model::dequant::{self, DequantArgs};
use fwaun_tools_core::model::info::{self, ModelInfo};
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
    /// Supports Krea 2 and Anima key conventions. Inputs may be bf16/fp16/fp32 or
    /// int8_convrot (dequantized on load); an int8 target is written as bf16 unless
    /// `--requantize` is given. All math runs on CPU in f32, streaming key-by-key
    /// so peak RAM stays small.
    MergeDiff(MergeCommand),

    /// Quantize a bf16/fp16 checkpoint to int8 + ConvRot (comfy-kitchen layout).
    ///
    /// Auto-detects the per-token block linears (attention + FFN), rotates each
    /// with a block-Hadamard at the best power-of-4 group size, and stores int8
    /// weights + per-channel scales + a `comfy_quant` config. All math is CPU/f32
    /// and parallelized across cores. Run with `--dry-run` first on a new
    /// architecture to review the plan.
    QuantInt8(QuantCommand),

    /// Dequantize an int8_convrot checkpoint to bf16 (or fp16/fp32).
    ///
    /// For GPUs without comfy-kitchen int8 support. Every int8 layer is
    /// un-rotated and written as a plain float weight; its weight_scale /
    /// comfy_quant companions are dropped and all other tensors are copied
    /// unchanged. The result carries the int8 rounding error: it is not the
    /// original bf16 model.
    Dequant(DequantCommand),

    /// Show a checkpoint's metadata, dtype mix, quantization, and key prefix.
    ///
    /// Reads only the header (and the tiny comfy_quant configs), so it is instant
    /// even for 20+ GB files. Also reports whether the file is complete: a
    /// truncated download still opens but is missing tensor data.
    Info(InfoCommand),

    /// Extract a low-rank LoRA from a full fine-tune: SVD of (tuned - base).
    ///
    /// base/tuned may be bf16/fp16/fp32 or int8_convrot (dequantized on load).
    /// For every shared 2D linear weight, factorizes the fine-tune delta into
    /// lora_up/lora_down at the requested rank and writes a kohya-ss/ComfyUI
    /// (`lora_unet_*`) LoRA. Reports the per-module energy captured so you can
    /// tell whether the rank is high enough to track the fine-tune. All math is
    /// CPU/f32 and parallelized across cores.
    ExtractLora(ExtractCommand),
}

#[derive(Args)]
pub struct ExtractCommand {
    /// Original model the fine-tune started from (bf16/fp16/fp32 or int8_convrot).
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
pub struct DequantCommand {
    /// Source checkpoint (.safetensors, int8 / int8_convrot).
    src: std::path::PathBuf,

    /// Output path. If omitted, derived from SRC (int8_convrot -> bf16).
    dst: Option<std::path::PathBuf>,

    /// Output dtype for the dequantized layers (bf16, fp16, fp32).
    #[arg(long, default_value = "bf16")]
    dtype: String,
}

#[derive(Args)]
pub struct InfoCommand {
    /// Checkpoint to inspect (.safetensors).
    file: std::path::PathBuf,

    /// Print machine-readable JSON instead of text.
    #[arg(long)]
    json: bool,

    /// Also list tensors (key, dtype, shape), optionally only keys matching REGEX.
    #[arg(long, value_name = "REGEX", num_args = 0..=1, default_missing_value = "")]
    tensors: Option<String>,

    /// Print only the `__metadata__` entries, in full (the summary view shortens
    /// long values).
    #[arg(long, conflicts_with = "tensors")]
    metadata_only: bool,
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

    /// Override output dtype for merged keys (bf16, fp16, fp32). Default: keep target's
    /// dtype; int8 target layers are written dequantized as this dtype (default bf16).
    #[arg(long)]
    save_dtype: Option<String>,

    /// With an int8_convrot target, re-quantize merged layers to int8_convrot and copy
    /// the other int8 layers as-is, instead of writing a dequantized model.
    #[arg(long)]
    requantize: bool,

    /// Key-prefix convention: auto (default), krea2, or anima.
    #[arg(long, default_value = "auto")]
    model: String,
}

pub fn run(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::MergeDiff(cmd) => {
            let save_dtype = cmd
                .save_dtype
                .as_deref()
                .map(Dtype::parse_save_dtype)
                .transpose()?;
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
                    requantize: cmd.requantize,
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
        ModelCommand::Dequant(cmd) => dequant::run(
            DequantArgs {
                src: cmd.src,
                dst: cmd.dst,
                dtype: Dtype::parse_save_dtype(&cmd.dtype)?,
            },
            &mut StreamProgress::stdout(),
        ),
        ModelCommand::Info(cmd) => run_info(cmd),
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

/// Metadata values longer than this are shortened in the summary view.
const METADATA_PREVIEW_CHARS: usize = 600;

fn run_info(cmd: InfoCommand) -> Result<()> {
    let info = info::inspect(&cmd.file)?;
    // `--tensors` with no REGEX arrives as "" (list everything).
    let filter = match cmd.tensors.as_deref() {
        None | Some("") => None,
        Some(re) => Some(regex::Regex::new(re)?),
    };
    let tensors = cmd.tensors.is_some().then_some(filter.as_ref());

    if cmd.json {
        if cmd.metadata_only {
            println!("{}", serde_json::to_string_pretty(&info.metadata)?);
        } else {
            println!("{}", info.to_json(tensors));
        }
        return Ok(());
    }
    if cmd.metadata_only {
        print_metadata(&info, None);
        return Ok(());
    }

    println!(
        "file       : {} ({})",
        info.path,
        info::human_bytes(info.file_size)
    );
    println!("status     : {}", info::status_line(&info));
    println!(
        "tensors    : {} ({})",
        info.tensor_count,
        info::human_bytes(info.tensor_bytes)
    );
    let dtypes: Vec<String> = info
        .dtypes
        .iter()
        .map(|(d, s)| format!("{d} ×{} ({})", s.count, info::human_bytes(s.bytes)))
        .collect();
    println!("dtypes     : {}", dtypes.join(", "));
    println!("quant      : {}", info::quant_line(&info.quantization));
    if info.quantization.comfy_quant.len() > 1 || info.quantization.kind == "mixed" {
        for (body, n) in &info.quantization.comfy_quant {
            println!("             ×{n:<4} {body}");
        }
    }
    println!("key prefix : {}", info::prefix_line(&info.key_prefix));
    println!();
    print_metadata(&info, Some(METADATA_PREVIEW_CHARS));

    if let Some(filter) = tensors {
        println!();
        let list: Vec<_> = info.tensors_matching(filter).collect();
        println!("tensors ({} of {}):", list.len(), info.tensor_count);
        for t in list {
            println!(
                "  {:<5} {:<16} {}",
                t.dtype,
                format!("{:?}", t.shape),
                t.key
            );
        }
    }
    Ok(())
}

/// Print `__metadata__`, JSON values pretty-printed and indented under their key.
/// `limit` shortens each value to that many characters.
fn print_metadata(info: &ModelInfo, limit: Option<usize>) {
    if info.metadata.is_empty() {
        println!("metadata   : (none)");
        return;
    }
    println!("metadata ({}):", info.metadata.len());
    for (key, value) in &info.metadata {
        let mut pretty = info::pretty_metadata_value(value);
        if let Some(limit) = limit
            && pretty.chars().count() > limit
        {
            let total = pretty.chars().count();
            pretty = pretty.chars().take(limit).collect();
            pretty.push_str(&format!(" … ({total} chars; --metadata-only shows all)"));
        }
        let mut lines = pretty.lines();
        println!("  {key}: {}", lines.next().unwrap_or(""));
        for line in lines {
            println!("    {line}");
        }
    }
}
