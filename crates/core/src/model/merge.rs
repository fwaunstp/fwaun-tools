//! Task-vector merge: transfer a full fine-tune delta onto another checkpoint.
//!
//! For every key, streaming and low-RAM:
//!
//! ```text
//! output[k] = target[k] + multiplier * (tuned[k] - base[k])
//! ```
//!
//! This is a Rust port of musubi-tuner's `krea2_merge_diff.py`, generalized to
//! also cover Anima checkpoints (which namespace their DiT tensors under `net.`
//! rather than `model.diffusion_model.`). The math is architecture-agnostic; the
//! only model-specific piece is how keys are normalized so that base/tuned deltas
//! line up with a differently-prefixed target.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Result, bail};

use super::safetensors::{Dtype, OutputTensor, SafeTensorsFile, StreamWriter, f32_to_bytes};

/// Which key-prefix conventions to normalize away when matching tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    /// Union of all known prefixes — works for any supported checkpoint.
    Auto,
    /// Krea 2 full fine-tune workflow (ComfyUI/Civitai `model.diffusion_model.`).
    Krea2,
    /// Anima DiT (`net.` prefix, as saved by sd-scripts / official weights).
    Anima,
}

impl ModelArch {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" => ModelArch::Auto,
            "krea2" | "krea" => ModelArch::Krea2,
            "anima" => ModelArch::Anima,
            other => bail!("unknown model '{other}' (expected auto, krea2, or anima)"),
        })
    }

    /// DiT key prefixes to strip so differently-namespaced checkpoints line up.
    fn prefixes(self) -> &'static [&'static str] {
        match self {
            // Longest-first so `model.diffusion_model.` is tried before `diffusion_model.`.
            ModelArch::Auto => &["model.diffusion_model.", "diffusion_model.", "net."],
            ModelArch::Krea2 => &["model.diffusion_model.", "diffusion_model."],
            ModelArch::Anima => &["net.", "model.diffusion_model.", "diffusion_model."],
        }
    }

    fn strip_prefix(self, key: &str) -> &str {
        for p in self.prefixes() {
            if let Some(rest) = key.strip_prefix(p) {
                return rest;
            }
        }
        key
    }

    /// Public wrapper so other subcommands (e.g. LoRA extraction) can normalize
    /// keys with the same prefix conventions.
    pub fn strip_prefix_pub(self, key: &str) -> &str {
        self.strip_prefix(key)
    }
}

/// Parsed arguments for the merge subcommand.
pub struct MergeArgs {
    pub base: PathBuf,
    pub tuned: PathBuf,
    pub target: PathBuf,
    pub output: PathBuf,
    pub multiplier: f32,
    pub save_dtype: Option<Dtype>,
    pub arch: ModelArch,
}

/// fp8 scaled checkpoints store a separate per-tensor scale next to each
/// quantized weight; a bf16 delta cannot be added to an fp8 weight without also
/// touching its scale, so we refuse those (matching the reference).
fn looks_fp8_scaled(f: &SafeTensorsFile) -> bool {
    f.keys().any(|k| {
        k.ends_with("_scale") || k.ends_with(".scale_weight") || k.ends_with(".weight_scale")
    })
}

pub fn run(args: MergeArgs) -> Result<()> {
    eprintln!("base   (org)  : {}", args.base.display());
    eprintln!("tuned  (ft)   : {}", args.tuned.display());
    eprintln!("target (recv) : {}", args.target.display());
    eprintln!("output        : {}", args.output.display());
    eprintln!("multiplier    : {}", args.multiplier);
    eprintln!("model         : {:?}", args.arch);

    let base = SafeTensorsFile::open(&args.base)?;
    let tuned = SafeTensorsFile::open(&args.tuned)?;
    let target = SafeTensorsFile::open(&args.target)?;

    if looks_fp8_scaled(&target) {
        bail!(
            "target looks like an fp8_scaled checkpoint (has *_scale keys). A bf16 delta cannot be \
             cleanly added to fp8 weights. Use a bf16 target."
        );
    }
    if looks_fp8_scaled(&tuned) || looks_fp8_scaled(&base) {
        bail!("base/tuned look fp8_scaled; this merge expects a bf16 base + bf16 fine-tune.");
    }

    // Normalize both sides to the bare DiT key: bare_key -> actual key in that file.
    let base_norm: BTreeMap<&str, &String> =
        base.keys().map(|k| (args.arch.strip_prefix(k), k)).collect();
    let tuned_norm: BTreeMap<&str, &String> =
        tuned.keys().map(|k| (args.arch.strip_prefix(k), k)).collect();

    // Report keys present in the fine-tune but not usable (diagnostics only).
    let base_bare: std::collections::BTreeSet<&str> = base_norm.keys().copied().collect();
    let tuned_bare: std::collections::BTreeSet<&str> = tuned_norm.keys().copied().collect();
    let only_tuned: Vec<&str> = tuned_bare.difference(&base_bare).copied().collect();
    let only_base: Vec<&str> = base_bare.difference(&tuned_bare).copied().collect();
    if !only_tuned.is_empty() {
        eprintln!(
            "warning: {} keys only in tuned (ignored), e.g. {:?}",
            only_tuned.len(),
            &only_tuned[..only_tuned.len().min(3)]
        );
    }
    if !only_base.is_empty() {
        eprintln!(
            "warning: {} keys only in base (ignored), e.g. {:?}",
            only_base.len(),
            &only_base[..only_base.len().min(3)]
        );
    }

    // Decide, per target key, whether it receives a delta and what dtype it gets.
    // This is done from headers alone (no tensor data read) so the output layout
    // can be planned before any bytes are written.
    struct Plan {
        key: String,
        out_dtype: Dtype,
        has_delta: bool,
    }
    let mut plans: Vec<Plan> = Vec::new();
    let mut output_tensors: Vec<OutputTensor> = Vec::new();
    let mut missing_in_target = 0usize;

    // Track which delta keys never landed on the target, for a diagnostic.
    let mut delta_bare_seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();

    for key in target.keys() {
        let tinfo = target.info(key).unwrap();
        let bare = args.arch.strip_prefix(key);
        let has_base = base_norm.contains_key(bare);
        let has_tuned = tuned_norm.contains_key(bare);

        let mut has_delta = false;
        if has_base && has_tuned && tinfo.dtype.is_float() {
            let binfo = base.info(base_norm[bare]).unwrap();
            let uinfo = tuned.info(tuned_norm[bare]).unwrap();
            if binfo.shape == tinfo.shape && uinfo.shape == tinfo.shape {
                has_delta = true;
                delta_bare_seen.insert(bare);
            } else {
                eprintln!(
                    "warning: shape mismatch on {key}: base={:?} tuned={:?} target={:?} -> copying target unchanged",
                    binfo.shape, uinfo.shape, tinfo.shape
                );
            }
        }

        // save_dtype only overrides keys that actually receive a delta; pass-through
        // keys keep the target's original dtype (matching the reference).
        let out_dtype = if has_delta {
            args.save_dtype.unwrap_or(tinfo.dtype)
        } else {
            tinfo.dtype
        };
        let nbytes = tinfo.numel() * out_dtype.element_size();
        output_tensors.push(OutputTensor {
            key: key.clone(),
            dtype: out_dtype,
            shape: tinfo.shape.clone(),
            nbytes,
        });
        plans.push(Plan { key: key.clone(), out_dtype, has_delta });
    }

    // Delta keys defined by base∩tuned that the target does not carry.
    for bare in base_bare.intersection(&tuned_bare) {
        if !delta_bare_seen.contains(bare) {
            missing_in_target += 1;
        }
    }
    if missing_in_target > 0 {
        eprintln!(
            "warning: {missing_in_target} delta keys are absent (or shape-mismatched) in target and were skipped. \
             Are base/target the same architecture?"
        );
    }

    // Carry the target's metadata (keeps modelspec.architecture etc.) plus notes.
    let mut metadata = target.metadata().clone();
    metadata.insert("merged_from_target".to_string(), args.target.display().to_string());
    metadata.insert("merged_delta_base".to_string(), args.base.display().to_string());
    metadata.insert("merged_delta_tuned".to_string(), args.tuned.display().to_string());
    metadata.insert("merged_multiplier".to_string(), args.multiplier.to_string());

    let applied = plans.iter().filter(|p| p.has_delta).count();
    let carried = plans.len() - applied;

    // Stream the output: header first, then each tensor's bytes in plan order.
    let mut writer = StreamWriter::begin(&args.output, output_tensors, &metadata)?;

    let mut max_abs = 0.0f32;
    let mut sum_mean_abs = 0.0f64;

    for plan in &plans {
        let key = &plan.key;
        if !plan.has_delta {
            // Pass-through: copy the target's raw bytes unchanged (dtype preserved).
            writer.write_tensor(key, target.raw_bytes(key)?)?;
            continue;
        }

        let bare = args.arch.strip_prefix(key);
        let tgt = target.to_f32(key)?;
        let b = base.to_f32(base_norm[bare])?;
        let t = tuned.to_f32(tuned_norm[bare])?;

        let mut merged = Vec::with_capacity(tgt.len());
        let mut local_max = 0.0f32;
        let mut local_sum = 0.0f64;
        for i in 0..tgt.len() {
            let delta = t[i] - b[i];
            let d_abs = delta.abs();
            if d_abs > local_max {
                local_max = d_abs;
            }
            local_sum += d_abs as f64;
            merged.push(tgt[i] + args.multiplier * delta);
        }
        if local_max > max_abs {
            max_abs = local_max;
        }
        if !tgt.is_empty() {
            sum_mean_abs += local_sum / tgt.len() as f64;
        }

        let bytes = f32_to_bytes(&merged, plan.out_dtype)?;
        writer.write_tensor(key, &bytes)?;
    }

    writer.finish()?;

    eprintln!("applied delta to {applied} keys; carried {carried} target keys unchanged");
    eprintln!("delta magnitude: max|Δ|={max_abs:.3e}, sum of per-key mean|Δ|={sum_mean_abs:.3e}");
    if max_abs < 1e-4 {
        eprintln!(
            "warning: delta is nearly zero — the fine-tune barely changed the weights. \
             The merged model will be ~identical to the target."
        );
    }
    eprintln!("wrote {} tensors to {}", plans.len(), args.output.display());
    eprintln!("done.");
    Ok(())
}
