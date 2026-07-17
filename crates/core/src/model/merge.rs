//! Task-vector merge: transfer a full fine-tune delta onto another checkpoint.
//!
//! For every key, streaming and low-RAM:
//!
//! ```text
//! output[k] = target[k] + multiplier * (tuned[k] - base[k])
//! ```
//!
//! The math is architecture-agnostic; the only model-specific piece is how keys
//! are normalized so that base/tuned deltas line up with a differently-prefixed
//! target. Covers Krea 2 checkpoints (ComfyUI/Civitai `model.diffusion_model.`)
//! and Anima checkpoints (which namespace their DiT tensors under `net.` rather
//! than `model.diffusion_model.`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Result, bail};

use super::progress::ProgressSink;
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

pub fn run(args: MergeArgs, p: &mut dyn ProgressSink) -> Result<()> {
    p.log(&format!("base   (org)  : {}", args.base.display()));
    p.log(&format!("tuned  (ft)   : {}", args.tuned.display()));
    p.log(&format!("target (recv) : {}", args.target.display()));
    p.log(&format!("output        : {}", args.output.display()));
    p.log(&format!("multiplier    : {}", args.multiplier));
    p.log(&format!("model         : {:?}", args.arch));

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
        p.log(&format!(
            "warning: {} keys only in tuned (ignored), e.g. {:?}",
            only_tuned.len(),
            &only_tuned[..only_tuned.len().min(3)]
        ));
    }
    if !only_base.is_empty() {
        p.log(&format!(
            "warning: {} keys only in base (ignored), e.g. {:?}",
            only_base.len(),
            &only_base[..only_base.len().min(3)]
        ));
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
                p.log(&format!(
                    "warning: shape mismatch on {key}: base={:?} tuned={:?} target={:?} -> copying target unchanged",
                    binfo.shape, uinfo.shape, tinfo.shape
                ));
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
        p.log(&format!(
            "warning: {missing_in_target} delta keys are absent (or shape-mismatched) in target and were skipped. \
             Are base/target the same architecture?"
        ));
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

    let total = plans.len();
    for (i, plan) in plans.iter().enumerate() {
        p.tick(i + 1, total);
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

    p.log(&format!(
        "applied delta to {applied} keys; carried {carried} target keys unchanged"
    ));
    p.log(&format!(
        "delta magnitude: max|Δ|={max_abs:.3e}, sum of per-key mean|Δ|={sum_mean_abs:.3e}"
    ));
    if max_abs < 1e-4 {
        p.log(
            "warning: delta is nearly zero — the fine-tune barely changed the weights. \
             The merged model will be ~identical to the target.",
        );
    }
    p.log(&format!(
        "wrote {} tensors to {}",
        plans.len(),
        args.output.display()
    ));
    p.log("done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::progress::ProgressSink;
    use crate::model::safetensors::{OutputTensor, StreamWriter, f32_to_bytes};
    use std::collections::BTreeMap;

    /// A sink that records every log line and progress tick, for assertions.
    #[derive(Default)]
    struct Capture {
        logs: Vec<String>,
        last_tick: Option<(usize, usize)>,
        ticks: usize,
    }
    impl ProgressSink for Capture {
        fn log(&mut self, line: &str) {
            self.logs.push(line.to_string());
        }
        fn tick(&mut self, done: usize, total: usize) {
            self.last_tick = Some((done, total));
            self.ticks += 1;
        }
    }

    /// Write a one-tensor f32 safetensors file with the given key and values.
    fn write_one(path: &std::path::Path, key: &str, shape: Vec<usize>, vals: &[f32]) {
        let dtype = Dtype::parse_save_dtype("fp32").unwrap();
        let nbytes = vals.len() * dtype.element_size();
        let plan = vec![OutputTensor {
            key: key.to_string(),
            dtype,
            shape,
            nbytes,
        }];
        let mut w = StreamWriter::begin(path, plan, &BTreeMap::new()).unwrap();
        w.write_tensor(key, &f32_to_bytes(vals, dtype).unwrap()).unwrap();
        w.finish().unwrap();
    }

    #[test]
    fn merge_reports_progress_and_writes_output() {
        // Isolated temp dir (no external tempfile dep; unique per process).
        let dir = std::env::temp_dir().join(format!("fwaun-merge-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // One shared DiT weight; target carries it under a different namespace
        // so prefix-stripping has to line them up. delta = tuned - base = 1.0.
        let key_bt = "net.block.0.weight";
        let key_target = "model.diffusion_model.block.0.weight";
        let base = dir.join("base.safetensors");
        let tuned = dir.join("tuned.safetensors");
        let target = dir.join("target.safetensors");
        let out = dir.join("out.safetensors");
        write_one(&base, key_bt, vec![2, 2], &[0.0, 0.0, 0.0, 0.0]);
        write_one(&tuned, key_bt, vec![2, 2], &[1.0, 1.0, 1.0, 1.0]);
        write_one(&target, key_target, vec![2, 2], &[5.0, 5.0, 5.0, 5.0]);

        let mut cap = Capture::default();
        run(
            MergeArgs {
                base,
                tuned,
                target,
                output: out.clone(),
                multiplier: 1.0,
                save_dtype: None,
                arch: ModelArch::Auto,
            },
            &mut cap,
        )
        .unwrap();

        // Output written, and the delta landed: 5.0 + 1.0*(1.0-0.0) = 6.0.
        let merged = SafeTensorsFile::open(&out).unwrap();
        let vals = merged.to_f32(key_target).unwrap();
        assert_eq!(vals, vec![6.0, 6.0, 6.0, 6.0]);

        // Progress reached completion (target has one key -> tick (1, 1)).
        assert!(cap.ticks >= 1, "expected at least one progress tick");
        assert_eq!(cap.last_tick, Some((1, 1)));
        assert_eq!(cap.logs.last().map(String::as_str), Some("done."));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
