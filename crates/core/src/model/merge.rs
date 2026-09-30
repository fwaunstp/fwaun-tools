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
//!
//! Any of the three inputs may be an int8 / int8_convrot checkpoint (see
//! [`super::quantized`]): int8 layers are dequantized to f32 before the math.
//! An int8 target is written dequantized (bf16 by default) unless
//! `requantize` is set, in which case merged layers go back to int8 + ConvRot
//! and untouched int8 layers are copied as-is.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Result, bail};

use super::progress::ProgressSink;
use super::quant::{build_hadamard, comfy_quant_json, quantize_convrot};
use super::quantized::ModelFile;
use super::safetensors::{Dtype, OutputTensor, StreamWriter, f32_to_bytes};

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
    /// With an int8 target, re-quantize merged layers to int8 + ConvRot instead
    /// of writing them (and every other int8 layer) dequantized.
    pub requantize: bool,
}

/// How one logical target tensor is written.
enum Emit {
    /// Copy the target's raw bytes unchanged (plus int8 companions, if any).
    Copy,
    /// Decode to f32 (dequantizing int8), add the delta if any, cast to `dtype`.
    Float { dtype: Dtype, delta: bool },
    /// Add the delta and re-quantize to int8 + ConvRot at group size `gs`.
    Requant { gs: usize, out: usize, in_: usize },
}

pub fn run(args: MergeArgs, p: &mut dyn ProgressSink) -> Result<()> {
    p.log(&format!("base   (org)  : {}", args.base.display()));
    p.log(&format!("tuned  (ft)   : {}", args.tuned.display()));
    p.log(&format!("target (recv) : {}", args.target.display()));
    p.log(&format!("output        : {}", args.output.display()));
    p.log(&format!("multiplier    : {}", args.multiplier));
    p.log(&format!("model         : {:?}", args.arch));

    let base = ModelFile::open(&args.base)?;
    let tuned = ModelFile::open(&args.tuned)?;
    let target = ModelFile::open(&args.target)?;
    p.log(&format!(
        "format        : base={} tuned={} target={}",
        base.quant_label(),
        tuned.quant_label(),
        target.quant_label()
    ));
    if base.quant_label() != tuned.quant_label() {
        p.log(
            "warning: base and tuned are stored differently (float vs int8). Quantization error \
             between them ends up in the delta; use the exact base the fine-tune started from.",
        );
    }
    if args.requantize && !target.is_quantized() {
        bail!(
            "--requantize needs an int8 target; for a float target, run quant-int8 on the output."
        );
    }
    // Dequantized int8 layers have no original dtype to keep; they default to bf16.
    let dequant_dtype = args.save_dtype.unwrap_or(Dtype::Bf16);

    // Normalize both sides to the bare DiT key: bare_key -> actual key in that file.
    let base_norm: BTreeMap<&str, &String> = base
        .keys()
        .map(|k| (args.arch.strip_prefix(k), k))
        .collect();
    let tuned_norm: BTreeMap<&str, &String> = tuned
        .keys()
        .map(|k| (args.arch.strip_prefix(k), k))
        .collect();

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

    // Decide, per target key, whether it receives a delta and how it is written.
    // This is done from headers alone (no tensor data read) so the output layout
    // can be planned before any bytes are written.
    let mut plans: Vec<(String, Emit)> = Vec::new();
    let mut output_tensors: Vec<OutputTensor> = Vec::new();
    let mut missing_in_target = 0usize;

    // Track which delta keys never landed on the target, for a diagnostic.
    let mut delta_bare_seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();

    for key in target.keys() {
        let tinfo = target.raw().info(key).unwrap();
        let bare = args.arch.strip_prefix(key);
        let int8 = target.int8_layer(key);

        let mut has_delta = false;
        if let (Some(bk), Some(uk)) = (base_norm.get(bare), tuned_norm.get(bare))
            && target.is_numeric(key)
            && base.is_numeric(bk)
            && tuned.is_numeric(uk)
        {
            let (bshape, ushape) = (base.shape(bk).unwrap(), tuned.shape(uk).unwrap());
            if bshape == tinfo.shape && ushape == tinfo.shape {
                has_delta = true;
                delta_bare_seen.insert(bare);
            } else {
                p.log(&format!(
                    "warning: shape mismatch on {key}: base={bshape:?} tuned={ushape:?} target={:?} -> copying target unchanged",
                    tinfo.shape
                ));
            }
        }

        let write = match (int8, has_delta) {
            (Some(layer), true) if args.requantize => match layer.convrot_gs {
                Some(gs) => Emit::Requant {
                    gs,
                    out: tinfo.shape[0],
                    in_: tinfo.shape[1],
                },
                None => {
                    bail!("--requantize supports int8_convrot layers only; {key} is plain int8")
                }
            },
            (Some(_), _) if args.requantize => Emit::Copy,
            (Some(_), delta) => Emit::Float {
                dtype: dequant_dtype,
                delta,
            },
            // save_dtype only overrides keys that actually receive a delta; pass-through
            // keys keep the target's original dtype (matching the reference).
            (None, true) => Emit::Float {
                dtype: args.save_dtype.unwrap_or(tinfo.dtype),
                delta: true,
            },
            (None, false) => Emit::Copy,
        };

        match &write {
            Emit::Copy => {
                output_tensors.push(OutputTensor {
                    key: key.clone(),
                    dtype: tinfo.dtype,
                    shape: tinfo.shape.clone(),
                    nbytes: tinfo.end - tinfo.begin,
                });
                for c in int8.map(|l| l.companions.as_slice()).unwrap_or_default() {
                    let ci = target.raw().info(c).unwrap();
                    output_tensors.push(OutputTensor {
                        key: c.clone(),
                        dtype: ci.dtype,
                        shape: ci.shape.clone(),
                        nbytes: ci.end - ci.begin,
                    });
                }
            }
            Emit::Float { dtype, .. } => output_tensors.push(OutputTensor {
                key: key.clone(),
                dtype: *dtype,
                shape: tinfo.shape.clone(),
                nbytes: tinfo.numel() * dtype.element_size(),
            }),
            Emit::Requant { gs, out, in_ } => {
                let layer = key.strip_suffix(".weight").unwrap_or(key);
                let cq = comfy_quant_json(*gs);
                output_tensors.push(OutputTensor {
                    key: key.clone(),
                    dtype: Dtype::I8,
                    shape: vec![*out, *in_],
                    nbytes: out * in_,
                });
                output_tensors.push(OutputTensor {
                    key: format!("{layer}.weight_scale"),
                    dtype: Dtype::F32,
                    shape: vec![*out, 1],
                    nbytes: out * 4,
                });
                output_tensors.push(OutputTensor {
                    key: format!("{layer}.comfy_quant"),
                    dtype: Dtype::U8,
                    shape: vec![cq.len()],
                    nbytes: cq.len(),
                });
            }
        }
        plans.push((key.clone(), write));
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
    if target.is_quantized() && !args.requantize {
        // Every int8 layer is written dequantized, so the output is no longer int8.
        metadata.remove("quant_format");
        p.log(&format!(
            "target is {}: writing its int8 layers dequantized as {} (use --requantize to keep int8)",
            target.quant_label(),
            dequant_dtype.tag()
        ));
    }
    metadata.insert(
        "merged_from_target".to_string(),
        args.target.display().to_string(),
    );
    metadata.insert(
        "merged_delta_base".to_string(),
        args.base.display().to_string(),
    );
    metadata.insert(
        "merged_delta_tuned".to_string(),
        args.tuned.display().to_string(),
    );
    metadata.insert("merged_multiplier".to_string(), args.multiplier.to_string());

    let applied = plans
        .iter()
        .filter(|(_, w)| matches!(w, Emit::Float { delta: true, .. } | Emit::Requant { .. }))
        .count();
    let carried = plans.len() - applied;

    // Stream the output: header first, then each tensor's bytes in plan order.
    let mut writer = StreamWriter::begin(&args.output, output_tensors, &metadata)?;

    let mut max_abs = 0.0f32;
    let mut sum_mean_abs = 0.0f64;
    let mut max_requant_err = 0.0f64;

    let total = plans.len();
    for (i, (key, write)) in plans.iter().enumerate() {
        p.tick(i + 1, total);
        let delta = match write {
            Emit::Copy => {
                // Pass-through: copy the target's raw bytes unchanged (dtype preserved).
                writer.write_tensor(key, target.raw().raw_bytes(key)?)?;
                for c in target
                    .int8_layer(key)
                    .map(|l| l.companions.as_slice())
                    .unwrap_or_default()
                {
                    writer.write_tensor(c, target.raw().raw_bytes(c)?)?;
                }
                continue;
            }
            Emit::Float { delta, .. } => *delta,
            Emit::Requant { .. } => true,
        };

        let mut merged = target.to_f32(key)?;
        if delta {
            let bare = args.arch.strip_prefix(key);
            let b = base.to_f32(base_norm[bare])?;
            let t = tuned.to_f32(tuned_norm[bare])?;

            let mut local_max = 0.0f32;
            let mut local_sum = 0.0f64;
            for i in 0..merged.len() {
                let d = t[i] - b[i];
                let d_abs = d.abs();
                if d_abs > local_max {
                    local_max = d_abs;
                }
                local_sum += d_abs as f64;
                merged[i] += args.multiplier * d;
            }
            if local_max > max_abs {
                max_abs = local_max;
            }
            if !merged.is_empty() {
                sum_mean_abs += local_sum / merged.len() as f64;
            }
        }

        match write {
            Emit::Float { dtype, .. } => {
                writer.write_tensor(key, &f32_to_bytes(&merged, *dtype)?)?;
            }
            Emit::Requant { gs, out, in_ } => {
                let layer = key.strip_suffix(".weight").unwrap_or(key);
                let r = quantize_convrot(&merged, *out, *in_, *gs, &build_hadamard(*gs));
                max_requant_err = max_requant_err.max(r.relerr());
                let qbytes: Vec<u8> = r.qdata.iter().map(|&v| v as u8).collect();
                writer.write_tensor(key, &qbytes)?;
                writer.write_tensor(
                    &format!("{layer}.weight_scale"),
                    &f32_to_bytes(&r.scale, Dtype::F32)?,
                )?;
                writer.write_tensor(&format!("{layer}.comfy_quant"), &comfy_quant_json(*gs))?;
            }
            Emit::Copy => unreachable!(),
        }
    }

    writer.finish()?;

    p.log(&format!(
        "applied delta to {applied} keys; carried {carried} target keys unchanged"
    ));
    p.log(&format!(
        "delta magnitude: max|Δ|={max_abs:.3e}, sum of per-key mean|Δ|={sum_mean_abs:.3e}"
    ));
    if args.requantize {
        p.log(&format!(
            "re-quantized merged layers to int8_convrot: max relerr {max_requant_err:.2}%"
        ));
    }
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
    use crate::model::safetensors::{OutputTensor, SafeTensorsFile, StreamWriter, f32_to_bytes};
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
        w.write_tensor(key, &f32_to_bytes(vals, dtype).unwrap())
            .unwrap();
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
                requantize: false,
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

    #[test]
    fn merge_int8_convrot_inputs_dequantized_or_requantized() {
        use crate::model::quantized::ModelFile;
        use crate::model::quantized::tests::{TestTensor, temp_dir, weights, write_model};

        let dir = temp_dir("merge-int8");
        let (out_n, in_n, gs) = (8, 64, 16);
        let n = out_n * in_n;
        let key = "transformer_blocks.0.attn.to_q.weight";
        let (wb, wt, wg) = (weights(n, 1), weights(n, 2), weights(n, 3));
        let (base, tuned, target) = (
            dir.join("base.safetensors"),
            dir.join("tuned.safetensors"),
            dir.join("target.safetensors"),
        );
        for (path, w) in [(&base, &wb), (&tuned, &wt), (&target, &wg)] {
            write_model(
                path,
                &[
                    (key, TestTensor::Int8(out_n, in_n, gs, w.clone())),
                    ("norm.weight", TestTensor::Float(vec![4], vec![1.0; 4])),
                ],
            );
        }
        // Expected = dequantized target + (dequantized tuned - dequantized base).
        let deq = |p: &std::path::Path| ModelFile::open(p).unwrap().to_f32(key).unwrap();
        let (db, dt, dg) = (deq(&base), deq(&tuned), deq(&target));
        let expected: Vec<f32> = (0..n).map(|i| dg[i] + dt[i] - db[i]).collect();
        let max_err = |got: &[f32]| {
            got.iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };

        // Default: the int8 target is written dequantized as bf16.
        let out = dir.join("out.safetensors");
        let args = |output: PathBuf, requantize| MergeArgs {
            base: base.clone(),
            tuned: tuned.clone(),
            target: target.clone(),
            output,
            multiplier: 1.0,
            save_dtype: None,
            arch: ModelArch::Auto,
            requantize,
        };
        run(args(out.clone(), false), &mut Capture::default()).unwrap();
        let f = SafeTensorsFile::open(&out).unwrap();
        assert_eq!(f.info(key).unwrap().dtype, Dtype::Bf16);
        assert!(
            f.info("transformer_blocks.0.attn.to_q.weight_scale")
                .is_none()
        );
        assert!(
            f.info("transformer_blocks.0.attn.to_q.comfy_quant")
                .is_none()
        );
        assert!(max_err(&f.to_f32(key).unwrap()) < 0.02);

        // --requantize: stays int8_convrot and reads back close to the float merge.
        let out_q = dir.join("out_q.safetensors");
        run(args(out_q.clone(), true), &mut Capture::default()).unwrap();
        let m = ModelFile::open(&out_q).unwrap();
        assert_eq!(m.quant_label(), "int8_convrot");
        assert_eq!(m.int8_layer(key).unwrap().convrot_gs, Some(gs));
        assert!(max_err(&m.to_f32(key).unwrap()) < 0.1);
        assert_eq!(m.to_f32("norm.weight").unwrap(), vec![1.0; 4]);

        // --requantize is meaningless for a float target.
        let float_target = dir.join("float.safetensors");
        write_one(&float_target, key, vec![out_n, in_n], &wg);
        let mut a = args(dir.join("x.safetensors"), true);
        a.target = float_target;
        assert!(run(a, &mut Capture::default()).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
