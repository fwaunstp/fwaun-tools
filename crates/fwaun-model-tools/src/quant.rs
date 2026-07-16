//! INT8 + ConvRot quantizer for comfy-kitchen — auto layer detection.
//!
//! A CPU/f32 Rust port of comfy-model-tools' `quant_int8_convrot.py`. It quantizes
//! the per-token block linears (attention + FFN) and passes everything else
//! through unchanged. The recipe per quantized layer is:
//!
//!   1. upcast the weight to f32,
//!   2. apply a block-Hadamard rotation at the best power-of-4 group size,
//!   3. take a per-channel (per-row) absmax scale and round to int8,
//!   4. emit `<layer>.weight` (int8), `<layer>.weight_scale` (f32), and a
//!      `<layer>.comfy_quant` (uint8 JSON) config the comfy-kitchen loader reads.
//!
//! Only the *file* is produced here — dequantization at inference time still runs
//! comfy-kitchen's CUDA ops, which is unaffected by generating the file on CPU.
//!
//! Notes vs. the Python reference:
//! - safetensors input only (no torch `.pth`/`.pt` pickle path).
//! - fp8_scaled sources are rejected rather than dequantized (matching `merge-diff`).
//! - absmax scaling only (no `--mseclip` grid search yet).
//! - Reconstruction error (relerr/cosine) is computed in the rotated space. The
//!   Hadamard is orthogonal, so norms and inner products are preserved and the
//!   numbers equal the original-space error the Python script reports — for free,
//!   with no un-rotation pass.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rayon::prelude::*;
use regex::Regex;

use crate::safetensors::{Dtype, OutputTensor, SafeTensorsFile, StreamWriter, f32_to_bytes};

/// ConvRot Hadamard sizes: powers of 4, largest preferred (matches the reference).
const VALID_GS: [usize; 3] = [256, 64, 16];

/// Parsed arguments for the `quant-int8` subcommand.
pub struct QuantArgs {
    pub src: PathBuf,
    pub dst: Option<PathBuf>,
    pub dry_run: bool,
    pub exclude: Option<String>,
    pub include: Option<String>,
    pub min_gemm: usize,
    pub downcast_fp32: bool,
    pub warn_thresh: f32,
    pub verify_report: Option<PathBuf>,
}

/// Largest valid group size dividing `k`, or `None` if none does.
fn best_gs(k: usize) -> Option<usize> {
    VALID_GS.into_iter().find(|&g| k.is_multiple_of(g))
}

/// Layers we never quantize: not per-token GEMMs, or identity/quality-critical.
/// Ported verbatim from the reference's `EXCLUDE_SEG`; applied per dot-segment.
fn exclude_seg() -> Regex {
    Regex::new(
        r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|\
gate_logits|router|routing|logit|temperature|\
(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|\
(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)",
    )
    .expect("static EXCLUDE_SEG regex is valid")
}

/// Decide whether a 2-D weight is an eligible block linear. Mirrors the
/// reference `classify`: returns `Some(gs)` to quantize, or `Err(reason)` to skip.
fn classify(key: &str, shape: &[usize], deny: &Regex) -> std::result::Result<usize, &'static str> {
    if shape.len() != 2 {
        return Err("not-2d");
    }
    let (n, k) = (shape[0], shape[1]);
    if n < 8 {
        return Err("small-N");
    }
    let gs = best_gs(k).ok_or("ineligible-K")?;
    let segs: Vec<&str> = key.split('.').collect();
    // In an indexed block = an integer segment with named structure after it
    // (blocks.5.attn.q). A trailing integer is a Sequential index, not a block.
    let in_block = segs[..segs.len() - 1]
        .iter()
        .any(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
    if !in_block {
        return Err("not-in-indexed-block");
    }
    if segs.iter().any(|s| deny.is_match(s)) {
        return Err("denylist(scale_shift/embed/gate/time/head/refiner_blocks/adapter)");
    }
    Ok(gs)
}

/// Build the normalized regular (ConvRot) Hadamard matrix of `size`, row-major.
///
/// `size` must be a power of 4. The matrix is the Kronecker power of H4 divided by
/// `sqrt(size)`; it is symmetric, orthogonal, and involutory (H·H = I), so the same
/// operation both rotates and un-rotates.
fn build_hadamard(size: usize) -> Vec<f32> {
    assert!(
        size >= 4 && size.is_power_of_two() && size.trailing_zeros().is_multiple_of(2),
        "Hadamard size must be a power of 4, got {size}"
    );
    #[rustfmt::skip]
    let h4: [f32; 16] = [
         1.0,  1.0,  1.0, -1.0,
         1.0,  1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,  1.0,
    ];
    let mut dim = 4usize;
    let mut h = h4.to_vec();
    // Kronecker with H4 until we reach `size`: result[(i*4+p)][(j*4+q)] = A[i][j]*B[p][q].
    while dim < size {
        let nd = dim * 4;
        let mut next = vec![0.0f32; nd * nd];
        for i in 0..dim {
            for j in 0..dim {
                let a = h[i * dim + j];
                for p in 0..4 {
                    for q in 0..4 {
                        next[(i * 4 + p) * nd + (j * 4 + q)] = a * h4[p * 4 + q];
                    }
                }
            }
        }
        h = next;
        dim = nd;
    }
    let inv = 1.0 / (size as f32).sqrt();
    for v in &mut h {
        *v *= inv;
    }
    h
}

/// Per-layer quantization result: int8 weight (row-major), per-row scales, and
/// reduction sums for the rotated-space error metrics.
struct QuantOut {
    qdata: Vec<i8>,
    scale: Vec<f32>,
    sum_sq_err: f64,
    sum_sq_ref: f64,
    dot: f64,
    sum_sq_deq: f64,
}

impl QuantOut {
    /// Relative reconstruction error (%), = ||dequant - source|| / ||source||.
    fn relerr(&self) -> f64 {
        (self.sum_sq_err.sqrt() / self.sum_sq_ref.sqrt().max(1e-30)) * 100.0
    }
    /// Cosine similarity between dequantized and source weights.
    fn cosine(&self) -> f64 {
        self.dot / (self.sum_sq_deq.sqrt() * self.sum_sq_ref.sqrt()).max(1e-30)
    }
}

/// Rotate + per-channel absmax quantize a 2-D weight. Rows are independent, so the
/// heavy Hadamard matmul runs in parallel across cores.
fn quantize_convrot(w: &[f32], out: usize, in_: usize, gs: usize, h: &[f32]) -> QuantOut {
    let n_groups = in_ / gs;
    // (row_qdata, row_scale, se, sr, dot, sd) per row, in row order.
    let rows: Vec<(Vec<i8>, f32, f64, f64, f64, f64)> = (0..out)
        .into_par_iter()
        .map(|r| {
            let base = r * in_;
            // Rotate: for each group, wr[j] = sum_i w[i] * H[j][i] (H symmetric).
            let mut wr = vec![0.0f32; in_];
            for g in 0..n_groups {
                let gb = g * gs;
                for j in 0..gs {
                    let hj = &h[j * gs..j * gs + gs];
                    let mut acc = 0.0f32;
                    for i in 0..gs {
                        acc += w[base + gb + i] * hj[i];
                    }
                    wr[gb + j] = acc;
                }
            }
            let amax = wr.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-30);
            let scale = (amax / 127.0).max(1e-30);
            let mut q = vec![0i8; in_];
            let (mut se, mut sr, mut dot, mut sd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for i in 0..in_ {
                let qi = (wr[i] / scale).round_ties_even().clamp(-127.0, 127.0);
                q[i] = qi as i8;
                let deq = (qi * scale) as f64;
                let wri = wr[i] as f64;
                se += (deq - wri) * (deq - wri);
                sr += wri * wri;
                dot += deq * wri;
                sd += deq * deq;
            }
            (q, scale, se, sr, dot, sd)
        })
        .collect();

    let mut qdata = Vec::with_capacity(out * in_);
    let mut scale = Vec::with_capacity(out);
    let (mut sum_sq_err, mut sum_sq_ref, mut dot, mut sum_sq_deq) = (0.0, 0.0, 0.0, 0.0);
    for (q, s, se, sr, d, sd) in rows {
        qdata.extend_from_slice(&q);
        scale.push(s);
        sum_sq_err += se;
        sum_sq_ref += sr;
        dot += d;
        sum_sq_deq += sd;
    }
    QuantOut { qdata, scale, sum_sq_err, sum_sq_ref, dot, sum_sq_deq }
}

/// The embedded per-layer config the comfy-kitchen loader reads (uint8 JSON bytes).
/// Byte-for-byte identical to the reference's `json.dumps` output.
fn comfy_quant_json(gs: usize) -> Vec<u8> {
    format!("{{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": {gs}}}")
        .into_bytes()
}

/// Derive an output path when `dst` is omitted: swap a bf16/fp16/fp32 dtype token
/// for `int8_convrot` (else append `_int8_convrot`), always `.safetensors`.
fn derive_dst(src: &Path) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let re = Regex::new(r"(?i)bf16|fp16|fp32").unwrap();
    let new = if re.is_match(stem) {
        re.replace(stem, "int8_convrot").into_owned()
    } else {
        format!("{stem}_int8_convrot")
    };
    src.with_file_name(format!("{new}.safetensors"))
}

/// True if the file carries per-tensor fp8 scales we cannot cleanly consume here.
fn looks_fp8_scaled(f: &SafeTensorsFile) -> bool {
    f.keys().any(|k| k.ends_with(".weight_scale") || k.ends_with("_scale") || k.ends_with(".scale_weight"))
}

/// Collapse digit runs to `N` so sibling block layers group under one pattern.
fn pattern(key: &str) -> String {
    let re = Regex::new(r"\d+").unwrap();
    re.replace_all(key, "N").into_owned()
}

/// What to do with each source tensor, in the exact order it is written.
enum Action {
    /// Copy the source tensor's raw bytes unchanged.
    Passthrough { key: String },
    /// Passthrough f32 weight downcast to the compute dtype (`--downcast-fp32`).
    Downcast { key: String, dtype: Dtype },
    /// Quantize this `.weight`; emits weight + weight_scale + comfy_quant.
    Quantize { key: String, base: String, out: usize, in_: usize, gs: usize },
}

pub fn run(args: QuantArgs) -> Result<()> {
    let src = SafeTensorsFile::open(&args.src)?;
    if looks_fp8_scaled(&src) {
        bail!(
            "source looks like an fp8_scaled checkpoint (has *_scale keys). This port quantizes \
             bf16/fp16/fp32 sources only — use a non-fp8 checkpoint."
        );
    }

    let dst = args.dst.clone().unwrap_or_else(|| derive_dst(&args.src));
    if args.dst.is_none() && !args.dry_run {
        println!("auto dst -> {}", dst.display());
    }

    let deny = exclude_seg();
    let exc = args.exclude.as_deref().map(Regex::new).transpose()?;
    let inc = args.include.as_deref().map(Regex::new).transpose()?;

    // Compute/passthrough dtype = dominant non-fp8 float weight dtype (F16 vs BF16).
    let (mut n_f16, mut n_bf16) = (0usize, 0usize);
    for key in src.keys() {
        if key.ends_with(".weight") {
            match src.info(key).unwrap().dtype {
                Dtype::F16 => n_f16 += 1,
                Dtype::Bf16 => n_bf16 += 1,
                _ => {}
            }
        }
    }
    let target = if n_f16 >= n_bf16 && n_f16 > 0 { Dtype::F16 } else { Dtype::Bf16 };

    // ---- plan ----
    let mut actions: Vec<Action> = Vec::new();
    let mut out_tensors: Vec<OutputTensor> = Vec::new();
    let mut skip: HashMap<&'static str, usize> = HashMap::new();
    // pattern -> (count, shape, gs) for the quantize report.
    let mut by_pat: BTreeMap<String, (usize, [usize; 2], usize)> = BTreeMap::new();
    // Per-layer group-size histogram (a single pattern can mix group sizes).
    let mut gsdist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut qparams: u64 = 0;

    for key in src.keys() {
        let info = src.info(key).unwrap();
        if !key.ends_with(".weight") {
            actions.push(Action::Passthrough { key: key.clone() });
            out_tensors.push(OutputTensor {
                key: key.clone(),
                dtype: info.dtype,
                shape: info.shape.clone(),
                nbytes: info.end - info.begin,
            });
            continue;
        }
        let base = key[..key.len() - ".weight".len()].to_string();

        // classify + flag overrides, mirroring the reference precedence.
        let mut decision = classify(&base, &info.shape, &deny);
        if exc.as_ref().is_some_and(|re| re.is_match(&base)) {
            decision = Err("excluded(flag)");
        }
        if inc.as_ref().is_some_and(|re| re.is_match(&base))
            && info.shape.len() == 2
            && info.shape[0] >= 8
            && let Some(gs) = best_gs(info.shape[1])
        {
            decision = Ok(gs);
        }
        if decision.is_ok() {
            let m = info.shape[0].min(info.shape[1]);
            if args.min_gemm > 0 && m < args.min_gemm {
                decision = Err("below-min-gemm");
            }
        }

        match decision {
            Ok(gs) => {
                let (out, in_) = (info.shape[0], info.shape[1]);
                qparams += (out * in_) as u64;
                let e = by_pat.entry(pattern(&base)).or_insert((0, [out, in_], gs));
                e.0 += 1;
                *gsdist.entry(gs).or_default() += 1;
                let cq = comfy_quant_json(gs);
                out_tensors.push(OutputTensor {
                    key: key.clone(),
                    dtype: Dtype::I8,
                    shape: vec![out, in_],
                    nbytes: out * in_,
                });
                out_tensors.push(OutputTensor {
                    key: format!("{base}.weight_scale"),
                    dtype: Dtype::F32,
                    shape: vec![out, 1],
                    nbytes: out * 4,
                });
                out_tensors.push(OutputTensor {
                    key: format!("{base}.comfy_quant"),
                    dtype: Dtype::U8,
                    shape: vec![cq.len()],
                    nbytes: cq.len(),
                });
                actions.push(Action::Quantize { key: key.clone(), base, out, in_, gs });
            }
            Err(reason) => {
                *skip.entry(reason).or_default() += 1;
                let downcast = args.downcast_fp32
                    && info.dtype == Dtype::F32
                    && !base.ends_with(".scale")
                    && !base.rsplit('.').next().is_some_and(|s| deny.is_match(s));
                if downcast {
                    out_tensors.push(OutputTensor {
                        key: key.clone(),
                        dtype: target,
                        shape: info.shape.clone(),
                        nbytes: info.numel() * target.element_size(),
                    });
                    actions.push(Action::Downcast { key: key.clone(), dtype: target });
                } else {
                    out_tensors.push(OutputTensor {
                        key: key.clone(),
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        nbytes: info.end - info.begin,
                    });
                    actions.push(Action::Passthrough { key: key.clone() });
                }
            }
        }
    }

    // ---- report ----
    let n_quant = actions.iter().filter(|a| matches!(a, Action::Quantize { .. })).count();
    println!("SRC {}", args.src.display());
    println!("compute/passthrough dtype: {}", target.tag());
    println!("\nQUANTIZE {n_quant} layers (int8+convrot, absmax):");
    for (pat, (c, shape, gs)) in &by_pat {
        println!("  x{c:<4} gs{gs:<3} {:<16} {pat}", format!("{shape:?}"));
    }
    println!(
        "  groupsizes: {gsdist:?}   quantized params: {:.2}B (~{:.1} GB int8)",
        qparams as f64 / 1e9,
        qparams as f64 / 1e9,
    );
    let n_skip: usize = skip.values().sum();
    println!("\nLEAVE AS-IS ({n_skip} weights):");
    let mut skip_sorted: Vec<_> = skip.iter().collect();
    skip_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, c) in skip_sorted {
        println!("  x{c:<4} {reason}");
    }
    if args.dry_run {
        println!("\n[dry-run] nothing written.");
        return Ok(());
    }

    // ---- execute ----
    let mut hadamard_cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut metadata = src.metadata().clone();
    metadata.insert("quantized_from".to_string(), args.src.display().to_string());
    metadata.insert("quant_format".to_string(), "int8_convrot".to_string());

    let mut writer = StreamWriter::begin(&dst, out_tensors, &metadata)?;
    // (relerr, cosine, gs, base) per quantized layer, for the error report.
    let mut errs: Vec<(f64, f64, usize, String)> = Vec::new();
    let mut nq = 0usize;

    for action in &actions {
        match action {
            Action::Passthrough { key } => {
                writer.write_tensor(key, src.raw_bytes(key)?)?;
            }
            Action::Downcast { key, dtype } => {
                let vals = src.to_f32(key)?;
                writer.write_tensor(key, &f32_to_bytes(&vals, *dtype)?)?;
            }
            Action::Quantize { key, base, out, in_, gs } => {
                let h = hadamard_cache.entry(*gs).or_insert_with(|| build_hadamard(*gs));
                let w = src.to_f32(key)?;
                let r = quantize_convrot(&w, *out, *in_, *gs, h);
                let (relerr, cos) = (r.relerr(), r.cosine());
                if cos <= 0.99 {
                    bail!(
                        "BROKEN quant (rotation/format?) {base} cos={cos:.5} relerr={relerr:.2}%"
                    );
                }
                if relerr > args.warn_thresh as f64 {
                    println!("  WARN high error: {base} gs={gs} relerr={relerr:.2}% cos={cos:.5}");
                }
                errs.push((relerr, cos, *gs, base.clone()));

                let qbytes: Vec<u8> = r.qdata.iter().map(|&v| v as u8).collect();
                writer.write_tensor(key, &qbytes)?;
                writer.write_tensor(&format!("{base}.weight_scale"), &f32_to_bytes(&r.scale, Dtype::F32)?)?;
                writer.write_tensor(&format!("{base}.comfy_quant"), &comfy_quant_json(*gs))?;

                nq += 1;
                if nq.is_multiple_of(100) {
                    println!("  {nq}/{n_quant} ... {base} gs={gs} relerr={relerr:.2}% cos={cos:.5}");
                }
            }
        }
    }
    writer.finish()?;
    println!("DONE: quantized {nq} layers -> {}", dst.display());

    // ---- per-layer error report ----
    if !errs.is_empty() {
        errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // worst relerr first
        let rvals: Vec<f64> = errs.iter().map(|e| e.0).collect();
        let mean = rvals.iter().sum::<f64>() / rvals.len() as f64;
        let (min, max) = (
            rvals.iter().cloned().fold(f64::INFINITY, f64::min),
            rvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        let mut per_gs: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for (r, _, gs, _) in &errs {
            per_gs.entry(*gs).or_default().push(*r);
        }
        println!("\n=== quant error (relerr = ||dequant-source|| / ||source||) ===");
        println!("  mean {mean:.3}%   min {min:.3}%   max {max:.3}%   layers {}", errs.len());
        let per_gs_str: Vec<String> = per_gs
            .iter()
            .map(|(gs, v)| {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                format!("gs{gs}: mean {m:.3}% max {mx:.3}% (x{})", v.len())
            })
            .collect();
        println!("  per groupsize: {}", per_gs_str.join("  "));
        println!("  worst 8 layers:");
        for (r, c, gs, b) in errs.iter().take(8) {
            println!("    {r:6.3}%  cos {c:.5}  gs{gs:<3} {b}");
        }
        let over = errs.iter().filter(|e| e.0 > args.warn_thresh as f64).count();
        if over > 0 {
            println!("  !! {over} layer(s) over --warn-thresh ({}%) — review above", args.warn_thresh);
        }
        if let Some(path) = &args.verify_report {
            use std::io::Write;
            let mut f = std::fs::File::create(path)?;
            writeln!(f, "relerr_pct\tcosine\tgroupsize\tlayer")?;
            for (r, c, gs, b) in &errs {
                writeln!(f, "{r:.4}\t{c:.6}\t{gs}\t{b}")?;
            }
            println!("  full per-layer table -> {}", path.display());
        }
    }
    Ok(())
}
