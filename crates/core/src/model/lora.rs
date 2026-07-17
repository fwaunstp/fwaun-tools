//! Extract a low-rank LoRA from a full fine-tune, by SVD of the weight delta.
//!
//! For every 2D linear weight present in both the base and the fine-tune, we form
//! the task-vector delta (the same `tuned - base` used by `merge-diff`) and
//! approximate it with a rank-`r` factorization:
//!
//! ```text
//! ΔW = W_tuned - W_base  ≈  U_r · S_r · V_rᵀ
//! lora_up   (B) = U_r · √S_r          shape [out, r]
//! lora_down (A) = √S_r · V_rᵀ         shape [r, in]
//! ```
//!
//! At load time a LoRA reconstructs `ΔW ≈ up @ down · (alpha / dim)`. We store
//! `alpha = dim = r` per module so that, at multiplier 1, `up @ down` reproduces
//! the truncated delta exactly (`--alpha` rescales the factors if a different
//! nominal alpha is wanted). The rank-`r` truncation is inherently lossy — a
//! full fine-tune is generally full-rank — so higher ranks track the fine-tune
//! more faithfully at the cost of a larger file. The per-module *energy captured*
//! (Σσ²_kept / ‖ΔW‖²_F) is reported so you can tell whether the rank is enough.
//!
//! Keys are emitted in the kohya-ss / ComfyUI convention used by FLUX-family
//! LoRAs on Civitai: the bare DiT module path with `.` → `_` under a `lora_unet_`
//! prefix, e.g. `double_blocks.0.img_attn.qkv` → `lora_unet_double_blocks_0_img_attn_qkv`.
//!
//! The heavy linear algebra runs on CPU in f32, parallelized across cores with
//! rayon, one module at a time so peak RAM stays near a single weight matrix.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use regex::Regex;

use super::merge::ModelArch;
use super::progress::ProgressSink;
use super::safetensors::{Dtype, OutputTensor, SafeTensorsFile, StreamWriter, f32_to_bytes};

/// Parsed arguments for the `extract-lora` subcommand.
pub struct ExtractArgs {
    pub base: PathBuf,
    pub tuned: PathBuf,
    pub output: PathBuf,
    pub rank: usize,
    /// Nominal alpha to store. `None` means "use each module's own rank" (scale 1).
    pub alpha: Option<f32>,
    pub save_dtype: Dtype,
    pub arch: ModelArch,
    pub include: Option<String>,
    pub exclude: Option<String>,
    /// Power iterations in the randomized range finder (higher = more accurate, slower).
    pub niter: usize,
    /// Oversampling added to the rank before projection (improves accuracy).
    pub oversample: usize,
}

/// A module selected for extraction: its bare path, kohya name, and dimensions.
struct Module {
    /// Actual key in the base file (e.g. `model.diffusion_model.…weight`).
    base_key: String,
    /// Actual key in the tuned file.
    tuned_key: String,
    /// kohya module name, e.g. `lora_unet_double_blocks_0_img_attn_qkv`.
    lora_name: String,
    out: usize,
    in_: usize,
    /// Effective rank, `min(rank, out, in)`.
    r_eff: usize,
}

fn looks_fp8_scaled(f: &SafeTensorsFile) -> bool {
    f.keys().any(|k| {
        k.ends_with("_scale") || k.ends_with(".scale_weight") || k.ends_with(".weight_scale")
    })
}

/// Map a bare DiT weight path (`double_blocks.0.img_attn.qkv.weight`) to the kohya
/// LoRA module name (`lora_unet_double_blocks_0_img_attn_qkv`).
fn kohya_name(bare_weight_key: &str) -> String {
    let module = bare_weight_key.strip_suffix(".weight").unwrap_or(bare_weight_key);
    format!("lora_unet_{}", module.replace('.', "_"))
}

pub fn run(args: ExtractArgs, p: &mut dyn ProgressSink) -> Result<()> {
    p.log(&format!("base   (org) : {}", args.base.display()));
    p.log(&format!("tuned  (ft)  : {}", args.tuned.display()));
    p.log(&format!("output       : {}", args.output.display()));
    p.log(&format!("rank         : {}", args.rank));
    p.log(&format!(
        "alpha        : {}",
        args.alpha.map(|a| a.to_string()).unwrap_or_else(|| "per-module (= rank)".into())
    ));
    p.log(&format!("save dtype   : {}", args.save_dtype.tag()));
    p.log(&format!("model        : {:?}", args.arch));

    if args.rank == 0 {
        bail!("--rank must be >= 1");
    }

    let base = SafeTensorsFile::open(&args.base)?;
    let tuned = SafeTensorsFile::open(&args.tuned)?;
    if looks_fp8_scaled(&base) || looks_fp8_scaled(&tuned) {
        bail!(
            "base/tuned look like fp8_scaled checkpoints (have *_scale keys). Extraction needs a \
             bf16/fp16/fp32 base + fine-tune so a plain delta is meaningful."
        );
    }

    let include = args.include.as_deref().map(Regex::new).transpose().context("bad --include regex")?;
    let exclude = args.exclude.as_deref().map(Regex::new).transpose().context("bad --exclude regex")?;

    // Bare-key -> actual key, for both sides (prefix-normalized so differently
    // namespaced checkpoints line up, exactly as merge-diff does).
    let base_norm: BTreeMap<&str, &String> =
        base.keys().map(|k| (args.arch.strip_prefix_pub(k), k)).collect();
    let tuned_norm: BTreeMap<&str, &String> =
        tuned.keys().map(|k| (args.arch.strip_prefix_pub(k), k)).collect();

    // Select every 2D float `.weight` present in both, with matching shapes.
    let mut modules: Vec<Module> = Vec::new();
    let mut skipped_shape = 0usize;
    for (bare, bkey) in &base_norm {
        if !bare.ends_with(".weight") {
            continue;
        }
        let Some(tkey) = tuned_norm.get(bare) else { continue };
        let binfo = base.info(bkey).unwrap();
        let tinfo = tuned.info(tkey).unwrap();
        if !binfo.dtype.is_float() || binfo.shape.len() != 2 {
            continue;
        }
        if binfo.shape != tinfo.shape {
            skipped_shape += 1;
            continue;
        }
        if let Some(re) = &include
            && !re.is_match(bare)
        {
            continue;
        }
        if let Some(re) = &exclude
            && re.is_match(bare)
        {
            continue;
        }
        let out = binfo.shape[0];
        let in_ = binfo.shape[1];
        modules.push(Module {
            base_key: (*bkey).clone(),
            tuned_key: (*tkey).clone(),
            lora_name: kohya_name(bare),
            out,
            in_,
            r_eff: args.rank.min(out).min(in_),
        });
    }
    // Deterministic order (also keeps the streamed output stable run-to-run).
    modules.sort_by(|a, b| a.lora_name.cmp(&b.lora_name));

    if modules.is_empty() {
        bail!("no matching 2D weight tensors found in both base and tuned (check --model / --include)");
    }
    if skipped_shape > 0 {
        p.log(&format!(
            "warning: {skipped_shape} shared 2D weights had mismatched shapes and were skipped."
        ));
    }
    p.log(&format!(
        "selected {} linear modules for extraction",
        modules.len()
    ));

    // Plan the output: three tensors per module (down, up, alpha), in a fixed order.
    let mut plan: Vec<OutputTensor> = Vec::with_capacity(modules.len() * 3);
    for m in &modules {
        let dsz = args.save_dtype.element_size();
        plan.push(OutputTensor {
            key: format!("{}.lora_down.weight", m.lora_name),
            dtype: args.save_dtype,
            shape: vec![m.r_eff, m.in_],
            nbytes: m.r_eff * m.in_ * dsz,
        });
        plan.push(OutputTensor {
            key: format!("{}.lora_up.weight", m.lora_name),
            dtype: args.save_dtype,
            shape: vec![m.out, m.r_eff],
            nbytes: m.out * m.r_eff * dsz,
        });
        plan.push(OutputTensor {
            key: format!("{}.alpha", m.lora_name),
            dtype: args.save_dtype,
            shape: vec![],
            nbytes: dsz,
        });
    }

    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    metadata.insert("ss_network_module".into(), "networks.lora".into());
    metadata.insert("ss_network_dim".into(), args.rank.to_string());
    metadata.insert(
        "ss_network_alpha".into(),
        args.alpha.map(|a| a.to_string()).unwrap_or_else(|| args.rank.to_string()),
    );
    metadata.insert("lora_extracted_from_base".into(), args.base.display().to_string());
    metadata.insert("lora_extracted_from_tuned".into(), args.tuned.display().to_string());

    let mut writer = StreamWriter::begin(&args.output, plan, &metadata)?;

    let mut min_energy = f32::INFINITY;
    let mut sum_energy = 0.0f64;
    let mut worst_name = String::new();

    for (idx, m) in modules.iter().enumerate() {
        // ΔW = tuned - base, reusing tuned's buffer. Also track ‖ΔW‖²_F for the
        // energy metric.
        let mut delta = tuned.to_f32(&m.tuned_key)?;
        let b = base.to_f32(&m.base_key)?;
        debug_assert_eq!(delta.len(), b.len());
        let mut fro2 = 0.0f64;
        for i in 0..delta.len() {
            delta[i] -= b[i];
            fro2 += (delta[i] as f64) * (delta[i] as f64);
        }
        drop(b);

        // Vary the random projection per module but keep it reproducible.
        let seed = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let svd = randomized_svd(&delta, m.out, m.in_, m.r_eff, args.oversample, args.niter, seed);
        drop(delta);

        // Energy captured by the retained singular values.
        let kept: f64 = svd.s.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let energy = if fro2 > 0.0 { (kept / fro2) as f32 } else { 1.0 };
        if energy < min_energy {
            min_energy = energy;
            worst_name = m.lora_name.clone();
        }
        sum_energy += energy as f64;

        // Fold √S into up/down, optionally rescaling to hit a user-fixed alpha.
        // up @ down · (alpha/dim) must equal the truncated ΔW. With alpha = dim
        // that is up @ down = ΔW_r (factors carry √S). For a fixed alpha `a`,
        // scale both factors by √(dim/a) so the runtime `a/dim` cancels it.
        let dim = m.r_eff as f32;
        let alpha_val = args.alpha.unwrap_or(dim);
        let rescale = (dim / alpha_val).sqrt();

        let (up, down) = fold_factors(&svd, m.out, m.in_, rescale);

        writer.write_tensor(
            &format!("{}.lora_down.weight", m.lora_name),
            &f32_to_bytes(&down, args.save_dtype)?,
        )?;
        writer.write_tensor(
            &format!("{}.lora_up.weight", m.lora_name),
            &f32_to_bytes(&up, args.save_dtype)?,
        )?;
        writer.write_tensor(
            &format!("{}.alpha", m.lora_name),
            &f32_to_bytes(&[alpha_val], args.save_dtype)?,
        )?;

        p.tick(idx + 1, modules.len());
        if (idx + 1) % 32 == 0 || idx + 1 == modules.len() {
            p.log(&format!("  extracted {}/{} modules", idx + 1, modules.len()));
        }
    }

    writer.finish()?;

    let mean_energy = sum_energy / modules.len() as f64;
    p.log(&format!(
        "energy captured: mean {:.1}%, min {:.1}% ({})",
        mean_energy * 100.0,
        min_energy * 100.0,
        worst_name
    ));
    if min_energy < 0.90 {
        p.log(&format!(
            "note: some modules capture <90% of their delta at rank {}. Raise --rank for a closer \
             match to the fine-tune (at the cost of a larger LoRA).",
            args.rank
        ));
    }
    p.log(&format!(
        "wrote {} tensors to {}",
        modules.len() * 3,
        args.output.display()
    ));
    p.log("done.");
    Ok(())
}

/// Result of a randomized SVD: `u` is [m, r], `vh` is [r, n], `s` holds the r
/// singular values (unfolded).
struct Svd {
    u: Vec<f32>,
    s: Vec<f32>,
    vh: Vec<f32>,
    r: usize,
}

/// Build LoRA `up` [out, r] and `down` [r, in] from an SVD, folding √S into each
/// factor and applying `rescale` (for a user-fixed alpha; 1.0 for the default).
fn fold_factors(svd: &Svd, out: usize, in_: usize, rescale: f32) -> (Vec<f32>, Vec<f32>) {
    let r = svd.r;
    let sqrt_s: Vec<f32> = svd.s.iter().map(|&x| x.max(0.0).sqrt()).collect();

    let mut up = vec![0f32; out * r];
    up.par_chunks_mut(r).enumerate().for_each(|(x, o)| {
        let urow = &svd.u[x * r..x * r + r];
        for i in 0..r {
            o[i] = urow[i] * sqrt_s[i] * rescale;
        }
    });

    let mut down = vec![0f32; r * in_];
    down.par_chunks_mut(in_).enumerate().for_each(|(i, o)| {
        let f = sqrt_s[i] * rescale;
        let vrow = &svd.vh[i * in_..i * in_ + in_];
        for j in 0..in_ {
            o[j] = f * vrow[j];
        }
    });

    (up, down)
}

/// Randomized SVD (Halko, Martinsson & Tropp 2011) with `niter` power iterations,
/// returning the top-`rank` singular triplets of the m×n matrix `a` (row-major).
fn randomized_svd(
    a: &[f32],
    m: usize,
    n: usize,
    rank: usize,
    oversample: usize,
    niter: usize,
    seed: u64,
) -> Svd {
    let r = rank.min(m).min(n);
    let k = (rank + oversample).min(m).min(n);

    // Y = A Ω, with Ω a random n×k Gaussian; Q spans the sampled column space.
    let omega = gaussian(n * k, seed);
    let mut q = mgs_qr(mm_ab(a, m, n, &omega, k), m, k); // [m, k]

    // Subspace (power) iteration: sharpen the range toward A's dominant left
    // singular space, re-orthonormalizing every half-step so the smaller
    // singular directions survive in f32 (a plain (AAᵀ)^q product would collapse
    // them into the top one).
    for _ in 0..niter {
        let zt = mm_atb(a, m, n, &q, k); // Aᵀ Q  -> [n, k]
        let qz = mgs_qr(zt, n, k);
        let y = mm_ab(a, m, n, &qz, k); // A Qz  -> [m, k]
        q = mgs_qr(y, m, k);
    }

    let bmat = mm_qta(&q, m, k, a, n); // B = Qᵀ A, [k, n]

    // SVD of the small B via the eigendecomposition of G = B Bᵀ (k×k). The left
    // singular vectors of B are the eigenvectors of G; σ = √λ.
    let mut g = vec![0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut acc = 0.0f64;
            for t in 0..n {
                acc += bmat[i * n + t] as f64 * bmat[j * n + t] as f64;
            }
            g[i * k + j] = acc;
            g[j * k + i] = acc;
        }
    }
    let (evals, evecs) = jacobi_eigh(&g, k); // evals desc; evecs columns

    // Assemble the top-r triplets. U = Q · Ũ, Vh = Σ⁻¹ Ũᵀ B.
    let mut u = vec![0f32; m * r];
    let mut vh = vec![0f32; r * n];
    let mut s = vec![0f32; r];
    for i in 0..r {
        let sigma = evals[i].max(0.0).sqrt();
        s[i] = sigma as f32;
        // Eigenvector i (column i of `evecs`, stored row-major).
        let ecol: Vec<f64> = (0..k).map(|t| evecs[t * k + i]).collect();

        // U[:, i] = Q · ecol
        for x in 0..m {
            let mut acc = 0.0f64;
            let qrow = &q[x * k..x * k + k];
            for t in 0..k {
                acc += qrow[t] as f64 * ecol[t];
            }
            u[x * r + i] = acc as f32;
        }

        // Vh[i, :] = (1/σ) · ecolᵀ B  (zero row if σ underflows).
        if sigma > 1e-12 {
            let inv = 1.0 / sigma;
            let row = &mut vh[i * n..i * n + n];
            for t in 0..k {
                let e = ecol[t] * inv;
                let brow = &bmat[t * n..t * n + n];
                for j in 0..n {
                    row[j] += (e * brow[j] as f64) as f32;
                }
            }
        }
    }

    Svd { u, s, vh, r }
}

/// C = A · B, with A [m×n] and B [n×k], all row-major. Parallel over rows of C.
fn mm_ab(a: &[f32], m: usize, n: usize, b: &[f32], k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * k];
    c.par_chunks_mut(k).enumerate().for_each(|(row, o)| {
        let arow = &a[row * n..row * n + n];
        for t in 0..n {
            let av = arow[t];
            let brow = &b[t * k..t * k + k];
            for j in 0..k {
                o[j] += av * brow[j];
            }
        }
    });
    c
}

/// C = Aᵀ · Y, with A [m×n] and Y [m×k]; result [n×k]. Parallel over rows of C.
fn mm_atb(a: &[f32], m: usize, n: usize, y: &[f32], k: usize) -> Vec<f32> {
    let mut c = vec![0f32; n * k];
    c.par_chunks_mut(k).enumerate().for_each(|(i, o)| {
        for row in 0..m {
            let av = a[row * n + i];
            let yrow = &y[row * k..row * k + k];
            for j in 0..k {
                o[j] += av * yrow[j];
            }
        }
    });
    c
}

/// C = Qᵀ · A, with Q [m×k] and A [m×n]; result [k×n]. Parallel over rows of C.
fn mm_qta(q: &[f32], m: usize, k: usize, a: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0f32; k * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, o)| {
        for row in 0..m {
            let qv = q[row * k + i];
            let arow = &a[row * n..row * n + n];
            for j in 0..n {
                o[j] += qv * arow[j];
            }
        }
    });
    c
}

/// Thin QR by Gram–Schmidt with reorthogonalization: orthonormalize the k columns
/// of Y [m×k], returning Q [m×k] with orthonormal columns.
///
/// When the input is rank-deficient (as `A Ω` is when the rank we ask for exceeds
/// the delta's true rank), the surplus columns collapse to f32 noise under
/// orthogonalization. Normalizing that noise would produce spurious, non-orthogonal
/// unit vectors, so such columns are zeroed using a *relative* tolerance against
/// each column's original norm. The projection pass is run twice ("twice is
/// enough") so the retained columns stay orthonormal to near machine precision.
fn mgs_qr(mut q: Vec<f32>, m: usize, k: usize) -> Vec<f32> {
    let col_norm = |q: &[f32], j: usize| -> f64 {
        let mut s = 0.0f64;
        for x in 0..m {
            let v = q[x * k + j] as f64;
            s += v * v;
        }
        s.sqrt()
    };

    for j in 0..k {
        let n0 = col_norm(&q, j);
        // Two passes of projection removal against the already-orthonormal columns.
        for _pass in 0..2 {
            for i in 0..j {
                let mut dot = 0.0f64;
                for x in 0..m {
                    dot += q[x * k + i] as f64 * q[x * k + j] as f64;
                }
                let dot = dot as f32;
                for x in 0..m {
                    q[x * k + j] -= dot * q[x * k + i];
                }
            }
        }
        let nrm = col_norm(&q, j);
        // Keep only if a real independent direction survived (relative to n0).
        if nrm > 1e-6 * n0.max(1e-30) && nrm > 1e-20 {
            let inv = (1.0 / nrm) as f32;
            for x in 0..m {
                q[x * k + j] *= inv;
            }
        } else {
            for x in 0..m {
                q[x * k + j] = 0.0;
            }
        }
    }
    q
}

/// Cyclic Jacobi eigendecomposition of a symmetric k×k matrix (row-major, f64).
/// Returns `(evals, evecs)` with eigenvalues in descending order and eigenvector
/// `i` stored as column `i` of the row-major `evecs`.
fn jacobi_eigh(g: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = g.to_vec();
    let mut v = vec![0f64; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }

    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[p * k + q] * a[p * k + q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * k + p];
                let aqq = a[q * k + q];
                let phi = 0.5 * (aqq - app) / apq;
                let t = if phi == 0.0 {
                    1.0
                } else {
                    let sign = if phi >= 0.0 { 1.0 } else { -1.0 };
                    sign / (phi.abs() + (phi * phi + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate columns p,q then rows p,q of A, and columns p,q of V.
                for i in 0..k {
                    let aip = a[i * k + p];
                    let aiq = a[i * k + q];
                    a[i * k + p] = c * aip - s * aiq;
                    a[i * k + q] = s * aip + c * aiq;
                }
                for i in 0..k {
                    let api = a[p * k + i];
                    let aqi = a[q * k + i];
                    a[p * k + i] = c * api - s * aqi;
                    a[q * k + i] = s * api + c * aqi;
                }
                for i in 0..k {
                    let vip = v[i * k + p];
                    let viq = v[i * k + q];
                    v[i * k + p] = c * vip - s * viq;
                    v[i * k + q] = s * vip + c * viq;
                }
            }
        }
    }

    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&i, &j| a[j * k + j].partial_cmp(&a[i * k + i]).unwrap());

    let evals: Vec<f64> = idx.iter().map(|&i| a[i * k + i]).collect();
    let mut evecs = vec![0f64; k * k];
    for (new_col, &old_col) in idx.iter().enumerate() {
        for t in 0..k {
            evecs[t * k + new_col] = v[t * k + old_col];
        }
    }
    (evals, evecs)
}

/// A tiny reproducible xorshift64* stream of standard-normal f32 values.
fn gaussian(count: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 0xDEAD_BEEF_CAFE_F00D } else { seed };
    let mut next_unit = || {
        // xorshift64* -> (0, 1].
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40; // top 24 bits
        (bits as f32 + 1.0) / (1u64 << 24) as f32
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        // Box–Muller: two independent normals per pair of uniforms.
        let u1 = next_unit();
        let u2 = next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        out.push(radius * (std::f32::consts::TAU * u2).cos());
        if out.len() < count {
            out.push(radius * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_reconstructs_symmetric() {
        // G = M Mᵀ for a small random M, then check V Λ Vᵀ ≈ G and orthonormal V.
        let k = 5;
        let m = gaussian(k * k, 3);
        let mut g = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0f64;
                for t in 0..k {
                    acc += m[i * k + t] as f64 * m[j * k + t] as f64;
                }
                g[i * k + j] = acc;
            }
        }
        let (evals, evecs) = jacobi_eigh(&g, k);
        // Reconstruct G' = Σ_i λ_i v_i v_iᵀ (v_i = column i of evecs).
        let mut max_err = 0.0f64;
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += evals[i] * evecs[a * k + i] * evecs[b * k + i];
                }
                max_err = max_err.max((acc - g[a * k + b]).abs());
            }
        }
        assert!(max_err < 1e-9, "jacobi eigenvectors wrong: {max_err}");
    }

    /// A rank-`true_rank` matrix must be reconstructed near-exactly when the
    /// requested rank exceeds it — including when many surplus projection columns
    /// are rank-deficient (the case that requires robust QR).
    #[test]
    fn extracts_low_rank_delta() {
        // Many surplus columns: extract rank 16 (+8 oversample = 24 columns) from
        // a true-rank-8 matrix, so 16 of the 24 sampled columns are dependent.
        let (m, n, true_rank) = (96usize, 64usize, 8usize);
        let p = gaussian(m * true_rank, 11);
        let q = gaussian(n * true_rank, 29);
        let mut a = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..true_rank {
                    acc += p[i * true_rank + t] * q[j * true_rank + t];
                }
                a[i * n + j] = acc * 0.05; // fine-tune-sized delta magnitude
            }
        }

        let svd = randomized_svd(&a, m, n, 16, 8, 2, 1);
        let (up, down) = fold_factors(&svd, m, n, 1.0); // up[m,r], down[r,n]
        let r = svd.r;

        // The retained singular values must account for ~all of ‖A‖²_F.
        let fro2: f64 = a.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let kept: f64 = svd.s.iter().map(|&x| (x as f64) * (x as f64)).sum();
        assert!((kept / fro2 - 1.0).abs() < 1e-3, "energy = {}", kept / fro2);

        // Reconstruct up @ down and compare to A.
        let mut max_err = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..r {
                    acc += up[i * r + t] * down[t * n + j];
                }
                max_err = max_err.max((acc - a[i * n + j]).abs());
            }
        }
        let scale = a.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
        assert!(max_err < 1e-3 * scale.max(1.0), "max_err={max_err}, scale={scale}");
    }
}
