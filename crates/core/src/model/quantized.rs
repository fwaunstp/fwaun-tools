//! Read comfy-kitchen int8 checkpoints (the `quant-int8` / ai-toolkit
//! `int8_convrot` layout) as plain f32 tensors.
//!
//! A quantized linear is stored as three tensors:
//!
//! ```text
//! <layer>.weight       I8   [N, K]   (rotated when convrot is on)
//! <layer>.weight_scale F32  [N, 1]   per-row absmax scale (or a single value)
//! <layer>.comfy_quant  U8   JSON     {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
//! ```
//!
//! [`ModelFile`] hides that: `<layer>.weight` is exposed as one logical tensor
//! that [`ModelFile::to_f32`] dequantizes (`(q · scale) · H_blockdiag`), and the
//! companion keys are left out of [`ModelFile::keys`]. Everything else reads
//! exactly as it does from a [`SafeTensorsFile`], so `merge-diff` and
//! `extract-lora` can take int8 and float checkpoints alike.
//!
//! fp8_scaled checkpoints (and comfy_quant formats other than int8) are still
//! rejected at open time: their scales are not handled here.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

use super::quant::{build_hadamard, dequantize_int8};
use super::safetensors::{Dtype, SafeTensorsFile};

/// Suffixes stored next to a quantized `<layer>.weight`, as `<layer>.<suffix>`.
const COMPANION_SUFFIXES: [&str; 3] = ["weight_scale", "comfy_quant", "input_scale"];

/// Per-layer config of one int8 linear, parsed from its `comfy_quant` JSON.
#[derive(Debug, Clone)]
pub struct Int8Layer {
    /// ConvRot block-Hadamard group size; `None` for plain (unrotated) int8.
    pub convrot_gs: Option<usize>,
    /// The companion keys present in the file (`weight_scale`, `comfy_quant`, …).
    pub companions: Vec<String>,
}

/// A safetensors checkpoint whose int8 layers read back as f32.
pub struct ModelFile {
    file: SafeTensorsFile,
    /// `<layer>.weight` key -> its int8 config.
    int8: BTreeMap<String, Int8Layer>,
    /// Every companion key, so [`ModelFile::keys`] can skip them.
    companions: std::collections::BTreeSet<String>,
}

impl ModelFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = SafeTensorsFile::open(path)?;
        let mut int8 = BTreeMap::new();
        let mut companions = std::collections::BTreeSet::new();

        let cq_keys: Vec<String> = file
            .keys()
            .filter(|k| k.ends_with(".comfy_quant"))
            .cloned()
            .collect();
        for cq_key in cq_keys {
            let layer = cq_key.strip_suffix(".comfy_quant").unwrap();
            let weight_key = format!("{layer}.weight");
            let cfg = parse_comfy_quant(&file, &cq_key)
                .with_context(|| format!("{}: {cq_key}", path.display()))?;
            let convrot_gs = validate_layer(&file, layer, &weight_key, &cfg)
                .with_context(|| format!("{}: {layer}", path.display()))?;
            let present: Vec<String> = COMPANION_SUFFIXES
                .iter()
                .map(|s| format!("{layer}.{s}"))
                .filter(|k| file.info(k).is_some())
                .collect();
            companions.extend(present.iter().cloned());
            int8.insert(
                weight_key,
                Int8Layer {
                    convrot_gs,
                    companions: present,
                },
            );
        }

        if let Some(key) = fp8_scaled_marker(&file, &companions) {
            bail!(
                "{} looks like an fp8_scaled checkpoint ({key}). fp8 scales are not supported; \
                 use a bf16/fp16/fp32 or int8_convrot checkpoint.",
                path.display()
            );
        }

        Ok(Self {
            file,
            int8,
            companions,
        })
    }

    /// The underlying file, for raw byte copies and metadata.
    pub fn raw(&self) -> &SafeTensorsFile {
        &self.file
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        self.file.metadata()
    }

    /// Whether any layer is int8-quantized.
    pub fn is_quantized(&self) -> bool {
        !self.int8.is_empty()
    }

    /// Short label for logs: `int8_convrot`, `int8`, or `float`.
    pub fn quant_label(&self) -> &'static str {
        if self.int8.is_empty() {
            "float"
        } else if self.int8.values().any(|l| l.convrot_gs.is_some()) {
            "int8_convrot"
        } else {
            "int8"
        }
    }

    /// Logical tensor keys: every key except int8 companions.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.file.keys().filter(|k| !self.companions.contains(*k))
    }

    /// The int8 config if `key` is a quantized `<layer>.weight`.
    pub fn int8_layer(&self, key: &str) -> Option<&Int8Layer> {
        self.int8.get(key)
    }

    /// Whether `key` decodes to f32 values (a float tensor or an int8 weight).
    pub fn is_numeric(&self, key: &str) -> bool {
        self.int8.contains_key(key) || self.file.info(key).is_some_and(|i| i.dtype.is_float())
    }

    /// Logical shape of `key`.
    pub fn shape(&self, key: &str) -> Option<&[usize]> {
        self.file.info(key).map(|i| i.shape.as_slice())
    }

    /// Decode `key` into f32, dequantizing int8 weights.
    pub fn to_f32(&self, key: &str) -> Result<Vec<f32>> {
        let Some(layer) = self.int8.get(key) else {
            return self.file.to_f32(key);
        };
        let info = self.file.info(key).unwrap();
        let (out, in_) = (info.shape[0], info.shape[1]);
        let q: Vec<i8> = self.file.raw_bytes(key)?.iter().map(|&b| b as i8).collect();
        let scale_key = format!("{}_scale", key);
        let scale = self.file.to_f32(&scale_key)?;
        let h = layer.convrot_gs.map(build_hadamard);
        let rot = layer.convrot_gs.zip(h.as_deref());
        Ok(dequantize_int8(&q, &scale, out, in_, rot))
    }
}

/// Parse a `<layer>.comfy_quant` tensor (uint8 bytes of a JSON object).
fn parse_comfy_quant(file: &SafeTensorsFile, key: &str) -> Result<Value> {
    let bytes = file.raw_bytes(key)?;
    let v: Value = serde_json::from_slice(bytes).context("comfy_quant is not valid JSON")?;
    if !v.is_object() {
        bail!("comfy_quant is not a JSON object: {v}");
    }
    Ok(v)
}

/// Check one int8 layer's tensors against its config; returns the ConvRot group
/// size (`None` when the layer is not rotated).
fn validate_layer(
    file: &SafeTensorsFile,
    layer: &str,
    weight_key: &str,
    cfg: &Value,
) -> Result<Option<usize>> {
    let format = cfg.get("format").and_then(Value::as_str).unwrap_or("");
    if format != "int8_tensorwise" {
        bail!("unsupported comfy_quant format {cfg} (only int8_tensorwise is supported)");
    }
    let w = file
        .info(weight_key)
        .ok_or_else(|| anyhow!("has comfy_quant but no {weight_key}"))?;
    if w.dtype != Dtype::I8 || w.shape.len() != 2 {
        bail!(
            "expected a 2-D I8 weight, got {} {:?}",
            w.dtype.tag(),
            w.shape
        );
    }
    let (out, in_) = (w.shape[0], w.shape[1]);
    let scale_key = format!("{layer}.weight_scale");
    let s = file
        .info(&scale_key)
        .ok_or_else(|| anyhow!("missing {scale_key}"))?;
    if !s.dtype.is_float() || (s.numel() != out && s.numel() != 1) {
        bail!(
            "weight_scale must be a float with {out} or 1 values, got {} {:?}",
            s.dtype.tag(),
            s.shape
        );
    }

    let convrot = cfg.get("convrot").and_then(Value::as_bool).unwrap_or(false);
    if !convrot {
        return Ok(None);
    }
    let gs = cfg
        .get("convrot_groupsize")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("convrot without convrot_groupsize: {cfg}"))? as usize;
    let power_of_4 = gs >= 4 && gs.is_power_of_two() && gs.trailing_zeros().is_multiple_of(2);
    if !power_of_4 || !in_.is_multiple_of(gs) {
        bail!("convrot_groupsize {gs} must be a power of 4 dividing K={in_}");
    }
    Ok(Some(gs))
}

/// Header-level quantization summary of a checkpoint. Unlike [`ModelFile::open`]
/// it never fails: fp8_scaled, unknown comfy_quant formats, and unreadable
/// configs are reported instead of rejected, so `model info` can describe any file.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct QuantSummary {
    /// `float`, `int8_convrot`, `int8`, `fp8_scaled`, or `mixed`.
    pub kind: String,
    /// Layers whose comfy_quant format is int8_tensorwise.
    pub int8_layers: usize,
    /// ConvRot group size -> number of int8 layers using it.
    pub convrot_groupsizes: BTreeMap<usize, usize>,
    /// Distinct comfy_quant JSON bodies -> number of layers carrying each.
    pub comfy_quant: BTreeMap<String, usize>,
    /// The key that marks the file as fp8_scaled, if any.
    pub fp8_marker: Option<String>,
}

/// Classify a checkpoint's quantization from its header and comfy_quant configs.
pub fn summarize(file: &SafeTensorsFile) -> QuantSummary {
    let mut s = QuantSummary::default();
    let mut int8_companions = std::collections::BTreeSet::new();
    let mut unrotated = 0usize;
    let mut other_formats = 0usize;
    for key in file.keys().filter(|k| k.ends_with(".comfy_quant")) {
        let layer = key.strip_suffix(".comfy_quant").unwrap();
        let body = match file.raw_bytes(key) {
            Ok(b) => String::from_utf8_lossy(b).into_owned(),
            Err(_) => "<unreadable: data past end of file>".to_string(),
        };
        *s.comfy_quant.entry(body.clone()).or_default() += 1;
        let cfg: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        if cfg.get("format").and_then(Value::as_str) != Some("int8_tensorwise") {
            other_formats += 1;
            continue;
        }
        s.int8_layers += 1;
        int8_companions.extend(COMPANION_SUFFIXES.iter().map(|x| format!("{layer}.{x}")));
        let gs = cfg.get("convrot_groupsize").and_then(Value::as_u64);
        match (cfg.get("convrot").and_then(Value::as_bool), gs) {
            (Some(true), Some(gs)) => *s.convrot_groupsizes.entry(gs as usize).or_default() += 1,
            _ => unrotated += 1,
        }
    }
    s.fp8_marker = fp8_scaled_marker(file, &int8_companions);

    s.kind = match (s.int8_layers, other_formats, &s.fp8_marker) {
        (0, 0, None) => "float",
        (0, 0, Some(_)) => "fp8_scaled",
        (n, 0, None) if unrotated == 0 && n > 0 => "int8_convrot",
        (n, 0, None) if unrotated == n => "int8",
        _ => "mixed",
    }
    .to_string();
    s
}

/// A key that marks an fp8_scaled checkpoint, if any: an fp8 tensor, or a scale
/// that does not belong to an int8 layer.
fn fp8_scaled_marker(
    file: &SafeTensorsFile,
    int8_companions: &std::collections::BTreeSet<String>,
) -> Option<String> {
    file.keys()
        .find(|k| {
            let fp8 = matches!(
                file.info(k).map(|i| i.dtype),
                Some(Dtype::F8E4M3 | Dtype::F8E5M2)
            );
            let stray_scale = (k.ends_with("_scale") || k.ends_with(".scale_weight"))
                && !int8_companions.contains(*k);
            fp8 || stray_scale || k.as_str() == "scaled_fp8"
        })
        .cloned()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::quant::{comfy_quant_json, quantize_convrot};
    use crate::model::safetensors::{OutputTensor, StreamWriter, f32_to_bytes};

    /// A tensor to write into a test checkpoint.
    pub(crate) enum TestTensor {
        /// Float weight stored as f32.
        Float(Vec<usize>, Vec<f32>),
        /// 2-D weight quantized to int8 + ConvRot at group size `gs`.
        Int8(usize, usize, usize, Vec<f32>),
    }

    /// Write a checkpoint mixing f32 and int8_convrot tensors.
    pub(crate) fn write_model(path: &Path, tensors: &[(&str, TestTensor)]) {
        let mut plan = Vec::new();
        let mut blobs: Vec<(String, Vec<u8>)> = Vec::new();
        for (key, t) in tensors {
            match t {
                TestTensor::Float(shape, vals) => {
                    plan.push(OutputTensor {
                        key: key.to_string(),
                        dtype: Dtype::F32,
                        shape: shape.clone(),
                        nbytes: vals.len() * 4,
                    });
                    blobs.push((key.to_string(), f32_to_bytes(vals, Dtype::F32).unwrap()));
                }
                TestTensor::Int8(out, in_, gs, vals) => {
                    let layer = key.strip_suffix(".weight").unwrap();
                    let r = quantize_convrot(vals, *out, *in_, *gs, &build_hadamard(*gs));
                    let cq = comfy_quant_json(*gs);
                    let parts = [
                        (
                            key.to_string(),
                            Dtype::I8,
                            vec![*out, *in_],
                            r.qdata.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
                        ),
                        (
                            format!("{layer}.weight_scale"),
                            Dtype::F32,
                            vec![*out, 1],
                            f32_to_bytes(&r.scale, Dtype::F32).unwrap(),
                        ),
                        (
                            format!("{layer}.comfy_quant"),
                            Dtype::U8,
                            vec![cq.len()],
                            cq,
                        ),
                    ];
                    for (k, dtype, shape, bytes) in parts {
                        plan.push(OutputTensor {
                            key: k.clone(),
                            dtype,
                            shape,
                            nbytes: bytes.len(),
                        });
                        blobs.push((k, bytes));
                    }
                }
            }
        }
        let mut w = StreamWriter::begin(path, plan, &BTreeMap::new()).unwrap();
        for (k, b) in &blobs {
            w.write_tensor(k, b).unwrap();
        }
        w.finish().unwrap();
    }

    /// Deterministic pseudo-random weights in roughly [-1, 1).
    pub(crate) fn weights(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    pub(crate) fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fwaun-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn int8_convrot_round_trips_and_hides_companions() {
        let dir = temp_dir("quantized-roundtrip");
        let path = dir.join("m.safetensors");
        let (out, in_) = (8, 64);
        let w = weights(out * in_, 7);
        write_model(
            &path,
            &[
                (
                    "blocks.0.attn.q.weight",
                    TestTensor::Int8(out, in_, 16, w.clone()),
                ),
                (
                    "blocks.0.attn.q.bias",
                    TestTensor::Float(vec![out], vec![0.5; out]),
                ),
            ],
        );

        let m = ModelFile::open(&path).unwrap();
        assert!(m.is_quantized());
        assert_eq!(m.quant_label(), "int8_convrot");
        let keys: Vec<&String> = m.keys().collect();
        assert_eq!(keys, ["blocks.0.attn.q.bias", "blocks.0.attn.q.weight"]);
        assert!(m.is_numeric("blocks.0.attn.q.weight"));
        assert_eq!(
            m.int8_layer("blocks.0.attn.q.weight")
                .unwrap()
                .companions
                .len(),
            2
        );

        // Dequantized weight must match the source up to int8 rounding.
        let deq = m.to_f32("blocks.0.attn.q.weight").unwrap();
        let err: f32 = deq
            .iter()
            .zip(&w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.05, "max abs err {err}");
        assert_eq!(m.to_f32("blocks.0.attn.q.bias").unwrap(), vec![0.5; out]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fp8_scaled_is_still_rejected() {
        let dir = temp_dir("quantized-fp8");
        let path = dir.join("m.safetensors");
        write_model(
            &path,
            &[
                (
                    "blocks.0.q.weight",
                    TestTensor::Float(vec![2, 2], vec![1.0; 4]),
                ),
                (
                    "blocks.0.q.scale_weight",
                    TestTensor::Float(vec![], vec![1.0]),
                ),
            ],
        );
        let err = ModelFile::open(&path).err().unwrap().to_string();
        assert!(err.contains("fp8_scaled"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn summarize_classifies_without_failing() {
        let dir = temp_dir("quantized-summarize");
        let kind = |name: &str, tensors: &[(&str, TestTensor)]| {
            let path = dir.join(name);
            write_model(&path, tensors);
            summarize(&SafeTensorsFile::open(&path).unwrap())
        };

        let f = kind(
            "f.safetensors",
            &[("a.weight", TestTensor::Float(vec![2], vec![1.0; 2]))],
        );
        assert_eq!(f.kind, "float");

        let q = kind(
            "q.safetensors",
            &[(
                "blocks.0.q.weight",
                TestTensor::Int8(8, 64, 16, weights(8 * 64, 3)),
            )],
        );
        assert_eq!(q.kind, "int8_convrot");
        assert_eq!(q.int8_layers, 1);
        assert_eq!(q.convrot_groupsizes[&16], 1);
        assert!(q.fp8_marker.is_none());

        let fp8 = kind(
            "fp8.safetensors",
            &[
                (
                    "blocks.0.q.weight",
                    TestTensor::Float(vec![2, 2], vec![1.0; 4]),
                ),
                (
                    "blocks.0.q.scale_weight",
                    TestTensor::Float(vec![], vec![1.0]),
                ),
            ],
        );
        assert_eq!(fp8.kind, "fp8_scaled");
        assert_eq!(fp8.fp8_marker.as_deref(), Some("blocks.0.q.scale_weight"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
