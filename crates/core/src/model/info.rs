//! Describe a safetensors checkpoint from its header: metadata, dtype mix,
//! quantization scheme, key prefix, and whether the file is complete.
//!
//! Only the header and the tiny `comfy_quant` config tensors are read, so this is
//! instant even for 20+ GB files. Shared by `fwaun-tools model info` and the GUI's
//! "Model info" mode; the formatting helpers here keep their output consistent.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use regex::Regex;
use serde::Serialize;

use super::quantized::{QuantSummary, summarize};
use super::safetensors::SafeTensorsFile;

/// Key prefixes the `model` tools normalize, with the `--model` value they imply.
const KNOWN_PREFIXES: [(&str, &str); 3] = [
    ("model.diffusion_model.", "krea2"),
    ("diffusion_model.", "krea2"),
    ("net.", "anima"),
];

/// Everything `model info` reports about one checkpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub path: String,
    pub file_size: u64,
    /// Bytes of the 8-byte length prefix + header JSON.
    pub header_size: u64,
    /// File size the header implies; larger than `file_size` when truncated.
    pub expected_size: u64,
    /// Tensors whose data lies (partly) past the end of the file.
    pub truncated_tensors: usize,
    pub metadata: BTreeMap<String, String>,
    pub tensor_count: usize,
    pub tensor_bytes: u64,
    /// dtype tag -> count and bytes.
    pub dtypes: BTreeMap<String, DtypeStat>,
    pub quantization: QuantSummary,
    pub key_prefix: KeyPrefix,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DtypeStat {
    pub count: usize,
    pub bytes: u64,
}

/// The dominant known DiT key prefix, if any.
#[derive(Debug, Clone, Serialize)]
pub struct KeyPrefix {
    /// e.g. `model.diffusion_model.`; `None` for bare keys.
    pub prefix: Option<String>,
    /// Keys carrying `prefix` (or, for bare keys, none of the known prefixes).
    pub count: usize,
    /// `--model` value for merge-diff / extract-lora (`auto` handles all of them).
    pub model_hint: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorEntry {
    pub key: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

impl ModelInfo {
    /// True when every tensor's data is present on disk.
    pub fn is_complete(&self) -> bool {
        self.truncated_tensors == 0 && self.file_size >= self.expected_size
    }

    /// Tensors whose key matches `filter` (all of them when `None`).
    pub fn tensors_matching<'a>(
        &'a self,
        filter: Option<&'a Regex>,
    ) -> impl Iterator<Item = &'a TensorEntry> {
        self.tensors
            .iter()
            .filter(move |t| filter.is_none_or(|re| re.is_match(&t.key)))
    }

    /// Pretty JSON for `--json` / the GUI's "Copy JSON". The tensor list is
    /// included only when `tensors` is `Some` (filtered by the regex inside).
    pub fn to_json(&self, tensors: Option<Option<&Regex>>) -> String {
        let mut v = serde_json::to_value(self).expect("ModelInfo serializes");
        match tensors {
            Some(filter) => {
                let list: Vec<&TensorEntry> = self.tensors_matching(filter).collect();
                v["tensors"] = serde_json::to_value(list).expect("tensors serialize");
            }
            None => {
                v.as_object_mut().unwrap().remove("tensors");
            }
        }
        serde_json::to_string_pretty(&v).expect("JSON value serializes")
    }
}

/// Read `path`'s header and build its [`ModelInfo`].
pub fn inspect(path: &Path) -> Result<ModelInfo> {
    let file = SafeTensorsFile::open(path)?;
    let file_size = file.file_len() as u64;
    let data_start = file.data_start() as u64;

    let mut dtypes: BTreeMap<String, DtypeStat> = BTreeMap::new();
    let mut tensors = Vec::new();
    let mut data_end = 0u64;
    let mut truncated_tensors = 0usize;
    for key in file.keys() {
        let info = file.info(key).unwrap();
        let bytes = (info.end - info.begin) as u64;
        data_end = data_end.max(info.end as u64);
        if data_start + info.end as u64 > file_size {
            truncated_tensors += 1;
        }
        let stat = dtypes.entry(info.dtype.tag().to_string()).or_default();
        stat.count += 1;
        stat.bytes += bytes;
        tensors.push(TensorEntry {
            key: key.clone(),
            dtype: info.dtype.tag().to_string(),
            shape: info.shape.clone(),
            bytes,
        });
    }

    Ok(ModelInfo {
        path: path.display().to_string(),
        file_size,
        header_size: data_start,
        expected_size: data_start + data_end,
        truncated_tensors,
        metadata: file.metadata().clone(),
        tensor_count: tensors.len(),
        tensor_bytes: tensors.iter().map(|t| t.bytes).sum(),
        dtypes,
        quantization: summarize(&file),
        key_prefix: detect_prefix(file.keys()),
        tensors,
    })
}

/// Pick the known prefix most keys carry; bare keys win when none dominates.
fn detect_prefix<'a>(keys: impl Iterator<Item = &'a String>) -> KeyPrefix {
    let mut counts = [0usize; KNOWN_PREFIXES.len()];
    let mut bare = 0usize;
    for key in keys {
        // Longest-first order means `model.diffusion_model.` is not also counted
        // as `diffusion_model.` (it doesn't start with it anyway).
        match KNOWN_PREFIXES.iter().position(|(p, _)| key.starts_with(p)) {
            Some(i) => counts[i] += 1,
            None => bare += 1,
        }
    }
    let (best, &count) = counts.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap();
    if count > bare {
        KeyPrefix {
            prefix: Some(KNOWN_PREFIXES[best].0.to_string()),
            count,
            model_hint: KNOWN_PREFIXES[best].1.to_string(),
        }
    } else {
        KeyPrefix {
            prefix: None,
            count: bare,
            model_hint: "auto".to_string(),
        }
    }
}

/// `7256783320` -> `6.76 GiB`.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[unit])
    }
}

/// A metadata value for display: JSON objects/arrays (ai-toolkit's
/// `training_info`, kohya's `ss_*` tables) pretty-printed, anything else as-is.
pub fn pretty_metadata_value(value: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(value) {
        Ok(v @ (serde_json::Value::Object(_) | serde_json::Value::Array(_))) => {
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| value.to_string())
        }
        _ => value.to_string(),
    }
}

/// One-line quantization description, e.g. `int8_convrot — 192 layers (gs256 ×192)`.
pub fn quant_line(q: &QuantSummary) -> String {
    let mut s = q.kind.clone();
    if q.int8_layers > 0 {
        s.push_str(&format!(" — {} int8 layers", q.int8_layers));
        if !q.convrot_groupsizes.is_empty() {
            let gs: Vec<String> = q
                .convrot_groupsizes
                .iter()
                .map(|(gs, n)| format!("gs{gs} ×{n}"))
                .collect();
            s.push_str(&format!(" ({})", gs.join(", ")));
        }
    }
    if let Some(m) = &q.fp8_marker {
        s.push_str(&format!(" — marker: {m}"));
    }
    s
}

/// One-line key-prefix description, e.g. `(none) — 649 keys; --model auto`.
pub fn prefix_line(p: &KeyPrefix) -> String {
    format!(
        "{} — {} keys; --model {}",
        p.prefix.as_deref().unwrap_or("(none)"),
        p.count,
        p.model_hint
    )
}

/// One-line completeness description.
pub fn status_line(info: &ModelInfo) -> String {
    if info.is_complete() {
        "complete".to_string()
    } else {
        format!(
            "TRUNCATED — {} missing, {} of {} tensors past end of file",
            human_bytes(info.expected_size.saturating_sub(info.file_size)),
            info.truncated_tensors,
            info.tensor_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::quantized::tests::{TestTensor, temp_dir, weights, write_model};

    #[test]
    fn describes_int8_convrot_checkpoint() {
        let dir = temp_dir("info");
        let path = dir.join("m.safetensors");
        write_model(
            &path,
            &[
                (
                    "model.diffusion_model.blocks.0.q.weight",
                    TestTensor::Int8(8, 64, 16, weights(8 * 64, 1)),
                ),
                (
                    "model.diffusion_model.norm.weight",
                    TestTensor::Float(vec![4], vec![1.0; 4]),
                ),
            ],
        );

        let info = inspect(&path).unwrap();
        assert!(info.is_complete());
        assert_eq!(info.tensor_count, 4);
        assert_eq!(info.dtypes["I8"].count, 1);
        assert_eq!(info.dtypes["F32"].count, 2);
        assert_eq!(info.quantization.kind, "int8_convrot");
        assert_eq!(info.quantization.convrot_groupsizes[&16], 1);
        assert_eq!(
            info.key_prefix.prefix.as_deref(),
            Some("model.diffusion_model.")
        );
        assert_eq!(info.key_prefix.model_hint, "krea2");

        let re = Regex::new("norm").unwrap();
        assert_eq!(info.tensors_matching(Some(&re)).count(), 1);
        let json: serde_json::Value = serde_json::from_str(&info.to_json(Some(Some(&re)))).unwrap();
        assert_eq!(json["tensors"].as_array().unwrap().len(), 1);
        let json: serde_json::Value = serde_json::from_str(&info.to_json(None)).unwrap();
        assert!(json.get("tensors").is_none());

        // Chop the file: info must still open it and report the truncation.
        let bytes = std::fs::read(&path).unwrap();
        let cut = dir.join("cut.safetensors");
        std::fs::write(&cut, &bytes[..bytes.len() - 100]).unwrap();
        let info = inspect(&cut).unwrap();
        assert!(!info.is_complete());
        assert!(info.truncated_tensors >= 1);
        assert!(status_line(&info).starts_with("TRUNCATED"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn formatting_helpers() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(7_256_783_320), "6.76 GiB");
        assert_eq!(pretty_metadata_value("pt"), "pt");
        assert_eq!(pretty_metadata_value("123"), "123");
        assert_eq!(
            pretty_metadata_value(r#"{"step": 4800}"#),
            "{\n  \"step\": 4800\n}"
        );
    }
}
