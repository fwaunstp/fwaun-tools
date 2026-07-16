//! Minimal, memory-efficient safetensors I/O.
//!
//! Reading is done by memory-mapping the file and parsing the JSON header, so a
//! tensor's bytes are only paged in when accessed. Writing is streamed: the
//! header is computed and written first, then each tensor's bytes are appended
//! in order, so peak RAM stays at a few tensors rather than a whole model.
//!
//! The on-disk layout matches `safetensors` (and musubi-tuner's
//! `mem_eff_save_file`): an 8-byte little-endian header length, the header JSON
//! padded with spaces to a 256-byte boundary, then the tensor data blob.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use serde_json::Value;

/// Alignment used for the header JSON padding, matching musubi-tuner's writer.
const HEADER_ALIGN: usize = 256;

/// Tensor element type, mirroring the safetensors dtype tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
    F8E5M2,
    F8E4M3,
}

impl Dtype {
    pub fn from_tag(tag: &str) -> Result<Self> {
        Ok(match tag {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E4M3" => Dtype::F8E4M3,
            other => bail!("unsupported safetensors dtype tag: {other}"),
        })
    }

    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E4M3 => "F8_E4M3",
        }
    }

    pub fn element_size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool | Dtype::F8E5M2 | Dtype::F8E4M3 => 1,
        }
    }

    /// Whether task-vector arithmetic (done in f32) is meaningful for this dtype.
    /// fp8 is intentionally excluded: it carries per-tensor scales that a plain
    /// bf16 delta cannot be added to, so those checkpoints are rejected upstream.
    pub fn is_float(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::Bf16)
    }

    /// Parse a user-supplied `--save-dtype` value.
    pub fn parse_save_dtype(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Dtype::Bf16,
            "fp16" | "float16" => Dtype::F16,
            "fp32" | "float32" => Dtype::F32,
            other => bail!("unsupported save dtype: {other} (expected bf16, fp16, or fp32)"),
        })
    }
}

/// A single tensor's location and description within a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte offsets relative to the start of the data blob (i.e. after the header).
    pub begin: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        // Empty shape (a 0-d scalar) yields the empty product, 1, which is correct.
        self.shape.iter().product()
    }
}

/// A memory-mapped safetensors file: header parsed, tensor bytes on demand.
pub struct SafeTensorsFile {
    mmap: Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: we only read from the mapping; the file is not mutated elsewhere.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?;

        if mmap.len() < 8 {
            bail!("{}: too small to be a safetensors file", path.display());
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .filter(|&e| e <= mmap.len())
            .ok_or_else(|| anyhow!("{}: header length {header_len} exceeds file size", path.display()))?;

        let header: Value = serde_json::from_slice(&mmap[8..header_end])
            .with_context(|| format!("parsing header JSON of {}", path.display()))?;
        let obj = header
            .as_object()
            .ok_or_else(|| anyhow!("{}: header is not a JSON object", path.display()))?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (key, val) in obj {
            if key == "__metadata__" {
                if let Some(m) = val.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = Dtype::from_tag(
                val.get("dtype").and_then(Value::as_str).ok_or_else(|| anyhow!("{key}: missing dtype"))?,
            )?;
            let shape = val
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("{key}: missing shape"))?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| anyhow!("{key}: bad shape entry")))
                .collect::<Result<Vec<_>>>()?;
            let offsets = val
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("{key}: missing data_offsets"))?;
            if offsets.len() != 2 {
                bail!("{key}: data_offsets must have 2 entries");
            }
            let begin = offsets[0].as_u64().ok_or_else(|| anyhow!("{key}: bad offset"))? as usize;
            let end = offsets[1].as_u64().ok_or_else(|| anyhow!("{key}: bad offset"))? as usize;
            tensors.insert(key.clone(), TensorInfo { dtype, shape, begin, end });
        }

        Ok(Self { mmap, data_start: header_end, tensors, metadata })
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn info(&self, key: &str) -> Option<&TensorInfo> {
        self.tensors.get(key)
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Raw little-endian bytes for a tensor (a view into the mmap, no copy).
    pub fn raw_bytes(&self, key: &str) -> Result<&[u8]> {
        let info = self.tensors.get(key).ok_or_else(|| anyhow!("tensor '{key}' not found"))?;
        let start = self.data_start + info.begin;
        let end = self.data_start + info.end;
        self.mmap
            .get(start..end)
            .ok_or_else(|| anyhow!("tensor '{key}' data range out of bounds"))
    }

    /// Decode a float tensor into an f32 vector for arithmetic.
    pub fn to_f32(&self, key: &str) -> Result<Vec<f32>> {
        let info = self.tensors.get(key).ok_or_else(|| anyhow!("tensor '{key}' not found"))?;
        bytes_to_f32(self.raw_bytes(key)?, info.dtype)
    }
}

/// Decode raw little-endian bytes of a float dtype into f32 values.
pub fn bytes_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    let esz = dtype.element_size();
    if !bytes.len().is_multiple_of(esz) {
        bail!("byte length {} not a multiple of element size {esz}", bytes.len());
    }
    let n = bytes.len() / esz;
    let mut out = Vec::with_capacity(n);
    match dtype {
        Dtype::F32 => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        Dtype::F16 => {
            for c in bytes.chunks_exact(2) {
                out.push(f16::from_le_bytes(c.try_into().unwrap()).to_f32());
            }
        }
        Dtype::Bf16 => {
            for c in bytes.chunks_exact(2) {
                out.push(bf16::from_le_bytes(c.try_into().unwrap()).to_f32());
            }
        }
        Dtype::F64 => {
            for c in bytes.chunks_exact(8) {
                out.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        _ => bail!("dtype {} is not a float", dtype.tag()),
    }
    Ok(out)
}

/// Encode f32 values into little-endian bytes of the target float dtype.
pub fn f32_to_bytes(vals: &[f32], dtype: Dtype) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(vals.len() * dtype.element_size());
    match dtype {
        Dtype::F32 => {
            for &v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Dtype::F16 => {
            for &v in vals {
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        Dtype::Bf16 => {
            for &v in vals {
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        Dtype::F64 => {
            for &v in vals {
                out.extend_from_slice(&(v as f64).to_le_bytes());
            }
        }
        _ => bail!("dtype {} is not a float", dtype.tag()),
    }
    Ok(out)
}

/// A tensor slot in the output file: its dtype/shape/offsets are known up front,
/// its bytes are produced lazily during the streaming write.
pub struct OutputTensor {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub nbytes: usize,
}

/// Streaming safetensors writer.
///
/// Usage: build the full list of [`OutputTensor`] slots (with known dtype/shape),
/// call [`StreamWriter::begin`] to write the header, then feed each tensor's bytes
/// in the same order via [`StreamWriter::write_tensor`].
pub struct StreamWriter {
    inner: BufWriter<File>,
    plan: Vec<OutputTensor>,
    next: usize,
}

impl StreamWriter {
    pub fn begin(
        path: &Path,
        plan: Vec<OutputTensor>,
        metadata: &BTreeMap<String, String>,
    ) -> Result<Self> {
        // Build the header JSON with cumulative, data-relative offsets.
        let mut header = serde_json::Map::new();
        if !metadata.is_empty() {
            let mut meta = serde_json::Map::new();
            for (k, v) in metadata {
                meta.insert(k.clone(), Value::String(v.clone()));
            }
            header.insert("__metadata__".to_string(), Value::Object(meta));
        }
        let mut offset = 0usize;
        for t in &plan {
            let begin = offset;
            let end = offset + t.nbytes;
            offset = end;
            header.insert(
                t.key.clone(),
                serde_json::json!({
                    "dtype": t.dtype.tag(),
                    "shape": t.shape,
                    "data_offsets": [begin, end],
                }),
            );
        }

        let mut hjson = serde_json::to_vec(&Value::Object(header))?;
        // Pad with spaces so (8 + header_len) is a multiple of HEADER_ALIGN.
        let pad = (HEADER_ALIGN - ((hjson.len() + 8) % HEADER_ALIGN)) % HEADER_ALIGN;
        hjson.extend(std::iter::repeat_n(b' ', pad));

        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut inner = BufWriter::new(file);
        inner.write_all(&(hjson.len() as u64).to_le_bytes())?;
        inner.write_all(&hjson)?;

        Ok(Self { inner, plan, next: 0 })
    }

    /// Write the next tensor's bytes. Must be called in plan order; `bytes` length
    /// must equal the slot's `nbytes`.
    pub fn write_tensor(&mut self, key: &str, bytes: &[u8]) -> Result<()> {
        let slot = self
            .plan
            .get(self.next)
            .ok_or_else(|| anyhow!("wrote more tensors than planned"))?;
        if slot.key != key {
            bail!("out-of-order write: expected '{}', got '{}'", slot.key, key);
        }
        if bytes.len() != slot.nbytes {
            bail!("'{key}': wrote {} bytes, planned {}", bytes.len(), slot.nbytes);
        }
        self.inner.write_all(bytes)?;
        self.next += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        if self.next != self.plan.len() {
            bail!("only wrote {} of {} planned tensors", self.next, self.plan.len());
        }
        self.inner.flush()?;
        Ok(())
    }
}
