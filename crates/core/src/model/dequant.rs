//! Dequantize an int8 / int8_convrot checkpoint back to a plain float one.
//!
//! The comfy-kitchen int8 kernels only run on recent GPUs; this writes the same
//! model as bf16 (or fp16/fp32) so older cards can load it. Every int8 layer is
//! decoded with [`ModelFile::to_f32`] (`(q · scale) · H_blockdiag`) and cast to
//! the requested dtype, its `weight_scale` / `comfy_quant` companions are
//! dropped, and every other tensor is copied byte-for-byte.
//!
//! The output carries the int8 rounding error: it is the dequantized model, not
//! the original bf16 the int8 file was made from.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use regex::Regex;

use super::progress::ProgressSink;
use super::quantized::ModelFile;
use super::safetensors::{Dtype, OutputTensor, StreamWriter, f32_to_bytes};

/// Parsed arguments for the `dequant` subcommand.
pub struct DequantArgs {
    pub src: PathBuf,
    /// Output path; derived from `src` when `None`.
    pub dst: Option<PathBuf>,
    /// dtype for the dequantized layers (bf16, fp16, or fp32).
    pub dtype: Dtype,
}

/// Derive an output path when `dst` is omitted: swap an `int8_convrot` / `int8`
/// token for the dtype name (else append `_<dtype>`), always `.safetensors`.
fn derive_dst(src: &Path, dtype: Dtype) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let name = match dtype {
        Dtype::F16 => "fp16",
        Dtype::F32 => "fp32",
        _ => "bf16",
    };
    let re = Regex::new(r"(?i)int8_convrot|int8").unwrap();
    let new = if re.is_match(stem) {
        re.replace(stem, name).into_owned()
    } else {
        format!("{stem}_{name}")
    };
    src.with_file_name(format!("{new}.safetensors"))
}

pub fn run(args: DequantArgs, p: &mut dyn ProgressSink) -> Result<()> {
    if !matches!(args.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
        bail!(
            "unsupported output dtype {} (expected bf16, fp16, or fp32)",
            args.dtype.tag()
        );
    }
    let src = ModelFile::open(&args.src)?;
    if !src.is_quantized() {
        bail!(
            "{} has no int8 layers; nothing to dequantize",
            args.src.display()
        );
    }
    let dst = args
        .dst
        .clone()
        .unwrap_or_else(|| derive_dst(&args.src, args.dtype));
    // The source is memory-mapped while we write; overwriting it would corrupt the read.
    if dst.canonicalize().ok() == args.src.canonicalize().ok() {
        bail!("output must differ from the source ({})", dst.display());
    }

    p.log(&format!("SRC    : {}", args.src.display()));
    p.log(&format!("DST    : {}", dst.display()));
    p.log(&format!(
        "format : {} -> {}",
        src.quant_label(),
        args.dtype.tag()
    ));

    let keys: Vec<String> = src.keys().cloned().collect();
    let mut plan = Vec::with_capacity(keys.len());
    let mut n_int8 = 0usize;
    for key in &keys {
        let info = src.raw().info(key).unwrap();
        let (dtype, nbytes) = if src.int8_layer(key).is_some() {
            n_int8 += 1;
            (args.dtype, info.numel() * args.dtype.element_size())
        } else {
            (info.dtype, info.end - info.begin)
        };
        plan.push(OutputTensor {
            key: key.clone(),
            dtype,
            shape: info.shape.clone(),
            nbytes,
        });
    }
    p.log(&format!(
        "dequantizing {n_int8} int8 layers; copying {} other tensors unchanged",
        keys.len() - n_int8
    ));

    let mut metadata = src.metadata().clone();
    metadata.remove("quant_format");
    metadata.insert(
        "dequantized_from".to_string(),
        args.src.display().to_string(),
    );

    let mut writer = StreamWriter::begin(&dst, plan, &metadata)?;
    let mut done = 0usize;
    for key in &keys {
        if src.int8_layer(key).is_some() {
            let w = src.to_f32(key)?;
            writer.write_tensor(key, &f32_to_bytes(&w, args.dtype)?)?;
            done += 1;
            p.tick(done, n_int8);
        } else {
            writer.write_tensor(key, src.raw().raw_bytes(key)?)?;
        }
    }
    writer.finish()?;

    p.log(&format!(
        "DONE: wrote {} tensors -> {}",
        keys.len(),
        dst.display()
    ));
    p.log("note: the output carries the int8 rounding error; it is not the original bf16 model.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::quantized::tests::{TestTensor, temp_dir, weights, write_model};
    use crate::model::safetensors::SafeTensorsFile;

    struct Quiet;
    impl ProgressSink for Quiet {
        fn log(&mut self, _: &str) {}
        fn tick(&mut self, _: usize, _: usize) {}
    }

    #[test]
    fn derive_dst_swaps_int8_token() {
        let p = Path::new("/m/qwen_image_2.1_int8_convrot.safetensors");
        assert_eq!(
            derive_dst(p, Dtype::Bf16),
            Path::new("/m/qwen_image_2.1_bf16.safetensors")
        );
        assert_eq!(
            derive_dst(Path::new("/m/tune.safetensors"), Dtype::F16),
            Path::new("/m/tune_fp16.safetensors")
        );
    }

    #[test]
    fn dequantizes_int8_layers_and_copies_the_rest() {
        let dir = temp_dir("dequant");
        let src = dir.join("m_int8_convrot.safetensors");
        let (out, in_) = (8, 64);
        let key = "transformer_blocks.0.attn.to_q.weight";
        write_model(
            &src,
            &[
                (key, TestTensor::Int8(out, in_, 16, weights(out * in_, 5))),
                (
                    "norm.weight",
                    TestTensor::Float(vec![3], vec![1.0, 2.0, 3.0]),
                ),
            ],
        );

        run(
            DequantArgs {
                src: src.clone(),
                dst: None,
                dtype: Dtype::F32,
            },
            &mut Quiet,
        )
        .unwrap();

        let dst = dir.join("m_fp32.safetensors");
        let f = SafeTensorsFile::open(&dst).unwrap();
        assert_eq!(f.keys().count(), 2, "companions must be dropped");
        assert_eq!(f.info(key).unwrap().dtype, Dtype::F32);
        let expected = ModelFile::open(&src).unwrap().to_f32(key).unwrap();
        assert_eq!(f.to_f32(key).unwrap(), expected);
        assert_eq!(f.to_f32("norm.weight").unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(f.metadata().contains_key("dequantized_from"));

        // Already-float input and in-place output are refused.
        let float_src = dst.clone();
        let args = |src: PathBuf, dst: Option<PathBuf>| DequantArgs {
            src,
            dst,
            dtype: Dtype::Bf16,
        };
        assert!(run(args(float_src, None), &mut Quiet).is_err());
        assert!(run(args(src.clone(), Some(src.clone())), &mut Quiet).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
