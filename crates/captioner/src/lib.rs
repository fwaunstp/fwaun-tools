//! Florence-2 ONNX captioner.
//!
//! Loads the four ONNX submodels exported by HuggingFace
//! (`vision_encoder`, `embed_tokens`, `encoder_model`, `decoder_model`) plus a
//! BPE `tokenizer.json`, runs greedy autoregressive decoding without KV cache,
//! and returns the decoded caption.
//!
//! No KV cache means the decoder cost is roughly O(n²) in the output length.
//! For caption tasks (~50–200 tokens) on CPU this is workable; KV cache can be
//! wired in later by switching to `decoder_model_merged.onnx`.

use std::path::Path;

use anima_tagger_core::config::CaptionerProfile;
use image::DynamicImage;
use image::imageops::FilterType;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use thiserror::Error;
use tokenizers::Tokenizer;

/// BART-family decoder convention used by Florence-2.
const DECODER_START_TOKEN_ID: i64 = 2;
const EOS_TOKEN_ID: i64 = 2;

#[derive(Debug, Error)]
pub enum CaptionerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ort: {0}")]
    Ort(String),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("model output shape unexpected: {0}")]
    Shape(String),
}

impl<F> From<ort::Error<F>> for CaptionerError {
    fn from(e: ort::Error<F>) -> Self {
        CaptionerError::Ort(e.to_string())
    }
}

pub struct Captioner {
    vision: Session,
    embed: Session,
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    prompt_token_ids: Vec<i64>,
    input_size: u32,
    max_new_tokens: usize,
}

impl Captioner {
    pub fn from_profile(profile: &CaptionerProfile) -> Result<Self, CaptionerError> {
        let dir = &profile.model_dir;
        let vision = build_session(&dir.join("vision_encoder.onnx"), "vision_encoder")?;
        let embed = build_session(&dir.join("embed_tokens.onnx"), "embed_tokens")?;
        let encoder = build_session(&dir.join("encoder_model.onnx"), "encoder_model")?;
        let decoder = build_session(&dir.join("decoder_model.onnx"), "decoder_model")?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            CaptionerError::Tokenizer(format!("loading {}: {e}", tokenizer_path.display()))
        })?;

        let encoding = tokenizer
            .encode(profile.prompt.as_str(), true)
            .map_err(|e| CaptionerError::Tokenizer(e.to_string()))?;
        let prompt_token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        Ok(Self {
            vision,
            embed,
            encoder,
            decoder,
            tokenizer,
            prompt_token_ids,
            input_size: profile.input_size,
            max_new_tokens: profile.max_new_tokens,
        })
    }

    pub fn caption_image(&mut self, image_path: &Path) -> Result<String, CaptionerError> {
        let img = image::open(image_path)?;

        // 1. Vision encoder: image → image_features [1, V, D]
        let pixel_data = preprocess_florence2(&img, self.input_size);
        let s = self.input_size as i64;
        let vision_input = Tensor::from_array(([1_i64, 3, s, s], pixel_data))?;
        let vision_outputs = self
            .vision
            .run(ort::inputs![vision_input])
            .map_err(|e| CaptionerError::Ort(format!("vision_encoder: {e}")))?;
        let (v, d, vision_data) = {
            let (shape, data) = vision_outputs[0].try_extract_tensor::<f32>()?;
            check_rank(shape, 3, "vision_encoder output")?;
            (shape[1] as usize, shape[2] as usize, data.to_vec())
        };
        drop(vision_outputs);

        // 2. Embed prompt tokens: input_ids → text_embeds [1, T, D]
        let t = self.prompt_token_ids.len();
        let prompt_input =
            Tensor::from_array(([1_i64, t as i64], self.prompt_token_ids.clone()))?;
        let embed_outputs = self
            .embed
            .run(ort::inputs![prompt_input])
            .map_err(|e| CaptionerError::Ort(format!("embed_tokens: {e}")))?;
        let embed_data = {
            let (shape, data) = embed_outputs[0].try_extract_tensor::<f32>()?;
            check_rank(shape, 3, "embed_tokens output")?;
            if (shape[2] as usize) != d {
                return Err(CaptionerError::Shape(format!(
                    "embed hidden dim {} != vision hidden dim {d}",
                    shape[2]
                )));
            }
            data.to_vec()
        };
        drop(embed_outputs);

        // 3. Concat along seq axis → [1, V+T, D]; attention_mask all-ones.
        let mut concat = Vec::with_capacity((v + t) * d);
        concat.extend_from_slice(&vision_data);
        concat.extend_from_slice(&embed_data);
        let attention_mask = vec![1_i64; v + t];
        let concat_input =
            Tensor::from_array(([1_i64, (v + t) as i64, d as i64], concat))?;
        let mask_input =
            Tensor::from_array(([1_i64, (v + t) as i64], attention_mask.clone()))?;

        // 4. Text encoder
        let encoder_outputs = self
            .encoder
            .run(ort::inputs! {
                "inputs_embeds" => concat_input,
                "attention_mask" => mask_input,
            })
            .map_err(|e| CaptionerError::Ort(format!("encoder_model: {e}")))?;
        let (enc_seq, enc_dim, enc_data) = {
            let (shape, data) = encoder_outputs[0].try_extract_tensor::<f32>()?;
            check_rank(shape, 3, "encoder_model output")?;
            (shape[1] as usize, shape[2] as usize, data.to_vec())
        };
        drop(encoder_outputs);

        // 5. Greedy decoder loop (no KV cache).
        //
        // This export's decoder takes `inputs_embeds` (not `input_ids`), so each
        // step we re-embed the full decoder sequence via `embed_tokens` and
        // feed the resulting embeddings to the decoder. The decoder also emits
        // `present.*.{key,value}` KV-cache tensors that we discard — wiring
        // those into a separate `decoder_with_past_model.onnx` would speed up
        // generation from O(n²) to O(n), but is left for a follow-up.
        let mut decoder_ids: Vec<i64> = vec![DECODER_START_TOKEN_ID];
        let mut generated: Vec<u32> = Vec::new();

        for _ in 0..self.max_new_tokens {
            let cur_len = decoder_ids.len();

            // Embed the current decoder sequence.
            let dec_ids_tensor =
                Tensor::from_array(([1_i64, cur_len as i64], decoder_ids.clone()))?;
            let dec_embed_outputs = self
                .embed
                .run(ort::inputs![dec_ids_tensor])
                .map_err(|e| CaptionerError::Ort(format!("embed_tokens (decode step): {e}")))?;
            let dec_embeds_data = {
                let (shape, data) = dec_embed_outputs[0].try_extract_tensor::<f32>()?;
                check_rank(shape, 3, "embed_tokens (decode step) output")?;
                data.to_vec()
            };
            drop(dec_embed_outputs);

            let enc_state_tensor = Tensor::from_array((
                [1_i64, enc_seq as i64, enc_dim as i64],
                enc_data.clone(),
            ))?;
            let enc_mask_tensor =
                Tensor::from_array(([1_i64, enc_seq as i64], attention_mask.clone()))?;
            let dec_embeds_tensor = Tensor::from_array((
                [1_i64, cur_len as i64, enc_dim as i64],
                dec_embeds_data,
            ))?;

            let dec_outputs = self
                .decoder
                .run(ort::inputs! {
                    "encoder_attention_mask" => enc_mask_tensor,
                    "encoder_hidden_states" => enc_state_tensor,
                    "inputs_embeds" => dec_embeds_tensor,
                })
                .map_err(|e| CaptionerError::Ort(format!("decoder_model: {e}")))?;
            let next_id = {
                let (shape, data) = dec_outputs[0].try_extract_tensor::<f32>()?;
                check_rank(shape, 3, "decoder_model logits output")?;
                let vocab = shape[2] as usize;
                let last_offset = (cur_len - 1) * vocab;
                argmax_i64(&data[last_offset..last_offset + vocab])
            };
            drop(dec_outputs);

            if next_id == EOS_TOKEN_ID {
                break;
            }
            generated.push(next_id as u32);
            decoder_ids.push(next_id);
        }

        // 6. Detokenize, strip special tokens.
        let caption = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| CaptionerError::Tokenizer(e.to_string()))?;
        Ok(caption.trim().to_string())
    }
}

fn build_session(path: &Path, label: &str) -> Result<Session, CaptionerError> {
    let bytes = std::fs::read(path)?;
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_memory(&bytes)?;
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    eprintln!(
        "[captioner:{label}] inputs={inputs:?} outputs={outputs:?}"
    );
    Ok(session)
}

fn check_rank(shape: &[i64], expected: usize, label: &str) -> Result<(), CaptionerError> {
    if shape.len() != expected {
        return Err(CaptionerError::Shape(format!(
            "{label}: expected rank {expected}, got {:?}",
            shape
        )));
    }
    Ok(())
}

fn argmax_i64(slice: &[f32]) -> i64 {
    let mut best = 0usize;
    let mut best_score = f32::MIN;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_score {
            best_score = v;
            best = i;
        }
    }
    best as i64
}

/// Florence-2 preprocessing: resize to size×size (bilinear), RGB,
/// per-channel ImageNet normalization, layout NCHW.
fn preprocess_florence2(img: &DynamicImage, size: u32) -> Vec<f32> {
    let resized = img
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8();
    let s = size as usize;
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    let mut data = vec![0.0_f32; 3 * s * s];
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        let xy = (y as usize) * s + (x as usize);
        let rgb = [r, g, b];
        for (c, &channel) in rgb.iter().enumerate() {
            data[c * s * s + xy] = (channel as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    data
}
