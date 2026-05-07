//! Qwen3-VL ONNX backend (default: `onnx-community/Qwen3-4B-VL-ONNX`,
//! 4B vision-fp32 / text-int4 variant). Selected via
//! `[captioner.<name>] kind = "onnx"` in `anima-tagger.toml`.
//!
//! Three sessions:
//! ```text
//! pixel_values + image_grid_thw ─► qwen3vl-vision.onnx     ─► vision_hidden_states [Nv, 2560]
//! input_ids + vision_hidden_states ─► qwen3vl-embedding.onnx ─► inputs_embeds [1, S, 2560]
//!                                                  │
//!                                                  ▼
//!                                          model.onnx (greedy decode loop with KV cache)
//! ```
//!
//! Notable contrasts with Qwen2-VL (in case anyone re-reads this after porting
//! between the two):
//!
//! 1. **Embedding is fused.** Qwen3-VL's `qwen3vl-embedding.onnx` runs
//!    `embed_tokens(input_ids)` then in-graph scatters `vision_hidden_states`
//!    into the rows where `input_ids == <|image_pad|>` (151655). The caller's
//!    job shrinks to: tokenize the chat template, expand the single
//!    `<|image_pad|>` to `n_vision_tokens` copies, hand both tensors to the
//!    embedding session — no manual row overwrite.
//!
//! 2. **Patch geometry differs.** patch_size=16 (was 14), merge_size=2 still,
//!    so smart_resize aligns to `factor = 32` (was 28). Each pixel_values
//!    row is `3 * temporal_patch_size * patch_size² = 3 * 2 * 16 * 16 = 1536`.
//!    Vision output hidden dim is 2560 (matches the LLM hidden_size, no
//!    cross-modal projection needed).
//!
//! 3. **Normalization is plain (0.5, 0.5, 0.5) mean/std**, not CLIP. Easy to
//!    miss when porting.
//!
//! 4. **Decoder is bigger.** 36 layers, 32 attention heads, 8 KV heads (GQA),
//!    head_dim 128, hidden 2560, vocab 151936. KV cache is 36×2 = 72 tensors
//!    per call, shape `[1, 8, past, 128]`.
//!
//! 5. **Decoder is INT4-quantized** (the only prebuilt 4B variant). External
//!    weights live in `model.onnx.data` (~2.4 GB), loaded via ort's external-
//!    data path resolution, so the decoder is committed from a file path
//!    rather than from in-memory bytes.
//!
//! 6. **Chat template has no system message.** Just user → assistant.
//!
//! 7. **3D MROPE position_ids** still apply (`[3, 1, S]`). The internal RoPE
//!    splits change (`mrope_section = [24, 20, 20]`) but that's invisible to
//!    the caller — the position-id construction algorithm is the same as
//!    Qwen2-VL.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use anima_tagger_core::config::OnnxCaptionerProfile;
use anima_tagger_core::hub;
use image::DynamicImage;
use image::imageops::FilterType;
use ort::memory::Allocator;
use ort::session::{Session, SessionInputValue};
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::CaptionerError;

// Image / patch geometry (Qwen3-VL preprocessor_config.json).
const PATCH_SIZE: u32 = 16;
const MERGE_SIZE: u32 = 2;
const TEMPORAL_PATCH_SIZE: u32 = 2;
const FACTOR: u32 = PATCH_SIZE * MERGE_SIZE; // 32
/// Floor for smart_resize. Qwen3-VL's own preprocessor uses 65_536 (256×256);
/// we relax it so very small training crops still go through the model
/// without being upsampled aggressively.
const MIN_PIXELS: u32 = 32 * 32;

// Qwen3-VL normalizes with plain (0.5, 0.5, 0.5) — different from Qwen2-VL's
// CLIP mean/std. Stays in fp32, RGB, NCHW-flattened per patch.
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

// Decoder geometry (Qwen3-VL-4B text_config from the model card).
const HIDDEN_SIZE: usize = 2_560;
const NUM_LAYERS: usize = 36;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

// Token ids (from added_tokens.json / generation_config.json).
const IMAGE_PAD_TOKEN_ID: i64 = 151_655;
const IM_END_TOKEN_ID: i64 = 151_645;
const ENDOFTEXT_TOKEN_ID: i64 = 151_643;

const LOG_FIRST_N_STEPS: usize = 8;

pub struct OnnxCaptioner {
    vision: Session,
    embed: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    max_pixels: u32,
    max_new_tokens: usize,
}

impl OnnxCaptioner {
    pub fn from_profile(profile: &OnnxCaptionerProfile) -> Result<Self, CaptionerError> {
        let prefix = profile.subdir.as_deref().unwrap_or("");
        let vision_rel = join_subdir(prefix, "qwen3vl-vision.onnx");
        let embed_rel = join_subdir(prefix, "qwen3vl-embedding.onnx");
        let decoder_rel = join_subdir(prefix, "model.onnx");
        let decoder_data_rel = join_subdir(prefix, "model.onnx.data");
        let tokenizer_rel = join_subdir(prefix, "tokenizer.json");

        let files = hub::fetch_files(
            &profile.repo,
            profile.revision.as_deref(),
            &[
                &vision_rel,
                &embed_rel,
                &decoder_rel,
                &decoder_data_rel,
                &tokenizer_rel,
            ],
        )?;
        let vision_path: &PathBuf = &files[0];
        let embed_path: &PathBuf = &files[1];
        let decoder_path: &PathBuf = &files[2];
        // files[3] is `model.onnx.data` — its presence in the same directory
        // is what matters; ort resolves the external-data ref from `model.onnx`.
        let tokenizer_path: &PathBuf = &files[4];

        let vision = build_session_from_file(vision_path, "qwen3vl-vision")?;
        let embed = build_session_from_file(embed_path, "qwen3vl-embedding")?;
        let decoder = build_session_from_file(decoder_path, "decoder")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            CaptionerError::Tokenizer(format!("loading {}: {e}", tokenizer_path.display()))
        })?;

        Ok(Self {
            vision,
            embed,
            decoder,
            tokenizer,
            max_pixels: profile.max_pixels,
            max_new_tokens: profile.max_new_tokens,
        })
    }

    pub fn caption_image(
        &mut self,
        image_path: &Path,
        prompt: &str,
        context: Option<&str>,
    ) -> Result<String, CaptionerError> {
        let img = image::open(image_path)?;

        // 1. Preprocess image → flattened patches + image_grid_thw.
        let (pixel_values, grid_thw) = preprocess_qwen3vl(&img, self.max_pixels);
        let [grid_t, grid_h, grid_w] = grid_thw;
        let n_patches = (grid_t * grid_h * grid_w) as usize;
        let n_vision_tokens = n_patches / ((MERGE_SIZE * MERGE_SIZE) as usize);
        let row_dim: usize = (3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE) as usize;
        eprintln!(
            "[captioner:image] resized to {}x{} → grid {grid_t}x{grid_h}x{grid_w} → {n_vision_tokens} vision tokens",
            grid_w * PATCH_SIZE,
            grid_h * PATCH_SIZE
        );

        // 2. Vision encoder → vision_hidden_states [Nv, 2560].
        let pixel_tensor = Tensor::from_array((
            [n_patches as i64, row_dim as i64],
            pixel_values,
        ))?;
        let grid_thw_tensor = Tensor::from_array((
            [1_i64, 3_i64],
            vec![grid_t as i64, grid_h as i64, grid_w as i64],
        ))?;
        let vision_out = self
            .vision
            .run(ort::inputs! {
                "pixel_values" => pixel_tensor,
                "image_grid_thw" => grid_thw_tensor,
            })
            .map_err(|e| CaptionerError::Ort(format!("vision_encoder: {e}")))?;
        let vision_hidden_states: Vec<f32> = {
            let (shape, data) = vision_out[0].try_extract_tensor::<f32>()?;
            if shape.len() != 2 || shape[1] as usize != HIDDEN_SIZE {
                return Err(CaptionerError::Shape(format!(
                    "vision output {shape:?} != [_, {HIDDEN_SIZE}]"
                )));
            }
            if shape[0] as usize != n_vision_tokens {
                return Err(CaptionerError::Shape(format!(
                    "vision produced {} tokens, expected {n_vision_tokens}",
                    shape[0]
                )));
            }
            data.to_vec()
        };
        drop(vision_out);

        // 3. Render chat template, tokenize.
        let chat_prompt = build_chat_prompt(prompt, context);
        let encoding = self
            .tokenizer
            .encode(chat_prompt, false)
            .map_err(|e| CaptionerError::Tokenizer(e.to_string()))?;
        let prompt_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        // 4. Expand the single <|image_pad|> token to n_vision_tokens copies.
        let pad_count = prompt_ids
            .iter()
            .filter(|&&id| id == IMAGE_PAD_TOKEN_ID)
            .count();
        if pad_count != 1 {
            return Err(CaptionerError::ImagePadCount(pad_count));
        }
        let pad_pos = prompt_ids
            .iter()
            .position(|&id| id == IMAGE_PAD_TOKEN_ID)
            .expect("checked above");
        let mut input_ids: Vec<i64> = Vec::with_capacity(prompt_ids.len() + n_vision_tokens - 1);
        input_ids.extend_from_slice(&prompt_ids[..pad_pos]);
        for _ in 0..n_vision_tokens {
            input_ids.push(IMAGE_PAD_TOKEN_ID);
        }
        input_ids.extend_from_slice(&prompt_ids[pad_pos + 1..]);
        let s = input_ids.len();

        // 5. Embedding session does the splice in-graph: it returns
        //    inputs_embeds with vision_hidden_states already scattered into
        //    the image-pad rows. We don't post-process the embeds.
        let input_ids_tensor = Tensor::from_array(([1_i64, s as i64], input_ids.clone()))?;
        let vis_tensor = Tensor::from_array((
            [n_vision_tokens as i64, HIDDEN_SIZE as i64],
            vision_hidden_states,
        ))?;
        let embed_out = self
            .embed
            .run(ort::inputs! {
                "input_ids" => input_ids_tensor,
                "vision_hidden_states" => vis_tensor,
            })
            .map_err(|e| CaptionerError::Ort(format!("embedding: {e}")))?;
        let embeds: Vec<f32> = {
            let (shape, data) = embed_out[0].try_extract_tensor::<f32>()?;
            if shape.len() != 3 || shape[2] as usize != HIDDEN_SIZE {
                return Err(CaptionerError::Shape(format!(
                    "embedding output {shape:?} != [1, S, {HIDDEN_SIZE}]"
                )));
            }
            data.to_vec()
        };
        drop(embed_out);

        // 6. Position ids for prefill + counter for decode steps.
        let (prefill_pos_ids, mut next_text_pos) =
            build_position_ids(s, pad_pos, n_vision_tokens, grid_t, grid_h, grid_w);

        // 7. Empty initial KV cache.
        let mut past_kv: Vec<KvTensor> = Vec::with_capacity(NUM_LAYERS * 2);
        for _ in 0..NUM_LAYERS {
            past_kv.push(KvTensor::empty());
            past_kv.push(KvTensor::empty());
        }

        // 8. Greedy decode loop.
        let mut attention_mask: Vec<i64> = vec![1; s];
        let mut cur_embeds = embeds;
        let mut cur_pos_ids = prefill_pos_ids;
        let mut cur_seq_len = s;
        let mut generated: Vec<u32> = Vec::new();

        for step in 0..self.max_new_tokens {
            let embeds_tensor = Tensor::from_array((
                [1_i64, cur_seq_len as i64, HIDDEN_SIZE as i64],
                cur_embeds.clone(),
            ))?;
            let mask_tensor = Tensor::from_array((
                [1_i64, attention_mask.len() as i64],
                attention_mask.clone(),
            ))?;
            let pos_tensor = Tensor::from_array((
                [3_i64, 1_i64, cur_seq_len as i64],
                cur_pos_ids.clone(),
            ))?;

            let mut named: Vec<(Cow<'static, str>, SessionInputValue)> =
                Vec::with_capacity(3 + NUM_LAYERS * 2);
            named.push((Cow::Borrowed("inputs_embeds"), embeds_tensor.into()));
            named.push((Cow::Borrowed("attention_mask"), mask_tensor.into()));
            named.push((Cow::Borrowed("position_ids"), pos_tensor.into()));
            for layer in 0..NUM_LAYERS {
                let k_kv = &past_kv[layer * 2];
                let v_kv = &past_kv[layer * 2 + 1];
                let k_tensor = k_kv.to_tensor()?;
                let v_tensor = v_kv.to_tensor()?;
                named.push((
                    Cow::Owned(format!("past_key_values.{layer}.key")),
                    k_tensor.into(),
                ));
                named.push((
                    Cow::Owned(format!("past_key_values.{layer}.value")),
                    v_tensor.into(),
                ));
            }

            let dec_out = self
                .decoder
                .run(named)
                .map_err(|e| CaptionerError::Ort(format!("decoder (step {step}): {e}")))?;

            // Output 0: logits [1, S, V]. Outputs 1.. are present.{i}.{key,value}.
            let next_id = {
                let (shape, data) = dec_out[0].try_extract_tensor::<f32>()?;
                if shape.len() != 3 {
                    return Err(CaptionerError::Shape(format!(
                        "decoder logits {shape:?}, expected rank 3"
                    )));
                }
                let s_out = shape[1] as usize;
                let vocab = shape[2] as usize;
                let last = (s_out - 1) * vocab;
                argmax_i64(&data[last..last + vocab])
            };

            if step < LOG_FIRST_N_STEPS {
                let surface = self
                    .tokenizer
                    .id_to_token(next_id as u32)
                    .unwrap_or_default();
                eprintln!("[captioner:step {step}] next_id={next_id} surface={surface:?}");
            }

            if next_id == IM_END_TOKEN_ID || next_id == ENDOFTEXT_TOKEN_ID {
                break;
            }
            generated.push(next_id as u32);

            for layer in 0..NUM_LAYERS {
                let k_idx = 1 + layer * 2;
                let v_idx = 1 + layer * 2 + 1;
                past_kv[layer * 2] = KvTensor::extract(&dec_out[k_idx])?;
                past_kv[layer * 2 + 1] = KvTensor::extract(&dec_out[v_idx])?;
            }
            drop(dec_out);

            // Embed only the newly chosen token for the next step.
            let next_id_tensor = Tensor::from_array(([1_i64, 1_i64], vec![next_id]))?;
            // Decode-step calls don't need vision input; we pass an
            // intentionally-empty vision tensor with 0 rows. The embedding
            // session's masked_scatter is a no-op when no input_id matches
            // <|image_pad|>.
            // `Tensor::from_array` rejects any dim < 1 in ort 2.0.0-rc.12, so
            // the zero-row vision input has to be constructed via the
            // allocator-based ctor instead.
            let empty_vision = Tensor::<f32>::new(
                &Allocator::default(),
                [0_i64, HIDDEN_SIZE as i64],
            )?;
            let next_embed_out = self
                .embed
                .run(ort::inputs! {
                    "input_ids" => next_id_tensor,
                    "vision_hidden_states" => empty_vision,
                })
                .map_err(|e| CaptionerError::Ort(format!("embedding (decode): {e}")))?;
            cur_embeds = {
                let (shape, data) = next_embed_out[0].try_extract_tensor::<f32>()?;
                if shape.len() != 3 || shape[2] as usize != HIDDEN_SIZE {
                    return Err(CaptionerError::Shape(format!(
                        "embedding (decode) output {shape:?}"
                    )));
                }
                data.to_vec()
            };
            drop(next_embed_out);
            cur_seq_len = 1;
            attention_mask.push(1);
            cur_pos_ids = vec![next_text_pos, next_text_pos, next_text_pos];
            next_text_pos += 1;
        }

        eprintln!("[captioner:done] generated {} tokens", generated.len());

        let caption = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| CaptionerError::Tokenizer(e.to_string()))?;
        Ok(caption.trim().to_string())
    }
}

fn join_subdir(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

/// One key-or-value tensor of the KV cache: shape
/// `[1, NUM_KV_HEADS, seq, HEAD_DIM]` stored row-major.
struct KvTensor {
    data: Vec<f32>,
    seq: usize,
}

impl KvTensor {
    fn empty() -> Self {
        Self {
            data: Vec::new(),
            seq: 0,
        }
    }

    fn shape_i64(&self) -> [i64; 4] {
        [1, NUM_KV_HEADS as i64, self.seq as i64, HEAD_DIM as i64]
    }

    /// Build the per-step decoder input tensor. On the first step `seq == 0`,
    /// and `Tensor::from_array` rejects any dim < 1 in ort 2.0.0-rc.12, so the
    /// empty cache has to go through the allocator-based ctor instead.
    fn to_tensor(&self) -> Result<Tensor<f32>, CaptionerError> {
        if self.seq == 0 {
            Ok(Tensor::<f32>::new(&Allocator::default(), self.shape_i64())?)
        } else {
            Ok(Tensor::from_array((self.shape_i64(), self.data.clone()))?)
        }
    }

    fn extract(value: &ort::value::DynValue) -> Result<Self, CaptionerError> {
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.len() != 4
            || shape[0] != 1
            || shape[1] as usize != NUM_KV_HEADS
            || shape[3] as usize != HEAD_DIM
        {
            return Err(CaptionerError::Shape(format!(
                "kv tensor {shape:?} != [1, {NUM_KV_HEADS}, _, {HEAD_DIM}]"
            )));
        }
        Ok(Self {
            data: data.to_vec(),
            seq: shape[2] as usize,
        })
    }
}

fn build_session_from_file(path: &Path, label: &str) -> Result<Session, CaptionerError> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?;
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    eprintln!("[captioner:{label}] inputs={inputs:?} outputs={outputs:?}");
    Ok(session)
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

fn build_chat_prompt(user_instruction: &str, context: Option<&str>) -> String {
    // Qwen3-VL's default chat template emits no system block when none is
    // supplied (different from Qwen2-VL). Single image + single user turn.
    //
    // Caller-supplied `context` (character names / positions / scene
    // continuity) is image-specific reference info, so we embed it inside
    // the user turn next to the image rather than as a system turn — that
    // way the model sees image, context, and instruction together as one
    // unit instead of a free-floating persona-style preamble.
    let body = crate::build_user_text(user_instruction, context);
    format!(
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{body}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Build flat `[3 * S]` row-major position ids (row 0 first, then row 1, then
/// row 2) for the prefill pass, plus the next text position for decode steps.
///
/// Same algorithm as Qwen2-VL's `get_rope_index` for a single image — only
/// the internal RoPE splits change between the two models, not the caller's
/// position-id layout.
fn build_position_ids(
    s: usize,
    pad_pos: usize,
    n_vision_tokens: usize,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
) -> (Vec<i64>, i64) {
    let mut row_t = Vec::with_capacity(s);
    let mut row_h = Vec::with_capacity(s);
    let mut row_w = Vec::with_capacity(s);

    for i in 0..pad_pos {
        let p = i as i64;
        row_t.push(p);
        row_h.push(p);
        row_w.push(p);
    }

    let base = pad_pos as i64;
    let mut max_image_pos: i64 = base - 1;

    let gh_merged = (grid_h / MERGE_SIZE) as i64;
    let gw_merged = (grid_w / MERGE_SIZE) as i64;
    let _ = grid_t; // grid_t == 1 for a single image; t_idx is always 0.
    let tokens_per_t = gh_merged * gw_merged;

    for k in 0..n_vision_tokens {
        let k64 = k as i64;
        let t_idx = k64 / tokens_per_t;
        let inner = k64 % tokens_per_t;
        let h_idx = inner / gw_merged;
        let w_idx = inner % gw_merged;
        let pt = base + t_idx;
        let ph = base + h_idx;
        let pw = base + w_idx;
        max_image_pos = max_image_pos.max(pt).max(ph).max(pw);
        row_t.push(pt);
        row_h.push(ph);
        row_w.push(pw);
    }

    let after_base = max_image_pos + 1;
    let post_image_start = pad_pos + n_vision_tokens;
    for i in post_image_start..s {
        let offset = (i - post_image_start) as i64;
        let p = after_base + offset;
        row_t.push(p);
        row_h.push(p);
        row_w.push(p);
    }

    let next_text_pos = if s > post_image_start {
        after_base + (s - post_image_start) as i64
    } else {
        after_base
    };

    let mut flat = Vec::with_capacity(3 * s);
    flat.extend_from_slice(&row_t);
    flat.extend_from_slice(&row_h);
    flat.extend_from_slice(&row_w);
    (flat, next_text_pos)
}

/// Smart-resize an image to a 32-aligned grid bounded by `max_pixels` (and
/// floored at `MIN_PIXELS`), then produce flattened patches `[N, 1536]` and
/// `image_grid_thw = [grid_t, grid_h, grid_w]`. For a single still image
/// `grid_t = 1` and the temporal axis is filled by duplicating the image.
fn preprocess_qwen3vl(img: &DynamicImage, max_pixels: u32) -> (Vec<f32>, [u32; 3]) {
    let rgb = img.to_rgb8();
    let (w_orig, h_orig) = rgb.dimensions();
    let (h_bar, w_bar) = smart_resize(h_orig, w_orig, FACTOR, MIN_PIXELS, max_pixels);
    let resized = image::imageops::resize(&rgb, w_bar, h_bar, FilterType::CatmullRom);

    let grid_h = h_bar / PATCH_SIZE;
    let grid_w = w_bar / PATCH_SIZE;
    let grid_t = 1u32;
    let n = (grid_t * grid_h * grid_w) as usize;
    let row_len = (3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE) as usize; // 1536

    // Output: [N, 1536] in row-major. Row order iterates
    // (gt, gh_merged, gw_merged, mh, mw); within each row,
    // (channel, temporal, patch_h, patch_w). Mirrors HF
    // `_get_pixel_values_from_image` after the transpose+reshape.
    let mut out = vec![0.0f32; n * row_len];

    let gh_m = grid_h / MERGE_SIZE;
    let gw_m = grid_w / MERGE_SIZE;
    let mut row_idx = 0usize;
    for _gt in 0..grid_t {
        for gh in 0..gh_m {
            for gw in 0..gw_m {
                for mh in 0..MERGE_SIZE {
                    for mw in 0..MERGE_SIZE {
                        let y_base = (gh * MERGE_SIZE + mh) * PATCH_SIZE;
                        let x_base = (gw * MERGE_SIZE + mw) * PATCH_SIZE;
                        let row_offset = row_idx * row_len;
                        for c in 0..3usize {
                            for t in 0..TEMPORAL_PATCH_SIZE as usize {
                                let _ = t; // both temporal frames hold the same image
                                let plane_offset = c * (TEMPORAL_PATCH_SIZE as usize) + t;
                                for ph in 0..PATCH_SIZE {
                                    for pw in 0..PATCH_SIZE {
                                        let y = y_base + ph;
                                        let x = x_base + pw;
                                        let pixel = resized.get_pixel(x, y);
                                        let raw = pixel.0[c] as f32 / 255.0;
                                        let normed = (raw - MEAN[c]) / STD[c];
                                        let inner = plane_offset
                                            * (PATCH_SIZE * PATCH_SIZE) as usize
                                            + (ph * PATCH_SIZE + pw) as usize;
                                        out[row_offset + inner] = normed;
                                    }
                                }
                            }
                        }
                        row_idx += 1;
                    }
                }
            }
        }
    }

    (out, [grid_t, grid_h, grid_w])
}

fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: u32, max_pixels: u32) -> (u32, u32) {
    let mut h_bar = round_to(h, factor).max(factor);
    let mut w_bar = round_to(w, factor).max(factor);
    let cur = h_bar.saturating_mul(w_bar);
    if cur > max_pixels {
        let beta = ((h as f64) * (w as f64) / (max_pixels as f64)).sqrt();
        h_bar = floor_to(((h as f64) / beta).max(1.0) as u32, factor).max(factor);
        w_bar = floor_to(((w as f64) / beta).max(1.0) as u32, factor).max(factor);
    } else if cur < min_pixels {
        let beta = (min_pixels as f64 / ((h as f64) * (w as f64))).sqrt();
        h_bar = ceil_to(((h as f64) * beta) as u32, factor).max(factor);
        w_bar = ceil_to(((w as f64) * beta) as u32, factor).max(factor);
    }
    (h_bar, w_bar)
}

fn round_to(n: u32, factor: u32) -> u32 {
    let q = (n as f64 / factor as f64).round() as u32;
    q * factor
}

fn floor_to(n: u32, factor: u32) -> u32 {
    (n / factor) * factor
}

fn ceil_to(n: u32, factor: u32) -> u32 {
    n.div_ceil(factor) * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_caps_to_max_pixels() {
        let (h, w) = smart_resize(2048, 4096, FACTOR, MIN_PIXELS, 1_000_000);
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
        assert!(h * w <= 1_000_000);
    }

    #[test]
    fn smart_resize_floors_at_min_pixels() {
        let (h, w) = smart_resize(20, 20, FACTOR, MIN_PIXELS, 1_000_000);
        assert!(h * w >= MIN_PIXELS);
        assert_eq!(h % FACTOR, 0);
        assert_eq!(w % FACTOR, 0);
    }

    #[test]
    fn build_position_ids_simple() {
        // 3 text tokens, then image span of 4 vision tokens (2x2 merged grid),
        // then 2 text tokens. grid_h=4, grid_w=4 → gh_m=2, gw_m=2.
        let (flat, next) = build_position_ids(9, 3, 4, 1, 4, 4);
        assert_eq!(flat.len(), 27);
        let row_t = &flat[..9];
        let row_h = &flat[9..18];
        let row_w = &flat[18..];

        assert_eq!(&row_t[..3], &[0, 1, 2]);
        assert_eq!(&row_h[..3], &[0, 1, 2]);
        assert_eq!(&row_w[..3], &[0, 1, 2]);

        assert_eq!(&row_t[3..7], &[3, 3, 3, 3]);
        assert_eq!(&row_h[3..7], &[3, 3, 4, 4]);
        assert_eq!(&row_w[3..7], &[3, 4, 3, 4]);

        assert_eq!(&row_t[7..], &[5, 6]);
        assert_eq!(&row_h[7..], &[5, 6]);
        assert_eq!(&row_w[7..], &[5, 6]);

        assert_eq!(next, 7);
    }
}
