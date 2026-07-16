//! WD14-family ONNX tagger. Loads a model + selected_tags.csv, runs inference
//! per image, returns ranked `AutoTag`s with category metadata preserved.

use std::path::Path;

use fwaun_tools_core::config::TaggerProfile;
use fwaun_tools_core::hub;
use fwaun_tools_core::sidecar::AutoTag;
#[cfg(feature = "onnx")]
use image::imageops::FilterType;
#[cfg(feature = "onnx")]
use image::{DynamicImage, Rgb, RgbImage};
#[cfg(feature = "onnx")]
use ndarray::Array4;
#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "onnx")]
use ort::value::Tensor;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TaggerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ort: {0}")]
    Ort(String),
    #[error(
        "this is a light build without the local ONNX tagger; install the full build to run WD14 tagging"
    )]
    Unsupported,
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("csv parse on {path}: {source}")]
    Csv {
        path: std::path::PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("hub: {0}")]
    Hub(#[from] hub::HubError),
    #[error(
        "model output count ({actual}) does not match tag dictionary size ({expected}) — model and tag CSV are mismatched"
    )]
    OutputMismatch { expected: usize, actual: usize },
}

#[cfg(feature = "onnx")]
impl<F> From<ort::Error<F>> for TaggerError {
    fn from(e: ort::Error<F>) -> Self {
        TaggerError::Ort(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct TagDef {
    pub name: String,
    pub category: String,
}

#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(rename = "tag_id")]
    _tag_id: i64,
    name: String,
    category: i32,
    #[serde(default, rename = "count")]
    _count: Option<i64>,
}

pub fn load_tags(path: &Path) -> Result<Vec<TagDef>, TaggerError> {
    let mut rdr = csv::Reader::from_path(path).map_err(|source| TaggerError::Csv {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = Vec::new();
    for row in rdr.deserialize::<CsvRow>() {
        let row = row.map_err(|source| TaggerError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        out.push(TagDef {
            name: row.name,
            category: category_name(row.category),
        });
    }
    Ok(out)
}

fn category_name(id: i32) -> String {
    match id {
        0 => "general".to_string(),
        1 => "artist".to_string(),
        3 => "copyright".to_string(),
        4 => "character".to_string(),
        5 => "meta".to_string(),
        9 => "rating".to_string(),
        other => format!("category_{other}"),
    }
}

/// WD14-style preprocessing: pad to square (white), bicubic resize, BGR uint8 → f32 NHWC.
#[cfg(feature = "onnx")]
pub fn preprocess_wd14(img: &DynamicImage, size: u32) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let max_side = w.max(h);
    let mut square = RgbImage::from_pixel(max_side, max_side, Rgb([255, 255, 255]));
    let dx = ((max_side - w) / 2) as i64;
    let dy = ((max_side - h) / 2) as i64;
    image::imageops::overlay(&mut square, &rgb, dx, dy);

    let resized = image::imageops::resize(&square, size, size, FilterType::CatmullRom);

    let s = size as usize;
    let mut arr = Array4::<f32>::zeros((1, s, s, 3));
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        arr[(0, y as usize, x as usize, 0)] = b as f32;
        arr[(0, y as usize, x as usize, 1)] = g as f32;
        arr[(0, y as usize, x as usize, 2)] = r as f32;
    }
    arr
}

#[cfg(feature = "onnx")]
pub struct Tagger {
    session: Session,
    tags: Vec<TagDef>,
    input_size: u32,
}

#[cfg(feature = "onnx")]
impl Tagger {
    pub fn from_profile(profile: &TaggerProfile) -> Result<Self, TaggerError> {
        // SmilingWolf's WD14 repos lay out `model.onnx` and `selected_tags.csv`
        // at the root, so the file list is the same across model variants.
        let files = hub::fetch_files(
            &profile.repo,
            profile.revision.as_deref(),
            &["model.onnx", "selected_tags.csv"],
        )?;
        let model_path = &files[0];
        let tags_path = &files[1];

        let bytes = std::fs::read(model_path)?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(&bytes)?;
        let tags = load_tags(tags_path)?;
        Ok(Self {
            session,
            tags,
            input_size: profile.input_size,
        })
    }

    pub fn num_tags(&self) -> usize {
        self.tags.len()
    }

    pub fn tag_image(
        &mut self,
        image_path: &Path,
        threshold: f32,
    ) -> Result<Vec<AutoTag>, TaggerError> {
        let img = image::open(image_path)?;
        let arr = preprocess_wd14(&img, self.input_size);

        let s = self.input_size as i64;
        let shape = [1_i64, s, s, 3];
        let data: Vec<f32> = arr.iter().copied().collect();
        let tensor = Tensor::from_array((shape, data))?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;

        let total = out_data.len();
        if total != self.tags.len() {
            return Err(TaggerError::OutputMismatch {
                expected: self.tags.len(),
                actual: total,
            });
        }

        let mut results: Vec<AutoTag> = Vec::new();
        for (i, &score) in out_data.iter().enumerate() {
            if score < threshold {
                continue;
            }
            let tag = &self.tags[i];
            results.push(AutoTag {
                tag: tag.name.clone(),
                score,
                category: tag.category.clone(),
            });
        }
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }
}

/// Light-build stub: API-compatible with the real [`Tagger`] so callers
/// (CLI/GUI) need no `cfg`, but every operation returns
/// [`TaggerError::Unsupported`]. Local WD14 inference requires the `onnx`
/// feature (the "full" build).
#[cfg(not(feature = "onnx"))]
pub struct Tagger(());

#[cfg(not(feature = "onnx"))]
impl Tagger {
    pub fn from_profile(_profile: &TaggerProfile) -> Result<Self, TaggerError> {
        Err(TaggerError::Unsupported)
    }

    pub fn num_tags(&self) -> usize {
        0
    }

    pub fn tag_image(
        &mut self,
        _image_path: &Path,
        _threshold: f32,
    ) -> Result<Vec<AutoTag>, TaggerError> {
        Err(TaggerError::Unsupported)
    }
}
