//! Batch image upscaling over a ComfyUI server.
//!
//! Each image is uploaded, run through an upscale workflow, and the result
//! pulled back. Two workflow sources are supported:
//!
//! * **Built-in** — a model-based (ESRGAN-style) graph:
//!   `LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage`.
//!   Configure only the `upscale_model` filename (e.g. `RealESRGAN_x4plus.pth`).
//! * **Custom template** — an API-format workflow JSON exported from ComfyUI
//!   (*Save (API Format)*). The single `LoadImage` node's `image` input is
//!   rewritten per image; the result is read from the `SaveImage` node. This
//!   covers anything ComfyUI can do (Ultimate SD Upscale, tiled diffusion, …).
//!
//! Model upscalers emit a fixed multiplier (usually ×4). [`Options::max_edge`]
//! optionally shrinks the result so its longest edge fits a cap, keeping a
//! dataset's resolutions bounded.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::{Client, ComfyError, Result};

/// How to run the upscale. Mirrors the configurable knobs; the CLI builds this
/// from an `[upscaler.<name>]` profile plus flag overrides.
#[derive(Debug, Clone)]
pub struct Options {
    /// ComfyUI server root, e.g. `http://127.0.0.1:8188`.
    pub base_url: String,
    /// Upscale-model filename for the built-in workflow. Ignored (and may be
    /// `None`) when `workflow_template` is set.
    pub upscale_model: Option<String>,
    /// API-format workflow JSON to use instead of the built-in graph.
    pub workflow_template: Option<PathBuf>,
    /// Cap the upscaled result's longest edge to this many pixels (downscaling
    /// with Lanczos3 when exceeded). `0` keeps the model's native output size.
    pub max_edge: u32,
    /// Per-request / whole-job timeout in seconds.
    pub timeout_secs: u64,
    /// Pause between `/history` polls, in milliseconds.
    pub poll_interval_ms: u64,
}

/// The upscale workflow graph, resolved once at construction.
#[derive(Debug, Clone)]
enum Workflow {
    /// Built-in ESRGAN-style graph parameterized by the model filename.
    Builtin { model: String },
    /// A user template with the detected `LoadImage` / `SaveImage` node ids.
    Custom {
        graph: Value,
        load_node: String,
        save_node: String,
    },
}

/// Node id of the `SaveImage` node in the built-in graph.
const BUILTIN_SAVE_NODE: &str = "13";
/// Node id of the `LoadImage` node in the built-in graph.
const BUILTIN_LOAD_NODE: &str = "10";

impl Workflow {
    fn save_node(&self) -> &str {
        match self {
            Workflow::Builtin { .. } => BUILTIN_SAVE_NODE,
            Workflow::Custom { save_node, .. } => save_node,
        }
    }

    /// Produce the API-format graph for one input image (the value a
    /// `LoadImage` node's `image` input expects — see
    /// [`crate::UploadRef::load_image_value`]).
    fn build_graph(&self, input_image: &str) -> Value {
        match self {
            Workflow::Builtin { model } => json!({
                BUILTIN_LOAD_NODE: {
                    "class_type": "LoadImage",
                    "inputs": { "image": input_image }
                },
                "11": {
                    // The Load-Upscale-Model node's input widget is `model_name`;
                    // its output (type UPSCALE_MODEL) is what node 12 consumes.
                    "class_type": "UpscaleModelLoader",
                    "inputs": { "model_name": model }
                },
                "12": {
                    "class_type": "ImageUpscaleWithModel",
                    "inputs": { "upscale_model": ["11", 0], "image": [BUILTIN_LOAD_NODE, 0] }
                },
                BUILTIN_SAVE_NODE: {
                    "class_type": "SaveImage",
                    "inputs": { "images": ["12", 0], "filename_prefix": "fwaun_upscaled" }
                }
            }),
            Workflow::Custom {
                graph, load_node, ..
            } => {
                let mut g = graph.clone();
                // Safe: `load_node` was validated to exist with an object
                // `inputs` at construction. Overwrite just the `image` field.
                if let Some(inputs) = g
                    .get_mut(load_node)
                    .and_then(|n| n.get_mut("inputs"))
                    .and_then(Value::as_object_mut)
                {
                    inputs.insert("image".into(), Value::String(input_image.to_string()));
                }
                g
            }
        }
    }
}

pub struct Upscaler {
    client: Client,
    workflow: Workflow,
    max_edge: u32,
    timeout: Duration,
    poll: Duration,
}

impl Upscaler {
    /// Build an upscaler from [`Options`], resolving the workflow up front so a
    /// bad template / missing model fails once, before any image is processed.
    pub fn new(opts: Options) -> Result<Self> {
        let timeout = Duration::from_secs(opts.timeout_secs.max(1));
        let workflow = match &opts.workflow_template {
            Some(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    ComfyError::Workflow(format!("reading {}: {e}", path.display()))
                })?;
                let graph: Value = serde_json::from_str(&text).map_err(|e| {
                    ComfyError::Workflow(format!("parsing {} as JSON: {e}", path.display()))
                })?;
                parse_template(graph)?
            }
            None => {
                let model = opts.upscale_model.clone().ok_or_else(|| {
                    ComfyError::Workflow(
                        "no upscale model configured: set `upscale_model` (e.g. \
                         RealESRGAN_x4plus.pth) or provide a workflow template"
                            .into(),
                    )
                })?;
                Workflow::Builtin { model }
            }
        };
        Ok(Self {
            client: Client::new(&opts.base_url, timeout),
            workflow,
            max_edge: opts.max_edge,
            timeout,
            poll: Duration::from_millis(opts.poll_interval_ms.max(50)),
        })
    }

    /// The upscale-model filenames the server offers (built-in workflow only
    /// uses one of these). Handy for validating config or listing choices.
    pub fn list_models(&self) -> Result<Vec<String>> {
        self.client.list_upscale_models()
    }

    /// Upscale one image file end-to-end and return the resulting PNG bytes
    /// (post-`max_edge`). Does not touch the filesystem beyond reading `path`.
    pub fn upscale_file(&self, path: &Path) -> Result<Vec<u8>> {
        let bytes = std::fs::read(path)?;
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.png");
        let uploaded = self.client.upload_image(filename, &bytes)?;
        let graph = self.workflow.build_graph(&uploaded.load_image_value());
        let prompt_id = self.client.queue_prompt(&graph)?;
        let image =
            self.client
                .wait_for_output(&prompt_id, self.workflow.save_node(), self.timeout, self.poll)?;
        let raw = self.client.download(&image)?;
        self.fit_max_edge(raw)
    }

    /// Shrink to `max_edge` if the longest side exceeds it; otherwise return
    /// the server bytes untouched (no needless re-encode).
    fn fit_max_edge(&self, raw: Vec<u8>) -> Result<Vec<u8>> {
        if self.max_edge == 0 {
            return Ok(raw);
        }
        let img = image::load_from_memory(&raw)?;
        let (w, h) = (img.width(), img.height());
        let long = w.max(h);
        if long <= self.max_edge {
            return Ok(raw);
        }
        let scale = self.max_edge as f32 / long as f32;
        let nw = ((w as f32 * scale).round() as u32).max(1);
        let nh = ((h as f32 * scale).round() as u32).max(1);
        let resized = img.resize(nw, nh, image::imageops::FilterType::Lanczos3);
        let mut out = std::io::Cursor::new(Vec::new());
        resized.write_to(&mut out, image::ImageFormat::Png)?;
        Ok(out.into_inner())
    }
}

/// Validate a user API-format graph and locate the single `LoadImage` node
/// (where each image is injected) and a `SaveImage` node (where the result is
/// read). Rejects the UI-format export up front with a pointer to the right
/// menu item — that mistake is common and otherwise fails obscurely.
fn parse_template(graph: Value) -> Result<Workflow> {
    let obj = graph.as_object().ok_or_else(|| {
        ComfyError::Workflow(
            "template is not a JSON object of nodes. Export via ComfyUI's \
             'Save (API Format)' — the plain workflow save is UI format and \
             won't work here."
                .into(),
        )
    })?;
    // UI-format exports carry a top-level `nodes` array; the API format is a
    // flat map of id → node.
    if obj.contains_key("nodes") && obj.get("nodes").is_some_and(Value::is_array) {
        return Err(ComfyError::Workflow(
            "this is a UI-format workflow (has a top-level \"nodes\" array). \
             Re-export with 'Save (API Format)' to get the API-format graph."
                .into(),
        ));
    }

    let ids_of = |class: &str| -> Vec<String> {
        obj.iter()
            .filter(|(_, n)| n.get("class_type").and_then(Value::as_str) == Some(class))
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>()
    };

    let loads = ids_of("LoadImage");
    let load_node = match loads.as_slice() {
        [one] => one.clone(),
        [] => {
            return Err(ComfyError::Workflow(
                "no LoadImage node in the template; the batch upscaler needs one \
                 LoadImage node to inject each dataset image into"
                    .into(),
            ));
        }
        many => {
            return Err(ComfyError::Workflow(format!(
                "template has {} LoadImage nodes ({}); it must have exactly one so \
                 the upscaler knows where to inject each image",
                many.len(),
                many.join(", "),
            )));
        }
    };

    let saves = ids_of("SaveImage");
    let save_node = saves.first().cloned().ok_or_else(|| {
        ComfyError::Workflow(
            "no SaveImage node in the template; add one so the upscaled result \
             can be retrieved"
                .into(),
        )
    })?;

    Ok(Workflow::Custom {
        graph,
        load_node,
        save_node,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_graph_injects_input_and_model() {
        let wf = Workflow::Builtin {
            model: "RealESRGAN_x4plus.pth".into(),
        };
        let g = wf.build_graph("cat.png");
        assert_eq!(g[BUILTIN_LOAD_NODE]["inputs"]["image"], json!("cat.png"));
        // The UpscaleModelLoader widget is `model_name`, and node 12 wires to
        // node 11's UPSCALE_MODEL output.
        assert_eq!(g["11"]["inputs"]["model_name"], json!("RealESRGAN_x4plus.pth"));
        assert_eq!(g["12"]["inputs"]["upscale_model"], json!(["11", 0]));
        assert_eq!(wf.save_node(), BUILTIN_SAVE_NODE);
    }

    #[test]
    fn custom_template_detects_nodes_and_injects() {
        let graph = json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "placeholder.png" } },
            "2": { "class_type": "UpscaleModelLoader", "inputs": { "upscale_model": "x.pth" } },
            "3": { "class_type": "ImageUpscaleWithModel",
                   "inputs": { "upscale_model": ["2", 0], "image": ["1", 0] } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["3", 0] } }
        });
        let wf = parse_template(graph).expect("valid template");
        match &wf {
            Workflow::Custom { load_node, save_node, .. } => {
                assert_eq!(load_node, "1");
                assert_eq!(save_node, "9");
            }
            _ => panic!("expected custom workflow"),
        }
        let g = wf.build_graph("dog.png");
        assert_eq!(g["1"]["inputs"]["image"], json!("dog.png"));
    }

    #[test]
    fn ui_format_export_is_rejected() {
        let graph = json!({ "nodes": [], "links": [], "version": 0.4 });
        let err = parse_template(graph).unwrap_err();
        assert!(matches!(err, ComfyError::Workflow(_)));
        assert!(err.to_string().contains("API Format"));
    }

    #[test]
    fn missing_load_image_is_rejected() {
        let graph = json!({
            "9": { "class_type": "SaveImage", "inputs": { "images": ["3", 0] } }
        });
        assert!(parse_template(graph).is_err());
    }

    #[test]
    fn multiple_load_image_is_rejected() {
        let graph = json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } },
            "2": { "class_type": "LoadImage", "inputs": { "image": "b.png" } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["1", 0] } }
        });
        assert!(parse_template(graph).is_err());
    }
}
