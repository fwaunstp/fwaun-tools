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

use crate::workflow::{self, CustomTemplate};
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
    Custom(CustomTemplate),
}

/// Node id of the `SaveImage` node in the built-in graph.
const BUILTIN_SAVE_NODE: &str = "13";
/// Node id of the `LoadImage` node in the built-in graph.
const BUILTIN_LOAD_NODE: &str = "10";

impl Workflow {
    fn save_node(&self) -> &str {
        match self {
            Workflow::Builtin { .. } => BUILTIN_SAVE_NODE,
            Workflow::Custom(template) => template.save_node(),
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
            Workflow::Custom(template) => template.with_input(input_image),
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
            Some(path) => Workflow::Custom(CustomTemplate::from_file(path)?),
            None => {
                let model = opts.upscale_model.clone().ok_or_else(|| {
                    ComfyError::Config(
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
        let uploaded = workflow::upload_file(&self.client, path)?;
        let graph = self.workflow.build_graph(&uploaded);
        let raw = workflow::run_graph(
            &self.client,
            &graph,
            self.workflow.save_node(),
            self.timeout,
            self.poll,
        )?;
        workflow::fit_max_edge(raw, self.max_edge)
    }
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
        assert_eq!(
            g["11"]["inputs"]["model_name"],
            json!("RealESRGAN_x4plus.pth")
        );
        assert_eq!(g["12"]["inputs"]["upscale_model"], json!(["11", 0]));
        assert_eq!(wf.save_node(), BUILTIN_SAVE_NODE);
    }

    #[test]
    fn custom_template_injects_each_image() {
        let template = CustomTemplate::parse(json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "placeholder.png" } },
            "2": { "class_type": "UpscaleModelLoader", "inputs": { "upscale_model": "x.pth" } },
            "3": { "class_type": "ImageUpscaleWithModel",
                   "inputs": { "upscale_model": ["2", 0], "image": ["1", 0] } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["3", 0] } }
        }))
        .expect("valid template");
        let wf = Workflow::Custom(template);
        assert_eq!(wf.save_node(), "9");
        let g = wf.build_graph("dog.png");
        assert_eq!(g["1"]["inputs"]["image"], json!("dog.png"));
    }
}
