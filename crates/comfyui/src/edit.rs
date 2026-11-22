//! Batch image *editing* over a ComfyUI server — one prompt applied to every
//! image in a dataset.
//!
//! Two built-in graphs are available, picked by [`Backend`]:
//!
//! * [`Backend::Gemini`] drives ComfyUI's Gemini image API node
//!   ([`EDIT_NODE_CLASS`], shown in the UI as *Nano Banana Pro (Google Gemini
//!   Image)*), whose `model` widget covers both Nano Banana 2 and Nano Banana
//!   Pro:
//!
//!   ```text
//!   LoadImage → GeminiImage2Node(prompt, model, …) → SaveImage
//!   ```
//!
//! * [`Backend::MiniMaxH3`] runs MiniMax-H3-Image locally on the server's GPU,
//!   following ComfyUI's bundled *MiniMax-H3-Image: Image Edit* template (minus
//!   its optional prompt-enhancer LLM):
//!
//!   ```text
//!   LoadImage → ImageScaleToTotalPixels → GetImageSize ─┐
//!   UNETLoader / CLIPLoader / VAELoader ──→ MiniMaxH3ImageConditioning
//!     → KSampler (negative = ConditioningZeroOut) → VAEDecode → SaveImage
//!   ```
//!
//! The motivating case is a dataset of illustrated characters drawn against
//! illustrated backgrounds, re-rendered with photographic backgrounds so a LoRA
//! learns the "drawn subject over a real scene" look. The instruction ("replace
//! the background with a photorealistic one, keep the character exactly as
//! drawn") is the same for every image, so it lives in the profile rather than
//! being retyped per shot.
//!
//! As with [`crate::upscale`], a [`Options::workflow_template`] takes over
//! entirely when set: any API-format export with one `LoadImage` and a
//! `SaveImage` works, and the prompt is injected into a node carrying a string
//! `prompt` widget (auto-detected, or named via [`Options::prompt_node`]).
//!
//! **The Gemini node bills per image.** A run over a whole dataset is a real
//! charge on the ComfyUI account, so callers should offer a dry run and a way
//! to cap the batch before pointing it at a few hundred images. The MiniMax
//! backend costs only GPU time.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::workflow::{self, CustomTemplate};
use crate::{Client, ComfyError, Result};

/// ComfyUI class name of the Gemini image node the built-in graph drives. Its
/// `model` widget selects between Nano Banana 2 and Nano Banana Pro; run
/// [`Editor::list_models`] to read the exact strings this server offers.
pub const EDIT_NODE_CLASS: &str = "GeminiImage2Node";

/// ComfyUI class name of the local MiniMax-H3-Image edit conditioning node.
pub const MINIMAX_COND_CLASS: &str = "MiniMaxH3ImageConditioning";

/// Node ids of the built-in Gemini graph.
const BUILTIN_LOAD_NODE: &str = "1";
const BUILTIN_EDIT_NODE: &str = "2";
const BUILTIN_SAVE_NODE: &str = "3";

/// Node ids of the built-in MiniMax H3 graph.
mod mm {
    pub const LOAD: &str = "1";
    pub const SCALE: &str = "2";
    pub const SIZE: &str = "3";
    pub const UNET: &str = "4";
    pub const CLIP: &str = "5";
    pub const VAE: &str = "6";
    pub const COND: &str = "7";
    pub const NEG: &str = "8";
    pub const SAMPLER: &str = "9";
    pub const DECODE: &str = "10";
    pub const SAVE: &str = "11";
}

/// Which built-in graph does the edit. Irrelevant when
/// [`Options::workflow_template`] is set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    /// The Gemini image (Nano Banana) API node — paid, needs an API key.
    #[default]
    Gemini,
    /// MiniMax-H3-Image run locally, configured by [`MiniMaxH3Options`].
    MiniMaxH3,
}

/// Knobs for [`Backend::MiniMaxH3`]. Defaults live in the config layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3Options {
    /// `UNETLoader` diffusion model file.
    pub unet: String,
    /// `CLIPLoader` text encoder file (loaded as type `minimax`).
    pub clip: String,
    /// `VAELoader` VAE file.
    pub vae: String,
    /// Target area, in megapixels, each input is rescaled to before editing.
    pub megapixels: f64,
    pub steps: u32,
    pub cfg: f64,
    pub sampler: String,
    pub scheduler: String,
}

/// How to run the edit. Mirrors the configurable knobs; the CLI builds this
/// from an `[editor.<name>]` profile plus flag overrides.
#[derive(Debug, Clone)]
pub struct Options {
    /// Which built-in graph to run when no `workflow_template` is set.
    pub backend: Backend,
    /// Settings for [`Backend::MiniMaxH3`]; ignored by the other backends.
    pub minimax_h3: MiniMaxH3Options,
    /// ComfyUI server root, e.g. `http://127.0.0.1:8188`.
    pub base_url: String,
    /// comfy.org account API key. Required in practice: the Gemini node is a
    /// paid API node and rejects a run without one ("Please login first to use
    /// this node"), since being logged into the web UI doesn't authenticate
    /// requests made over the HTTP API.
    pub api_key: Option<String>,
    /// The edit instruction applied to every image. Required by the built-in
    /// graph; may be empty with a `workflow_template` that bakes its own in.
    pub prompt: String,
    /// `model` widget value for the built-in graph, e.g.
    /// `"Nano Banana 2 (Gemini 3.1 Flash Image)"`. Ignored when
    /// `workflow_template` is set.
    pub model: String,
    /// `auto` matches each input image's aspect ratio — the right choice for a
    /// dataset pass, since anything else re-crops the subject.
    pub aspect_ratio: String,
    /// Output resolution tier: `1K`, `2K`, or `4K`.
    pub resolution: String,
    /// Passed through to the node. The API only makes a best effort at
    /// reproducing a seed, so this bounds drift rather than eliminating it.
    pub seed: u64,
    /// Override the node's default system prompt. `None` keeps the node's own.
    pub system_prompt: Option<String>,
    /// API-format workflow JSON to use instead of the built-in graph.
    pub workflow_template: Option<PathBuf>,
    /// Node id in that template to inject `prompt` into. `None` auto-detects
    /// the single node with a string `prompt` widget.
    pub prompt_node: Option<String>,
    /// Cap the result's longest edge to this many pixels (Lanczos3 downscale
    /// when exceeded). `0` keeps the model's native output size.
    pub max_edge: u32,
    /// Per-request / whole-job timeout in seconds.
    pub timeout_secs: u64,
    /// Pause between `/history` polls, in milliseconds.
    pub poll_interval_ms: u64,
}

/// The edit workflow graph, resolved once at construction.
#[derive(Debug, Clone)]
enum Workflow {
    /// Built-in `LoadImage → GeminiImage2Node → SaveImage` graph.
    Builtin {
        model: String,
        aspect_ratio: String,
        resolution: String,
        seed: u64,
        system_prompt: Option<String>,
    },
    /// Built-in local MiniMax-H3-Image graph.
    MiniMaxH3 { opts: MiniMaxH3Options, seed: u64 },
    /// A user template, with the node to inject the prompt into (`None` when
    /// no prompt was configured — the template supplies its own).
    Custom {
        template: CustomTemplate,
        prompt_node: Option<String>,
    },
}

impl Workflow {
    fn save_node(&self) -> &str {
        match self {
            Workflow::Builtin { .. } => BUILTIN_SAVE_NODE,
            Workflow::MiniMaxH3 { .. } => mm::SAVE,
            Workflow::Custom { template, .. } => template.save_node(),
        }
    }

    /// Produce the API-format graph for one input image and prompt.
    fn build_graph(&self, input_image: &str, prompt: &str) -> Value {
        match self {
            Workflow::Builtin {
                model,
                aspect_ratio,
                resolution,
                seed,
                system_prompt,
            } => {
                let mut edit = json!({
                    "class_type": EDIT_NODE_CLASS,
                    "inputs": {
                        "prompt": prompt,
                        "model": model,
                        "seed": seed,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        // Image-only: the node's other output is a text
                        // response we have nowhere to put.
                        "response_modalities": "IMAGE",
                        "images": [BUILTIN_LOAD_NODE, 0],
                    }
                });
                if let Some(sp) = system_prompt
                    && let Some(inputs) = edit.get_mut("inputs").and_then(Value::as_object_mut)
                {
                    inputs.insert("system_prompt".into(), Value::String(sp.clone()));
                }
                json!({
                    BUILTIN_LOAD_NODE: {
                        "class_type": "LoadImage",
                        "inputs": { "image": input_image }
                    },
                    BUILTIN_EDIT_NODE: edit,
                    BUILTIN_SAVE_NODE: {
                        "class_type": "SaveImage",
                        "inputs": {
                            "images": [BUILTIN_EDIT_NODE, 0],
                            "filename_prefix": "fwaun_edited"
                        }
                    }
                })
            }
            Workflow::MiniMaxH3 { opts, seed } => json!({
                mm::LOAD: {
                    "class_type": "LoadImage",
                    "inputs": { "image": input_image }
                },
                // The model is poor at low resolution, so every input is
                // brought to a fixed area first (sides snapped to the latent's
                // multiple of 32), as the stock template does.
                mm::SCALE: {
                    "class_type": "ImageScaleToTotalPixels",
                    "inputs": {
                        "image": [mm::LOAD, 0],
                        "upscale_method": "area",
                        "megapixels": opts.megapixels,
                        "resolution_steps": 32,
                    }
                },
                mm::SIZE: {
                    "class_type": "GetImageSize",
                    "inputs": { "image": [mm::SCALE, 0] }
                },
                mm::UNET: {
                    "class_type": "UNETLoader",
                    "inputs": { "unet_name": opts.unet, "weight_dtype": "default" }
                },
                mm::CLIP: {
                    "class_type": "CLIPLoader",
                    "inputs": { "clip_name": opts.clip, "type": "minimax", "device": "default" }
                },
                mm::VAE: {
                    "class_type": "VAELoader",
                    "inputs": { "vae_name": opts.vae }
                },
                // image_1 is the picture to edit (`<Picture 1>` in the
                // prompt); with the VAE connected it is also anchored as a
                // reference latent. Outputs: 0 = positive, 1 = empty latent
                // at width × height.
                mm::COND: {
                    "class_type": MINIMAX_COND_CLASS,
                    "inputs": {
                        "clip": [mm::CLIP, 0],
                        "vae": [mm::VAE, 0],
                        "images.image_1": [mm::SCALE, 0],
                        "prompt": prompt,
                        "width": [mm::SIZE, 0],
                        "height": [mm::SIZE, 1],
                    }
                },
                mm::NEG: {
                    "class_type": "ConditioningZeroOut",
                    "inputs": { "conditioning": [mm::COND, 0] }
                },
                mm::SAMPLER: {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": [mm::UNET, 0],
                        "positive": [mm::COND, 0],
                        "negative": [mm::NEG, 0],
                        "latent_image": [mm::COND, 1],
                        "seed": seed,
                        "steps": opts.steps,
                        "cfg": opts.cfg,
                        "sampler_name": opts.sampler,
                        "scheduler": opts.scheduler,
                        "denoise": 1.0,
                    }
                },
                mm::DECODE: {
                    "class_type": "VAEDecode",
                    "inputs": { "samples": [mm::SAMPLER, 0], "vae": [mm::VAE, 0] }
                },
                mm::SAVE: {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": [mm::DECODE, 0],
                        "filename_prefix": "fwaun_edited"
                    }
                }
            }),
            Workflow::Custom {
                template,
                prompt_node,
            } => {
                let mut g = template.with_input(input_image);
                if let Some(node) = prompt_node {
                    workflow::set_input(&mut g, node, "prompt", Value::String(prompt.to_string()));
                }
                g
            }
        }
    }
}

pub struct Editor {
    client: Client,
    workflow: Workflow,
    prompt: String,
    max_edge: u32,
    timeout: Duration,
    poll: Duration,
}

impl Editor {
    /// Build an editor from [`Options`], resolving the workflow up front so a
    /// bad template / missing prompt fails once, before any image is sent (and
    /// so before anything is billed).
    pub fn new(opts: Options) -> Result<Self> {
        let timeout = Duration::from_secs(opts.timeout_secs.max(1));
        let prompt = opts.prompt.trim().to_string();

        let workflow = match &opts.workflow_template {
            Some(path) => {
                let template = CustomTemplate::from_file(path)?;
                let prompt_node = if prompt.is_empty() {
                    // No prompt to inject — the template carries its own text.
                    None
                } else {
                    Some(resolve_prompt_node(&template, opts.prompt_node.as_deref())?)
                };
                Workflow::Custom {
                    template,
                    prompt_node,
                }
            }
            None => {
                if prompt.is_empty() {
                    return Err(ComfyError::Config(
                        "no edit prompt configured: set `prompt` on the editor profile \
                         (or pass --prompt) describing the edit to apply to every image"
                            .into(),
                    ));
                }
                match opts.backend {
                    Backend::Gemini => Workflow::Builtin {
                        model: opts.model.clone(),
                        aspect_ratio: opts.aspect_ratio.clone(),
                        resolution: opts.resolution.clone(),
                        seed: opts.seed,
                        system_prompt: opts.system_prompt.clone(),
                    },
                    Backend::MiniMaxH3 => {
                        let m = &opts.minimax_h3;
                        // ImageScaleToTotalPixels accepts at most 16 MP; catching
                        // it here fails before any image is uploaded.
                        if m.megapixels <= 0.0 || m.megapixels > 16.0 || m.megapixels.is_nan() {
                            return Err(ComfyError::Config(format!(
                                "minimax_h3: megapixels must be in (0, 16], got {}",
                                m.megapixels
                            )));
                        }
                        if m.steps == 0 {
                            return Err(ComfyError::Config(
                                "minimax_h3: steps must be at least 1".into(),
                            ));
                        }
                        Workflow::MiniMaxH3 {
                            opts: m.clone(),
                            seed: opts.seed,
                        }
                    }
                }
            }
        };

        Ok(Self {
            client: Client::new(&opts.base_url, timeout).with_comfy_api_key(opts.api_key.clone()),
            workflow,
            prompt,
            max_edge: opts.max_edge,
            timeout,
            poll: Duration::from_millis(opts.poll_interval_ms.max(50)),
        })
    }

    /// The `model` values this server's Gemini image node offers. Doubles as a
    /// cheap check that the API node is installed at all.
    pub fn list_models(&self) -> Result<Vec<String>> {
        self.client.list_node_enum(EDIT_NODE_CLASS, "model")
    }

    /// Whether this editor runs a paid API node that needs a comfy.org key:
    /// true for the built-in Gemini graph, false for the local MiniMax one.
    /// Unknown for a custom template, which is reported as `false` — callers
    /// should only warn about a missing key when this is true or a template
    /// is in use.
    pub fn needs_api_key(&self) -> bool {
        matches!(self.workflow, Workflow::Builtin { .. })
    }

    /// The prompt every image is edited with (trimmed; empty when a custom
    /// template supplies its own).
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Whether a comfy.org API key was supplied. Without one the built-in
    /// workflow's paid node will reject every image, so callers should say so
    /// before starting a batch.
    pub fn has_api_key(&self) -> bool {
        self.client.has_comfy_api_key()
    }

    /// Edit one image file end-to-end and return the resulting PNG bytes
    /// (post-`max_edge`). Does not touch the filesystem beyond reading `path`.
    pub fn edit_file(&self, path: &Path) -> Result<Vec<u8>> {
        self.edit_file_with_prompt(path, &self.prompt)
    }

    /// [`Self::edit_file`] with a per-image instruction, for callers that
    /// derive the prompt from the image's sidecar rather than the profile.
    pub fn edit_file_with_prompt(&self, path: &Path, prompt: &str) -> Result<Vec<u8>> {
        let uploaded = workflow::upload_file(&self.client, path)?;
        let graph = self.workflow.build_graph(&uploaded, prompt);
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

/// Pick the template node to write the prompt into: the caller's explicit
/// choice (validated to exist), else the sole node with a string `prompt`
/// widget. Ambiguity is an error rather than a guess — editing the wrong node
/// would silently produce a dataset of unedited images at API prices.
fn resolve_prompt_node(template: &CustomTemplate, explicit: Option<&str>) -> Result<String> {
    if let Some(id) = explicit {
        if !template.has_node(id) {
            return Err(ComfyError::Workflow(format!(
                "prompt_node `{id}` is not in the template; its nodes with a prompt \
                 widget are [{}]",
                template.prompt_nodes().join(", "),
            )));
        }
        return Ok(id.to_string());
    }
    match template.prompt_nodes().as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(ComfyError::Workflow(
            "a prompt was configured but the template has no node with a `prompt` \
             widget to put it in; either drop the prompt so the template's own text \
             is used, or set `prompt_node` to the node id to overwrite"
                .into(),
        )),
        many => Err(ComfyError::Workflow(format!(
            "template has {} nodes with a `prompt` widget ({}); set `prompt_node` to \
             the one the edit instruction belongs in",
            many.len(),
            many.join(", "),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_with(system_prompt: Option<&str>) -> Workflow {
        Workflow::Builtin {
            model: "Nano Banana 2 (Gemini 3.1 Flash Image)".into(),
            aspect_ratio: "auto".into(),
            resolution: "1K".into(),
            seed: 42,
            system_prompt: system_prompt.map(str::to_string),
        }
    }

    fn builtin() -> Workflow {
        builtin_with(None)
    }

    #[test]
    fn builtin_graph_wires_image_and_prompt() {
        let g = builtin().build_graph("cat.png", "replace the background");
        assert_eq!(g[BUILTIN_LOAD_NODE]["inputs"]["image"], json!("cat.png"));
        let edit = &g[BUILTIN_EDIT_NODE];
        assert_eq!(edit["class_type"], json!(EDIT_NODE_CLASS));
        assert_eq!(edit["inputs"]["prompt"], json!("replace the background"));
        assert_eq!(
            edit["inputs"]["model"],
            json!("Nano Banana 2 (Gemini 3.1 Flash Image)")
        );
        // `auto` keeps each image's own framing.
        assert_eq!(edit["inputs"]["aspect_ratio"], json!("auto"));
        assert_eq!(edit["inputs"]["response_modalities"], json!("IMAGE"));
        // The node's IMAGE output is slot 0; the LoadImage feeds its `images`.
        assert_eq!(edit["inputs"]["images"], json!([BUILTIN_LOAD_NODE, 0]));
        assert_eq!(
            g[BUILTIN_SAVE_NODE]["inputs"]["images"],
            json!([BUILTIN_EDIT_NODE, 0])
        );
        assert_eq!(builtin().save_node(), BUILTIN_SAVE_NODE);
    }

    #[test]
    fn builtin_omits_system_prompt_unless_set() {
        let g = builtin().build_graph("a.png", "p");
        assert!(
            g[BUILTIN_EDIT_NODE]["inputs"]
                .get("system_prompt")
                .is_none()
        );

        let g = builtin_with(Some("be literal")).build_graph("a.png", "p");
        assert_eq!(
            g[BUILTIN_EDIT_NODE]["inputs"]["system_prompt"],
            json!("be literal")
        );
    }

    fn gemini_template() -> CustomTemplate {
        CustomTemplate::parse(json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "placeholder.png" } },
            "4": { "class_type": EDIT_NODE_CLASS,
                   "inputs": { "prompt": "baked in", "model": "m", "images": ["1", 0] } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["4", 0] } }
        }))
        .expect("valid template")
    }

    #[test]
    fn custom_template_injects_image_and_prompt() {
        let wf = Workflow::Custom {
            template: gemini_template(),
            prompt_node: Some("4".into()),
        };
        let g = wf.build_graph("dog.png", "make the background photographic");
        assert_eq!(g["1"]["inputs"]["image"], json!("dog.png"));
        assert_eq!(
            g["4"]["inputs"]["prompt"],
            json!("make the background photographic")
        );
        assert_eq!(wf.save_node(), "9");
    }

    #[test]
    fn custom_template_keeps_its_own_prompt_when_none_configured() {
        let wf = Workflow::Custom {
            template: gemini_template(),
            prompt_node: None,
        };
        let g = wf.build_graph("dog.png", "");
        assert_eq!(g["4"]["inputs"]["prompt"], json!("baked in"));
    }

    #[test]
    fn prompt_node_auto_detected() {
        let node = resolve_prompt_node(&gemini_template(), None).expect("sole prompt node");
        assert_eq!(node, "4");
    }

    #[test]
    fn ambiguous_prompt_node_is_an_error() {
        let template = CustomTemplate::parse(json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } },
            "4": { "class_type": EDIT_NODE_CLASS, "inputs": { "prompt": "x", "images": ["1", 0] } },
            "5": { "class_type": "CLIPTextEncode", "inputs": { "prompt": "y" } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["4", 0] } }
        }))
        .expect("valid template");
        let err = resolve_prompt_node(&template, None).unwrap_err();
        assert!(err.to_string().contains("prompt_node"));
        // An explicit choice resolves it.
        assert_eq!(resolve_prompt_node(&template, Some("5")).unwrap(), "5");
        // A typo'd choice is caught rather than silently ignored.
        assert!(resolve_prompt_node(&template, Some("55")).is_err());
    }

    #[test]
    fn no_prompt_widget_is_an_error() {
        let template = CustomTemplate::parse(json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["1", 0] } }
        }))
        .expect("valid template");
        assert!(resolve_prompt_node(&template, None).is_err());
    }

    fn minimax_opts() -> MiniMaxH3Options {
        MiniMaxH3Options {
            unet: "u.safetensors".into(),
            clip: "c.safetensors".into(),
            vae: "v.safetensors".into(),
            megapixels: 4.0,
            steps: 6,
            cfg: 1.0,
            sampler: "euler".into(),
            scheduler: "simple".into(),
        }
    }

    #[test]
    fn minimax_graph_wires_image_prompt_and_models() {
        let wf = Workflow::MiniMaxH3 {
            opts: minimax_opts(),
            seed: 7,
        };
        let g = wf.build_graph("cat.png", "Replace the background of <Picture 1>");
        assert_eq!(wf.save_node(), mm::SAVE);
        assert_eq!(g[mm::LOAD]["inputs"]["image"], json!("cat.png"));
        assert_eq!(g[mm::SCALE]["inputs"]["image"], json!([mm::LOAD, 0]));
        assert_eq!(g[mm::SCALE]["inputs"]["megapixels"], json!(4.0));
        assert_eq!(g[mm::UNET]["inputs"]["unet_name"], json!("u.safetensors"));
        assert_eq!(g[mm::CLIP]["inputs"]["clip_name"], json!("c.safetensors"));
        assert_eq!(g[mm::CLIP]["inputs"]["type"], json!("minimax"));
        assert_eq!(g[mm::VAE]["inputs"]["vae_name"], json!("v.safetensors"));

        let cond = &g[mm::COND];
        assert_eq!(cond["class_type"], json!(MINIMAX_COND_CLASS));
        assert_eq!(
            cond["inputs"]["prompt"],
            json!("Replace the background of <Picture 1>")
        );
        assert_eq!(cond["inputs"]["images.image_1"], json!([mm::SCALE, 0]));
        assert_eq!(cond["inputs"]["width"], json!([mm::SIZE, 0]));
        assert_eq!(cond["inputs"]["height"], json!([mm::SIZE, 1]));

        let ks = &g[mm::SAMPLER]["inputs"];
        assert_eq!(ks["seed"], json!(7));
        assert_eq!(ks["steps"], json!(6));
        assert_eq!(ks["positive"], json!([mm::COND, 0]));
        assert_eq!(ks["negative"], json!([mm::NEG, 0]));
        assert_eq!(ks["latent_image"], json!([mm::COND, 1]));
        assert_eq!(g[mm::SAVE]["inputs"]["images"], json!([mm::DECODE, 0]));

        // Every link points at a node that exists in the graph.
        let obj = g.as_object().unwrap();
        for node in obj.values() {
            for v in node["inputs"].as_object().unwrap().values() {
                if let Some([Value::String(id), _]) = v.as_array().map(Vec::as_slice) {
                    assert!(obj.contains_key(id), "dangling link to {id}");
                }
            }
        }
    }

    fn options(backend: Backend, prompt: &str) -> Options {
        Options {
            backend,
            minimax_h3: minimax_opts(),
            base_url: "http://127.0.0.1:8188".into(),
            api_key: None,
            prompt: prompt.into(),
            model: "m".into(),
            aspect_ratio: "auto".into(),
            resolution: "1K".into(),
            seed: 42,
            system_prompt: None,
            workflow_template: None,
            prompt_node: None,
            max_edge: 0,
            timeout_secs: 600,
            poll_interval_ms: 750,
        }
    }

    #[test]
    fn backend_selects_graph_and_key_requirement() {
        let gemini = Editor::new(options(Backend::Gemini, "p")).expect("gemini editor");
        assert!(gemini.needs_api_key());
        let local = Editor::new(options(Backend::MiniMaxH3, "p")).expect("minimax editor");
        assert!(!local.needs_api_key());
        assert_eq!(local.workflow.save_node(), mm::SAVE);
    }

    #[test]
    fn minimax_rejects_degenerate_settings() {
        let mut o = options(Backend::MiniMaxH3, "p");
        o.minimax_h3.megapixels = 0.0;
        assert!(Editor::new(o).is_err());
        let mut o = options(Backend::MiniMaxH3, "p");
        o.minimax_h3.megapixels = 20.0;
        assert!(Editor::new(o).is_err());
        let mut o = options(Backend::MiniMaxH3, "p");
        o.minimax_h3.steps = 0;
        assert!(Editor::new(o).is_err());
        // The prompt requirement applies to every built-in backend.
        assert!(Editor::new(options(Backend::MiniMaxH3, " ")).is_err());
    }

    #[test]
    fn builtin_requires_a_prompt() {
        let made = Editor::new(Options {
            backend: Backend::Gemini,
            minimax_h3: minimax_opts(),
            base_url: "http://127.0.0.1:8188".into(),
            api_key: None,
            prompt: "   ".into(),
            model: "m".into(),
            aspect_ratio: "auto".into(),
            resolution: "1K".into(),
            seed: 42,
            system_prompt: None,
            workflow_template: None,
            prompt_node: None,
            max_edge: 0,
            timeout_secs: 600,
            poll_interval_ms: 750,
        });
        match made {
            Err(e) => assert!(e.to_string().contains("prompt")),
            Ok(_) => panic!("a whitespace-only prompt must not build a built-in editor"),
        }
    }
}
