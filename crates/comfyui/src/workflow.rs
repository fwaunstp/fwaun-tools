//! Pieces shared by every ComfyUI-backed batch pass.
//!
//! Each pass ([`crate::upscale`], [`crate::edit`], …) has the same shape: take
//! one dataset image, push it through a workflow graph on the server, pull the
//! result back. What differs is only the graph in the middle. This module holds
//! the parts that don't:
//!
//! * [`CustomTemplate`] — a user-supplied API-format workflow export, validated
//!   once and rewritten per image.
//! * [`upload_file`] / [`run_graph`] — the upload → queue → poll → download
//!   round trip.
//! * [`fit_max_edge`] — the optional post-run downscale that keeps a dataset's
//!   resolutions bounded.

use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use crate::{Client, ComfyError, Result};

/// A user API-format graph (ComfyUI's *Save (API Format)* export) plus the node
/// ids a batch pass needs: where to inject each dataset image, and where to
/// read the result from.
#[derive(Debug, Clone)]
pub struct CustomTemplate {
    graph: Value,
    load_node: String,
    save_node: String,
}

impl CustomTemplate {
    /// Validate a graph and locate its single `LoadImage` node (where each
    /// image is injected) and a `SaveImage` node (where the result is read).
    ///
    /// Rejects the UI-format export up front with a pointer to the right menu
    /// item — that mistake is common and otherwise fails obscurely.
    pub fn parse(graph: Value) -> Result<Self> {
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
                    "no LoadImage node in the template; a batch pass needs one \
                     LoadImage node to inject each dataset image into"
                        .into(),
                ));
            }
            many => {
                return Err(ComfyError::Workflow(format!(
                    "template has {} LoadImage nodes ({}); it must have exactly one so \
                     the batch pass knows where to inject each image",
                    many.len(),
                    many.join(", "),
                )));
            }
        };

        let saves = ids_of("SaveImage");
        let save_node = saves.first().cloned().ok_or_else(|| {
            ComfyError::Workflow(
                "no SaveImage node in the template; add one so the result can be \
                 retrieved"
                    .into(),
            )
        })?;

        Ok(Self {
            graph,
            load_node,
            save_node,
        })
    }

    /// Read and parse a template file, tagging both failures with the path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ComfyError::Workflow(format!("reading {}: {e}", path.display())))?;
        let graph: Value = serde_json::from_str(&text).map_err(|e| {
            ComfyError::Workflow(format!("parsing {} as JSON: {e}", path.display()))
        })?;
        Self::parse(graph)
    }

    pub fn load_node(&self) -> &str {
        &self.load_node
    }

    pub fn save_node(&self) -> &str {
        &self.save_node
    }

    /// The graph with the `LoadImage` node pointed at `input_image` (the value
    /// [`crate::UploadRef::load_image_value`] produces).
    pub fn with_input(&self, input_image: &str) -> Value {
        let mut g = self.graph.clone();
        // Safe: `load_node` was validated to exist with an object `inputs` at
        // parse time. Overwrite just the `image` field.
        set_input(
            &mut g,
            &self.load_node,
            "image",
            Value::String(input_image.to_string()),
        );
        g
    }

    /// Whether the graph has a node with this id.
    pub fn has_node(&self, id: &str) -> bool {
        self.graph.get(id).is_some()
    }

    /// Ids of every node carrying a string-valued `prompt` input — the
    /// injection points for a batch pass that supplies its own prompt text.
    /// A prompt wired in from another node (an array `[node, slot]` link
    /// rather than a literal) is not a widget we can overwrite, so it doesn't
    /// count.
    pub fn prompt_nodes(&self) -> Vec<String> {
        let Some(obj) = self.graph.as_object() else {
            return Vec::new();
        };
        let mut ids: Vec<String> = obj
            .iter()
            .filter(|(_, n)| {
                n.pointer("/inputs/prompt")
                    .is_some_and(|p| p.as_str().is_some())
            })
            .map(|(k, _)| k.clone())
            .collect();
        ids.sort();
        ids
    }
}

/// Overwrite one widget input on a node, returning whether the node (and its
/// `inputs` object) existed.
pub fn set_input(graph: &mut Value, node: &str, field: &str, value: Value) -> bool {
    match graph
        .get_mut(node)
        .and_then(|n| n.get_mut("inputs"))
        .and_then(Value::as_object_mut)
    {
        Some(inputs) => {
            inputs.insert(field.to_string(), value);
            true
        }
        None => false,
    }
}

/// Upload one image file into the server's `input/` dir, returning the value a
/// `LoadImage` node's `image` input expects.
pub fn upload_file(client: &Client, path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("input.png");
    Ok(client.upload_image(filename, &bytes)?.load_image_value())
}

/// Queue `graph`, wait for `save_node` to produce an image, and download it.
pub fn run_graph(
    client: &Client,
    graph: &Value,
    save_node: &str,
    timeout: Duration,
    poll: Duration,
) -> Result<Vec<u8>> {
    let prompt_id = client.queue_prompt(graph)?;
    let image = client.wait_for_output(&prompt_id, save_node, timeout, poll)?;
    client.download(&image)
}

/// Shrink `raw` so its longest side fits `max_edge`; otherwise return the
/// server bytes untouched (no needless re-encode). `max_edge == 0` disables the
/// cap entirely.
pub fn fit_max_edge(raw: Vec<u8>, max_edge: u32) -> Result<Vec<u8>> {
    if max_edge == 0 {
        return Ok(raw);
    }
    let img = image::load_from_memory(&raw)?;
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    if long <= max_edge {
        return Ok(raw);
    }
    let scale = max_edge as f32 / long as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let resized = img.resize(nw, nh, image::imageops::FilterType::Lanczos3);
    let mut out = std::io::Cursor::new(Vec::new());
    resized.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn upscale_template() -> Value {
        json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "placeholder.png" } },
            "2": { "class_type": "UpscaleModelLoader", "inputs": { "upscale_model": "x.pth" } },
            "3": { "class_type": "ImageUpscaleWithModel",
                   "inputs": { "upscale_model": ["2", 0], "image": ["1", 0] } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["3", 0] } }
        })
    }

    #[test]
    fn detects_nodes_and_injects_input() {
        let t = CustomTemplate::parse(upscale_template()).expect("valid template");
        assert_eq!(t.load_node(), "1");
        assert_eq!(t.save_node(), "9");
        let g = t.with_input("dog.png");
        assert_eq!(g["1"]["inputs"]["image"], json!("dog.png"));
        // The original is untouched, so the template can be reused per image.
        assert_eq!(
            t.with_input("cat.png")["1"]["inputs"]["image"],
            json!("cat.png")
        );
    }

    #[test]
    fn ui_format_export_is_rejected() {
        let graph = json!({ "nodes": [], "links": [], "version": 0.4 });
        let err = CustomTemplate::parse(graph).unwrap_err();
        assert!(matches!(err, ComfyError::Workflow(_)));
        assert!(err.to_string().contains("API Format"));
    }

    #[test]
    fn missing_load_image_is_rejected() {
        let graph = json!({
            "9": { "class_type": "SaveImage", "inputs": { "images": ["3", 0] } }
        });
        assert!(CustomTemplate::parse(graph).is_err());
    }

    #[test]
    fn multiple_load_image_is_rejected() {
        let graph = json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } },
            "2": { "class_type": "LoadImage", "inputs": { "image": "b.png" } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["1", 0] } }
        });
        assert!(CustomTemplate::parse(graph).is_err());
    }

    #[test]
    fn missing_save_image_is_rejected() {
        let graph = json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } }
        });
        assert!(CustomTemplate::parse(graph).is_err());
    }

    #[test]
    fn prompt_nodes_finds_string_widgets_only() {
        let graph = json!({
            "1": { "class_type": "LoadImage", "inputs": { "image": "a.png" } },
            "4": { "class_type": "GeminiImage2Node",
                   "inputs": { "prompt": "make it real", "images": ["1", 0] } },
            // A prompt wired in from another node is a link, not a widget.
            "5": { "class_type": "SomeOtherNode", "inputs": { "prompt": ["6", 0] } },
            "9": { "class_type": "SaveImage", "inputs": { "images": ["4", 0] } }
        });
        let t = CustomTemplate::parse(graph).expect("valid template");
        assert_eq!(t.prompt_nodes(), vec!["4".to_string()]);
        assert!(t.has_node("4"));
        assert!(!t.has_node("77"));
    }

    #[test]
    fn set_input_reports_missing_node() {
        let mut g = upscale_template();
        assert!(set_input(&mut g, "1", "image", json!("x.png")));
        assert!(!set_input(&mut g, "404", "image", json!("x.png")));
    }

    #[test]
    fn fit_max_edge_passes_bytes_through_when_disabled() {
        let raw = vec![1, 2, 3];
        assert_eq!(fit_max_edge(raw.clone(), 0).expect("passthrough"), raw);
    }
}
