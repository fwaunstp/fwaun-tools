//! Thin HTTP client over the ComfyUI server API. Endpoint shapes follow the
//! stock ComfyUI server (`/upload/image`, `/prompt`, `/history/{id}`, `/view`,
//! `/object_info`).

use std::io::Read;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ComfyError, Result};

/// Where an uploaded input image lives on the server, as returned by
/// `POST /upload/image`. `subfolder` is usually empty.
#[derive(Debug, Clone, Deserialize)]
pub struct UploadRef {
    pub name: String,
    #[serde(default)]
    pub subfolder: String,
    #[serde(default, rename = "type")]
    pub kind: String,
}

impl UploadRef {
    /// The value a `LoadImage` node's `image` input expects: `name` alone, or
    /// `subfolder/name` when the upload landed in a subfolder.
    pub fn load_image_value(&self) -> String {
        if self.subfolder.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.subfolder, self.name)
        }
    }
}

/// A produced output image, as listed under a node's `images` in `/history`.
#[derive(Debug, Clone)]
pub struct ImageRef {
    pub filename: String,
    pub subfolder: String,
    pub kind: String,
}

pub struct Client {
    base_url: String,
    client_id: String,
    agent: ureq::Agent,
}

impl Client {
    /// Connect to a ComfyUI server. `base_url` is the server root, e.g.
    /// `http://127.0.0.1:8188`; a trailing slash is tolerated. `timeout` caps
    /// each individual HTTP request (upload, queue, one history poll, one
    /// download) — not the whole job.
    pub fn new(base_url: &str, timeout: Duration) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_read(timeout)
            .timeout_write(timeout)
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            // ComfyUI accepts any client_id string; it only scopes websocket
            // progress events, which we don't use (we poll /history instead).
            client_id: format!("fwaun-tools-{}", std::process::id()),
            agent,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `POST /upload/image` (multipart). Uploads `bytes` as `filename` into the
    /// server's `input/` dir with `overwrite=true`, so re-runs reuse the slot
    /// instead of accumulating `foo (1).png` copies.
    pub fn upload_image(&self, filename: &str, bytes: &[u8]) -> Result<UploadRef> {
        let boundary = format!("----fwauntoolsboundary{}", std::process::id());
        let mut body: Vec<u8> = Vec::with_capacity(bytes.len() + 512);
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; \
                 filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
        for (name, value) in [("type", "input"), ("overwrite", "true")] {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; \
                     name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let url = format!("{}/upload/image", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send_bytes(&body)
            .map_err(http_err)?;
        Ok(resp.into_json()?)
    }

    /// `POST /prompt`. Queues a workflow graph (API format) and returns its
    /// `prompt_id`. Surfaces per-node validation failures as
    /// [`ComfyError::NodeErrors`].
    pub fn queue_prompt(&self, graph: &Value) -> Result<String> {
        let url = format!("{}/prompt", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .send_json(json!({ "prompt": graph, "client_id": self.client_id }))
            .map_err(http_err)?;
        let v: Value = resp.into_json()?;
        if let Some(errs) = v.get("node_errors").and_then(Value::as_object)
            && !errs.is_empty()
        {
            return Err(ComfyError::NodeErrors(
                serde_json::to_string(errs).unwrap_or_else(|_| "<unprintable>".into()),
            ));
        }
        v.get("prompt_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ComfyError::Http("no prompt_id in /prompt response".into()))
    }

    /// Poll `GET /history/{prompt_id}` until the run produces an output image
    /// on `save_node` (falling back to any node that emitted images), erroring
    /// on a reported job failure or when `timeout` elapses. `poll` is the pause
    /// between polls.
    pub fn wait_for_output(
        &self,
        prompt_id: &str,
        save_node: &str,
        timeout: Duration,
        poll: Duration,
    ) -> Result<ImageRef> {
        let url = format!("{}/history/{}", self.base_url, prompt_id);
        let deadline = Instant::now() + timeout;
        loop {
            let v: Value = self.agent.get(&url).call().map_err(http_err)?.into_json()?;
            if let Some(entry) = v.get(prompt_id) {
                if let Some("error") = entry
                    .pointer("/status/status_str")
                    .and_then(Value::as_str)
                {
                    let detail = entry.get("status").map(Value::to_string).unwrap_or_default();
                    return Err(ComfyError::Http(format!("ComfyUI job errored: {detail}")));
                }
                if let Some(img) = extract_output(entry, save_node) {
                    return Ok(img);
                }
                // Completed but nothing matched — the graph has no reachable
                // image output; polling longer won't change that.
                if entry
                    .pointer("/status/completed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return Err(ComfyError::NoOutput(prompt_id.to_string()));
                }
            }
            if Instant::now() >= deadline {
                return Err(ComfyError::Timeout(timeout));
            }
            std::thread::sleep(poll);
        }
    }

    /// `GET /view`. Downloads the raw bytes of a produced image.
    pub fn download(&self, image: &ImageRef) -> Result<Vec<u8>> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base_url,
            enc(&image.filename),
            enc(&image.subfolder),
            enc(&image.kind),
        );
        let resp = self.agent.get(&url).call().map_err(http_err)?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf)?;
        Ok(buf)
    }

    /// The list of upscale-model filenames the server offers, read from
    /// `GET /object_info/UpscaleModelLoader` (the enum ComfyUI populates from
    /// `models/upscale_models/`). Sorted, case-insensitively.
    ///
    /// The node's input widget is named `model_name` (not `upscale_model` —
    /// that's the type of the *output* it feeds into `ImageUpscaleWithModel`).
    pub fn list_upscale_models(&self) -> Result<Vec<String>> {
        self.list_node_enum("UpscaleModelLoader", "model_name")
    }

    /// Read one combo/enum-typed input's allowed values from a node's
    /// `/object_info` entry (e.g. `model_name` of `UpscaleModelLoader`, or
    /// `ckpt_name` of `CheckpointLoaderSimple`). Generic so future ComfyUI
    /// batch features can discover their own model lists. On a name mismatch
    /// the error lists the node's actual input names, so the fix is obvious.
    pub fn list_node_enum(&self, class_type: &str, input: &str) -> Result<Vec<String>> {
        let url = format!("{}/object_info/{}", self.base_url, class_type);
        let v: Value = self.agent.get(&url).call().map_err(http_err)?.into_json()?;
        let node = v.get(class_type).ok_or_else(|| {
            ComfyError::Http(format!(
                "/object_info has no entry for `{class_type}` (is the node installed \
                 on this server?)"
            ))
        })?;

        // Look under both required and optional, tolerating either combo
        // serialization ComfyUI has shipped (see `combo_choices`).
        let spec = node
            .pointer(&format!("/input/required/{input}"))
            .or_else(|| node.pointer(&format!("/input/optional/{input}")));
        let mut names = match spec.and_then(combo_choices) {
            Some(n) => n,
            None => {
                let available = ["required", "optional"]
                    .into_iter()
                    .filter_map(|sect| node.pointer(&format!("/input/{sect}")))
                    .filter_map(Value::as_object)
                    .flat_map(|o| o.keys().cloned())
                    .collect::<Vec<_>>();
                return Err(ComfyError::Http(format!(
                    "could not find combo values at {class_type}.input.*.{input}; \
                     the node's actual inputs are [{}]",
                    available.join(", "),
                )));
            }
        };
        names.sort_by_key(|s| s.to_lowercase());
        Ok(names)
    }
}

/// Extract the choice strings from a combo input spec, tolerating both
/// serializations ComfyUI has used:
///
/// * **Legacy** — `[["a.pth", "b.pth"], {…opts}]`: the choices are the first
///   element (a bare list).
/// * **Current** — `["COMBO", {"options": ["a.pth", "b.pth"], …}]`: the type
///   name is the first element and the choices live under `options` (falling
///   back to `values`) in the trailing options object.
fn combo_choices(spec: &Value) -> Option<Vec<String>> {
    let arr = spec.as_array()?;
    let strs = |list: &[Value]| -> Vec<String> {
        list.iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect()
    };
    // Legacy: first element is the choices list itself.
    if let Some(list) = arr.first().and_then(Value::as_array) {
        return Some(strs(list));
    }
    // Current: choices under the options object.
    if let Some(opts) = arr.get(1).and_then(Value::as_object) {
        for key in ["options", "values"] {
            if let Some(list) = opts.get(key).and_then(Value::as_array) {
                return Some(strs(list));
            }
        }
    }
    None
}

/// Pick the output image from a `/history` entry: prefer the node we asked to
/// save on, else the first node that emitted a non-`temp` image.
fn extract_output(entry: &Value, save_node: &str) -> Option<ImageRef> {
    let outputs = entry.get("outputs")?.as_object()?;
    let node = outputs
        .get(save_node)
        .filter(|n| n.get("images").is_some())
        .or_else(|| outputs.values().find(|n| n.get("images").is_some()))?;
    let images = node.get("images")?.as_array()?;
    let img = images
        .iter()
        .find(|im| im.get("type").and_then(Value::as_str) != Some("temp"))
        .or_else(|| images.first())?;
    Some(ImageRef {
        filename: img.get("filename")?.as_str()?.to_string(),
        subfolder: img
            .get("subfolder")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        kind: img
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("output")
            .to_string(),
    })
}

/// Percent-encode a `/view` query value. Keeps RFC-3986 unreserved chars plus
/// `/` (subfolders are sent as a path-like value), encodes everything else so
/// spaces and other bytes in ComfyUI-chosen filenames survive.
fn enc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Fold a `ureq` error into [`ComfyError::Http`], pulling the response body out
/// of a non-2xx status so ComfyUI's validation detail isn't lost.
fn http_err(e: ureq::Error) -> ComfyError {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let body = body.trim();
            if body.is_empty() {
                ComfyError::Http(format!("HTTP {code}"))
            } else {
                ComfyError::Http(format!("HTTP {code}: {body}"))
            }
        }
        ureq::Error::Transport(t) => ComfyError::Http(t.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_choices_current_format() {
        // ComfyUI's current serialization: ["COMBO", {"options": [...]}].
        let spec = json!(["COMBO", {
            "multiselect": false,
            "options": ["b.pth", "A.safetensors", "c.pth"]
        }]);
        let got = combo_choices(&spec).expect("choices");
        assert_eq!(got, vec!["b.pth", "A.safetensors", "c.pth"]);
    }

    #[test]
    fn combo_choices_legacy_format() {
        // Legacy serialization: [[choices], {opts}].
        let spec = json!([["x.pth", "y.pth"], {}]);
        let got = combo_choices(&spec).expect("choices");
        assert_eq!(got, vec!["x.pth", "y.pth"]);
    }

    #[test]
    fn combo_choices_values_key_fallback() {
        let spec = json!(["COMBO", { "values": ["m.pth"] }]);
        assert_eq!(combo_choices(&spec).unwrap(), vec!["m.pth"]);
    }

    #[test]
    fn combo_choices_none_when_no_list() {
        let spec = json!(["INT", { "default": 1, "min": 0 }]);
        assert!(combo_choices(&spec).is_none());
    }

    #[test]
    fn extract_output_prefers_save_node() {
        let entry = json!({
            "outputs": {
                "7": { "images": [{ "filename": "prev.png", "subfolder": "", "type": "temp" }] },
                "13": { "images": [{ "filename": "out.png", "subfolder": "s", "type": "output" }] }
            }
        });
        let img = extract_output(&entry, "13").expect("image");
        assert_eq!(img.filename, "out.png");
        assert_eq!(img.subfolder, "s");
        assert_eq!(img.kind, "output");
    }

    #[test]
    fn extract_output_falls_back_to_any_image_node() {
        let entry = json!({
            "outputs": { "9": { "images": [{ "filename": "z.png", "subfolder": "", "type": "output" }] } }
        });
        // Asked for node "13" (absent) → falls back to the node that has images.
        let img = extract_output(&entry, "13").expect("image");
        assert_eq!(img.filename, "z.png");
    }
}
