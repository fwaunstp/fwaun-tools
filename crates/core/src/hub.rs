//! HuggingFace Hub resolution. Both the tagger and captioner crates need to
//! turn `(repo, revision, [files...])` into a list of local cached file paths,
//! so the hf-hub interaction lives here in core to keep ML crates focused.
//!
//! Cache location follows hf-hub's defaults (`$HF_HOME/hub` or
//! `~/.cache/huggingface/hub`), shared with `huggingface_hub` / sd-scripts /
//! diffusers. Already-downloaded models are reused for free.
//!
//! We build the API via [`ApiBuilder::from_env`] so the standard HuggingFace
//! environment variables are honored: `HF_HOME` (cache location) and
//! `HF_ENDPOINT` (base URL, e.g. `https://hf-mirror.com` for users who cannot
//! reach `huggingface.co` directly).

use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    #[error("hf-hub: {0}")]
    Api(String),
}

impl From<hf_hub::api::sync::ApiError> for HubError {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        HubError::Api(e.to_string())
    }
}

/// Fetch `files` from `repo` (optionally pinned to `revision`) into the local
/// hf-hub cache and return their absolute paths in the same order.
pub fn fetch_files(
    repo: &str,
    revision: Option<&str>,
    files: &[&str],
) -> Result<Vec<PathBuf>, HubError> {
    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let repo_handle = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Model,
            rev.to_string(),
        )),
        None => api.model(repo.to_string()),
    };
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        eprintln!("[hub] resolving {repo}:{f}");
        let path = repo_handle.get(f)?;
        out.push(path);
    }
    Ok(out)
}
