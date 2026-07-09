//! Booru API tag fetching. Computes the image's MD5 hash and looks it up on
//! a booru-style API to recover the tags already curated by humans.
//!
//! Currently supports Danbooru (`https://danbooru.donmai.us`). The shape of
//! [`BooruClient`] is generic enough that other boorus (Gelbooru, Konachan,
//! Yande.re) can be added with adapter functions later.

use std::path::Path;

use fwaun_tagger_core::sidecar::{BooruInfo, BooruTag};
use chrono::Utc;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BooruError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("http: {0}")]
    Http(String),
    #[error("post not found for md5 {0}")]
    NotFound(String),
}

pub struct BooruClient {
    base_url: String,
    user_agent: String,
}

impl BooruClient {
    pub fn danbooru() -> Self {
        Self {
            base_url: "https://danbooru.donmai.us".to_string(),
            user_agent: format!("fwaun-tagger/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn fetch_for_image(
        &self,
        path: &Path,
    ) -> Result<(Vec<BooruTag>, BooruInfo), BooruError> {
        let bytes = std::fs::read(path)?;
        let digest = md5::compute(&bytes);
        let hex = format!("{:x}", digest);
        let url = format!(
            "{}/posts.json?tags=md5:{}&limit=1",
            self.base_url, hex
        );

        let posts: Vec<DanbooruPost> = ureq::get(&url)
            .set("User-Agent", &self.user_agent)
            .call()
            .map_err(|e| BooruError::Http(e.to_string()))?
            .into_json()
            .map_err(|e| BooruError::Http(e.to_string()))?;

        let post = posts
            .into_iter()
            .next()
            .ok_or_else(|| BooruError::NotFound(hex.clone()))?;

        let mut tags: Vec<BooruTag> = Vec::new();
        for (cat, s) in [
            ("artist", &post.tag_string_artist),
            ("copyright", &post.tag_string_copyright),
            ("character", &post.tag_string_character),
            ("general", &post.tag_string_general),
            ("meta", &post.tag_string_meta),
        ] {
            for tag in s.split_whitespace() {
                tags.push(BooruTag {
                    tag: tag.to_string(),
                    category: cat.to_string(),
                });
            }
        }

        let info = BooruInfo {
            source: "danbooru".to_string(),
            post_id: Some(post.id),
            post_url: Some(format!("{}/posts/{}", self.base_url, post.id)),
            fetched_at: Utc::now(),
        };
        Ok((tags, info))
    }
}

#[derive(Debug, Deserialize)]
struct DanbooruPost {
    id: u64,
    #[serde(default)]
    tag_string_artist: String,
    #[serde(default)]
    tag_string_copyright: String,
    #[serde(default)]
    tag_string_character: String,
    #[serde(default)]
    tag_string_general: String,
    #[serde(default)]
    tag_string_meta: String,
}
