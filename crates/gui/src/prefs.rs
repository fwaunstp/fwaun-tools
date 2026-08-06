//! Machine-local GUI preferences (`gui-prefs.toml`).
//!
//! Distinct from `fwaun-tools.toml`, which describes a *dataset* and is meant
//! to be committed alongside it. Everything here is a property of this
//! machine / this install — the UI language, whether to keep a thumbnail
//! cache on disk and how big it may get — so it lives in the platform config
//! directory instead: `%APPDATA%` on Windows, `$XDG_CONFIG_HOME`
//! (falling back to `~/.config`) on Linux, `~/Library/Application Support`
//! on macOS.
//!
//! The file is read once at startup and rewritten whole on every change.
//! Unknown keys are dropped on rewrite — acceptable for a file only this app
//! writes, and it keeps the struct the single source of truth for defaults.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::i18n::Lang;

/// Everything persisted in `gui-prefs.toml`.
///
/// `language` stays an `Option<String>` (rather than `Lang`) so an absent or
/// unrecognized value falls back to host-locale detection instead of failing
/// the whole parse and silently resetting the cache settings too.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiPrefs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub thumb_cache: ThumbCachePrefs,
}

/// On-disk thumbnail cache settings. See [`crate::thumb_cache`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThumbCachePrefs {
    /// Master switch. On by default: the cache is a pure speed-up, and a
    /// bounded few hundred MB under the OS cache directory is what that
    /// directory is for.
    pub enabled: bool,
    /// Hard ceiling in MiB. Once the cache exceeds it, the oldest entries are
    /// dropped until it's back under the limit. `0` disables the check.
    pub limit_mb: u64,
    /// Drop entries this many days old regardless of size. `0` disables the
    /// check. Entries are stamped at write time, so a long-lived dataset's
    /// thumbnails do get regenerated once per period — the cost is one slow
    /// folder open, and it keeps datasets the user has moved on from from
    /// squatting on the whole budget.
    pub max_age_days: u32,
}

impl Default for ThumbCachePrefs {
    fn default() -> Self {
        Self {
            enabled: true,
            limit_mb: 512,
            max_age_days: 90,
        }
    }
}

impl GuiPrefs {
    /// The stored language, or the host locale when unset / unrecognized.
    pub fn lang(&self) -> Lang {
        match self.language.as_deref() {
            Some("ja") => Lang::Ja,
            Some("en") => Lang::En,
            _ => Lang::detect_host(),
        }
    }

    pub fn set_lang(&mut self, lang: Lang) {
        self.language = Some(lang.code().to_string());
    }
}

fn prefs_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("fwaun-tools").join("gui-prefs.toml"))
}

/// Best-effort load. A missing or unparseable file yields defaults rather
/// than an error: there is nothing useful the user could do about it at
/// startup, and defaults are always a working configuration.
pub fn load() -> GuiPrefs {
    prefs_path()
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str::<GuiPrefs>(&s).ok())
        .unwrap_or_default()
}

/// Best-effort save. Failures are silent for the same reason: the app stays
/// usable with the in-memory value, and a modal about an unwritable config
/// directory on every toggle would be worse than losing the preference.
pub fn save(prefs: &GuiPrefs) {
    let Some(path) = prefs_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(body) = toml::to_string_pretty(prefs) {
        let _ = fs::write(&path, body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_only_file_keeps_cache_defaults() {
        // The pre-0.7 file format was a bare `language = "..."`.
        let p: GuiPrefs = toml::from_str(r#"language = "ja""#).unwrap();
        assert_eq!(p.lang(), Lang::Ja);
        assert_eq!(p.thumb_cache, ThumbCachePrefs::default());
    }

    #[test]
    fn round_trips() {
        let mut p = GuiPrefs::default();
        p.set_lang(Lang::En);
        p.thumb_cache.enabled = false;
        p.thumb_cache.limit_mb = 64;
        let text = toml::to_string_pretty(&p).unwrap();
        assert_eq!(toml::from_str::<GuiPrefs>(&text).unwrap(), p);
    }

    #[test]
    fn unset_language_falls_back_to_host_detection() {
        let p = GuiPrefs::default();
        assert_eq!(p.lang(), Lang::detect_host());
    }
}
