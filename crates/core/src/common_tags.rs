//! Dataset-wide common tag layer.
//!
//! A character LoRA usually wants the same handful of manual entries on
//! *every* image: the trigger word itself, plus `-foo` suppressions for the
//! traits that should be folded into that trigger (hair colour, eye colour,
//! …) rather than learned as separate tags. Writing those into each sidecar
//! means re-running `add-tag` every time images are added to the dataset.
//!
//! Instead they are declared once in `fwaun-tools.toml`:
//!
//! ```toml
//! common_tags = ["himeko", "-red_hair", "-yellow_eyes", "-horns"]
//! ```
//!
//! and applied as a *virtual* manual-tag layer sitting underneath each
//! image's own `manual_tags`. Nothing is written back to disk, so new images
//! are covered the moment they land in the directory, and dropping an entry
//! from the config drops it everywhere.
//!
//! Entries use exactly the manual-tag syntax: `foo` positive, `-foo`
//! suppression, `_foo` curation-only (see [`crate::sidecar`]).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;
use toml_edit::{Array, DocumentMut, Item, Value};

use crate::sidecar::{NEGATIVE_PREFIX, ORGANIZATIONAL_PREFIX, Sidecar};

/// Top-level key holding the layer in `fwaun-tools.toml`.
pub const COMMON_TAGS_KEY: &str = "common_tags";

/// Normalize a manual entry to the key used for override matching: trimmed,
/// with a leading `-` (suppression) and/or `_` (organizational) marker
/// stripped, lowercased.
///
/// `red_hair`, `-red_hair` and `_red_hair` therefore share the stem
/// `red_hair` — an image that names a tag in *any* form takes full control of
/// it, whatever form the common layer used.
pub fn tag_stem(entry: &str) -> String {
    let t = entry.trim();
    let t = t.strip_prefix(NEGATIVE_PREFIX).unwrap_or(t).trim_start();
    let t = t.strip_prefix(ORGANIZATIONAL_PREFIX).unwrap_or(t);
    t.trim().to_lowercase()
}

/// True if `entry` is a `-foo` suppression marker.
fn is_negative(entry: &str) -> bool {
    entry.trim_start().starts_with(NEGATIVE_PREFIX)
}

/// The resolved dataset-wide manual tag layer. Build one per directory from
/// [`crate::config::ProjectConfig::resolve_common_tags`]; the default is
/// empty, which makes every consumer behave exactly as it did before the
/// layer existed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommonTags {
    entries: Vec<String>,
}

impl CommonTags {
    /// Build from raw config entries, in declaration order.
    ///
    /// Blank entries and bare markers (`-`, `_`) are dropped. When two
    /// entries share a stem the first wins, so a config that lists both
    /// `red_hair` and `-red_hair` resolves the same way on every run instead
    /// of depending on iteration order downstream.
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut out: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for entry in entries {
            let trimmed = entry.as_ref().trim();
            let stem = tag_stem(trimmed);
            if stem.is_empty() || !seen.insert(stem) {
                continue;
            }
            out.push(trimmed.to_string());
        }
        Self { entries: out }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The normalized entries, in config order.
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// The manual-tag list in effect for `sc`: every common entry the image
    /// does not override, in config order, followed by the image's own
    /// entries verbatim.
    ///
    /// Common positives lead so a trigger word declared here heads the
    /// exported tag string. An image overrides a common entry by naming the
    /// same stem in any form — a common `-red_hair` is cancelled by a
    /// per-image `red_hair`, and a common `himeko` by a per-image `-himeko`.
    pub fn merged_manual_tags(&self, sc: &Sidecar) -> Vec<String> {
        if self.entries.is_empty() {
            return sc.manual_tags.clone();
        }
        let overridden: HashSet<String> = sc.manual_tags.iter().map(|t| tag_stem(t)).collect();
        let mut out: Vec<String> = self
            .entries
            .iter()
            .filter(|e| !overridden.contains(&tag_stem(e)))
            .cloned()
            .collect();
        out.extend(sc.manual_tags.iter().cloned());
        out
    }

    /// The common entry governing `tag`'s stem, if any — `Some("-red_hair")`
    /// for a suppression, `Some("himeko")` for a positive. Used by the GUI to
    /// label a chip as coming from the shared layer rather than the sidecar.
    pub fn entry_for(&self, tag: &str) -> Option<&str> {
        let stem = tag_stem(tag);
        if stem.is_empty() {
            return None;
        }
        self.entries
            .iter()
            .find(|e| tag_stem(e) == stem)
            .map(String::as_str)
    }

    /// True when the common layer contributes `tag` as a positive entry (so a
    /// caller about to add it to a sidecar can skip the redundant write).
    pub fn provides_positive(&self, tag: &str) -> bool {
        self.entry_for(tag).is_some_and(|e| !is_negative(e))
    }
}

// ───────── editing the layer in `fwaun-tools.toml` ─────────
//
// A bulk tag edit that spans the whole dataset belongs in the config, not in
// every sidecar — that's the point of the layer. The CLI's `add-tag` /
// `remove-tag` and the GUI's bulk panel therefore rewrite `common_tags` when
// their target is the dataset root, and go through these functions to do it.
//
// The rewrite is surgical (`toml_edit`, not a serialize round-trip) so the
// comments and layout of a hand-maintained config survive an edit made from
// the GUI.

#[derive(Debug, Error)]
pub enum CommonTagsError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse error on {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: Box<toml_edit::TomlError>,
    },
    #[error("`{COMMON_TAGS_KEY}` in {path} is not an array of strings — fix it by hand first")]
    NotAnArray { path: PathBuf },
}

/// What an [`add_to_config_file`] / [`remove_from_config_file`] call did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommonTagEdit {
    /// Tags whose presence in the array actually changed.
    pub applied: Vec<String>,
    /// Tags that were already in the requested state — nothing to do.
    pub unchanged: Vec<String>,
    /// The `common_tags` array as it stands after the edit.
    pub result: Vec<String>,
}

impl CommonTagEdit {
    pub fn changed(&self) -> bool {
        !self.applied.is_empty()
    }
}

/// Add `tags` to the `common_tags` array of the config at `path`, in the
/// exact form given (`foo` positive, `-foo` suppression, `_foo`
/// curation-only).
///
/// An entry naming the same tag in a *different* form is replaced in place,
/// so `add(["-red_hair"])` over an existing `red_hair` flips it rather than
/// leaving a contradictory pair behind. Entries already present verbatim are
/// reported as unchanged. With `dry_run` the file is left alone and the
/// returned edit describes what would have happened.
pub fn add_to_config_file(
    path: &Path,
    tags: &[String],
    dry_run: bool,
) -> Result<CommonTagEdit, CommonTagsError> {
    edit_config_file(path, dry_run, |arr, edit| {
        for tag in normalized_args(tags) {
            if index_of_exact(arr, &tag).is_some() {
                edit.unchanged.push(tag);
                continue;
            }
            match index_of_stem(arr, &tag_stem(&tag)) {
                Some(i) => {
                    arr.replace(i, tag.as_str());
                }
                None => arr.push(tag.as_str()),
            }
            edit.applied.push(tag);
        }
    })
}

/// Remove `tags` from the `common_tags` array of the config at `path`.
///
/// Matching is case-insensitive on the entry as written, so the leading `-`
/// is part of it: removing `red_hair` never removes `-red_hair`, and vice
/// versa. This mirrors `remove-tag`'s rule for sidecar manual entries.
pub fn remove_from_config_file(
    path: &Path,
    tags: &[String],
    dry_run: bool,
) -> Result<CommonTagEdit, CommonTagsError> {
    edit_config_file(path, dry_run, |arr, edit| {
        for tag in normalized_args(tags) {
            let key = tag.to_lowercase();
            let before = arr.len();
            arr.retain(|v| {
                v.as_str()
                    .map(|s| s.trim().to_lowercase() != key)
                    .unwrap_or(true)
            });
            if arr.len() == before {
                edit.unchanged.push(tag);
            } else {
                edit.applied.push(tag);
            }
        }
    })
}

/// Rename entries in the `common_tags` array of the config at `path`:
/// each `(from, to)` pair rewrites a matching entry **in place**, so the
/// order of a hand-maintained list survives a rename. Matching follows
/// [`remove_from_config_file`]'s rule (case-insensitive on the entry as
/// written, `-foo` distinct from `foo`); `to` is written exactly as given.
/// An entry already equal to `to` absorbs the renamed one instead of ending
/// up with two copies of it.
///
/// Pairs whose `from` isn't in the array are reported in
/// [`CommonTagEdit::unchanged`]; applied ones are reported as `"from -> to"`,
/// since neither half alone describes what happened.
pub fn replace_in_config_file(
    path: &Path,
    pairs: &[(String, String)],
    dry_run: bool,
) -> Result<CommonTagEdit, CommonTagsError> {
    edit_config_file(path, dry_run, |arr, edit| {
        for (from, to) in pairs {
            let (from, to) = (from.trim(), to.trim());
            if tag_stem(from).is_empty() || tag_stem(to).is_empty() {
                continue;
            }
            let label = format!("{from} -> {to}");
            let Some(i) = index_of_exact(arr, from) else {
                edit.unchanged.push(label);
                continue;
            };
            // `i` is the first entry naming this tag, so an exact hit here
            // means the list already says what the rename asks for.
            if arr
                .get(i)
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.trim() == to)
            {
                edit.unchanged.push(label);
                continue;
            }
            arr.replace(i, to);
            // Fold any other copy of `to` (pre-existing, or a second entry
            // that just got renamed onto it) into the slot we just wrote.
            let key = to.to_lowercase();
            let mut seen = false;
            arr.retain(|v| {
                let hit = v
                    .as_str()
                    .map(|s| s.trim().to_lowercase() == key)
                    .unwrap_or(false);
                if !hit {
                    return true;
                }
                let first = !seen;
                seen = true;
                first
            });
            edit.applied.push(label);
        }
    })
}

/// Trim the caller's tag arguments and drop blanks / bare markers.
fn normalized_args(tags: &[String]) -> Vec<String> {
    tags.iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !tag_stem(t).is_empty())
        .collect()
}

fn index_of_exact(arr: &Array, tag: &str) -> Option<usize> {
    let key = tag.to_lowercase();
    arr.iter()
        .position(|v| v.as_str().is_some_and(|s| s.trim().to_lowercase() == key))
}

fn index_of_stem(arr: &Array, stem: &str) -> Option<usize> {
    arr.iter()
        .position(|v| v.as_str().is_some_and(|s| tag_stem(s) == stem))
}

/// Load, mutate, and (unless `dry_run`) write back the `common_tags` array,
/// creating it when the config doesn't have one yet.
fn edit_config_file(
    path: &Path,
    dry_run: bool,
    mutate: impl FnOnce(&mut Array, &mut CommonTagEdit),
) -> Result<CommonTagEdit, CommonTagsError> {
    let text = fs::read_to_string(path).map_err(|source| CommonTagsError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut doc: DocumentMut = text.parse().map_err(|source| CommonTagsError::Parse {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;

    if doc.get(COMMON_TAGS_KEY).is_none() {
        doc[COMMON_TAGS_KEY] = Item::Value(Value::Array(Array::new()));
    }
    let was_multiline = doc[COMMON_TAGS_KEY].to_string().contains('\n');
    let arr = doc[COMMON_TAGS_KEY]
        .as_array_mut()
        .ok_or_else(|| CommonTagsError::NotAnArray {
            path: path.to_path_buf(),
        })?;

    let mut edit = CommonTagEdit::default();
    mutate(arr, &mut edit);
    // A long shared list is far easier to maintain one-per-line, and an
    // array the user already wrote that way stays that way.
    format_array(arr, was_multiline || arr.len() >= 3);
    edit.result = arr
        .iter()
        .filter_map(|v| v.as_str())
        .map(str::to_string)
        .collect();

    if edit.changed() && !dry_run {
        write_atomically(path, &doc.to_string())?;
    }
    Ok(edit)
}

/// Render the array one entry per line (with a trailing comma, so adding the
/// next entry is a one-line diff) or compactly on a single line.
fn format_array(arr: &mut Array, multiline: bool) {
    if !multiline || arr.is_empty() {
        arr.fmt();
        return;
    }
    for value in arr.iter_mut() {
        value.decor_mut().set_prefix("\n  ");
        value.decor_mut().set_suffix("");
    }
    arr.set_trailing_comma(true);
    arr.set_trailing("\n");
}

fn write_atomically(path: &Path, body: &str) -> Result<(), CommonTagsError> {
    let mut tmp_os = path.as_os_str().to_owned();
    tmp_os.push(".tmp");
    let tmp = PathBuf::from(tmp_os);
    fs::write(&tmp, body).map_err(|source| CommonTagsError::Io {
        path: tmp.clone(),
        source,
    })?;
    fs::rename(&tmp, path).map_err(|source| CommonTagsError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common(entries: &[&str]) -> CommonTags {
        CommonTags::new(entries)
    }

    fn sidecar(manual: &[&str]) -> Sidecar {
        Sidecar {
            manual_tags: manual.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn tag_stem_strips_both_markers_and_case() {
        assert_eq!(tag_stem("  Red_Hair "), "red_hair");
        assert_eq!(tag_stem("-Red_Hair"), "red_hair");
        assert_eq!(tag_stem("_Red_Hair"), "red_hair");
        assert_eq!(tag_stem("- red_hair"), "red_hair");
        assert_eq!(tag_stem("-"), "");
        assert_eq!(tag_stem("   "), "");
    }

    #[test]
    fn new_drops_blanks_and_keeps_first_of_conflicting_stems() {
        let c = common(&["himeko", "", "  ", "-", "-himeko", "-red_hair"]);
        assert_eq!(c.entries(), ["himeko", "-red_hair"]);
    }

    #[test]
    fn merged_puts_common_first_then_image_entries() {
        let c = common(&["himeko", "-red_hair"]);
        let sc = sidecar(&["smile", "-watermark"]);
        assert_eq!(
            c.merged_manual_tags(&sc),
            ["himeko", "-red_hair", "smile", "-watermark"]
        );
    }

    #[test]
    fn image_entry_overrides_common_entry_with_same_stem() {
        let c = common(&["himeko", "-red_hair"]);
        // This one image really is about the hair, and isn't Himeko.
        let sc = sidecar(&["red_hair", "-himeko"]);
        assert_eq!(c.merged_manual_tags(&sc), ["red_hair", "-himeko"]);
    }

    #[test]
    fn override_matches_across_marker_forms() {
        let c = common(&["-red_hair"]);
        // An organizational per-image entry still claims the stem.
        let sc = sidecar(&["_red_hair"]);
        assert_eq!(c.merged_manual_tags(&sc), ["_red_hair"]);
    }

    #[test]
    fn empty_layer_returns_image_entries_unchanged() {
        let c = CommonTags::default();
        let sc = sidecar(&["a", "-b"]);
        assert_eq!(c.merged_manual_tags(&sc), ["a", "-b"]);
    }

    #[test]
    fn entry_for_and_provides_positive() {
        let c = common(&["himeko", "-red_hair"]);
        assert_eq!(c.entry_for("HIMEKO"), Some("himeko"));
        assert_eq!(c.entry_for("red_hair"), Some("-red_hair"));
        assert_eq!(c.entry_for("smile"), None);
        assert!(c.provides_positive("himeko"));
        assert!(!c.provides_positive("red_hair"));
        assert!(!c.provides_positive("smile"));
    }

    // ───────── config-file editing ─────────

    /// Writes `body` to a uniquely-named temp config and removes it on drop.
    struct TempConfig(PathBuf);
    impl TempConfig {
        fn new(tag: &str, body: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "fwaun-tools-common-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join(crate::config::CONFIG_FILE);
            fs::write(&path, body).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
        fn read(&self) -> String {
            fs::read_to_string(&self.0).unwrap()
        }
    }
    impl Drop for TempConfig {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.0.parent().unwrap());
        }
    }

    fn args(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn add_creates_the_key_and_preserves_comments() {
        let cfg = TempConfig::new(
            "create",
            "# keep me\ndefault_profile = \"anima\"\n\n[export.anima]\nthreshold = 0.35\n",
        );
        let edit = add_to_config_file(cfg.path(), &args(&["himeko", "-red_hair"]), false).unwrap();
        assert_eq!(edit.applied, ["himeko", "-red_hair"]);
        assert_eq!(edit.result, ["himeko", "-red_hair"]);

        let text = cfg.read();
        assert!(text.contains("# keep me"), "comment lost: {text}");
        assert!(text.contains("[export.anima]"), "table lost: {text}");
        // Round-trips as the real config type, with the tags in place.
        let parsed: crate::config::ProjectConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed.common_tags, ["himeko", "-red_hair"]);
    }

    #[test]
    fn add_is_idempotent_and_reports_unchanged() {
        let cfg = TempConfig::new("idem", "common_tags = [\"himeko\"]\n");
        let edit = add_to_config_file(cfg.path(), &args(&["himeko"]), false).unwrap();
        assert!(!edit.changed());
        assert_eq!(edit.unchanged, ["himeko"]);
        assert_eq!(edit.result, ["himeko"]);
    }

    #[test]
    fn add_replaces_the_other_form_of_the_same_tag() {
        // Flipping a positive to a suppression must not leave both behind.
        let cfg = TempConfig::new("flip", "common_tags = [\"red_hair\", \"himeko\"]\n");
        let edit = add_to_config_file(cfg.path(), &args(&["-red_hair"]), false).unwrap();
        assert_eq!(edit.applied, ["-red_hair"]);
        // Replaced in place, so the original ordering survives.
        assert_eq!(edit.result, ["-red_hair", "himeko"]);
    }

    #[test]
    fn remove_matches_the_written_form_only() {
        let cfg = TempConfig::new("rm", "common_tags = [\"-red_hair\", \"himeko\"]\n");
        // `red_hair` must not take out the `-red_hair` marker …
        let edit = remove_from_config_file(cfg.path(), &args(&["red_hair"]), false).unwrap();
        assert!(!edit.changed());
        assert_eq!(edit.unchanged, ["red_hair"]);
        // … only `-red_hair` does.
        let edit = remove_from_config_file(cfg.path(), &args(&["-RED_HAIR"]), false).unwrap();
        assert_eq!(edit.applied, ["-RED_HAIR"]);
        assert_eq!(edit.result, ["himeko"]);
    }

    fn pairs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn replace_renames_in_place_and_keeps_the_order() {
        let cfg = TempConfig::new("ren", "common_tags = [\"himeko\", \"-red_hair\"]\n");
        let edit =
            replace_in_config_file(cfg.path(), &pairs(&[("HIMEKO", "himeko_v2")]), false).unwrap();
        assert_eq!(edit.applied, ["HIMEKO -> himeko_v2"]);
        assert_eq!(edit.result, ["himeko_v2", "-red_hair"]);
    }

    #[test]
    fn replace_reports_a_missing_source_without_adding_it() {
        let cfg = TempConfig::new("ren-miss", "common_tags = [\"himeko\"]\n");
        let edit =
            replace_in_config_file(cfg.path(), &pairs(&[("nope", "something")]), false).unwrap();
        assert!(!edit.changed());
        assert_eq!(edit.unchanged, ["nope -> something"]);
        assert_eq!(edit.result, ["himeko"]);
    }

    #[test]
    fn replace_collapses_onto_an_existing_target() {
        let cfg = TempConfig::new("ren-dup", "common_tags = [\"old\", \"keep\", \"new\"]\n");
        let edit = replace_in_config_file(cfg.path(), &pairs(&[("old", "new")]), false).unwrap();
        assert_eq!(edit.applied, ["old -> new"]);
        assert_eq!(edit.result, ["new", "keep"]);
    }

    #[test]
    fn replace_matches_the_written_form_only() {
        let cfg = TempConfig::new("ren-form", "common_tags = [\"-red_hair\"]\n");
        // The positive form isn't there, so this renames nothing …
        let edit =
            replace_in_config_file(cfg.path(), &pairs(&[("red_hair", "crimson_hair")]), false)
                .unwrap();
        assert!(!edit.changed());
        // … and the marker is renamed only when its `-` is passed too.
        let edit =
            replace_in_config_file(cfg.path(), &pairs(&[("-red_hair", "-crimson_hair")]), false)
                .unwrap();
        assert_eq!(edit.result, ["-crimson_hair"]);
    }

    #[test]
    fn dry_run_reports_without_writing() {
        let cfg = TempConfig::new("dry", "common_tags = [\"himeko\"]\n");
        let before = cfg.read();
        let edit = add_to_config_file(cfg.path(), &args(&["-red_hair"]), true).unwrap();
        assert_eq!(edit.applied, ["-red_hair"]);
        assert_eq!(edit.result, ["himeko", "-red_hair"]);
        assert_eq!(cfg.read(), before);
    }

    #[test]
    fn long_list_is_written_one_entry_per_line() {
        let cfg = TempConfig::new("multiline", "common_tags = []\n");
        add_to_config_file(
            cfg.path(),
            &args(&["himeko", "-red_hair", "-yellow_eyes"]),
            false,
        )
        .unwrap();
        let text = cfg.read();
        assert!(
            text.contains(
                "common_tags = [\n  \"himeko\",\n  \"-red_hair\",\n  \"-yellow_eyes\",\n]"
            ),
            "unexpected layout: {text}"
        );
        // Still valid TOML for the real parser.
        let parsed: crate::config::ProjectConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed.common_tags.len(), 3);
    }

    #[test]
    fn blank_and_bare_marker_args_are_ignored() {
        let cfg = TempConfig::new("blank", "common_tags = []\n");
        let edit = add_to_config_file(cfg.path(), &args(&["  ", "-", "_"]), false).unwrap();
        assert!(!edit.changed());
        assert!(edit.unchanged.is_empty());
        assert!(edit.result.is_empty());
    }

    #[test]
    fn non_array_common_tags_errors_instead_of_clobbering() {
        let cfg = TempConfig::new("badtype", "common_tags = \"himeko\"\n");
        match add_to_config_file(cfg.path(), &args(&["x"]), false) {
            Err(CommonTagsError::NotAnArray { .. }) => {}
            other => panic!("expected NotAnArray, got {other:?}"),
        }
        assert_eq!(cfg.read(), "common_tags = \"himeko\"\n");
    }
}
