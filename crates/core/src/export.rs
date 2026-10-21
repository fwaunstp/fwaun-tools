use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use rand::seq::SliceRandom;
use thiserror::Error;

use crate::common_tags::CommonTags;
use crate::config::{ExportProfile, TagGroup};
use crate::sidecar::{is_organizational, positive_entries, suppressed_stems, Sidecar};
use crate::tag_group::AffixSeed;

/// Extension of the per-image text files `export` writes unless `--ext`
/// says otherwise.
pub const DEFAULT_TEXT_EXTENSION: &str = "txt";

/// Separator between the two blocks of a combined (`tags+caption` /
/// `caption+tags`) export: a blank line. A single `\n` would not be a
/// block boundary for trainers that split a caption file on newlines into
/// variations (sd-scripts' `--shuffle_captions`), which would silently
/// train half a block as a full caption. The `tags` block itself is a
/// single comma-joined line (never contains newlines), so two consecutive
/// newlines are unambiguous.
pub const COMBINED_SEPARATOR: &str = "\n\n";

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "cannot derive an export text path from {path}: extension {ext:?} is empty or path-bearing"
    )]
    InvalidPath { path: PathBuf, ext: String },
}

/// What the `export` command writes into each text file, as one merged
/// block. The CLI's `--content` flag values are exactly the strings
/// returned by [`ExportContent::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportContent {
    /// Merged manual + auto + booru tags (the historical default), for
    /// tag-trained models (e.g. anime checkpoints via sd-scripts/musubi).
    /// Underscores are flattened to spaces and the tags comma-joined.
    Tags,
    /// The export caption — manual caption if set, else the joined active
    /// auto captions, with any configured prefixes/suffixes applied. For
    /// natural-language models (e.g. Qwen Image, Flux) that were never
    /// trained on danbooru-style tags. Images without a caption are skipped.
    Caption,
    /// Both blocks, tags first, separated by [`COMBINED_SEPARATOR`]. For
    /// models that take a tag-conditioned style/character embedding *and*
    /// a natural-language caption in one text input (e.g. MOSS Image).
    /// Skips images without a caption like [`Self::Caption`]; an image
    /// whose tag set is empty exports as caption-only.
    TagsCaption,
    /// Both blocks, caption first — the order some trainers' prompt
    /// templates expect (description, then trailing tags).
    CaptionTags,
}

impl ExportContent {
    /// The canonical CLI spelling of this content kind (also the string
    /// shown in `--help`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tags => "tags",
            Self::Caption => "caption",
            Self::TagsCaption => "tags+caption",
            Self::CaptionTags => "caption+tags",
        }
    }
}

/// Parses the canonical strings from [`ExportContent::as_str`],
/// case-insensitively; the historical kebab spellings (`tags-caption`) are
/// accepted too.
impl FromStr for ExportContent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = s.trim().replace('-', "+").to_lowercase();
        match v.as_str() {
            "tags" => Ok(Self::Tags),
            "caption" => Ok(Self::Caption),
            "tags+caption" => Ok(Self::TagsCaption),
            "caption+tags" => Ok(Self::CaptionTags),
            _ => Err(format!(
                "invalid content {s:?} (possible values: tags, caption, tags+caption, caption+tags)"
            )),
        }
    }
}


/// The default export text path for `image`: sibling in the same directory,
/// stem unchanged, [`DEFAULT_TEXT_EXTENSION`] extension. Kept as-is (an
/// image with no file name simply gets `txt` appended) for backward
/// compatibility — new callers should prefer [`export_text_path_with_ext`].
pub fn export_text_path(image: &Path) -> PathBuf {
    image.with_extension(DEFAULT_TEXT_EXTENSION)
}

/// The export text path for `image`: the sibling with the same parent and
/// stem but the given `extension` — normally `txt` (see
/// [`DEFAULT_TEXT_EXTENSION`]), but `export --ext` passes composite values
/// like `tags.txt` here, which is how several export kinds coexist in one
/// dataset folder without overwriting each other (`img.png` + `tags.txt` →
/// `<dir>/img.tags.txt`).
///
/// Built from `file_stem` + `set_extension` rather than
/// [`Path::with_extension`], and returns [`ExportError::InvalidPath`] when
/// the image has no file name to derive a stem from (a pathless path like
/// `<dir>/`, `..`, `.`, or a bare `C:\`) or when `extension` is empty or
/// itself contains a path separator.
pub fn export_text_path_with_ext(
    image: &Path,
    extension: &str,
) -> Result<PathBuf, ExportError> {
    let invalid = || ExportError::InvalidPath {
        path: image.to_path_buf(),
        ext: extension.to_string(),
    };
    if extension.is_empty() || extension.chars().any(std::path::is_separator) {
        return Err(invalid());
    }
    let mut out = image.to_path_buf();
    // `set_extension` refuses (returns false, leaving the path unchanged)
    // when the last component isn't a file name — root, `.`, `..`.
    if out.file_stem().is_none_or(|s| s.is_empty()) {
        return Err(invalid());
    }
    out.set_extension(extension);
    Ok(out)
}

/// Build the final ordered tag list for a single image, applying:
/// - the dataset-wide `common` layer underneath the image's own manual tags
/// - threshold + category exclusion to auto tags
/// - suppression of any auto/booru tag named in a `-foo` manual entry
/// - category prefix formatting (e.g. ANIMA artist `@`)
/// - dedup (manual wins on collision; comparison is prefix-stripped, lowercase)
/// - optional shuffle
///
/// Negative manual entries (`-foo`) are never emitted as positive tags.
/// Organizational manual entries (`_foo`) are curation-only and never
/// exported either, though they still count for tag-group classification.
///
/// Common-layer positives lead the list, so a trigger word declared in
/// `common_tags` heads every exported caption. Pass
/// [`CommonTags::default`] for no shared layer.
pub fn build_tags(sidecar: &Sidecar, profile: &ExportProfile, common: &CommonTags) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let manual = common.merged_manual_tags(sidecar);
    let suppressed = suppressed_stems(&manual);

    for raw in positive_entries(&manual) {
        let trimmed = raw.trim();
        if trimmed.is_empty() || is_organizational(trimmed) {
            continue;
        }
        let stem = normalize_stem(trimmed, profile);
        if seen.insert(stem) {
            out.push(trimmed.to_string());
        }
    }

    for at in &sidecar.auto_tags {
        if at.score < profile.threshold {
            continue;
        }
        if profile.exclude_categories.iter().any(|c| c == &at.category) {
            continue;
        }
        if suppressed.contains(&at.tag.to_lowercase()) {
            continue;
        }
        let formatted = format_external_tag(&at.tag, &at.category, profile);
        let stem = normalize_stem(&formatted, profile);
        if seen.insert(stem) {
            out.push(formatted);
        }
    }

    for bt in &sidecar.booru_tags {
        if profile.exclude_categories.iter().any(|c| c == &bt.category) {
            continue;
        }
        if suppressed.contains(&bt.tag.to_lowercase()) {
            continue;
        }
        let formatted = format_external_tag(&bt.tag, &bt.category, profile);
        let stem = normalize_stem(&formatted, profile);
        if seen.insert(stem) {
            out.push(formatted);
        }
    }

    if profile.shuffle {
        let mut rng = rand::rng();
        out.shuffle(&mut rng);
    }
    out
}

/// The tag list from [`build_tags`] rendered as training text: underscores
/// flattened to spaces, comma-joined — the exact string a `tags` export
/// writes. Empty tag sets yield an empty string (not `None`), matching the
/// historical `export_image` behavior of writing an empty file.
pub fn build_tags_text(tags: &[String]) -> String {
    tags.iter()
        .map(|t| t.replace('_', " "))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Caption text for export, with any configured `caption_prefixes` /
/// `caption_suffixes` applied.
///
/// The body comes from [`Sidecar::export_caption`] (manual caption wins,
/// else the joined active auto captions). For each `(tag, affix)` rule
/// whose tag matches one of the image's positive manual tags — compared
/// case-insensitively, ignoring a leading organizational `_` — the affix is
/// prepended (`caption_prefixes`) or appended (`caption_suffixes`) verbatim.
/// Matched affixes are emitted in the profile's key order (BTreeMap =
/// sorted), so output is deterministic even if several rules match. Tags
/// from the dataset-wide `common` layer match too, so a trigger word
/// declared once there can drive a caption affix on every image.
///
/// `seed` orders tag-group affixes that tie on `priority` — see
/// [`AffixSeed`]. It must be the same seed the captioner was primed with,
/// or the prefix baked into the generated body won't match the one export
/// prepends.
///
/// Returns `None` when the image has no caption body at all: a bare affix
/// without a caption isn't a useful training caption, so such images are
/// skipped by callers rather than emitted affix-only.
pub fn build_caption(
    sidecar: &Sidecar,
    profile: &ExportProfile,
    tag_groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
) -> Option<String> {
    let body = sidecar.export_caption()?;
    let present = present_caption_stems(sidecar, profile, common);
    // Tag-group affixes (keyed on a tag *combination*, priority-ordered) sit
    // outermost; legacy per-tag export-profile affixes nest just inside the
    // caption body. Both are honored so existing configs keep working.
    let group_prefix = crate::tag_group::resolved_caption_prefix(sidecar, tag_groups, common, seed);
    let group_suffix = crate::tag_group::resolved_caption_suffix(sidecar, tag_groups, common, seed);
    let legacy_prefix = matched_affixes(&profile.caption_prefixes, &present);
    let legacy_suffix = matched_affixes(&profile.caption_suffixes, &present);
    let prefix = format!("{group_prefix}{legacy_prefix}");
    let suffix = format!("{legacy_suffix}{group_suffix}");
    if prefix.is_empty() && suffix.is_empty() {
        Some(body)
    } else {
        Some(format!("{prefix}{body}{suffix}"))
    }
}

/// The full export text for `image` under `content`, in one merged block:
/// - [`ExportContent::Tags`] — [`build_tags_text`] over [`build_tags`]
/// - [`ExportContent::Caption`] — [`build_caption`]
/// - [`ExportContent::TagsCaption`] / [`ExportContent::CaptionTags`] —
///   both, in the order the variant name reads, joined by
///   [`COMBINED_SEPARATOR`]
///
/// Returns `None` when there is nothing to write *for this content*: no
/// caption body for any caption-carrying content (same skip rule as
/// [`build_caption`]). `Tags` never returns `None` (an empty tag set
/// exports as an empty file); in a combined export an empty tag set
/// degrades to caption-only.
pub fn build_export_text(
    sidecar: &Sidecar,
    profile: &ExportProfile,
    tag_groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
    content: ExportContent,
) -> Option<String> {
    let tags = || build_tags_text(&build_tags(sidecar, profile, common));
    let caption = || build_caption(sidecar, profile, tag_groups, common, seed);
    match content {
        ExportContent::Tags => Some(tags()),
        ExportContent::Caption => caption(),
        ExportContent::TagsCaption => {
            let t = tags();
            let c = caption()?;
            Some(if t.is_empty() {
                c
            } else {
                format!("{t}{COMBINED_SEPARATOR}{c}")
            })
        }
        ExportContent::CaptionTags => {
            let c = caption()?;
            let t = tags();
            Some(if t.is_empty() {
                c
            } else {
                format!("{c}{COMBINED_SEPARATOR}{t}")
            })
        }
    }
}

/// Stems of the image's positive manual tags (common layer included),
/// normalized for affix matching. Empty (and cheap) when neither affix table
/// is configured.
fn present_caption_stems(
    sidecar: &Sidecar,
    profile: &ExportProfile,
    common: &CommonTags,
) -> HashSet<String> {
    if profile.caption_prefixes.is_empty() && profile.caption_suffixes.is_empty() {
        return HashSet::new();
    }
    let manual = common.merged_manual_tags(sidecar);
    positive_entries(&manual)
        .map(caption_prefix_stem)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Concatenate the affixes whose tag stem is present, in map key order.
/// Empty when nothing matches.
fn matched_affixes(
    affixes: &BTreeMap<String, String>,
    present: &HashSet<String>,
) -> String {
    let mut out = String::new();
    for (tag, affix) in affixes {
        if present.contains(&caption_prefix_stem(tag)) {
            out.push_str(affix);
        }
    }
    out
}

/// Normalize a tag for caption-prefix matching: trim, drop a single leading
/// organizational `_`, lowercase. So a config key `realistic` matches a
/// sidecar tag `realistic` or the organizational `_realistic`.
fn caption_prefix_stem(s: &str) -> String {
    s.trim()
        .strip_prefix(crate::sidecar::ORGANIZATIONAL_PREFIX)
        .unwrap_or_else(|| s.trim())
        .to_lowercase()
}

/// Write `image`'s export text for `content` (see [`build_export_text`]) to
/// a sibling `<stem>.<ext>` (see [`export_text_path_with_ext`]).
///
/// Returns `Ok(None)` without touching the filesystem when there is nothing
/// to write for `content` — the no-caption skip described on
/// [`build_export_text`]. This is the single per-image entry point the CLI
/// `export` command uses for all four content kinds.
#[allow(clippy::too_many_arguments)] // one argument per input the command needs; callers use names
pub fn export_file(
    image: &Path,
    sidecar: &Sidecar,
    profile: &ExportProfile,
    tag_groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
    content: ExportContent,
    ext: &str,
) -> Result<Option<PathBuf>, ExportError> {
    let Some(text) = build_export_text(sidecar, profile, tag_groups, common, seed, content) else {
        return Ok(None);
    };
    Ok(Some(write_export_text(image, ext, text)?))
}

/// Write `image`'s merged tags (see [`build_tags`]) to `<image>.txt`. Thin
/// wrapper over [`build_export_text`] + [`export_text_path`] for existing
/// library callers; the CLI now goes through [`export_file`]. An empty
/// `tag_groups`/default seed means no affix processing — `Tags` never
/// consults either, so this parameterizes nothing observable here.
pub fn export_image(
    image: &Path,
    sidecar: &Sidecar,
    profile: &ExportProfile,
    common: &CommonTags,
) -> Result<PathBuf, ExportError> {
    let text = build_export_text(
        sidecar,
        profile,
        &BTreeMap::new(),
        common,
        AffixSeed::default(),
        ExportContent::Tags,
    )
    .expect("tags export never yields None");
    let out = export_text_path(image);
    fs::write(&out, text).map_err(|source| ExportError::Io {
        path: out.clone(),
        source,
    })?;
    Ok(out)
}

/// Write `image`'s export caption (see [`build_caption`]) to `<image>.txt`.
/// Returns `Ok(None)` without touching the filesystem when the image has no
/// caption body — callers should count that as skipped rather than writing
/// an empty file, the same way [`build_caption`] itself distinguishes "no
/// caption" from "caption with no affixes". Thin wrapper over
/// [`build_export_text`] for existing library callers; the CLI now goes
/// through [`export_file`].
pub fn export_caption_file(
    image: &Path,
    sidecar: &Sidecar,
    profile: &ExportProfile,
    tag_groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
) -> Result<Option<PathBuf>, ExportError> {
    let Some(text) = build_caption(sidecar, profile, tag_groups, common, seed) else {
        return Ok(None);
    };
    Ok(Some(write_export_text(
        image,
        DEFAULT_TEXT_EXTENSION,
        text,
    )?))
}

/// Write `text` to `image`'s sibling `<stem>.<ext>` and return its path.
fn write_export_text(image: &Path, ext: &str, text: String) -> Result<PathBuf, ExportError> {
    let out = export_text_path_with_ext(image, ext)?;
    fs::write(&out, text).map_err(|source| ExportError::Io {
        path: out.clone(),
        source,
    })?;
    Ok(out)
}

fn format_external_tag(tag: &str, category: &str, profile: &ExportProfile) -> String {
    match profile.category_prefix(category) {
        Some(p) => format!("{p}{tag}"),
        None => tag.to_string(),
    }
}

fn normalize_stem(s: &str, profile: &ExportProfile) -> String {
    let trimmed = s.trim();
    let mut current = trimmed;
    for prefix in profile.all_prefixes() {
        if let Some(stripped) = current.strip_prefix(prefix) {
            current = stripped;
            break;
        }
    }
    current.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidecar::{AutoTag, BooruTag};

    fn no_shuffle(mut p: ExportProfile) -> ExportProfile {
        p.shuffle = false;
        p
    }

    fn sidecar_with(tags: &[&str], caption: Option<&str>) -> Sidecar {
        let mut s = Sidecar {
            manual_tags: tags.iter().map(|t| t.to_string()).collect(),
            ..Default::default()
        };
        if let Some(c) = caption {
            s.set_caption("model", c);
        }
        s
    }

    #[test]
    fn manual_wins_over_auto_with_artist_prefix() {
        let sidecar = Sidecar {
            manual_tags: vec!["tezuka_osamu".into()],
            auto_tags: vec![
                AutoTag {
                    tag: "tezuka_osamu".into(),
                    score: 0.9,
                    category: "artist".into(),
                },
                AutoTag {
                    tag: "1girl".into(),
                    score: 0.95,
                    category: "general".into(),
                },
            ],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::anima());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["tezuka_osamu".to_string(), "1girl".to_string()]);
    }

    #[test]
    fn artist_prefix_applied_when_no_collision() {
        let sidecar = Sidecar {
            auto_tags: vec![AutoTag {
                tag: "tezuka_osamu".into(),
                score: 0.9,
                category: "artist".into(),
            }],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::anima());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["@tezuka_osamu".to_string()]);
    }

    #[test]
    fn threshold_filters_auto_only() {
        let sidecar = Sidecar {
            auto_tags: vec![
                AutoTag {
                    tag: "high".into(),
                    score: 0.9,
                    category: "general".into(),
                },
                AutoTag {
                    tag: "low".into(),
                    score: 0.1,
                    category: "general".into(),
                },
            ],
            ..Default::default()
        };
        let mut profile = no_shuffle(ExportProfile::default());
        profile.threshold = 0.5;
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["high".to_string()]);
    }

    #[test]
    fn excluded_category_dropped() {
        let sidecar = Sidecar {
            auto_tags: vec![
                AutoTag {
                    tag: "watermark".into(),
                    score: 0.9,
                    category: "meta".into(),
                },
                AutoTag {
                    tag: "1girl".into(),
                    score: 0.9,
                    category: "general".into(),
                },
            ],
            ..Default::default()
        };
        let mut profile = no_shuffle(ExportProfile::default());
        profile.exclude_categories = vec!["meta".into()];
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["1girl".to_string()]);
    }

    #[test]
    fn manual_order_preserved_when_no_shuffle() {
        let sidecar = Sidecar {
            manual_tags: vec!["my_trigger".into(), "outfit_a".into()],
            auto_tags: vec![AutoTag {
                tag: "1girl".into(),
                score: 0.9,
                category: "general".into(),
            }],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(
            tags,
            vec![
                "my_trigger".to_string(),
                "outfit_a".to_string(),
                "1girl".to_string()
            ]
        );
    }

    #[test]
    fn negative_manual_suppresses_auto() {
        let sidecar = Sidecar {
            manual_tags: vec!["-watermark".into()],
            auto_tags: vec![
                AutoTag {
                    tag: "watermark".into(),
                    score: 0.9,
                    category: "meta".into(),
                },
                AutoTag {
                    tag: "1girl".into(),
                    score: 0.9,
                    category: "general".into(),
                },
            ],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["1girl".to_string()]);
    }

    #[test]
    fn negative_manual_not_emitted_as_positive() {
        let sidecar = Sidecar {
            manual_tags: vec!["-foo".into(), "bar".into()],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["bar".to_string()]);
    }

    #[test]
    fn organizational_manual_tag_not_exported() {
        let sidecar = Sidecar {
            manual_tags: vec!["_no_character".into(), "1girl".into()],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["1girl".to_string()]);
    }

    #[test]
    fn organizational_tag_does_not_suppress_matching_external_tag() {
        // `_foo` is curation-only, NOT a suppression marker: an auto/booru
        // tag with the same stem (sans underscore) is still exported.
        let sidecar = Sidecar {
            manual_tags: vec!["_watermark".into()],
            auto_tags: vec![AutoTag {
                tag: "watermark".into(),
                score: 0.9,
                category: "meta".into(),
            }],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags, vec!["watermark".to_string()]);
    }

    #[test]
    fn booru_tags_exported_with_artist_prefix() {
        let sidecar = Sidecar {
            booru_tags: vec![
                BooruTag {
                    tag: "tezuka_osamu".into(),
                    category: "artist".into(),
                },
                BooruTag {
                    tag: "astro_boy".into(),
                    category: "copyright".into(),
                },
            ],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::anima());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"@tezuka_osamu".to_string()));
        assert!(tags.contains(&"astro_boy".to_string()));
    }

    fn with_caption_prefixes(pairs: &[(&str, &str)]) -> ExportProfile {
        ExportProfile {
            caption_prefixes: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn caption_prefix_prepended_when_tag_present() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["realistic".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a girl standing in a field");
        let profile = with_caption_prefixes(&[("realistic", "realistic proportions, ")]);
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("realistic proportions, a girl standing in a field")
        );
    }

    #[test]
    fn caption_prefix_absent_tag_is_untouched() {
        let mut sidecar = Sidecar::default();
        sidecar.set_caption("a", "a girl standing in a field");
        let profile = with_caption_prefixes(&[
            ("realistic", "realistic proportions, "),
            ("super_deformed", "super deformed, "),
        ]);
        // No proportion tag → bare caption (the default house style).
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("a girl standing in a field")
        );
    }

    #[test]
    fn caption_prefix_matches_organizational_and_case_insensitively() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["_Super_Deformed".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a cat");
        let profile = with_caption_prefixes(&[("super_deformed", "super deformed, ")]);
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("super deformed, a cat")
        );
    }

    #[test]
    fn caption_prefix_manual_caption_body_wins() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["realistic".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "auto text");
        sidecar.set_manual_caption("hand-edited body");
        let profile = with_caption_prefixes(&[("realistic", "realistic proportions, ")]);
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("realistic proportions, hand-edited body")
        );
    }

    #[test]
    fn caption_prefix_no_body_is_none() {
        // A matching prefix rule but no caption at all → None (skip), never
        // a prefix-only "caption".
        let sidecar = Sidecar {
            manual_tags: vec!["realistic".into()],
            ..Default::default()
        };
        let profile = with_caption_prefixes(&[("realistic", "realistic proportions, ")]);
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            ),
            None
        );
    }

    #[test]
    fn caption_suffix_appended_when_tag_present() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["realistic".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a girl standing in a field");
        let profile = ExportProfile {
            caption_suffixes: [(
                "realistic".to_string(),
                ", realistic proportions".to_string(),
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("a girl standing in a field, realistic proportions")
        );
    }

    #[test]
    fn caption_prefix_and_suffix_both_apply() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["super_deformed".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a cat");
        let profile = ExportProfile {
            caption_prefixes: [("super_deformed".to_string(), "PRE ".to_string())]
                .into_iter()
                .collect(),
            caption_suffixes: [("super_deformed".to_string(), " SUF".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("PRE a cat SUF")
        );
    }

    #[test]
    fn caption_no_prefixes_configured_equals_export_caption() {
        let mut sidecar = Sidecar::default();
        sidecar.set_caption("a", "plain caption");
        let profile = ExportProfile::default();
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default()
            ),
            sidecar.export_caption()
        );
    }

    fn single_tag_prefix_group(tag: &str, content: &str, priority: i64) -> TagGroup {
        TagGroup {
            tags: vec![tag.into()],
            caption_prefix: Some(crate::config::CaptionAffix {
                content: content.into(),
                priority,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn tag_group_prefix_priority_orders_concatenation() {
        // Character name (priority 0) precedes costume (priority 1) regardless
        // of the tags' alphabetical order or presence order.
        let mut sidecar = Sidecar {
            manual_tags: vec!["fantasy_knight".into(), "sayaka".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "she stands ready");
        let mut groups = BTreeMap::new();
        groups.insert(
            "fantasy_knight".into(),
            single_tag_prefix_group(
                "fantasy_knight",
                "She wears her Fantasy Knight costume. ",
                1,
            ),
        );
        groups.insert(
            "sayaka".into(),
            single_tag_prefix_group("sayaka", "Sayaka from Saru getchu. ", 0),
        );
        let profile = ExportProfile::default();
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &groups,
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("Sayaka from Saru getchu. She wears her Fantasy Knight costume. she stands ready")
        );
    }

    #[test]
    fn tag_group_caption_applies_only_when_all_tags_present() {
        // Conjunction match: a two-tag group needs both tags present.
        let mut sidecar = Sidecar {
            manual_tags: vec!["1girl".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a scene");
        let mut groups = BTreeMap::new();
        groups.insert(
            "girl_fourth_wall".into(),
            TagGroup {
                tags: vec!["1girl".into(), "breaking_through_fourth_wall".into()],
                exclusive: false,
                caption_prefix: Some(crate::config::CaptionAffix {
                    content: "PRE ".into(),
                    priority: 0,
                }),
                ..Default::default()
            },
        );
        let profile = ExportProfile::default();
        // Only `1girl` present → group does not fire.
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &groups,
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("a scene")
        );
        // Add the concept tag → both present → prefix folds in.
        sidecar.add_manual_tag("breaking_through_fourth_wall");
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &groups,
                &CommonTags::default(),
                AffixSeed::default()
            )
            .as_deref(),
            Some("PRE a scene")
        );
    }

    #[test]
    fn common_trigger_leads_and_common_suppression_drops_auto_tag() {
        // The dataset-wide layer as a character LoRA uses it: trigger word in
        // front, appearance traits folded into it rather than tagged.
        let sidecar = Sidecar {
            manual_tags: vec!["smile".into()],
            auto_tags: vec![
                AutoTag {
                    tag: "red_hair".into(),
                    score: 0.9,
                    category: "general".into(),
                },
                AutoTag {
                    tag: "1girl".into(),
                    score: 0.9,
                    category: "general".into(),
                },
            ],
            ..Default::default()
        };
        let common = CommonTags::new(["himeko", "-red_hair"]);
        let profile = no_shuffle(ExportProfile::default());
        assert_eq!(
            build_tags(&sidecar, &profile, &common),
            vec![
                "himeko".to_string(),
                "smile".to_string(),
                "1girl".to_string()
            ]
        );
    }

    #[test]
    fn per_image_entry_overrides_common_layer() {
        // This shot is genuinely about the hair, and isn't the character.
        let sidecar = Sidecar {
            manual_tags: vec!["red_hair".into(), "-himeko".into()],
            ..Default::default()
        };
        let common = CommonTags::new(["himeko", "-red_hair"]);
        let profile = no_shuffle(ExportProfile::default());
        assert_eq!(
            build_tags(&sidecar, &profile, &common),
            vec!["red_hair".to_string()]
        );
    }

    #[test]
    fn common_organizational_tag_is_not_exported() {
        let sidecar = Sidecar {
            manual_tags: vec!["1girl".into()],
            ..Default::default()
        };
        let common = CommonTags::new(["_reviewed"]);
        let profile = no_shuffle(ExportProfile::default());
        assert_eq!(
            build_tags(&sidecar, &profile, &common),
            vec!["1girl".to_string()]
        );
    }

    #[test]
    fn common_tag_drives_caption_prefix() {
        let mut sidecar = Sidecar::default();
        sidecar.set_caption("a", "a girl standing in a field");
        let profile = with_caption_prefixes(&[("himeko", "Himeko. ")]);
        assert_eq!(
            build_caption(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::new(["himeko"]),
                AffixSeed::default(),
            )
            .as_deref(),
            Some("Himeko. a girl standing in a field")
        );
    }

    #[test]
    fn export_caption_file_writes_body_and_returns_path() {
        let dir = std::env::temp_dir().join(format!(
            "fwaun-export-caption-test-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        fs::write(&image, b"").unwrap();

        let mut sidecar = Sidecar::default();
        sidecar.set_caption("a", "a girl standing in a field");
        let profile = no_shuffle(ExportProfile::default());
        let out = export_caption_file(
            &image,
            &sidecar,
            &profile,
            &BTreeMap::new(),
            &CommonTags::default(),
            AffixSeed::default(),
        )
        .unwrap();
        let out = out.expect("caption present");
        assert_eq!(out, export_text_path(&image));
        assert_eq!(
            fs::read_to_string(&out).unwrap(),
            "a girl standing in a field"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn export_caption_file_no_caption_writes_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "fwaun-export-caption-test-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        fs::write(&image, b"").unwrap();

        let sidecar = Sidecar::default();
        let profile = no_shuffle(ExportProfile::default());
        let out = export_caption_file(
            &image,
            &sidecar,
            &profile,
            &BTreeMap::new(),
            &CommonTags::default(),
            AffixSeed::default(),
        )
        .unwrap();
        assert!(out.is_none());
        assert!(!export_text_path(&image).exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn suppression_works_across_sources() {
        let sidecar = Sidecar {
            manual_tags: vec!["-1girl".into()],
            auto_tags: vec![AutoTag {
                tag: "1girl".into(),
                score: 0.9,
                category: "general".into(),
            }],
            booru_tags: vec![BooruTag {
                tag: "1girl".into(),
                category: "general".into(),
            }],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile, &CommonTags::default());
        assert!(tags.is_empty());
    }

    #[test]
    fn build_tags_text_flattens_and_joins() {
        let sidecar = sidecar_with(&["1girl", "solo", "blue_sky"], None);
        let tags = build_tags(
            &sidecar,
            &no_shuffle(ExportProfile::default()),
            &CommonTags::default(),
        );
        assert_eq!(build_tags_text(&tags), "1girl, solo, blue sky");
        assert_eq!(build_tags_text(&[]), "");
    }

    #[test]
    fn build_export_text_combined_orders_blocks_with_blank_line() {
        let sidecar = sidecar_with(&["1girl", "solo"], Some("A girl under the sky."));
        let profile = no_shuffle(ExportProfile::default());
        let seed = AffixSeed::default();
        assert_eq!(
            build_export_text(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                seed,
                ExportContent::TagsCaption
            )
            .as_deref(),
            Some("1girl, solo\n\nA girl under the sky.")
        );
        assert_eq!(
            build_export_text(
                &sidecar,
                &profile,
                &BTreeMap::new(),
                &CommonTags::default(),
                seed,
                ExportContent::CaptionTags
            )
            .as_deref(),
            Some("A girl under the sky.\n\n1girl, solo")
        );
    }

    #[test]
    fn build_export_text_combined_without_caption_is_none() {
        // The skip rule: no caption body → combined export skips the image,
        // even when tags are present.
        let sidecar = sidecar_with(&["1girl"], None);
        assert_eq!(
            build_export_text(
                &sidecar,
                &no_shuffle(ExportProfile::default()),
                &BTreeMap::new(),
                &CommonTags::default(),
                AffixSeed::default(),
                ExportContent::TagsCaption
            ),
            None
        );
    }

    #[test]
    fn build_export_text_combined_without_tags_is_caption_only() {
        // Empty tag set → no empty block, no leading blank line: just the
        // caption, same bytes a `caption` export would write.
        let sidecar = sidecar_with(&[], Some("only a caption"));
        for content in [ExportContent::TagsCaption, ExportContent::CaptionTags] {
            assert_eq!(
                build_export_text(
                    &sidecar,
                    &no_shuffle(ExportProfile::default()),
                    &BTreeMap::new(),
                    &CommonTags::default(),
                    AffixSeed::default(),
                    content
                )
                .as_deref(),
                Some("only a caption")
            );
        }
    }

    #[test]
    fn export_file_writes_content_to_custom_extension() {
        let dir = std::env::temp_dir().join(format!(
            "fwaun-export-file-test-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        fs::write(&image, b"").unwrap();

        let sidecar = sidecar_with(&["1girl", "solo"], Some("A girl."));
        let out = export_file(
            &image,
            &sidecar,
            &no_shuffle(ExportProfile::default()),
            &BTreeMap::new(),
            &CommonTags::default(),
            AffixSeed::default(),
            ExportContent::TagsCaption,
            "tags+caption.txt",
        )
        .unwrap()
        .expect("caption present");
        assert_eq!(out, dir.join("a.tags+caption.txt"));
        assert_eq!(fs::read_to_string(&out).unwrap(), "1girl, solo\n\nA girl.");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn export_file_combined_skips_image_without_caption() {
        let dir = std::env::temp_dir().join(format!(
            "fwaun-export-file-test-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        fs::write(&image, b"").unwrap();

        let sidecar = sidecar_with(&["1girl"], None);
        let out = export_file(
            &image,
            &sidecar,
            &no_shuffle(ExportProfile::default()),
            &BTreeMap::new(),
            &CommonTags::default(),
            AffixSeed::default(),
            ExportContent::CaptionTags,
            "tags.txt",
        )
        .unwrap();
        assert!(out.is_none());
        assert!(!dir.join("a.tags.txt").exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn export_text_path_with_ext_replaces_extension() {
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a.png"), "tags.txt").unwrap(),
            PathBuf::from("/d/a.tags.txt")
        );
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a.png"), "tags").unwrap(),
            PathBuf::from("/d/a.tags")
        );
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a"), "txt").unwrap(),
            PathBuf::from("/d/a.txt")
        );
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a.png"), "txt").unwrap(),
            export_text_path(Path::new("/d/a.png"))
        );
    }

    #[test]
    fn export_text_path_with_ext_uses_whole_extension() {
        // A composite extension with an inner dot is used verbatim — the
        // whole `extension` replaces the image's own extension.
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a.png"), "caption.txt").unwrap(),
            PathBuf::from("/d/a.caption.txt")
        );
        // A single-component extension keeps the old behavior.
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/a.png"), "txt").unwrap(),
            export_text_path(Path::new("/d/a.png"))
        );
        // A trailing slash is not pathless — the trailing component is the
        // file name, so the sibling is derived normally (unlike
        // `Path::with_extension`, which would overwrite the input itself).
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/e.png/"), "tags.txt").unwrap(),
            PathBuf::from("/d/e.tags.txt")
        );
        // The stem is preserved verbatim, dotfiles included.
        assert_eq!(
            export_text_path_with_ext(Path::new("/d/.hidden"), "caption.txt").unwrap(),
            PathBuf::from("/d/.hidden.caption.txt")
        );
    }

    #[test]
    fn export_text_path_with_ext_rejects_pathless_image_and_bad_extension() {
        // A pathless image (no file name) has no stem to keep — there is no
        // sibling to write.
        assert!(export_text_path_with_ext(Path::new(".."), "tags.txt").is_err());
        assert!(export_text_path_with_ext(Path::new("."), "tags.txt").is_err());
        assert!(export_text_path_with_ext(Path::new("/"), "tags.txt").is_err());
        // An empty or separator-bearing extension isn't an extension.
        assert!(export_text_path_with_ext(Path::new("/d/a.png"), "").is_err());
        assert!(export_text_path_with_ext(Path::new("/d/a.png"), "tags.txt/caption").is_err());
        // The legacy compat fn keeps `with_extension` semantics, which
        // likewise refuses to extend a pathless path (it returns the input
        // unchanged rather than inventing a sibling).
        assert_eq!(export_text_path(Path::new("..")), PathBuf::from(".."));
    }

    #[test]
    fn export_content_parses_all_values() {
        assert_eq!(
            ExportContent::from_str("tags").unwrap(),
            ExportContent::Tags
        );
        assert_eq!(
            ExportContent::from_str("caption").unwrap(),
            ExportContent::Caption
        );
        assert_eq!(
            ExportContent::from_str("tags+caption").unwrap(),
            ExportContent::TagsCaption
        );
        assert_eq!(
            ExportContent::from_str("caption+tags").unwrap(),
            ExportContent::CaptionTags
        );
        // Case-insensitive; the kebab spelling is accepted too.
        assert_eq!(
            ExportContent::from_str("Tags+Caption").unwrap(),
            ExportContent::TagsCaption
        );
        assert_eq!(
            ExportContent::from_str("caption-tags").unwrap(),
            ExportContent::CaptionTags
        );
        assert!(ExportContent::from_str("tags_caption").is_err());
        assert!(ExportContent::from_str("").is_err());
        assert!(ExportContent::from_str("tags,caption").is_err());
    }
}
