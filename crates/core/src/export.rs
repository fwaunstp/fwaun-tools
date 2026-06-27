use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use rand::seq::SliceRandom;
use thiserror::Error;

use crate::config::ExportProfile;
use crate::sidecar::{is_organizational, Sidecar};

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub fn export_text_path(image: &Path) -> PathBuf {
    image.with_extension("txt")
}

/// Build the final ordered tag list for a single image, applying:
/// - threshold + category exclusion to auto tags
/// - suppression of any auto/booru tag named in a `-foo` manual entry
/// - category prefix formatting (e.g. ANIMA artist `@`)
/// - dedup (manual wins on collision; comparison is prefix-stripped, lowercase)
/// - optional shuffle
///
/// Negative manual entries (`-foo`) are never emitted as positive tags.
/// Organizational manual entries (`_foo`) are curation-only and never
/// exported either, though they still count for tag-group classification.
pub fn build_tags(sidecar: &Sidecar, profile: &ExportProfile) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let suppressed = sidecar.suppressed_set();

    for raw in sidecar.manual_positive_tags() {
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
        let mut rng = rand::thread_rng();
        out.shuffle(&mut rng);
    }
    out
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
/// sorted), so output is deterministic even if several rules match.
///
/// Returns `None` when the image has no caption body at all: a bare affix
/// without a caption isn't a useful training caption, so such images are
/// skipped by callers rather than emitted affix-only.
pub fn build_caption(sidecar: &Sidecar, profile: &ExportProfile) -> Option<String> {
    let body = sidecar.export_caption()?;
    let present = present_caption_stems(sidecar, profile);
    let prefix = matched_affixes(&profile.caption_prefixes, &present);
    let suffix = matched_affixes(&profile.caption_suffixes, &present);
    if prefix.is_empty() && suffix.is_empty() {
        Some(body)
    } else {
        Some(format!("{prefix}{body}{suffix}"))
    }
}

/// Stems of the image's positive manual tags, normalized for affix
/// matching. Empty (and cheap) when neither affix table is configured.
fn present_caption_stems(sidecar: &Sidecar, profile: &ExportProfile) -> HashSet<String> {
    if profile.caption_prefixes.is_empty() && profile.caption_suffixes.is_empty() {
        return HashSet::new();
    }
    sidecar
        .manual_positive_tags()
        .map(caption_prefix_stem)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Concatenate the affixes whose tag stem is present, in map key order.
/// Empty when nothing matches.
fn matched_affixes(affixes: &std::collections::BTreeMap<String, String>, present: &HashSet<String>) -> String {
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

pub fn export_image(
    image: &Path,
    sidecar: &Sidecar,
    profile: &ExportProfile,
) -> Result<PathBuf, ExportError> {
    let tags = build_tags(sidecar, profile);
    let body = tags
        .iter()
        .map(|t| t.replace('_', " "))
        .collect::<Vec<_>>()
        .join(", ");
    let out = export_text_path(image);
    fs::write(&out, body).map_err(|source| ExportError::Io {
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
        assert_eq!(tags, vec!["1girl".to_string()]);
    }

    #[test]
    fn negative_manual_not_emitted_as_positive() {
        let sidecar = Sidecar {
            manual_tags: vec!["-foo".into(), "bar".into()],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile);
        assert_eq!(tags, vec!["bar".to_string()]);
    }

    #[test]
    fn organizational_manual_tag_not_exported() {
        let sidecar = Sidecar {
            manual_tags: vec!["_no_character".into(), "1girl".into()],
            ..Default::default()
        };
        let profile = no_shuffle(ExportProfile::default());
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
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
        let tags = build_tags(&sidecar, &profile);
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"@tezuka_osamu".to_string()));
        assert!(tags.contains(&"astro_boy".to_string()));
    }

    fn with_caption_prefixes(pairs: &[(&str, &str)]) -> ExportProfile {
        let mut p = ExportProfile::default();
        p.caption_prefixes = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        p
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
            build_caption(&sidecar, &profile).as_deref(),
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
            build_caption(&sidecar, &profile).as_deref(),
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
            build_caption(&sidecar, &profile).as_deref(),
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
            build_caption(&sidecar, &profile).as_deref(),
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
        assert_eq!(build_caption(&sidecar, &profile), None);
    }

    #[test]
    fn caption_suffix_appended_when_tag_present() {
        let mut sidecar = Sidecar {
            manual_tags: vec!["realistic".into()],
            ..Default::default()
        };
        sidecar.set_caption("a", "a girl standing in a field");
        let mut profile = ExportProfile::default();
        profile.caption_suffixes =
            [("realistic".to_string(), ", realistic proportions".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            build_caption(&sidecar, &profile).as_deref(),
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
        let mut profile = ExportProfile::default();
        profile.caption_prefixes = [("super_deformed".to_string(), "PRE ".to_string())]
            .into_iter()
            .collect();
        profile.caption_suffixes = [("super_deformed".to_string(), " SUF".to_string())]
            .into_iter()
            .collect();
        assert_eq!(
            build_caption(&sidecar, &profile).as_deref(),
            Some("PRE a cat SUF")
        );
    }

    #[test]
    fn caption_no_prefixes_configured_equals_export_caption() {
        let mut sidecar = Sidecar::default();
        sidecar.set_caption("a", "plain caption");
        let profile = ExportProfile::default();
        assert_eq!(
            build_caption(&sidecar, &profile),
            sidecar.export_caption()
        );
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
        let tags = build_tags(&sidecar, &profile);
        assert!(tags.is_empty());
    }
}
