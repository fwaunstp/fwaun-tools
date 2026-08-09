//! Classification of images against named tag groups (currently always
//! treated as mutually exclusive). Drives the CLI's `validate-tag-group`
//! command and the GUI Kanban view.
//!
//! Classification works on a *primitive* effective tag set —
//! `manual_positive ∪ auto_tags ∪ booru_tags` minus `-foo` suppressions —
//! rather than the export-profile-thresholded output of [`crate::export`].
//! Reason: kanban is curatorial. An auto-tag below the export threshold
//! still tells the user "the tagger thinks `school_uniform` is plausible
//! here" — hiding it would silently drop the image into the "unset"
//! bucket and the user would never see it.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::common_tags::CommonTags;
use crate::config::{CaptionAffix, TagGroup};
use crate::sidecar::{ORGANIZATIONAL_PREFIX, Sidecar, positive_entries, suppressed_stems};

/// Classification result for one image against one tag group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Classification {
    /// Exactly one of the group's tags is present.
    Tag(String),
    /// None of the group's tags are present.
    Unset,
    /// Two or more group tags coexist. Tags are returned in the group's
    /// declared order. Not an error — flagged for review.
    Violation(Vec<String>),
}

/// Drop target for the GUI Kanban view's drag-and-drop. The "Violation"
/// bucket is intentionally not a drop target — to consciously assign
/// multiple group tags to one image, the user edits manual_tags via the
/// detail panel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropTarget {
    Tag(String),
    Unset,
}

/// Build the lowercase-stem effective tag set for `sc`, with the
/// dataset-wide `common` layer merged underneath its own manual tags.
/// Suppressed (`-foo`) entries are removed — from either layer, so a
/// `common_tags` suppression drops the tag out of Kanban classification and
/// `mv` matching just as a per-image one does.
pub fn effective_tag_set(sc: &Sidecar, common: &CommonTags) -> HashSet<String> {
    let manual = common.merged_manual_tags(sc);
    let suppressed = suppressed_stems(&manual);
    let mut set: HashSet<String> = HashSet::new();
    for t in positive_entries(&manual) {
        let key = t.trim().to_lowercase();
        if !key.is_empty() {
            set.insert(key);
        }
    }
    for at in &sc.auto_tags {
        let key = at.tag.trim().to_lowercase();
        if !key.is_empty() {
            set.insert(key);
        }
    }
    for bt in &sc.booru_tags {
        let key = bt.tag.trim().to_lowercase();
        if !key.is_empty() {
            set.insert(key);
        }
    }
    for s in &suppressed {
        set.remove(s);
    }
    set
}

/// Classify `sc` against `group`.
pub fn classify(sc: &Sidecar, group: &TagGroup, common: &CommonTags) -> Classification {
    let eff = effective_tag_set(sc, common);
    let present: Vec<String> = group
        .tags
        .iter()
        .filter(|t| eff.contains(&t.trim().to_lowercase()))
        .cloned()
        .collect();
    match present.len() {
        0 => Classification::Unset,
        1 => Classification::Tag(present.into_iter().next().unwrap()),
        _ => Classification::Violation(present),
    }
}

// ───────── caption steering (hint / prefix / suffix) ─────────
//
// A tag group's `caption_hint` / `caption_prefix` / `caption_suffix` apply
// to an image when *all* of the group's tags are present (logical AND).
// Matching keys on the image's positive *manual* tags only — consistent
// with export caption-affix matching, and the deliberate choice for
// curation-driven steering — normalized the same way (trim, drop a single
// leading organizational `_`, lowercase). The dataset-wide `common_tags`
// layer counts as manual here, so a shared trigger word can steer captions
// across the whole set.

/// Normalize a tag for caption-steering matching.
fn caption_match_stem(s: &str) -> String {
    let t = s.trim();
    t.strip_prefix(ORGANIZATIONAL_PREFIX)
        .unwrap_or(t)
        .to_lowercase()
}

/// Lowercase, `_`-stripped stems of the image's positive manual tags, with
/// the dataset-wide common layer merged in.
fn manual_caption_stems(sc: &Sidecar, common: &CommonTags) -> HashSet<String> {
    let manual = common.merged_manual_tags(sc);
    positive_entries(&manual)
        .map(caption_match_stem)
        .filter(|s| !s.is_empty())
        .collect()
}

/// True when the group is non-empty and every one of its tags is present.
fn group_all_present(group: &TagGroup, stems: &HashSet<String>) -> bool {
    !group.tags.is_empty()
        && group
            .tags
            .iter()
            .all(|t| stems.contains(&caption_match_stem(t)))
}

/// Caption hints contributed by every tag group all of whose tags are
/// present on `sc`. Ordered by group name (BTreeMap iteration) for
/// determinism. Blank hints are skipped.
pub fn resolved_caption_hints(
    sc: &Sidecar,
    groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
) -> Vec<String> {
    let stems = manual_caption_stems(sc, common);
    groups
        .values()
        .filter(|g| group_all_present(g, &stems))
        .filter_map(|g| g.caption_hint.as_deref())
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .map(str::to_string)
        .collect()
}

// ───────── per-image affix ordering seed ─────────

/// FNV-1a 64-bit. Hand-rolled rather than `DefaultHasher` on purpose:
/// [`AffixSeed`] decides what ends up in exported captions, so the hash has
/// to produce the same number on every platform and every Rust version.
/// `std`'s default hasher guarantees neither.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Per-image seed that scatters the order of *equal-`priority`* caption
/// affixes.
///
/// Equal `priority` is a statement that the order doesn't matter, and for
/// style LoRAs it's actively better if it varies: training every image with
/// `"chibi style. realistic background. "` in that fixed order teaches the
/// order along with the concepts. So matching affixes that tie on priority
/// are ordered by a hash of `(seed, group name)` instead of by group name.
///
/// The seed is *pseudo*-random rather than random for two reasons:
///
/// 1. The captioner is primed with the resolved prefix at generation time
///    ([`resolved_caption_prefix`]) and export prepends it again later. A
///    fresh `rand` draw at each site would make the two disagree.
/// 2. Re-exporting an unchanged dataset must produce an unchanged
///    `meta.jsonl`, or every regeneration is a whole-file diff.
///
/// [`AffixSeed::for_image`] therefore derives it from *where the image sits
/// in the dataset*, not from anything that changes between runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AffixSeed(u64);

impl AffixSeed {
    /// Seed for `image`, derived from its path relative to `root` — the
    /// directory holding `fwaun-tools.toml` (see
    /// [`ProjectConfig::project_root`]). Relative rather than absolute so
    /// the same dataset orders its affixes identically after a move, a
    /// clone to another machine, or a checkout on another OS.
    ///
    /// The file extension is dropped: a sidecar is keyed on the stem, so
    /// `img_001.png` and the `img_001.jpg` a `dataset edit`/`upscale` pass
    /// turned it into are the same image and keep the same order.
    ///
    /// `root = None` (no project config found) falls back to the absolute
    /// path — still stable across re-runs, just not across machines.
    ///
    /// [`ProjectConfig::project_root`]: crate::config::ProjectConfig::project_root
    pub fn for_image(root: Option<&Path>, image: &Path) -> Self {
        // `project_root` canonicalizes, so canonicalize here too or the
        // prefix won't strip (`\\?\C:\…` vs `C:\…` on Windows).
        let abs = image.canonicalize();
        let abs = abs.as_deref().unwrap_or(image);
        let rel = root.and_then(|r| abs.strip_prefix(r).ok()).unwrap_or(abs);
        Self::from_key(&relative_key(rel))
    }

    /// Seed from an arbitrary stable identity string. Prefer
    /// [`Self::for_image`]; this exists for callers that already hold a
    /// dataset-relative key (and for tests).
    pub fn from_key(key: &str) -> Self {
        Self(fnv1a(key.as_bytes(), FNV_OFFSET))
    }

    /// Tie-break sort key for one affix. `domain` separates prefixes from
    /// suffixes so an image doesn't get the same permutation on both.
    fn tie_break(self, domain: &str, group_name: &str) -> u64 {
        let h = fnv1a(&self.0.to_le_bytes(), FNV_OFFSET);
        let h = fnv1a(domain.as_bytes(), h);
        fnv1a(group_name.as_bytes(), h)
    }
}

/// `a/b/img_001.png` → `a/b/img_001`. Separators are normalized to `/` so a
/// dataset seeds identically on Windows and Linux.
fn relative_key(rel: &Path) -> String {
    let mut parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if let Some(last) = parts.last_mut()
        && let Some(stem) = Path::new(last.as_str()).file_stem()
    {
        *last = stem.to_string_lossy().into_owned();
    }
    parts.join("/")
}

/// Concatenated caption prefix from every matching tag group, ordered by
/// ascending `priority`; groups that tie on priority are ordered
/// pseudo-randomly per image by `seed` (see [`AffixSeed`]). Empty when
/// nothing matches.
pub fn resolved_caption_prefix(
    sc: &Sidecar,
    groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
) -> String {
    resolved_affix(sc, groups, common, seed, "prefix", |g| {
        g.caption_prefix.as_ref()
    })
}

/// Concatenated caption suffix from every matching tag group. Same ordering
/// as [`resolved_caption_prefix`].
pub fn resolved_caption_suffix(
    sc: &Sidecar,
    groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
) -> String {
    resolved_affix(sc, groups, common, seed, "suffix", |g| {
        g.caption_suffix.as_ref()
    })
}

fn resolved_affix(
    sc: &Sidecar,
    groups: &BTreeMap<String, TagGroup>,
    common: &CommonTags,
    seed: AffixSeed,
    domain: &str,
    pick: impl Fn(&TagGroup) -> Option<&CaptionAffix>,
) -> String {
    let stems = manual_caption_stems(sc, common);
    let mut matched: Vec<(&str, u64, &CaptionAffix)> = groups
        .iter()
        .filter(|(_, g)| group_all_present(g, &stems))
        .filter_map(|(name, g)| pick(g).map(|a| (name.as_str(), seed.tie_break(domain, name), a)))
        .collect();
    // Ascending priority. Ties go to the per-image hash — keyed on the group
    // name alone, so adding or removing an unrelated group doesn't reshuffle
    // the ones that were already there. The name is the final tiebreak, for
    // the (astronomically unlikely) hash collision.
    matched.sort_by(|a, b| {
        a.2.priority
            .cmp(&b.2.priority)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.0.cmp(b.0))
    });
    matched
        .into_iter()
        .map(|(_, _, a)| a.content.as_str())
        .collect()
}

/// Apply a Kanban drop to `sc`, mutating its `manual_tags` so that the
/// classification result becomes `target`.
///
/// On `Tag(X)`: ensure `X` is a positive manual entry (clearing any `-X`
/// suppression), and for each *other* group tag `Y` that currently
/// appears in the effective tag set, replace it with a `-Y` suppression
/// marker. Tags that don't appear in any source are left untouched —
/// no eager suppression that would bloat the sidecar with `-Y` markers
/// for tags that may never appear.
///
/// On `Unset`: same as above but applied to *every* group tag currently
/// in the effective set.
///
/// A group tag supplied by the dataset-wide `common` layer is handled the
/// same way — dropping the image elsewhere writes a per-image `-Y` marker
/// that overrides the shared entry for that image alone.
pub fn apply_drop(sc: &mut Sidecar, group: &TagGroup, target: &DropTarget, common: &CommonTags) {
    let eff = effective_tag_set(sc, common);
    match target {
        DropTarget::Tag(x) => {
            let x_trimmed = x.trim();
            if x_trimmed.is_empty() {
                return;
            }
            sc.unsuppress(x_trimmed);
            // Add as positive only if not already effective: the common
            // layer may already supply it, in which case clearing any `-x`
            // above is enough and a per-image copy would just be noise.
            // `add_manual_tag` is a no-op for duplicates.
            if !common.provides_positive(x_trimmed) {
                sc.add_manual_tag(x_trimmed);
            }

            let x_key = x_trimmed.to_lowercase();
            for other in &group.tags {
                let other_trimmed = other.trim();
                if other_trimmed.is_empty() {
                    continue;
                }
                let other_key = other_trimmed.to_lowercase();
                if other_key == x_key {
                    continue;
                }
                if eff.contains(&other_key) {
                    sc.remove_manual_tag(other_trimmed);
                    sc.suppress(other_trimmed);
                }
            }
        }
        DropTarget::Unset => {
            for tag in &group.tags {
                let trimmed = tag.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let key = trimmed.to_lowercase();
                if eff.contains(&key) {
                    sc.remove_manual_tag(trimmed);
                    sc.suppress(trimmed);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidecar::{AutoTag, BooruTag};

    fn group(tags: &[&str]) -> TagGroup {
        TagGroup {
            tags: tags.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn auto(tag: &str) -> AutoTag {
        AutoTag {
            tag: tag.into(),
            score: 0.5,
            category: "general".into(),
        }
    }

    fn booru(tag: &str) -> BooruTag {
        BooruTag {
            tag: tag.into(),
            category: "general".into(),
        }
    }

    #[test]
    fn effective_tag_set_unions_all_sources() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("alpha".into());
        sc.auto_tags.push(auto("beta"));
        sc.booru_tags.push(booru("gamma"));
        let set = effective_tag_set(&sc, &CommonTags::default());
        assert!(set.contains("alpha"));
        assert!(set.contains("beta"));
        assert!(set.contains("gamma"));
    }

    #[test]
    fn effective_tag_set_strips_suppressed_entries() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("watermark"));
        sc.manual_tags.push("-watermark".into());
        let set = effective_tag_set(&sc, &CommonTags::default());
        assert!(!set.contains("watermark"));
    }

    #[test]
    fn classify_returns_unset_when_none_present() {
        let sc = Sidecar::default();
        let g = group(&["a", "b"]);
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Unset
        );
    }

    #[test]
    fn classify_returns_tag_when_one_present() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("a".into());
        let g = group(&["a", "b"]);
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("a".into())
        );
    }

    #[test]
    fn classify_returns_violation_when_two_present_in_group_order() {
        let mut sc = Sidecar::default();
        // mix sources to ensure both contribute
        sc.auto_tags.push(auto("b"));
        sc.booru_tags.push(booru("a"));
        let g = group(&["a", "b"]);
        match classify(&sc, &g, &CommonTags::default()) {
            Classification::Violation(tags) => {
                assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected Violation, got {other:?}"),
        }
    }

    #[test]
    fn organizational_tag_classifies_as_its_own_bucket() {
        // `_none` is curation-only (never exported) but a member of the group,
        // so a reviewed "deliberately none of these" image lands in the
        // `_none` bucket rather than Unset.
        let mut sc = Sidecar::default();
        sc.manual_tags.push("_none".into());
        let g = group(&["char_a", "char_b", "_none"]);
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("_none".into())
        );
    }

    #[test]
    fn classify_skips_tags_suppressed_by_negative_marker() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("a"));
        sc.manual_tags.push("-a".into());
        sc.manual_tags.push("b".into());
        let g = group(&["a", "b"]);
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("b".into())
        );
    }

    #[test]
    fn apply_drop_tag_adds_positive_and_suppresses_present_siblings() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y", "z"]);
        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );

        assert!(sc.manual_tags.contains(&"x".to_string()));
        assert!(sc.manual_tags.contains(&"-y".to_string()));
        // z was nowhere, so no `-z` written
        assert!(!sc.manual_tags.iter().any(|t| t == "-z"));
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("x".into())
        );
    }

    #[test]
    fn apply_drop_tag_clears_existing_negative_on_target() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("-x".into());
        let g = group(&["x", "y"]);
        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );
        assert!(!sc.manual_tags.iter().any(|t| t == "-x"));
        assert!(sc.manual_tags.contains(&"x".to_string()));
    }

    #[test]
    fn apply_drop_tag_replaces_positive_sibling_with_suppression() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("y".into());
        let g = group(&["x", "y"]);
        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );
        assert!(!sc.manual_tags.iter().any(|t| t == "y"));
        assert!(sc.manual_tags.contains(&"-y".to_string()));
        assert!(sc.manual_tags.contains(&"x".to_string()));
    }

    #[test]
    fn apply_drop_unset_suppresses_only_present_group_tags() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y", "z"]);
        apply_drop(&mut sc, &g, &DropTarget::Unset, &CommonTags::default());

        assert!(sc.manual_tags.contains(&"-y".to_string()));
        // x, z were absent → no eager suppression
        assert!(!sc.manual_tags.iter().any(|t| t == "-x"));
        assert!(!sc.manual_tags.iter().any(|t| t == "-z"));
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Unset
        );
    }

    #[test]
    fn apply_drop_is_idempotent() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y"]);
        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );
        let after_once = sc.manual_tags.clone();
        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );
        assert_eq!(sc.manual_tags, after_once);
    }

    #[test]
    fn resolved_caption_hints_fire_only_on_full_conjunction() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("1girl".into());
        let mut groups = BTreeMap::new();
        groups.insert(
            "g".into(),
            TagGroup {
                tags: vec!["1girl".into(), "breaking_through_fourth_wall".into()],
                exclusive: false,
                caption_hint: Some("A girl is breaking through the fourth wall.".into()),
                ..Default::default()
            },
        );
        // Only one of the two tags present → no hint.
        assert!(resolved_caption_hints(&sc, &groups, &CommonTags::default()).is_empty());
        sc.manual_tags.push("breaking_through_fourth_wall".into());
        assert_eq!(
            resolved_caption_hints(&sc, &groups, &CommonTags::default()),
            vec!["A girl is breaking through the fourth wall.".to_string()]
        );
    }

    #[test]
    fn resolved_caption_prefix_concatenates_by_priority() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("sayaka".into());
        sc.manual_tags.push("fantasy_knight".into());
        let mut groups = BTreeMap::new();
        // Group name order (z before a alphabetically is false) is deliberately
        // opposite to priority order to prove priority wins.
        groups.insert(
            "z_costume".into(),
            TagGroup {
                tags: vec!["fantasy_knight".into()],
                caption_prefix: Some(CaptionAffix {
                    content: "B".into(),
                    priority: 1,
                }),
                ..Default::default()
            },
        );
        groups.insert(
            "a_name".into(),
            TagGroup {
                tags: vec!["sayaka".into()],
                caption_prefix: Some(CaptionAffix {
                    content: "A".into(),
                    priority: 0,
                }),
                ..Default::default()
            },
        );
        assert_eq!(
            resolved_caption_prefix(&sc, &groups, &CommonTags::default(), AffixSeed::default()),
            "AB"
        );
    }

    #[test]
    fn resolved_caption_prefix_matches_organizational_and_case_insensitively() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("_Realistic".into());
        let mut groups = BTreeMap::new();
        groups.insert(
            "style".into(),
            TagGroup {
                tags: vec!["realistic".into()],
                caption_prefix: Some(CaptionAffix {
                    content: "realistic proportions, ".into(),
                    priority: 0,
                }),
                ..Default::default()
            },
        );
        assert_eq!(
            resolved_caption_prefix(&sc, &groups, &CommonTags::default(), AffixSeed::default()),
            "realistic proportions, "
        );
    }

    /// Two equal-priority steering groups, both firing: the "chibi style" /
    /// "realistic background" pair a style LoRA wants scattered.
    fn tied_style_groups() -> BTreeMap<String, TagGroup> {
        let affix = |content: &str| {
            Some(CaptionAffix {
                content: content.into(),
                priority: 0,
            })
        };
        let mut groups = BTreeMap::new();
        groups.insert(
            "style_chibi".to_string(),
            TagGroup {
                tags: vec!["chibi".into()],
                exclusive: false,
                caption_prefix: affix("chibi style. "),
                caption_suffix: affix(" [chibi]"),
                ..Default::default()
            },
        );
        groups.insert(
            "style_realistic_bg".to_string(),
            TagGroup {
                tags: vec!["realistic_background".into()],
                exclusive: false,
                caption_prefix: affix("realistic background. "),
                caption_suffix: affix(" [bg]"),
                ..Default::default()
            },
        );
        groups
    }

    fn both_styles() -> Sidecar {
        Sidecar {
            manual_tags: vec!["chibi".into(), "realistic_background".into()],
            ..Default::default()
        }
    }

    #[test]
    fn equal_priority_prefixes_scatter_across_images() {
        let sc = both_styles();
        let groups = tied_style_groups();
        let chibi_first = (0..200)
            .map(|i| AffixSeed::for_image(None, Path::new(&format!("img_{i:03}.png"))))
            .filter(|seed| {
                resolved_caption_prefix(&sc, &groups, &CommonTags::default(), *seed)
                    .starts_with("chibi")
            })
            .count();
        // Both orders must actually occur, and neither should dominate — a
        // 200-image dataset landing outside 25..175 means the hash isn't
        // mixing the filename in usefully.
        assert!(
            (25..175).contains(&chibi_first),
            "chibi-first for {chibi_first}/200 images — not a usable split"
        );
    }

    #[test]
    fn equal_priority_order_is_stable_for_one_image() {
        let sc = both_styles();
        let groups = tied_style_groups();
        let seed = AffixSeed::for_image(None, Path::new("dataset/img_007.png"));
        let first = resolved_caption_prefix(&sc, &groups, &CommonTags::default(), seed);
        for _ in 0..5 {
            assert_eq!(
                resolved_caption_prefix(&sc, &groups, &CommonTags::default(), seed),
                first
            );
        }
        // Exactly one of the two orders, never a partial or duplicated one.
        assert!(
            first == "chibi style. realistic background. "
                || first == "realistic background. chibi style. ",
            "unexpected prefix {first:?}"
        );
    }

    #[test]
    fn prefix_and_suffix_permutations_are_independent() {
        let sc = both_styles();
        let groups = tied_style_groups();
        // Same seed, different domain: the two must not move in lockstep, or
        // an image's suffix would just echo its prefix order.
        let disagreements = (0..64)
            .map(|i| AffixSeed::for_image(None, Path::new(&format!("img_{i:03}.png"))))
            .filter(|seed| {
                let p = resolved_caption_prefix(&sc, &groups, &CommonTags::default(), *seed);
                let s = resolved_caption_suffix(&sc, &groups, &CommonTags::default(), *seed);
                p.starts_with("chibi") != s.starts_with(" [chibi]")
            })
            .count();
        assert!(disagreements > 0, "prefix and suffix orders never differ");
    }

    #[test]
    fn adding_a_group_leaves_the_existing_tie_order_alone() {
        // The tie-break hashes the group name alone, so a config edit that
        // introduces a third steering group must not reshuffle the two that
        // were already there — otherwise every such edit rewrites the whole
        // exported metadata.
        let mut sc = both_styles();
        sc.manual_tags.push("watercolor".into());
        let groups = tied_style_groups();
        let mut grown = groups.clone();
        grown.insert(
            "style_watercolor".to_string(),
            TagGroup {
                tags: vec!["watercolor".into()],
                exclusive: false,
                caption_prefix: Some(CaptionAffix {
                    content: "watercolor. ".into(),
                    priority: 0,
                }),
                ..Default::default()
            },
        );
        for i in 0..32 {
            let seed = AffixSeed::for_image(None, Path::new(&format!("img_{i:03}.png")));
            let before = resolved_caption_prefix(&sc, &groups, &CommonTags::default(), seed);
            let after = resolved_caption_prefix(&sc, &grown, &CommonTags::default(), seed)
                .replace("watercolor. ", "");
            assert_eq!(before, after, "seed {i} reshuffled");
        }
    }

    #[test]
    fn affix_seed_is_relative_to_the_project_root() {
        // Same image, two checkouts of the same dataset → same seed.
        assert_eq!(
            AffixSeed::for_image(
                Some(Path::new("/home/a/ds")),
                Path::new("/home/a/ds/x/i.png")
            ),
            AffixSeed::for_image(Some(Path::new("/mnt/d/ds")), Path::new("/mnt/d/ds/x/i.png")),
        );
        // Different location inside the dataset → different seed.
        assert_ne!(
            AffixSeed::for_image(Some(Path::new("/ds")), Path::new("/ds/x/i.png")),
            AffixSeed::for_image(Some(Path::new("/ds")), Path::new("/ds/y/i.png")),
        );
    }

    #[test]
    fn affix_seed_ignores_the_file_extension() {
        // `dataset edit` / `upscale` can hand back a different container for
        // the same image; the sidecar (and the seed) key on the stem.
        assert_eq!(
            AffixSeed::for_image(Some(Path::new("/ds")), Path::new("/ds/i.png")),
            AffixSeed::for_image(Some(Path::new("/ds")), Path::new("/ds/i.webp")),
        );
    }

    #[test]
    fn common_layer_participates_in_classification() {
        let sc = Sidecar::default();
        let g = group(&["official_school_uniform", "official_lounge_wear"]);
        let common = CommonTags::new(["official_school_uniform"]);
        assert_eq!(
            classify(&sc, &g, &common),
            Classification::Tag("official_school_uniform".into())
        );
    }

    #[test]
    fn common_suppression_hides_auto_tag_from_classification() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("a"));
        let g = group(&["a", "b"]);
        assert_eq!(
            classify(&sc, &g, &CommonTags::new(["-a"])),
            Classification::Unset
        );
    }

    #[test]
    fn apply_drop_writes_per_image_override_against_common_layer() {
        // `y` comes from the shared layer; dropping the image into `x` has to
        // override it for this image only.
        let mut sc = Sidecar::default();
        let g = group(&["x", "y"]);
        let common = CommonTags::new(["y"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()), &common);
        assert!(sc.manual_tags.contains(&"x".to_string()));
        assert!(sc.manual_tags.contains(&"-y".to_string()));
        assert_eq!(classify(&sc, &g, &common), Classification::Tag("x".into()));
    }

    #[test]
    fn apply_drop_onto_common_positive_only_clears_the_override() {
        // `x` is already supplied by the shared layer, so re-selecting it
        // just drops the `-x` marker instead of writing a redundant copy.
        let mut sc = Sidecar::default();
        sc.manual_tags.push("-x".into());
        let g = group(&["x", "y"]);
        let common = CommonTags::new(["x"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()), &common);
        assert!(sc.manual_tags.is_empty());
        assert_eq!(classify(&sc, &g, &common), Classification::Tag("x".into()));
    }

    #[test]
    fn apply_drop_round_trip_tag_x_then_tag_y() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y"]);

        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("x".into()),
            &CommonTags::default(),
        );
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("x".into())
        );

        apply_drop(
            &mut sc,
            &g,
            &DropTarget::Tag("y".into()),
            &CommonTags::default(),
        );
        assert_eq!(
            classify(&sc, &g, &CommonTags::default()),
            Classification::Tag("y".into())
        );
        // After flipping back, `-x` should be present (since x was just
        // a positive manual tag) and `y` positive.
        assert!(sc.manual_tags.contains(&"y".to_string()));
        assert!(sc.manual_tags.contains(&"-x".to_string()));
    }
}
