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

use std::collections::HashSet;

use crate::config::TagGroup;
use crate::sidecar::Sidecar;

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

/// Build the lowercase-stem effective tag set for `sc`. Suppressed
/// (`-foo`) entries are removed.
pub fn effective_tag_set(sc: &Sidecar) -> HashSet<String> {
    let suppressed = sc.suppressed_set();
    let mut set: HashSet<String> = HashSet::new();
    for t in sc.manual_positive_tags() {
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
pub fn classify(sc: &Sidecar, group: &TagGroup) -> Classification {
    let eff = effective_tag_set(sc);
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
pub fn apply_drop(sc: &mut Sidecar, group: &TagGroup, target: &DropTarget) {
    let eff = effective_tag_set(sc);
    match target {
        DropTarget::Tag(x) => {
            let x_trimmed = x.trim();
            if x_trimmed.is_empty() {
                return;
            }
            sc.unsuppress(x_trimmed);
            // Add as positive only if not already a positive manual
            // entry. `add_manual_tag` is a no-op for duplicates.
            sc.add_manual_tag(x_trimmed);

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
        let set = effective_tag_set(&sc);
        assert!(set.contains("alpha"));
        assert!(set.contains("beta"));
        assert!(set.contains("gamma"));
    }

    #[test]
    fn effective_tag_set_strips_suppressed_entries() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("watermark"));
        sc.manual_tags.push("-watermark".into());
        let set = effective_tag_set(&sc);
        assert!(!set.contains("watermark"));
    }

    #[test]
    fn classify_returns_unset_when_none_present() {
        let sc = Sidecar::default();
        let g = group(&["a", "b"]);
        assert_eq!(classify(&sc, &g), Classification::Unset);
    }

    #[test]
    fn classify_returns_tag_when_one_present() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("a".into());
        let g = group(&["a", "b"]);
        assert_eq!(classify(&sc, &g), Classification::Tag("a".into()));
    }

    #[test]
    fn classify_returns_violation_when_two_present_in_group_order() {
        let mut sc = Sidecar::default();
        // mix sources to ensure both contribute
        sc.auto_tags.push(auto("b"));
        sc.booru_tags.push(booru("a"));
        let g = group(&["a", "b"]);
        match classify(&sc, &g) {
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
        assert_eq!(classify(&sc, &g), Classification::Tag("_none".into()));
    }

    #[test]
    fn classify_skips_tags_suppressed_by_negative_marker() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("a"));
        sc.manual_tags.push("-a".into());
        sc.manual_tags.push("b".into());
        let g = group(&["a", "b"]);
        assert_eq!(classify(&sc, &g), Classification::Tag("b".into()));
    }

    #[test]
    fn apply_drop_tag_adds_positive_and_suppresses_present_siblings() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y", "z"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));

        assert!(sc.manual_tags.contains(&"x".to_string()));
        assert!(sc.manual_tags.contains(&"-y".to_string()));
        // z was nowhere, so no `-z` written
        assert!(!sc.manual_tags.iter().any(|t| t == "-z"));
        assert_eq!(classify(&sc, &g), Classification::Tag("x".into()));
    }

    #[test]
    fn apply_drop_tag_clears_existing_negative_on_target() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("-x".into());
        let g = group(&["x", "y"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));
        assert!(!sc.manual_tags.iter().any(|t| t == "-x"));
        assert!(sc.manual_tags.contains(&"x".to_string()));
    }

    #[test]
    fn apply_drop_tag_replaces_positive_sibling_with_suppression() {
        let mut sc = Sidecar::default();
        sc.manual_tags.push("y".into());
        let g = group(&["x", "y"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));
        assert!(!sc.manual_tags.iter().any(|t| t == "y"));
        assert!(sc.manual_tags.contains(&"-y".to_string()));
        assert!(sc.manual_tags.contains(&"x".to_string()));
    }

    #[test]
    fn apply_drop_unset_suppresses_only_present_group_tags() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y", "z"]);
        apply_drop(&mut sc, &g, &DropTarget::Unset);

        assert!(sc.manual_tags.contains(&"-y".to_string()));
        // x, z were absent → no eager suppression
        assert!(!sc.manual_tags.iter().any(|t| t == "-x"));
        assert!(!sc.manual_tags.iter().any(|t| t == "-z"));
        assert_eq!(classify(&sc, &g), Classification::Unset);
    }

    #[test]
    fn apply_drop_is_idempotent() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y"]);
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));
        let after_once = sc.manual_tags.clone();
        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));
        assert_eq!(sc.manual_tags, after_once);
    }

    #[test]
    fn apply_drop_round_trip_tag_x_then_tag_y() {
        let mut sc = Sidecar::default();
        sc.auto_tags.push(auto("y"));
        let g = group(&["x", "y"]);

        apply_drop(&mut sc, &g, &DropTarget::Tag("x".into()));
        assert_eq!(classify(&sc, &g), Classification::Tag("x".into()));

        apply_drop(&mut sc, &g, &DropTarget::Tag("y".into()));
        assert_eq!(classify(&sc, &g), Classification::Tag("y".into()));
        // After flipping back, `-x` should be present (since x was just
        // a positive manual tag) and `y` positive.
        assert!(sc.manual_tags.contains(&"y".to_string()));
        assert!(sc.manual_tags.contains(&"-x".to_string()));
    }
}
