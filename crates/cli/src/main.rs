use std::path::PathBuf;

use anima_tagger_booru::{BooruClient, BooruError};
use anima_tagger_captioner::Captioner;
use anima_tagger_core::config::ProjectConfig;
use anima_tagger_core::export;
use anima_tagger_core::sidecar::{Sidecar, TaggerInfo};
use anima_tagger_core::walk::iter_images;
use anima_tagger_tagger::Tagger;
use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};

/// Whether to copy the generated reference caption into `manual_caption`
/// after captioning. Default depends on the resolved prompt count: a
/// single prompt promotes if-empty (the typical case where the auto
/// caption *is* the canonical one); multiple prompts default to `never`
/// so a comparison run doesn't randomly pick one to promote. Use
/// `always` to overwrite an existing manual caption (e.g. when promoting
/// a second prompt's caption over a previously-promoted one) without
/// clearing it in the GUI first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PromoteMode {
    Never,
    IfEmpty,
    /// Copy the resolved prompt's caption into `manual_caption`,
    /// overwriting any existing manual caption.
    Always,
}

/// Output layout for the `metadata` command.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum MetadataFormat {
    /// kohya-ss/sd-scripts fine-tune metadata: a single `meta.json` mapping
    /// each image's absolute path to `{ tags, caption }`. Tags and caption
    /// are kept in separate fields.
    SdScripts,
    /// musubi-tuner metadata JSONL (`image_jsonl_file`): one
    /// `{"image_path", "caption"}` object per line. Caption-only — tags are
    /// not emitted as a separate field; fold any trigger/proportion tags
    /// into the caption via `[export.<p>.caption_prefixes]`. Images without
    /// a caption are skipped.
    Musubi,
}

#[derive(Parser)]
#[command(
    name = "anima-tagger",
    about = "Manage manual + auto + booru tags and captions for ANIMA-style LoRA datasets"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the automatic tagger over images in a directory.
    Tag {
        dir: PathBuf,
        /// Name of a `[tagger.<name>]` profile in `anima-tagger.toml`.
        #[arg(long)]
        model: Option<String>,
        /// Re-tag images that already have an auto-tag record.
        #[arg(long)]
        force: bool,
        /// Override the storage threshold from the tagger profile.
        #[arg(long)]
        threshold: Option<f32>,
    },
    /// Run the automatic captioner over images in a directory.
    Caption {
        dir: PathBuf,
        /// Name of a `[captioner.<name>]` profile in `anima-tagger.toml`.
        #[arg(long)]
        model: Option<String>,
        /// Re-caption images that already have a caption record.
        #[arg(long)]
        force: bool,
        /// Comma-separated prompt names overriding the profile's
        /// `prompts` field for this run. Names must exist in
        /// `[captioner_prompts]` (or be the built-in `default`).
        #[arg(long, value_delimiter = ',')]
        prompts: Option<Vec<String>>,
        /// Copy the resolved single prompt's caption into the manual
        /// slot after generation. Requires exactly one resolved prompt.
        /// `if-empty` only fills an empty manual slot; `always` overwrites
        /// an existing manual caption; `never` skips promotion.
        /// Default: `if-empty` when 1 prompt is active, `never` otherwise.
        #[arg(long, value_enum)]
        promote_to_manual: Option<PromoteMode>,
    },
    /// Fetch tags from a booru API by image MD5 hash.
    Booru {
        dir: PathBuf,
        /// Booru source (`danbooru` is the only one currently implemented).
        #[arg(long, default_value = "danbooru")]
        source: String,
        /// Re-fetch images that already have booru data.
        #[arg(long)]
        force: bool,
    },
    /// Merge manual + auto + booru tags and write `<image>.txt` for training.
    Export {
        dir: PathBuf,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        threshold: Option<f32>,
    },
    /// Write a dataset metadata file for every image with a sidecar.
    /// `--format sd-scripts` (default) emits a kohya-ss/sd-scripts
    /// `meta.json` with tags + captions; `--format musubi` emits a
    /// musubi-tuner caption-only JSONL.
    Metadata {
        dir: PathBuf,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        threshold: Option<f32>,
        /// Metadata layout. `sd-scripts` → `meta.json`; `musubi` →
        /// caption-only `meta.jsonl`.
        #[arg(long, value_enum, default_value = "sd-scripts")]
        format: MetadataFormat,
        /// Output path (default: `<dir>/meta.json` for sd-scripts,
        /// `<dir>/meta.jsonl` for musubi).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Show sidecar status for images in a directory.
    Status { dir: PathBuf },
    /// Classify images against a `[tag_group.<name>]` from
    /// `anima-tagger.toml`. Each image is bucketed as one of the group's
    /// tags, "unset" (no group tag present), or "violation" (multiple).
    /// Violations are informational, not errors.
    ValidateTagGroup {
        dir: PathBuf,
        /// Name of the `[tag_group.<name>]` to check against.
        #[arg(long)]
        group: String,
        /// Show only unset + violation rows; hide cleanly-classified images.
        #[arg(long)]
        problems_only: bool,
        /// Emit one JSON object per line instead of the text table.
        #[arg(long)]
        json: bool,
    },
    /// Tokenize the would-be export text per image and flag overflows
    /// against the training context budget. Uses ANIMA's text encoder
    /// tokenizer (`Qwen/Qwen3-0.6B`).
    Tokens {
        dir: PathBuf,
        /// Export profile (same semantics as `export`).
        #[arg(long)]
        profile: Option<String>,
        /// Override the auto-tag score threshold from the export profile.
        #[arg(long)]
        threshold: Option<f32>,
        /// Token budget. Default 512 = ANIMA's qwen3 / t5
        /// max_token_length training cap.
        #[arg(long, default_value_t = 512)]
        limit: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Tag {
            dir,
            model,
            force,
            threshold,
        } => cmd_tag(dir, model, force, threshold),
        Command::Caption {
            dir,
            model,
            force,
            prompts,
            promote_to_manual,
        } => cmd_caption(dir, model, force, prompts, promote_to_manual),
        Command::Booru { dir, source, force } => cmd_booru(dir, source, force),
        Command::Export {
            dir,
            profile,
            threshold,
        } => cmd_export(dir, profile, threshold),
        Command::Metadata {
            dir,
            profile,
            threshold,
            format,
            output,
        } => cmd_metadata(dir, profile, threshold, format, output),
        Command::Status { dir } => cmd_status(dir),
        Command::ValidateTagGroup {
            dir,
            group,
            problems_only,
            json,
        } => cmd_validate_tag_group(dir, group, problems_only, json),
        Command::Tokens {
            dir,
            profile,
            threshold,
            limit,
        } => cmd_tokens(dir, profile, threshold, limit),
    }
}

fn cmd_tag(
    dir: PathBuf,
    model_name: Option<String>,
    force: bool,
    threshold_override: Option<f32>,
) -> Result<()> {
    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let (resolved_name, profile) = cfg.resolve_tagger(model_name.as_deref());
    let threshold = threshold_override.unwrap_or(profile.storage_threshold);

    eprintln!("loading tagger `{resolved_name}` from {} …", profile.repo);
    let mut tagger = Tagger::from_profile(&profile)?;
    eprintln!("model ready ({} tags)", tagger.num_tags());

    let mut tagged = 0usize;
    let mut skipped = 0usize;
    for image in iter_images(&dir) {
        let mut sc = Sidecar::load_or_default(&image)?;
        if !force && sc.is_auto_tagged() {
            skipped += 1;
            continue;
        }
        let tags = tagger.tag_image(&image, threshold)?;
        let n = tags.len();
        sc.auto_tags = tags;
        sc.tagger = Some(TaggerInfo {
            model: resolved_name.clone(),
            tagged_at: Utc::now(),
        });
        sc.save(&image)?;
        tagged += 1;
        println!("tagged {} ({n} tags)", image.display());
    }
    println!("done: {tagged} tagged, {skipped} skipped (use --force to retag)");
    Ok(())
}

fn cmd_caption(
    dir: PathBuf,
    model_name: Option<String>,
    force: bool,
    prompts_override: Option<Vec<String>>,
    promote_arg: Option<PromoteMode>,
) -> Result<()> {
    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let (resolved_name, mut profile) = cfg.resolve_captioner(model_name.as_deref());
    if let Some(names) = prompts_override {
        profile.set_prompt_names(names);
    }
    let library = cfg.prompt_library();
    let prompts = profile
        .resolved_prompts(&library)
        .with_context(|| format!("resolving prompts for captioner `{resolved_name}`"))?;

    let promote_mode = promote_arg.unwrap_or(if prompts.len() == 1 {
        PromoteMode::IfEmpty
    } else {
        PromoteMode::Never
    });
    if promote_mode != PromoteMode::Never && prompts.len() != 1 {
        anyhow::bail!(
            "--promote-to-manual requires exactly one resolved prompt; got {} \
             (use --prompts=<name> to narrow)",
            prompts.len()
        );
    }
    let promote_key = (promote_mode != PromoteMode::Never)
        .then(|| format!("{resolved_name}.{}", prompts[0].0));

    eprintln!(
        "loading captioner `{resolved_name}` from {} (prompts: {}, promote: {:?}) …",
        profile.source_label(),
        prompts.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "),
        promote_mode,
    );
    let mut captioner = Captioner::from_profile(&profile)?;
    eprintln!("captioner ready");

    let mut captioned = 0usize;
    let mut skipped = 0usize;
    let mut promoted = 0usize;
    for image in iter_images(&dir) {
        let mut sc = Sidecar::load_or_default(&image)?;
        let pending: Vec<(String, String, String)> = prompts
            .iter()
            .filter_map(|(pname, ptext)| {
                let key = format!("{resolved_name}.{pname}");
                if !force && sc.captions.contains_key(&key) {
                    None
                } else {
                    Some((key, pname.clone(), ptext.clone()))
                }
            })
            .collect();
        let mut dirty = false;
        let hint = sc.caption_hint.clone();
        if pending.is_empty() {
            skipped += 1;
        } else {
            for (key, pname, ptext) in pending {
                let caption = captioner.caption_image(&image, &ptext, hint.as_deref())?;
                let preview: String = caption.chars().take(60).collect();
                sc.set_caption(key, caption);
                dirty = true;
                println!("captioned {} [{pname}] — \"{preview}…\"", image.display());
            }
            captioned += 1;
        }

        // Promote step runs independently of generation, so a follow-up
        // run (`--prompts=default --promote-to-manual=if-empty`) can
        // copy an existing reference to manual without regenerating.
        if let Some(key) = promote_key.as_deref()
            && let Some(entry) = sc.captions.get(key)
        {
            let manual_empty = sc
                .manual_caption
                .as_deref()
                .map(str::trim)
                .map(|s| s.is_empty())
                .unwrap_or(true);
            let copy = match promote_mode {
                PromoteMode::IfEmpty => manual_empty,
                PromoteMode::Always => true,
                PromoteMode::Never => false,
            };
            if copy {
                let text = entry.caption.clone();
                sc.set_manual_caption(&text);
                dirty = true;
                promoted += 1;
                println!("promoted {} → manual ({key})", image.display());
            }
        }

        if dirty {
            sc.save(&image)?;
        }
    }
    println!(
        "done: {captioned} captioned, {skipped} skipped, {promoted} promoted to manual \
         (use --force to recaption)",
    );
    Ok(())
}

fn cmd_booru(dir: PathBuf, source: String, force: bool) -> Result<()> {
    let client = match source.as_str() {
        "danbooru" => BooruClient::danbooru(),
        other => anyhow::bail!(
            "unsupported booru source `{other}` (only 'danbooru' is implemented)"
        ),
    };

    let mut fetched = 0usize;
    let mut not_found = 0usize;
    let mut skipped = 0usize;
    for image in iter_images(&dir) {
        let mut sc = Sidecar::load_or_default(&image)?;
        if !force && sc.has_booru() {
            skipped += 1;
            continue;
        }
        match client.fetch_for_image(&image) {
            Ok((tags, info)) => {
                let n = tags.len();
                sc.booru_tags = tags;
                sc.booru = Some(info);
                sc.save(&image)?;
                fetched += 1;
                println!("fetched {} ({n} tags)", image.display());
            }
            Err(BooruError::NotFound(_)) => {
                not_found += 1;
                println!("not on booru: {}", image.display());
            }
            Err(e) => {
                eprintln!("error: {}: {e}", image.display());
            }
        }
    }
    println!("done: {fetched} fetched, {not_found} not found, {skipped} skipped");
    Ok(())
}

fn cmd_export(dir: PathBuf, profile_name: Option<String>, threshold: Option<f32>) -> Result<()> {
    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let mut profile = cfg.resolve_profile(profile_name.as_deref());
    if let Some(t) = threshold {
        profile.threshold = t;
    }

    let mut written = 0usize;
    let mut skipped = 0usize;
    for image in iter_images(&dir) {
        let sidecar = match Sidecar::load(&image)? {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };
        let out = export::export_image(&image, &sidecar, &profile)?;
        println!("wrote {}", out.display());
        written += 1;
    }
    println!("done: {written} written, {skipped} skipped (no sidecar)");
    Ok(())
}

fn cmd_metadata(
    dir: PathBuf,
    profile_name: Option<String>,
    threshold: Option<f32>,
    format: MetadataFormat,
    output: Option<PathBuf>,
) -> Result<()> {
    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let mut profile = cfg.resolve_profile(profile_name.as_deref());
    if let Some(t) = threshold {
        profile.threshold = t;
    }
    // The trainer shuffles tags at training time; metadata stays stable for diffability.
    profile.shuffle = false;

    match format {
        MetadataFormat::SdScripts => cmd_metadata_sd_scripts(&dir, &profile, output),
        MetadataFormat::Musubi => cmd_metadata_musubi(&dir, &profile, output),
    }
}

fn cmd_metadata_sd_scripts(
    dir: &std::path::Path,
    profile: &anima_tagger_core::config::ExportProfile,
    output: Option<PathBuf>,
) -> Result<()> {
    use std::collections::BTreeMap;

    let mut meta: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut count = 0usize;
    let mut skipped = 0usize;

    for image in iter_images(dir) {
        let sidecar = match Sidecar::load(&image)? {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };
        let tags = export::build_tags(&sidecar, profile);
        let mut entry = serde_json::Map::new();
        if !tags.is_empty() {
            let joined = tags
                .iter()
                .map(|t| t.replace('_', " "))
                .collect::<Vec<_>>()
                .join(", ");
            entry.insert("tags".to_string(), serde_json::Value::String(joined));
        }
        if let Some(cap) = export::build_caption(&sidecar, profile) {
            entry.insert("caption".to_string(), serde_json::Value::String(cap));
        }
        if entry.is_empty() {
            continue;
        }
        meta.insert(metadata_image_key(&image), serde_json::Value::Object(entry));
        count += 1;
    }

    let output_path = output.unwrap_or_else(|| dir.join("meta.json"));
    let json = serde_json::to_string_pretty(&meta)?;
    std::fs::write(&output_path, json)
        .with_context(|| format!("writing {}", output_path.display()))?;
    println!(
        "wrote {} ({count} entries, {skipped} images without sidecar skipped)",
        output_path.display()
    );
    Ok(())
}

fn cmd_metadata_musubi(
    dir: &std::path::Path,
    profile: &anima_tagger_core::config::ExportProfile,
    output: Option<PathBuf>,
) -> Result<()> {
    // (image_path, caption) pairs, sorted by path so the JSONL is stable
    // across runs (diff-friendly) regardless of directory iteration order.
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut no_sidecar = 0usize;
    let mut no_caption = 0usize;

    for image in iter_images(dir) {
        let sidecar = match Sidecar::load(&image)? {
            Some(s) => s,
            None => {
                no_sidecar += 1;
                continue;
            }
        };
        // musubi training here is caption-only: an image without a caption
        // has nothing to contribute, so skip it rather than emit a blank.
        match export::build_caption(&sidecar, profile) {
            Some(cap) => rows.push((metadata_image_key(&image), cap)),
            None => no_caption += 1,
        }
    }
    rows.sort();

    let mut body = String::new();
    for (image_path, caption) in &rows {
        let line = serde_json::json!({ "image_path": image_path, "caption": caption });
        body.push_str(&serde_json::to_string(&line)?);
        body.push('\n');
    }

    let output_path = output.unwrap_or_else(|| dir.join("meta.jsonl"));
    std::fs::write(&output_path, body)
        .with_context(|| format!("writing {}", output_path.display()))?;
    println!(
        "wrote {} ({} entries, {no_caption} without caption skipped, \
         {no_sidecar} without sidecar skipped)",
        output_path.display(),
        rows.len(),
    );
    Ok(())
}

/// Absolute path used as an image's metadata key (canonicalized when
/// possible, falling back to the display path for not-yet-existing inputs).
fn metadata_image_key(image: &std::path::Path) -> String {
    image
        .canonicalize()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| image.display().to_string())
}

fn cmd_status(dir: PathBuf) -> Result<()> {
    for image in iter_images(&dir) {
        match Sidecar::load(&image)? {
            None => println!("[   ] manual=0   {}", image.display()),
            Some(s) => {
                let auto = if s.is_auto_tagged() { 'T' } else { ' ' };
                let cap = if s.is_captioned() { 'C' } else { ' ' };
                let booru = if s.has_booru() { 'B' } else { ' ' };
                let n = s.manual_tags.len();
                println!("[{auto}{cap}{booru}] manual={n:<3} {}", image.display());
            }
        }
    }
    Ok(())
}

fn cmd_validate_tag_group(
    dir: PathBuf,
    group_name: String,
    problems_only: bool,
    json: bool,
) -> Result<()> {
    use anima_tagger_core::tag_group::{Classification, classify};

    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let group = cfg.tag_groups.get(&group_name).with_context(|| {
        format!(
            "tag_group `{group_name}` is not defined in any anima-tagger.toml \
             (project or user). Add a [tag_group.{group_name}] section."
        )
    })?;

    let mut tagged = 0usize;
    let mut unset = 0usize;
    let mut violations = 0usize;

    for image in iter_images(&dir) {
        let sc = Sidecar::load_or_default(&image)?;
        let classification = classify(&sc, group);
        let (state_text, state_json) = match &classification {
            Classification::Tag(t) => {
                tagged += 1;
                (format!("tag={t}"), serde_json::json!({"state": "tag", "tag": t}))
            }
            Classification::Unset => {
                unset += 1;
                ("unset".to_string(), serde_json::json!({"state": "unset"}))
            }
            Classification::Violation(tags) => {
                violations += 1;
                (
                    format!("violation={}", tags.join(",")),
                    serde_json::json!({"state": "violation", "tags": tags}),
                )
            }
        };

        let is_problem = matches!(
            classification,
            Classification::Unset | Classification::Violation(_)
        );
        if problems_only && !is_problem {
            continue;
        }

        if json {
            let mut obj = state_json.as_object().unwrap().clone();
            obj.insert(
                "image".to_string(),
                serde_json::Value::String(image.display().to_string()),
            );
            println!("{}", serde_json::Value::Object(obj));
        } else {
            println!("{state_text:<32} {}", image.display());
        }
    }

    if !json {
        eprintln!("{tagged} tagged, {unset} unset, {violations} violation");
    }
    Ok(())
}

fn cmd_tokens(
    dir: PathBuf,
    profile_name: Option<String>,
    threshold: Option<f32>,
    limit: usize,
) -> Result<()> {
    use anima_tagger_core::hub;
    use tokenizers::Tokenizer;

    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let mut profile = cfg.resolve_profile(profile_name.as_deref());
    if let Some(t) = threshold {
        profile.threshold = t;
    }
    profile.shuffle = false;

    eprintln!("[tokens] fetching Qwen/Qwen3-0.6B tokenizer...");
    let paths = hub::fetch_files("Qwen/Qwen3-0.6B", None, &["tokenizer.json"])
        .context("download Qwen3-0.6B tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&paths[0])
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let count = |s: &str| -> Result<usize> {
        if s.is_empty() {
            return Ok(0);
        }
        let enc = tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(enc.len())
    };

    let mut totals: Vec<usize> = Vec::new();
    let mut over: Vec<(PathBuf, usize, usize, usize)> = Vec::new();
    let mut analyzed = 0usize;
    let mut no_sidecar = 0usize;

    for image in iter_images(&dir) {
        let Some(sidecar) = Sidecar::load(&image)? else {
            no_sidecar += 1;
            continue;
        };
        let tags = anima_tagger_core::export::build_tags(&sidecar, &profile);
        let tags_text = tags
            .iter()
            .map(|t| t.replace('_', " "))
            .collect::<Vec<_>>()
            .join(", ");
        let caption_text = export::build_caption(&sidecar, &profile).unwrap_or_default();

        let tag_tok = count(&tags_text)?;
        let cap_tok = count(&caption_text)?;
        // The trainer concatenates them as one input; a single space
        // tokenizes to 0 or 1 BPE pieces, so plain sum is a tight upper
        // bound on the combined length.
        let total = tag_tok + cap_tok;
        totals.push(total);
        if total > limit {
            over.push((image.clone(), total, tag_tok, cap_tok));
        }
        analyzed += 1;
    }

    if analyzed == 0 {
        println!("no images with sidecar (no_sidecar={no_sidecar})");
        return Ok(());
    }

    totals.sort_unstable();
    let max = *totals.last().unwrap();
    let pct = |p: f32| -> usize {
        let i = ((totals.len() as f32 - 1.0) * p).round() as usize;
        totals[i.min(totals.len() - 1)]
    };

    println!(
        "analyzed {analyzed} images (skipped {no_sidecar} without sidecar) | budget {limit}"
    );
    println!(
        "tokens p50={} p90={} p99={} max={} | over budget: {}",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        max,
        over.len()
    );

    if !over.is_empty() {
        println!("\noverflows:");
        for (path, total, tag_tok, cap_tok) in &over {
            println!(
                "  {total:>4} (tags={tag_tok}, caption={cap_tok})  {}",
                path.display()
            );
        }
    }
    Ok(())
}
