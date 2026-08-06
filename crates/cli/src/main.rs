use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use fwaun_tools_booru::{BooruClient, BooruError};
use fwaun_tools_captioner::{Captioner, CaptionerError};
use fwaun_tools_comfyui::Client as ComfyClient;
use fwaun_tools_comfyui::upscale::{Options as UpscaleOptions, Upscaler};
use fwaun_tools_core::common_tags;
use fwaun_tools_core::config::ProjectConfig;
use fwaun_tools_core::export;
use fwaun_tools_core::sidecar::{Sidecar, TaggerInfo};
use fwaun_tools_core::walk::iter_images;
use fwaun_tools_tagger::Tagger;

mod model;
mod progress;

use progress::Progress;

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
    name = "fwaun-tools",
    about = "Tools for training fwaun models: LoRA dataset curation (`dataset`) and diffusion-checkpoint ops (`model`)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Top-level command groups: `dataset <verb>` and `model <verb>`.
#[derive(Subcommand)]
enum Command {
    /// Dataset curation — tag, caption, fetch booru, edit tags, and export
    /// captions/metadata for training.
    #[command(subcommand)]
    Dataset(DatasetCommand),
    /// Diffusion-checkpoint tools — task-vector merge, LoRA extraction, and
    /// INT8 quantization over safetensors files.
    #[command(subcommand)]
    Model(model::ModelCommand),
}

/// Dataset-curation subcommands (`fwaun-tools dataset <verb>`).
#[derive(Subcommand)]
enum DatasetCommand {
    /// Run the automatic tagger over images in a directory.
    Tag {
        dir: PathBuf,
        /// Name of a `[tagger.<name>]` profile in `fwaun-tools.toml`.
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
        /// Name of a `[captioner.<name>]` profile in `fwaun-tools.toml`.
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
    /// Batch-upscale every image in a directory by sending it to an existing
    /// ComfyUI server over its HTTP API, writing the results to a separate
    /// output directory (default: a `<dir>_upscaled` sibling). Uses the
    /// built-in ESRGAN-style workflow (set `--upscale-model` or the profile's
    /// `upscale_model`), or your own API-format workflow via `--workflow`.
    /// Each image's `.ron` sidecar is copied alongside so the output is a
    /// ready-to-train dataset. Run `dataset upscale-models` first to see which
    /// model filenames the server offers.
    Upscale {
        dir: PathBuf,
        /// Name of an `[upscaler.<name>]` profile in `fwaun-tools.toml`.
        #[arg(long)]
        profile: Option<String>,
        /// Output directory. Relative sub-paths are preserved under it.
        /// Default: a `<dir>_upscaled` sibling directory.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Override the ComfyUI server root (e.g. `http://127.0.0.1:8188`).
        #[arg(long)]
        base_url: Option<String>,
        /// Upscale-model filename for the built-in workflow
        /// (e.g. `RealESRGAN_x4plus.pth`). Overrides the profile.
        #[arg(long)]
        upscale_model: Option<String>,
        /// API-format workflow JSON to use instead of the built-in graph.
        /// Overrides the profile.
        #[arg(long)]
        workflow: Option<PathBuf>,
        /// Cap the upscaled longest edge to this many pixels (0 = keep the
        /// model's native output). Overrides the profile.
        #[arg(long)]
        max_edge: Option<u32>,
        /// Re-upscale images whose output file already exists.
        #[arg(long)]
        force: bool,
        /// List what would be upscaled without contacting ComfyUI.
        #[arg(long)]
        dry_run: bool,
    },
    /// List the upscale-model filenames the ComfyUI server offers (read from
    /// its `/object_info`). Use one of these as `--upscale-model` / the
    /// profile's `upscale_model`.
    UpscaleModels {
        /// Name of an `[upscaler.<name>]` profile to read `base_url` from.
        #[arg(long)]
        profile: Option<String>,
        /// Override the ComfyUI server root (e.g. `http://127.0.0.1:8188`).
        #[arg(long)]
        base_url: Option<String>,
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
    /// Move images (and their `.ron` sidecars) carrying given tag(s) into
    /// another directory. Matching uses the effective tag set
    /// (manual ∪ auto ∪ booru minus `-` suppressions), same as
    /// `validate-tag-group`. Sub-paths relative to `dir` are preserved
    /// under `dest`. The GUI has no folder concept, so this is CLI-only.
    Mv {
        /// Source directory to scan (recursively).
        dir: PathBuf,
        /// Destination directory. Created if missing.
        dest: PathBuf,
        /// Tag(s) to match, comma-separated or repeated. An image must
        /// carry *all* of them (AND). Case-insensitive.
        #[arg(long, value_delimiter = ',', required = true)]
        tags: Vec<String>,
        /// List what would move without touching the filesystem.
        #[arg(long)]
        dry_run: bool,
    },
    /// Append manual tag(s) to every image's sidecar in a directory.
    /// Tags are added verbatim to the manual layer: `foo` is a positive
    /// tag, `-foo` a suppression marker (removes a matching auto/booru tag
    /// from the export). Entries already present are left as-is; a sidecar
    /// is created for images that don't have one. Pass `-foo` after `=`
    /// (`--tags=-foo`) so it isn't parsed as a flag. To rename a tag, reach
    /// for `replace-tag` rather than this plus `remove-tag`: it keeps the
    /// entry's position and touches only the images that have it.
    ///
    /// When <DIR> is the dataset root (it holds the `fwaun-tools.toml`), the
    /// edit covers the whole dataset, so it is written to that file's
    /// `common_tags` list instead of to every sidecar — one line, and images
    /// added later are covered too. Use `--per-image` to force the old
    /// per-sidecar behaviour.
    AddTag {
        /// Directory to scan (recursively).
        dir: PathBuf,
        /// Tag(s) to add, comma-separated or repeated. Case preserved.
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            allow_hyphen_values = true
        )]
        tags: Vec<String>,
        /// List what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Always write to each image's sidecar, even at the dataset root.
        #[arg(long)]
        per_image: bool,
    },
    /// Remove manual tag(s) from every image's sidecar in a directory.
    /// Deletes the existing manual entry only (case-insensitive match),
    /// including `-foo` suppression markers when `-foo` is passed. It never
    /// adds a suppression marker and never touches auto/booru tags — to
    /// hide a model-produced tag, add `-foo` with `add-tag` instead. Images
    /// without the tag are left unchanged; sidecars are never created.
    ///
    /// When <DIR> is the dataset root (it holds the `fwaun-tools.toml`), the
    /// tags are dropped from that file's `common_tags` list instead of from
    /// the sidecars; per-image copies are reported but left alone, since
    /// they may be deliberate overrides. Use `--per-image` to clear those.
    RemoveTag {
        /// Directory to scan (recursively).
        dir: PathBuf,
        /// Tag(s) to remove, comma-separated or repeated. Match the leading
        /// `-` to drop a suppression marker (`--tags=-foo`).
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            allow_hyphen_values = true
        )]
        tags: Vec<String>,
        /// List what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Always write to each image's sidecar, even at the dataset root.
        #[arg(long)]
        per_image: bool,
    },
    /// Rename manual tag(s) across a directory: `--from OLD --to NEW`.
    ///
    /// Each match is rewritten where it sits, so the entry keeps its place in
    /// `manual_tags`, and images that don't carry OLD are left alone — the
    /// two things `remove-tag` + `add-tag` can't do. A NEW that an image
    /// already has absorbs the renamed entry instead of duplicating it.
    ///
    /// Matching is case-insensitive on the entry as written, like
    /// `remove-tag`: the leading `-` counts, so renaming `foo` never touches
    /// `-foo` (pass `--from=-foo --to=-bar` for that), and `--from Foo --to
    /// foo` fixes the casing of an entry. Repeat the pair (or comma-separate
    /// both lists) to run a whole rename table in one pass; the Nth `--from`
    /// goes with the Nth `--to`.
    ///
    /// Unlike `add-tag` / `remove-tag`, this always walks the sidecars — a
    /// rename has no whole-directory side effect to guard against. When <DIR>
    /// is the dataset root it *also* renames the entry in that config's
    /// `common_tags`; `--per-image` leaves the config alone.
    ReplaceTag {
        /// Directory to scan (recursively).
        dir: PathBuf,
        /// Tag(s) to rename. Comma-separated or repeated; pass a suppression
        /// marker after `=` (`--from=-foo`).
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            allow_hyphen_values = true
        )]
        from: Vec<String>,
        /// What each `--from` becomes, in the same order. Case preserved.
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            allow_hyphen_values = true
        )]
        to: Vec<String>,
        /// List what would change without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Skip the dataset root's `common_tags`; edit sidecars only.
        #[arg(long)]
        per_image: bool,
    },
    /// Show sidecar status for images in a directory.
    Status { dir: PathBuf },
    /// Classify images against a `[tag_group.<name>]` from
    /// `fwaun-tools.toml`. Each image is bucketed as one of the group's
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
        Command::Dataset(command) => run_dataset(command),
        Command::Model(command) => model::run(command),
    }
}

fn run_dataset(command: DatasetCommand) -> Result<()> {
    match command {
        DatasetCommand::Tag {
            dir,
            model,
            force,
            threshold,
        } => cmd_tag(dir, model, force, threshold),
        DatasetCommand::Caption {
            dir,
            model,
            force,
            prompts,
            promote_to_manual,
        } => cmd_caption(dir, model, force, prompts, promote_to_manual),
        DatasetCommand::Booru { dir, source, force } => cmd_booru(dir, source, force),
        DatasetCommand::Upscale {
            dir,
            profile,
            output,
            base_url,
            upscale_model,
            workflow,
            max_edge,
            force,
            dry_run,
        } => cmd_upscale(
            dir,
            profile,
            output,
            base_url,
            upscale_model,
            workflow,
            max_edge,
            force,
            dry_run,
        ),
        DatasetCommand::UpscaleModels { profile, base_url } => {
            cmd_upscale_models(profile, base_url)
        }
        DatasetCommand::Export {
            dir,
            profile,
            threshold,
        } => cmd_export(dir, profile, threshold),
        DatasetCommand::Metadata {
            dir,
            profile,
            threshold,
            format,
            output,
        } => cmd_metadata(dir, profile, threshold, format, output),
        DatasetCommand::Mv {
            dir,
            dest,
            tags,
            dry_run,
        } => cmd_mv(dir, dest, tags, dry_run),
        DatasetCommand::AddTag {
            dir,
            tags,
            dry_run,
            per_image,
        } => cmd_add_tag(dir, tags, dry_run, per_image),
        DatasetCommand::RemoveTag {
            dir,
            tags,
            dry_run,
            per_image,
        } => cmd_remove_tag(dir, tags, dry_run, per_image),
        DatasetCommand::ReplaceTag {
            dir,
            from,
            to,
            dry_run,
            per_image,
        } => cmd_replace_tag(dir, from, to, dry_run, per_image),
        DatasetCommand::Status { dir } => cmd_status(dir),
        DatasetCommand::ValidateTagGroup {
            dir,
            group,
            problems_only,
            json,
        } => cmd_validate_tag_group(dir, group, problems_only, json),
        DatasetCommand::Tokens {
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

    // Collected up front so the progress counter has a total to divide by;
    // the walk itself is a rounding error next to one inference per image.
    let images: Vec<PathBuf> = iter_images(&dir).collect();
    let progress = Progress::new("tag", images.len());

    let mut tagged = 0usize;
    let mut skipped = 0usize;
    for image in progress.wrap(images) {
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
        progress.println(format!("tagged {} ({n} tags)", image.display()));
    }
    progress.finish();
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
    let common = cfg.resolve_common_tags();
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
    let promote_key =
        (promote_mode != PromoteMode::Never).then(|| format!("{resolved_name}.{}", prompts[0].0));

    eprintln!(
        "loading captioner `{resolved_name}` from {} (prompts: {}, promote: {:?}) …",
        profile.source_label(),
        prompts
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        promote_mode,
    );
    let mut captioner = Captioner::from_profile(&profile)?;
    eprintln!("captioner ready");

    let images: Vec<PathBuf> = iter_images(&dir).collect();
    let progress = Progress::new("caption", images.len());

    let mut captioned = 0usize;
    let mut skipped = 0usize;
    let mut promoted = 0usize;
    let mut empty_failures = 0usize;
    for image in progress.wrap(images) {
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
        // Merge per-image hints with hints from tag groups whose tags are all
        // present, and resolve the tag-group caption prefix. The prefix is
        // embedded in the prompt (see `build_user_text`) so the model
        // continues from what export will prepend.
        let extra_hints =
            fwaun_tools_core::tag_group::resolved_caption_hints(&sc, &cfg.tag_groups, &common);
        let hint = sc.caption_hint_prompt_with(&extra_hints);
        let prefix =
            fwaun_tools_core::tag_group::resolved_caption_prefix(&sc, &cfg.tag_groups, &common);
        let prefix = (!prefix.is_empty()).then_some(prefix);
        if pending.is_empty() {
            skipped += 1;
        } else {
            let mut generated_any = false;
            for (key, pname, ptext) in pending {
                let caption = match captioner.caption_image(
                    &image,
                    &ptext,
                    hint.as_deref(),
                    prefix.as_deref(),
                ) {
                    Ok(c) => c,
                    // An all-empty result (after retries) is skipped rather than
                    // stored, so the key stays un-generated and a later run
                    // retries it. Don't abort the whole batch for one image.
                    Err(CaptionerError::EmptyCaption { attempts }) => {
                        empty_failures += 1;
                        progress.eprintln(format!(
                            "skipped {} [{pname}]: empty caption after {attempts} attempt(s)",
                            image.display()
                        ));
                        continue;
                    }
                    Err(e) => {
                        return Err(e)
                            .with_context(|| format!("captioning {} [{pname}]", image.display()));
                    }
                };
                let preview: String = caption.chars().take(60).collect();
                sc.set_caption(key, caption);
                dirty = true;
                generated_any = true;
                progress.println(format!(
                    "captioned {} [{pname}] — \"{preview}…\"",
                    image.display()
                ));
            }
            if generated_any {
                captioned += 1;
            }
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
                progress.println(format!("promoted {} → manual ({key})", image.display()));
            }
        }

        if dirty {
            sc.save(&image)?;
        }
    }
    progress.finish();
    println!(
        "done: {captioned} captioned, {skipped} skipped, {promoted} promoted to manual, \
         {empty_failures} empty (not saved) (use --force to recaption)",
    );
    Ok(())
}

fn cmd_booru(dir: PathBuf, source: String, force: bool) -> Result<()> {
    let client = match source.as_str() {
        "danbooru" => BooruClient::danbooru(),
        other => {
            anyhow::bail!("unsupported booru source `{other}` (only 'danbooru' is implemented)")
        }
    };

    let images: Vec<PathBuf> = iter_images(&dir).collect();
    let progress = Progress::new("booru", images.len());

    let mut fetched = 0usize;
    let mut not_found = 0usize;
    let mut skipped = 0usize;
    for image in progress.wrap(images) {
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
                progress.println(format!("fetched {} ({n} tags)", image.display()));
            }
            Err(BooruError::NotFound(_)) => {
                not_found += 1;
                progress.println(format!("not on booru: {}", image.display()));
            }
            Err(e) => {
                progress.eprintln(format!("error: {}: {e}", image.display()));
            }
        }
    }
    progress.finish();
    println!("done: {fetched} fetched, {not_found} not found, {skipped} skipped");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_upscale(
    dir: PathBuf,
    profile_name: Option<String>,
    output: Option<PathBuf>,
    base_url: Option<String>,
    upscale_model: Option<String>,
    workflow: Option<PathBuf>,
    max_edge: Option<u32>,
    force: bool,
    dry_run: bool,
) -> Result<()> {
    use fwaun_tools_core::sidecar::sidecar_path_for;

    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let (name, mut profile) = cfg.resolve_upscaler(profile_name.as_deref());

    // CLI flags override the resolved profile.
    if let Some(u) = base_url {
        profile.base_url = u;
    }
    if let Some(m) = upscale_model {
        profile.upscale_model = Some(m);
    }
    if let Some(w) = workflow {
        profile.workflow_template = Some(w);
    }
    if let Some(e) = max_edge {
        profile.max_edge = e;
    }

    // Default output: a `<dir>_upscaled` sibling so a re-run over `dir`
    // doesn't re-scan the results.
    let out_root = output.unwrap_or_else(|| match dir.file_name() {
        Some(fname) => {
            let mut n = fname.to_os_string();
            n.push("_upscaled");
            dir.with_file_name(n)
        }
        None => dir.join("upscaled"),
    });

    let upscaler = Upscaler::new(UpscaleOptions {
        base_url: profile.base_url.clone(),
        upscale_model: profile.upscale_model.clone(),
        workflow_template: profile.workflow_template.clone(),
        max_edge: profile.max_edge,
        timeout_secs: profile.timeout_secs,
        poll_interval_ms: profile.poll_interval_ms,
    })
    .context("configuring the ComfyUI upscaler")?;

    let how = match (&profile.workflow_template, &profile.upscale_model) {
        (Some(t), _) => format!("workflow template {}", t.display()),
        (None, Some(m)) => format!("built-in ESRGAN workflow, model {m}"),
        (None, None) => "built-in workflow".to_string(),
    };
    eprintln!(
        "upscaling via ComfyUI at {} (profile `{name}`, {how}) → {}",
        profile.base_url,
        out_root.display(),
    );

    // Never descend into our own output (matters when `out_root` sits inside
    // `dir`, e.g. the no-file-name fallback). Filtered here rather than in
    // the loop so the progress total counts only what will be worked on.
    let images: Vec<PathBuf> = iter_images(&dir)
        .filter(|image| !image.starts_with(&out_root))
        .collect();
    let progress = Progress::new("upscale", images.len());

    let mut done = 0usize;
    let mut would = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;
    for image in progress.wrap(images) {
        let rel = image.strip_prefix(&dir).unwrap_or(&image);
        // ComfyUI hands back PNG bytes regardless of the source format, so the
        // output always carries a `.png` extension. Sidecars are extension-
        // independent (`foo.*` → `foo.ron`), so the copied sidecar still lines
        // up with the renamed image.
        let target = out_root.join(rel).with_extension("png");
        if !force && target.exists() {
            skipped += 1;
            continue;
        }
        if dry_run {
            would += 1;
            progress.println(format!(
                "would upscale {} → {}",
                rel.display(),
                target.display()
            ));
            continue;
        }

        match upscaler.upscale_file(&image) {
            Ok(bytes) => {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
                std::fs::write(&target, bytes)
                    .with_context(|| format!("writing {}", target.display()))?;

                // Carry the sidecar over so the upscaled set stays a complete
                // dataset (tags/captions travel with the image).
                let src_side = sidecar_path_for(&image);
                if src_side.exists() {
                    let dst_side = sidecar_path_for(&target);
                    std::fs::copy(&src_side, &dst_side).with_context(|| {
                        format!(
                            "copying sidecar {} → {}",
                            src_side.display(),
                            dst_side.display()
                        )
                    })?;
                }
                done += 1;
                progress.println(format!("upscaled {} → {}", rel.display(), target.display()));
            }
            Err(e) => {
                errors += 1;
                progress.eprintln(format!("error: {}: {e}", image.display()));
            }
        }
    }
    progress.finish();

    if dry_run {
        println!("dry run: {would} would upscale, {skipped} already present");
    } else {
        println!("done: {done} upscaled, {skipped} skipped (already present), {errors} errored");
    }
    Ok(())
}

fn cmd_upscale_models(profile_name: Option<String>, base_url: Option<String>) -> Result<()> {
    use std::time::Duration;

    // Resolve base_url from the profile if present; a flag overrides it. Fall
    // back to the built-in default when there's no config at all.
    let base = base_url.unwrap_or_else(|| {
        let cfg = ProjectConfig::load_or_default(std::path::Path::new(".")).unwrap_or_default();
        let (_, profile) = cfg.resolve_upscaler(profile_name.as_deref());
        profile.base_url
    });

    eprintln!("querying ComfyUI at {base} …");
    let client = ComfyClient::new(&base, Duration::from_secs(30));
    let models = client
        .list_upscale_models()
        .with_context(|| format!("listing upscale models from {base}"))?;

    if models.is_empty() {
        println!("(no upscale models found — check the server's models/upscale_models dir)");
    } else {
        for m in &models {
            println!("{m}");
        }
        eprintln!("{} model(s)", models.len());
    }
    Ok(())
}

fn cmd_export(dir: PathBuf, profile_name: Option<String>, threshold: Option<f32>) -> Result<()> {
    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let common = cfg.resolve_common_tags();
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
        let out = export::export_image(&image, &sidecar, &profile, &common)?;
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
    let common = cfg.resolve_common_tags();
    let mut profile = cfg.resolve_profile(profile_name.as_deref());
    if let Some(t) = threshold {
        profile.threshold = t;
    }
    // The trainer shuffles tags at training time; metadata stays stable for diffability.
    profile.shuffle = false;

    match format {
        MetadataFormat::SdScripts => {
            cmd_metadata_sd_scripts(&dir, &profile, &cfg.tag_groups, &common, output)
        }
        MetadataFormat::Musubi => {
            cmd_metadata_musubi(&dir, &profile, &cfg.tag_groups, &common, output)
        }
    }
}

fn cmd_metadata_sd_scripts(
    dir: &std::path::Path,
    profile: &fwaun_tools_core::config::ExportProfile,
    tag_groups: &std::collections::BTreeMap<String, fwaun_tools_core::config::TagGroup>,
    common: &fwaun_tools_core::common_tags::CommonTags,
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
        let tags = export::build_tags(&sidecar, profile, common);
        let mut entry = serde_json::Map::new();
        if !tags.is_empty() {
            let joined = tags
                .iter()
                .map(|t| t.replace('_', " "))
                .collect::<Vec<_>>()
                .join(", ");
            entry.insert("tags".to_string(), serde_json::Value::String(joined));
        }
        if let Some(cap) = export::build_caption(&sidecar, profile, tag_groups, common) {
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
    profile: &fwaun_tools_core::config::ExportProfile,
    tag_groups: &std::collections::BTreeMap<String, fwaun_tools_core::config::TagGroup>,
    common: &fwaun_tools_core::common_tags::CommonTags,
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
        match export::build_caption(&sidecar, profile, tag_groups, common) {
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

fn cmd_mv(dir: PathBuf, dest: PathBuf, tags: Vec<String>, dry_run: bool) -> Result<()> {
    use fwaun_tools_core::sidecar::sidecar_path_for;
    use fwaun_tools_core::tag_group::effective_tag_set;

    // Matching honours the dataset-wide `common_tags` layer, so `mv --tags
    // <trigger>` selects everything when the trigger is declared there.
    let common = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?
        .resolve_common_tags();

    // Normalize query tags the same way the effective tag set is keyed:
    // trimmed + lowercased. Empty entries (e.g. from a trailing comma) are
    // dropped so they don't silently match everything.
    let wanted: Vec<String> = tags
        .iter()
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if wanted.is_empty() {
        anyhow::bail!("--tags must contain at least one non-empty tag");
    }

    // Collect all matches *before* moving anything. `dest` is allowed to
    // live inside `dir` (e.g. splitting `src/` into `src/<style>/`); moving
    // during the walk would let WalkDir descend into a freshly-created
    // destination subdir and re-encounter already-moved files.
    let mut matches: Vec<PathBuf> = Vec::new();
    for image in iter_images(&dir) {
        // No sidecar → no tags → can't match, so load_or_default is fine
        // (an empty effective set never contains a wanted tag).
        let sc = Sidecar::load_or_default(&image)?;
        let eff = effective_tag_set(&sc, &common);
        if wanted.iter().all(|t| eff.contains(t)) {
            matches.push(image);
        }
    }
    let matched = matches.len();

    let mut moved = 0usize;
    let mut skipped_exists = 0usize;
    for image in matches {
        // Preserve the sub-path relative to `dir` so recursive sources
        // don't collide when flattened into `dest`.
        let rel = image.strip_prefix(&dir).unwrap_or(&image);
        let target_image = dest.join(rel);
        let target_dir = target_image
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| dest.clone());

        let sidecar = sidecar_path_for(&image);
        let has_sidecar = sidecar.exists();
        let target_sidecar = sidecar_path_for(&target_image);

        // Never overwrite an existing file at the destination; report and
        // skip the whole pair so image and sidecar stay together.
        if target_image.exists() || (has_sidecar && target_sidecar.exists()) {
            eprintln!("skip (exists at dest): {}", rel.display());
            skipped_exists += 1;
            continue;
        }

        if dry_run {
            println!(
                "would move {}{}",
                rel.display(),
                if has_sidecar { " (+ sidecar)" } else { "" }
            );
            continue;
        }

        std::fs::create_dir_all(&target_dir)
            .with_context(|| format!("creating {}", target_dir.display()))?;
        move_file(&image, &target_image)?;
        if has_sidecar {
            move_file(&sidecar, &target_sidecar)?;
        }
        moved += 1;
        println!("moved {} → {}", rel.display(), target_image.display());
    }

    if dry_run {
        println!("dry run: {matched} would move (run without --dry-run to apply)");
    } else {
        println!("done: {moved} moved, {skipped_exists} skipped (already at dest)");
    }
    Ok(())
}

/// Move a file, falling back to copy+remove when `rename` fails because
/// source and destination are on different filesystems (`EXDEV`).
fn move_file(from: &std::path::Path, to: &std::path::Path) -> Result<()> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(e) if e.raw_os_error() == Some(libc_exdev()) => {
            std::fs::copy(from, to)
                .with_context(|| format!("copying {} → {}", from.display(), to.display()))?;
            std::fs::remove_file(from)
                .with_context(|| format!("removing {} after copy", from.display()))?;
            Ok(())
        }
        Err(e) => Err(e).with_context(|| format!("moving {} → {}", from.display(), to.display())),
    }
}

/// `EXDEV` errno ("cross-device link"). 18 on Linux and macOS/BSD alike.
fn libc_exdev() -> i32 {
    18
}

/// Normalize a `--tags` list: trim each entry and drop empties (e.g. from a
/// trailing comma) so they don't silently no-op or match nothing. Order and
/// case are preserved; a leading `-` (suppression marker) is kept intact.
fn normalize_tag_args(tags: &[String]) -> Vec<String> {
    tags.iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Route a whole-dataset bulk tag edit into `fwaun-tools.toml`'s
/// `common_tags` instead of into every sidecar.
///
/// `dir` being a dataset root means the edit covers the entire dataset, and
/// a dataset-wide tag belongs in the shared layer: one line in the config
/// instead of a copy in every `.ron`, and images added later are covered
/// without re-running the command. A subdirectory, or `--per-image`, falls
/// through to the per-sidecar path.
///
/// Returns `Ok(true)` when the config handled it.
fn try_common_tag_edit(
    dir: &std::path::Path,
    tags: &[String],
    dry_run: bool,
    per_image: bool,
    remove: bool,
) -> Result<bool> {
    if per_image {
        return Ok(false);
    }
    let Some(path) = ProjectConfig::dataset_root_config(dir) else {
        return Ok(false);
    };

    let edit = if remove {
        common_tags::remove_from_config_file(&path, tags, dry_run)
    } else {
        common_tags::add_to_config_file(&path, tags, dry_run)
    }
    .with_context(|| format!("editing common_tags in {}", path.display()))?;

    let verb = match (remove, dry_run) {
        (false, false) => "added to",
        (false, true) => "would add to",
        (true, false) => "removed from",
        (true, true) => "would remove from",
    };
    println!(
        "{} is the dataset root — editing the shared tag layer, not the sidecars.",
        dir.display()
    );
    if edit.applied.is_empty() {
        // Distinguish "the config already says this" from "these were never
        // in the config" — for `remove` the latter is the common case, and
        // the follow-up note below is what the user actually needs.
        let reason = if remove {
            "not in common_tags"
        } else {
            "already in common_tags"
        };
        println!("nothing to do: {} — {reason}", edit.unchanged.join(", "));
    } else {
        println!(
            "{verb} {} common_tags: {}",
            path.display(),
            edit.applied.join(", "),
        );
        if !edit.unchanged.is_empty() {
            println!("unchanged: {}", edit.unchanged.join(", "));
        }
    }
    println!("common_tags is now [{}]", edit.result.join(", "));

    // A dataset-wide remove leaves per-image copies untouched: they may be
    // deliberate per-image overrides. Say so rather than silently half-doing
    // the job.
    if remove {
        let stale = count_images_with_manual_tags(dir, tags)?;
        if stale > 0 {
            println!(
                "note: {stale} image(s) still carry one of these in their own manual_tags \
                 — re-run with --per-image to clear those too"
            );
        }
    }
    Ok(true)
}

/// How many images under `dir` have any of `tags` as their own manual entry
/// (case-insensitive, exact form).
fn count_images_with_manual_tags(dir: &std::path::Path, tags: &[String]) -> Result<usize> {
    let keys: Vec<String> = tags.iter().map(|t| t.trim().to_lowercase()).collect();
    let mut n = 0usize;
    for image in iter_images(dir) {
        let Some(sc) = Sidecar::load(&image)? else {
            continue;
        };
        if sc
            .manual_tags
            .iter()
            .any(|m| keys.contains(&m.trim().to_lowercase()))
        {
            n += 1;
        }
    }
    Ok(n)
}

fn cmd_add_tag(dir: PathBuf, tags: Vec<String>, dry_run: bool, per_image: bool) -> Result<()> {
    let wanted = normalize_tag_args(&tags);
    if wanted.is_empty() {
        anyhow::bail!("--tags must contain at least one non-empty tag");
    }
    if try_common_tag_edit(&dir, &wanted, dry_run, per_image, false)? {
        return Ok(());
    }

    let mut changed = 0usize;
    let mut unchanged = 0usize;
    for image in iter_images(&dir) {
        // load_or_default: a fresh sidecar is created for images that don't
        // have one so a directory-wide tag lands on every image.
        let mut sc = Sidecar::load_or_default(&image)?;
        let added: Vec<&str> = wanted
            .iter()
            .filter(|t| sc.add_manual_tag((*t).clone()))
            .map(|t| t.as_str())
            .collect();
        if added.is_empty() {
            unchanged += 1;
            continue;
        }
        if !dry_run {
            sc.save(&image)?;
        }
        changed += 1;
        println!(
            "{} {} (+{})",
            if dry_run { "would add" } else { "added" },
            image.display(),
            added.join(", "),
        );
    }

    if dry_run {
        println!("dry run: {changed} would change, {unchanged} already had the tag(s)");
    } else {
        println!("done: {changed} changed, {unchanged} unchanged");
    }
    Ok(())
}

fn cmd_remove_tag(dir: PathBuf, tags: Vec<String>, dry_run: bool, per_image: bool) -> Result<()> {
    let wanted = normalize_tag_args(&tags);
    if wanted.is_empty() {
        anyhow::bail!("--tags must contain at least one non-empty tag");
    }
    if try_common_tag_edit(&dir, &wanted, dry_run, per_image, true)? {
        return Ok(());
    }

    let mut changed = 0usize;
    let mut unchanged = 0usize;
    for image in iter_images(&dir) {
        // Only existing sidecars can carry a manual tag; skip the rest so
        // remove-tag never creates a sidecar just to change nothing.
        let Some(mut sc) = Sidecar::load(&image)? else {
            continue;
        };
        let removed: Vec<&str> = wanted
            .iter()
            .filter(|t| sc.remove_manual_tag_ci(t) > 0)
            .map(|t| t.as_str())
            .collect();
        if removed.is_empty() {
            unchanged += 1;
            continue;
        }
        if !dry_run {
            sc.save(&image)?;
        }
        changed += 1;
        println!(
            "{} {} (-{})",
            if dry_run { "would remove" } else { "removed" },
            image.display(),
            removed.join(", "),
        );
    }

    if dry_run {
        println!("dry run: {changed} would change, {unchanged} without the tag(s)");
    } else {
        println!("done: {changed} changed, {unchanged} unchanged");
    }
    Ok(())
}

fn cmd_replace_tag(
    dir: PathBuf,
    from: Vec<String>,
    to: Vec<String>,
    dry_run: bool,
    per_image: bool,
) -> Result<()> {
    let pairs = rename_pairs(&from, &to)?;

    // The config edit comes first so a `--dry-run` reads top-down: the shared
    // layer, then the sidecars. Unlike add/remove-tag it doesn't *replace* the
    // per-image pass — an entry can live in both places, and a rename that
    // half-applied would leave the two disagreeing.
    if !per_image && let Some(path) = ProjectConfig::dataset_root_config(&dir) {
        let edit = common_tags::replace_in_config_file(&path, &pairs, dry_run)
            .with_context(|| format!("editing common_tags in {}", path.display()))?;
        if edit.changed() {
            println!(
                "{} {} common_tags: {}",
                if dry_run {
                    "would rename in"
                } else {
                    "renamed in"
                },
                path.display(),
                edit.applied.join(", "),
            );
            println!("common_tags is now [{}]", edit.result.join(", "));
        }
    }

    let mut changed = 0usize;
    let mut unchanged = 0usize;
    for image in iter_images(&dir) {
        // Only an existing sidecar can carry the old tag, so a rename never
        // creates one just to change nothing.
        let Some(mut sc) = Sidecar::load(&image)? else {
            continue;
        };
        let renamed: Vec<String> = pairs
            .iter()
            .filter(|(f, t)| sc.replace_manual_tag_ci(f, t) > 0)
            .map(|(f, t)| format!("{f} -> {t}"))
            .collect();
        if renamed.is_empty() {
            unchanged += 1;
            continue;
        }
        if !dry_run {
            sc.save(&image)?;
        }
        changed += 1;
        println!(
            "{} {} ({})",
            if dry_run { "would rename" } else { "renamed" },
            image.display(),
            renamed.join(", "),
        );
    }

    if dry_run {
        println!("dry run: {changed} would change, {unchanged} without the tag(s)");
    } else {
        println!("done: {changed} changed, {unchanged} unchanged");
    }
    Ok(())
}

/// Zip `--from` / `--to` into rename pairs, rejecting the argument shapes
/// that would silently rename the wrong thing (a lopsided list, a blank
/// half). The pairing is positional, so `--from a,b --to x,y` is `a -> x`
/// and `b -> y`.
fn rename_pairs(from: &[String], to: &[String]) -> Result<Vec<(String, String)>> {
    let from = normalize_tag_args(from);
    let to = normalize_tag_args(to);
    if from.is_empty() || to.is_empty() {
        anyhow::bail!("--from and --to must each contain at least one non-empty tag");
    }
    if from.len() != to.len() {
        anyhow::bail!(
            "--from and --to must have the same number of tags \
             ({} vs {}) — they are paired in order",
            from.len(),
            to.len(),
        );
    }
    Ok(from.into_iter().zip(to).collect())
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
    use fwaun_tools_core::tag_group::{Classification, classify};

    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let common = cfg.resolve_common_tags();
    let group = cfg.tag_groups.get(&group_name).with_context(|| {
        format!(
            "tag_group `{group_name}` is not defined in any fwaun-tools.toml \
             (project or user). Add a [tag_group.{group_name}] section."
        )
    })?;

    let mut tagged = 0usize;
    let mut unset = 0usize;
    let mut violations = 0usize;

    for image in iter_images(&dir) {
        let sc = Sidecar::load_or_default(&image)?;
        let classification = classify(&sc, group, &common);
        let (state_text, state_json) = match &classification {
            Classification::Tag(t) => {
                tagged += 1;
                (
                    format!("tag={t}"),
                    serde_json::json!({"state": "tag", "tag": t}),
                )
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

        // Only exclusive groups have a notion of a "problem": Unset (image
        // not categorized) or Violation (two+ mutually-exclusive tags
        // coexisting). Non-exclusive groups exist for caption steering, where
        // co-occurrence is the whole point, so nothing is flagged.
        let is_problem = group.exclusive
            && matches!(
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
    use fwaun_tools_core::hub;
    use tokenizers::Tokenizer;

    let cfg = ProjectConfig::load_or_default(&dir)
        .with_context(|| format!("loading config in {}", dir.display()))?;
    let common = cfg.resolve_common_tags();
    let mut profile = cfg.resolve_profile(profile_name.as_deref());
    if let Some(t) = threshold {
        profile.threshold = t;
    }
    profile.shuffle = false;

    eprintln!("[tokens] fetching Qwen/Qwen3-0.6B tokenizer...");
    let paths = hub::fetch_files("Qwen/Qwen3-0.6B", None, &["tokenizer.json"])
        .context("download Qwen3-0.6B tokenizer.json")?;
    let tokenizer =
        Tokenizer::from_file(&paths[0]).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

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
        let tags = fwaun_tools_core::export::build_tags(&sidecar, &profile, &common);
        let tags_text = tags
            .iter()
            .map(|t| t.replace('_', " "))
            .collect::<Vec<_>>()
            .join(", ");
        let caption_text =
            export::build_caption(&sidecar, &profile, &cfg.tag_groups, &common).unwrap_or_default();

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

    println!("analyzed {analyzed} images (skipped {no_sidecar} without sidecar) | budget {limit}");
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
