//! "Model tools" tab — a GUI front-end for the `fwaun-tools model`
//! subcommands (`merge-diff`, `extract-lora`, `quant-int8`). This is a
//! plain batch-operation launcher: pick files, set a few knobs, hit Run.
//! It shares no state with the dataset editor.
//!
//! Each operation is one long CPU job. We run it on a background thread and
//! stream a tiny log back over an mpsc channel — the same pattern the
//! dataset side uses for the tagger/captioner — so the window keeps
//! repainting and the button re-enables when the job finishes.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::thread;

use eframe::egui;
use fwaun_tools_core::model::lora::{self, ExtractArgs};
use fwaun_tools_core::model::merge::{self, MergeArgs, ModelArch};
use fwaun_tools_core::model::quant::{self, QuantArgs};
use fwaun_tools_core::model::safetensors::Dtype;

use crate::i18n::{Lang, T};

/// Which subcommand the tab is currently showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelOp {
    Merge,
    Extract,
    Quant,
}

impl ModelOp {
    const ALL: [ModelOp; 3] = [ModelOp::Merge, ModelOp::Extract, ModelOp::Quant];

    fn label(self, t: T) -> &'static str {
        match self {
            ModelOp::Merge => t.model_op_merge(),
            ModelOp::Extract => t.model_op_extract(),
            ModelOp::Quant => t.model_op_quant(),
        }
    }

    fn desc(self, t: T) -> &'static str {
        match self {
            ModelOp::Merge => t.model_op_merge_desc(),
            ModelOp::Extract => t.model_op_extract_desc(),
            ModelOp::Quant => t.model_op_quant_desc(),
        }
    }
}

/// Output-dtype choice shared by the merge and extract forms. `Keep` maps to
/// "leave the target's dtype alone" (merge only); the extract form never
/// offers it since a fresh LoRA has no target dtype to keep.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DtypeChoice {
    Keep,
    Bf16,
    Fp16,
    Fp32,
}

impl DtypeChoice {
    /// The safetensors dtype string, or `None` for `Keep`.
    fn as_str(self) -> Option<&'static str> {
        match self {
            DtypeChoice::Keep => None,
            DtypeChoice::Bf16 => Some("bf16"),
            DtypeChoice::Fp16 => Some("fp16"),
            DtypeChoice::Fp32 => Some("fp32"),
        }
    }

    fn combo_label(self, t: T) -> &'static str {
        match self {
            DtypeChoice::Keep => t.model_dtype_keep(),
            DtypeChoice::Bf16 => "bf16",
            DtypeChoice::Fp16 => "fp16",
            DtypeChoice::Fp32 => "fp32",
        }
    }
}

fn arch_label(arch: ModelArch) -> &'static str {
    match arch {
        ModelArch::Auto => "auto",
        ModelArch::Krea2 => "krea2",
        ModelArch::Anima => "anima",
    }
}

const ARCHES: [ModelArch; 3] = [ModelArch::Auto, ModelArch::Krea2, ModelArch::Anima];

// ───────── per-operation form state ─────────
//
// Paths are kept as owned `String`s (not `PathBuf`) so the user can either
// type/paste a path or fill it via the Browse button. They're turned into
// `PathBuf`s at Run time.

struct MergeForm {
    base: String,
    tuned: String,
    target: String,
    output: String,
    multiplier: f32,
    save_dtype: DtypeChoice,
    arch: ModelArch,
}

impl Default for MergeForm {
    fn default() -> Self {
        Self {
            base: String::new(),
            tuned: String::new(),
            target: String::new(),
            output: String::new(),
            multiplier: 1.0,
            save_dtype: DtypeChoice::Keep,
            arch: ModelArch::Auto,
        }
    }
}

struct ExtractForm {
    base: String,
    tuned: String,
    output: String,
    rank: usize,
    alpha_enabled: bool,
    alpha: f32,
    save_dtype: DtypeChoice,
    arch: ModelArch,
    include: String,
    exclude: String,
    niter: usize,
    oversample: usize,
}

impl Default for ExtractForm {
    fn default() -> Self {
        Self {
            base: String::new(),
            tuned: String::new(),
            output: String::new(),
            rank: 32,
            alpha_enabled: false,
            alpha: 32.0,
            save_dtype: DtypeChoice::Fp16,
            arch: ModelArch::Auto,
            include: String::new(),
            exclude: String::new(),
            niter: 2,
            oversample: 8,
        }
    }
}

struct QuantForm {
    src: String,
    dst: String,
    dry_run: bool,
    include: String,
    exclude: String,
    min_gemm: usize,
    downcast_fp32: bool,
    warn_thresh: f32,
}

impl Default for QuantForm {
    fn default() -> Self {
        Self {
            src: String::new(),
            dst: String::new(),
            dry_run: false,
            include: String::new(),
            exclude: String::new(),
            min_gemm: 256,
            downcast_fp32: false,
            warn_thresh: 2.0,
        }
    }
}

/// Message from the worker thread back to the UI.
enum ModelMsg {
    /// Job finished. `Ok` on success, `Err(message)` on failure.
    Done(Result<(), String>),
}

pub struct ModelApp {
    op: ModelOp,
    merge: MergeForm,
    extract: ExtractForm,
    quant: QuantForm,
    /// `Some` while a job is in flight — the single source of truth for
    /// "running", used to disable the Run button.
    worker_rx: Option<Receiver<ModelMsg>>,
    /// Scrolling log of what has been run this session.
    log: Vec<String>,
}

impl ModelApp {
    pub fn new() -> Self {
        Self {
            op: ModelOp::Merge,
            merge: MergeForm::default(),
            extract: ExtractForm::default(),
            quant: QuantForm::default(),
            worker_rx: None,
            log: Vec::new(),
        }
    }

    fn running(&self) -> bool {
        self.worker_rx.is_some()
    }

    /// Drain the worker channel; when `Done` lands, clear the running flag and
    /// append the outcome to the log.
    fn poll_worker(&mut self, lang: Lang) {
        let t = T::new(lang);
        let op_name = self.op.label(t).to_string();
        if let Some(rx) = &self.worker_rx {
            if let Ok(msg) = rx.try_recv() {
                match msg {
                    ModelMsg::Done(Ok(())) => self.log.push(t.model_log_ok(&op_name)),
                    ModelMsg::Done(Err(e)) => self.log.push(t.model_log_err(&op_name, &e)),
                }
                self.worker_rx = None;
            }
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context, lang: Lang) {
        self.poll_worker(lang);
        let t = T::new(lang);

        // Left: operation picker. Right: the selected op's form + log.
        egui::SidePanel::left("model_op_list")
            .resizable(false)
            .default_width(160.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                for op in ModelOp::ALL {
                    ui.selectable_value(&mut self.op, op, op.label(t));
                }
            });

        egui::TopBottomPanel::bottom("model_log")
            .resizable(true)
            .default_height(140.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.log {
                            ui.label(line);
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading(self.op.label(t));
            ui.label(self.op.desc(t));
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| match self.op {
                ModelOp::Merge => self.ui_merge(ui, t),
                ModelOp::Extract => self.ui_extract(ui, t),
                ModelOp::Quant => self.ui_quant(ui, t),
            });
        });

        // Keep repainting while a job runs so the log updates promptly the
        // moment the worker sends `Done`.
        if self.running() {
            ctx.request_repaint();
        }
    }

    // ───────── per-op forms ─────────

    fn ui_merge(&mut self, ui: &mut egui::Ui, t: T) {
        let running = self.running();
        file_row(ui, t, t.model_field_base(), &mut self.merge.base, false);
        file_row(ui, t, t.model_field_tuned(), &mut self.merge.tuned, false);
        file_row(ui, t, t.model_field_target(), &mut self.merge.target, false);
        file_row(ui, t, t.model_field_output(), &mut self.merge.output, true);

        ui.horizontal(|ui| {
            ui.label(t.model_field_multiplier());
            ui.add(egui::DragValue::new(&mut self.merge.multiplier).speed(0.05));
        });
        arch_combo(ui, t, "merge_arch", &mut self.merge.arch);
        dtype_combo(ui, t, "merge_dtype", &mut self.merge.save_dtype, true);

        ui.separator();
        let ready = !self.merge.base.trim().is_empty()
            && !self.merge.tuned.trim().is_empty()
            && !self.merge.target.trim().is_empty()
            && !self.merge.output.trim().is_empty();
        self.run_row(ui, t, running, ready, |form| {
            let m = &form.merge;
            let save_dtype = parse_dtype(m.save_dtype)?;
            let args = MergeArgs {
                base: PathBuf::from(m.base.trim()),
                tuned: PathBuf::from(m.tuned.trim()),
                target: PathBuf::from(m.target.trim()),
                output: PathBuf::from(m.output.trim()),
                multiplier: m.multiplier,
                save_dtype,
                arch: m.arch,
            };
            Ok(Box::new(move || merge::run(args)))
        });
    }

    fn ui_extract(&mut self, ui: &mut egui::Ui, t: T) {
        let running = self.running();
        file_row(ui, t, t.model_field_base(), &mut self.extract.base, false);
        file_row(ui, t, t.model_field_tuned(), &mut self.extract.tuned, false);
        file_row(ui, t, t.model_field_output(), &mut self.extract.output, true);

        ui.horizontal(|ui| {
            ui.label(t.model_field_rank());
            ui.add(egui::DragValue::new(&mut self.extract.rank).range(1..=1024));
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.extract.alpha_enabled, t.model_field_alpha());
            ui.add_enabled(
                self.extract.alpha_enabled,
                egui::DragValue::new(&mut self.extract.alpha).speed(0.5),
            );
        });
        arch_combo(ui, t, "extract_arch", &mut self.extract.arch);
        dtype_combo(ui, t, "extract_dtype", &mut self.extract.save_dtype, false);

        ui.collapsing(t.model_field_include(), |ui| {
            labeled_text(ui, t.model_field_include(), &mut self.extract.include);
            labeled_text(ui, t.model_field_exclude(), &mut self.extract.exclude);
            ui.horizontal(|ui| {
                ui.label("niter");
                ui.add(egui::DragValue::new(&mut self.extract.niter).range(0..=16));
                ui.label("oversample");
                ui.add(egui::DragValue::new(&mut self.extract.oversample).range(0..=64));
            });
        });

        ui.separator();
        let ready = !self.extract.base.trim().is_empty()
            && !self.extract.tuned.trim().is_empty()
            && !self.extract.output.trim().is_empty();
        self.run_row(ui, t, running, ready, |form| {
            let e = &form.extract;
            let save_dtype = parse_dtype(e.save_dtype)?
                .ok_or_else(|| "save dtype required".to_string())?;
            let args = ExtractArgs {
                base: PathBuf::from(e.base.trim()),
                tuned: PathBuf::from(e.tuned.trim()),
                output: PathBuf::from(e.output.trim()),
                rank: e.rank,
                alpha: e.alpha_enabled.then_some(e.alpha),
                save_dtype,
                arch: e.arch,
                include: non_empty(&e.include),
                exclude: non_empty(&e.exclude),
                niter: e.niter,
                oversample: e.oversample,
            };
            Ok(Box::new(move || lora::run(args)))
        });
    }

    fn ui_quant(&mut self, ui: &mut egui::Ui, t: T) {
        let running = self.running();
        file_row(ui, t, t.model_field_src(), &mut self.quant.src, false);
        file_row(ui, t, t.model_field_dst(), &mut self.quant.dst, true);

        ui.checkbox(&mut self.quant.dry_run, t.model_field_dry_run());
        ui.checkbox(&mut self.quant.downcast_fp32, "downcast fp32 passthrough");
        ui.horizontal(|ui| {
            ui.label(t.model_field_min_gemm());
            ui.add(egui::DragValue::new(&mut self.quant.min_gemm).range(0..=8192));
        });
        ui.horizontal(|ui| {
            ui.label(t.model_field_warn_thresh());
            ui.add(egui::DragValue::new(&mut self.quant.warn_thresh).speed(0.1));
        });
        ui.collapsing(t.model_field_include(), |ui| {
            labeled_text(ui, t.model_field_include(), &mut self.quant.include);
            labeled_text(ui, t.model_field_exclude(), &mut self.quant.exclude);
        });

        ui.separator();
        let ready = !self.quant.src.trim().is_empty();
        self.run_row(ui, t, running, ready, |form| {
            let q = &form.quant;
            let dst = non_empty(&q.dst).map(PathBuf::from);
            let args = QuantArgs {
                src: PathBuf::from(q.src.trim()),
                dst,
                dry_run: q.dry_run,
                exclude: non_empty(&q.exclude),
                include: non_empty(&q.include),
                min_gemm: q.min_gemm,
                downcast_fp32: q.downcast_fp32,
                warn_thresh: q.warn_thresh,
                verify_report: None,
            };
            Ok(Box::new(move || quant::run(args)))
        });
    }

    /// Render the Run button + status text, and on click build the job (via
    /// `build`, which validates the form and returns a boxed closure) and
    /// spawn it on a worker thread. `build` runs on the UI thread so any
    /// validation error is reported immediately without touching the worker.
    fn run_row<F>(&mut self, ui: &mut egui::Ui, t: T, running: bool, ready: bool, build: F)
    where
        F: FnOnce(&Self) -> Result<Box<dyn FnOnce() -> anyhow::Result<()> + Send>, String>,
    {
        ui.horizontal(|ui| {
            let clicked = ui
                .add_enabled(!running && ready, egui::Button::new(t.model_run()))
                .clicked();
            if running {
                ui.spinner();
                ui.label(t.model_running());
            } else if !ready {
                ui.label(t.model_err_need_paths());
            }
            if clicked {
                match build(self) {
                    Ok(job) => {
                        let op_name = self.op.label(t).to_string();
                        self.log.push(t.model_log_start(&op_name));
                        let (tx, rx) = channel();
                        self.worker_rx = Some(rx);
                        thread::spawn(move || {
                            let res = job().map_err(|e| format!("{e:#}"));
                            let _ = tx.send(ModelMsg::Done(res));
                        });
                    }
                    Err(e) => self.log.push(e),
                }
            }
        });
    }
}

// ───────── small shared widgets ─────────

/// A label + path text box + Browse button on one row. `save = true` opens a
/// save dialog (for output paths); otherwise an open-file dialog.
fn file_row(ui: &mut egui::Ui, t: T, label: &str, buf: &mut String, save: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::TextEdit::singleline(buf)
                .desired_width(360.0)
                .hint_text(".safetensors"),
        );
        if ui.button(t.model_browse()).clicked() {
            let dialog = rfd::FileDialog::new().add_filter("safetensors", &["safetensors"]);
            let picked = if save {
                dialog.save_file()
            } else {
                dialog.pick_file()
            };
            if let Some(path) = picked {
                *buf = path.display().to_string();
            }
        }
    });
}

fn labeled_text(ui: &mut egui::Ui, label: &str, buf: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(buf).desired_width(280.0));
    });
}

fn arch_combo(ui: &mut egui::Ui, t: T, id: &str, arch: &mut ModelArch) {
    ui.horizontal(|ui| {
        ui.label(t.model_field_arch());
        egui::ComboBox::from_id_salt(id)
            .selected_text(arch_label(*arch))
            .show_ui(ui, |ui| {
                for a in ARCHES {
                    ui.selectable_value(arch, a, arch_label(a));
                }
            });
    });
}

fn dtype_combo(ui: &mut egui::Ui, t: T, id: &str, dtype: &mut DtypeChoice, allow_keep: bool) {
    let choices: &[DtypeChoice] = if allow_keep {
        &[
            DtypeChoice::Keep,
            DtypeChoice::Bf16,
            DtypeChoice::Fp16,
            DtypeChoice::Fp32,
        ]
    } else {
        &[DtypeChoice::Bf16, DtypeChoice::Fp16, DtypeChoice::Fp32]
    };
    ui.horizontal(|ui| {
        ui.label(t.model_field_save_dtype());
        egui::ComboBox::from_id_salt(id)
            .selected_text(dtype.combo_label(t))
            .show_ui(ui, |ui| {
                for &c in choices {
                    ui.selectable_value(dtype, c, c.combo_label(t));
                }
            });
    });
}

/// Parse a `DtypeChoice` into the core `Dtype`, or `None` for `Keep`.
fn parse_dtype(choice: DtypeChoice) -> Result<Option<Dtype>, String> {
    match choice.as_str() {
        None => Ok(None),
        Some(s) => Dtype::parse_save_dtype(s)
            .map(Some)
            .map_err(|e| format!("{e:#}")),
    }
}

/// Trim a text-field value to `None` if it's blank, else `Some(owned)`.
fn non_empty(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}
