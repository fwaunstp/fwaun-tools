//! Progress reporting for the long-running `model` operations.
//!
//! Each `run` takes a `&mut dyn ProgressSink` instead of hard-coding
//! `eprintln!`/`println!`, so the same operation can drive a CLI (print the
//! lines it always printed) or a GUI (forward structured updates to a worker
//! channel and drive a progress bar). See issue #26.

/// Sink for a model operation's human-readable log lines and coarse progress
/// ticks.
///
/// `log` carries the fully-formatted lines the operation used to print;
/// `tick` is an additional structured signal for a progress bar, with a
/// no-op default so a sink that only wants the log can ignore it.
pub trait ProgressSink {
    /// One fully-formatted log line (no trailing newline — the sink adds one).
    fn log(&mut self, line: &str);

    /// Coarse progress within the heavy loop: `done` of `total` units. Called
    /// frequently; `total` is the final count (non-zero). The default ignores
    /// it — the CLI's progress already arrives through [`log`](Self::log).
    fn tick(&mut self, done: usize, total: usize) {
        let _ = (done, total);
    }
}

/// A [`ProgressSink`] that prints each log line to stderr or stdout, matching
/// the stream each `model` subcommand historically used. `tick` stays the
/// no-op default: the CLI's progress already arrives through `log` as the
/// same lines it always printed.
pub struct StreamProgress {
    to_stdout: bool,
}

impl StreamProgress {
    /// Lines go to stderr (used by `merge-diff` and `extract-lora`).
    pub fn stderr() -> Self {
        Self { to_stdout: false }
    }

    /// Lines go to stdout (used by `quant-int8`).
    pub fn stdout() -> Self {
        Self { to_stdout: true }
    }
}

impl ProgressSink for StreamProgress {
    fn log(&mut self, line: &str) {
        if self.to_stdout {
            println!("{line}");
        } else {
            eprintln!("{line}");
        }
    }
}
