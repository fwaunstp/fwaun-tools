//! Progress counter for the per-image dataset commands.
//!
//! `tag` / `caption` / `booru` / `upscale` run model inference or network
//! I/O once per image — seconds to minutes each — and used to print only a
//! per-image result line, which says nothing about how much is left. A
//! captioning run over a few hundred images gave no way to tell "slow" from
//! "stuck". This adds a `[ 12/340 ] 3.5%` counter with elapsed time and an
//! ETA extrapolated from the images done so far.
//!
//! Two rules keep it out of the way of existing usage:
//!
//! - It writes to **stderr**, and the per-image result lines stay on stdout,
//!   so `fwaun-tools dataset tag . > tagged.log` still gets exactly the
//!   lines it used to. Result lines go through [`Progress::println`] so the
//!   transient counter is wiped before they are written and redrawn after.
//! - When stderr is not a terminal (redirected to a file, CI) the `\r`
//!   overwrite is replaced by one plain line per ~5% of the run, so a log
//!   doesn't collect thousands of counter updates.
//!
//! No progress-bar dependency: the whole thing is a `\r`-rewritten line, and
//! the CLI's "light" build is deliberately thin on dependencies.

use std::cell::RefCell;
use std::fmt::Display;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::{Duration, Instant};

/// Minimum gap between terminal redraws. Items that finish in microseconds
/// (an already-tagged image skipped without `--force`) would otherwise spend
/// more time painting the counter than doing the work.
const REDRAW_INTERVAL: Duration = Duration::from_millis(80);

/// Roughly how many counter lines a non-TTY run emits, i.e. one per 5%.
const LOG_LINES: usize = 20;

/// Display columns reserved for the name of the image being worked on. The
/// whole line has to fit in 80 columns, or `\r` returns to the start of the
/// wrapped row instead of the start of the line and leaves debris behind.
const NAME_WIDTH: usize = 20;

pub struct Progress {
    /// Command name, shown as the line's prefix (`tag`, `caption`, …).
    label: &'static str,
    total: usize,
    started: Instant,
    /// Whether stderr is a terminal — picks `\r` overwrite vs. plain lines.
    tty: bool,
    /// Non-TTY: emit a line every `step` completed items.
    step: usize,
    state: RefCell<State>,
}

#[derive(Default)]
struct State {
    done: usize,
    /// Name of the item currently being worked on, already truncated.
    current: Option<String>,
    /// Display width of the line on screen, so the next one can blank it.
    drawn: usize,
    last_draw: Option<Instant>,
    /// Non-TTY: `done` as of the last emitted line, so a boundary hit twice
    /// (advance + finish) doesn't log twice.
    last_logged: Option<usize>,
    /// Whether any item has been handed out yet. `advance` counts the
    /// *previous* item as done, so the first call must not increment.
    started_any: bool,
}

impl Progress {
    pub fn new(label: &'static str, total: usize) -> Self {
        Self {
            label,
            total,
            started: Instant::now(),
            tty: std::io::stderr().is_terminal(),
            step: total.div_ceil(LOG_LINES).max(1),
            state: RefCell::new(State::default()),
        }
    }

    /// Wrap an iterator so each item advances the counter. Doing it here
    /// rather than with an explicit `tick()` in the loop body means a `skip`
    /// path that `continue`s can't forget to count itself.
    pub fn wrap<I>(&self, items: I) -> ProgressIter<'_, I::IntoIter>
    where
        I: IntoIterator,
        I::Item: AsRef<Path>,
    {
        ProgressIter {
            progress: self,
            inner: items.into_iter(),
        }
    }

    /// Print a result line to stdout without the counter garbling it.
    pub fn println(&self, line: impl Display) {
        self.clear();
        println!("{line}");
        // Only redraw on a TTY: in a log, a counter line after every result
        // line is exactly the noise the `step` gate exists to avoid.
        self.render(self.tty);
    }

    /// Same as [`Progress::println`], for the diagnostic lines that belong
    /// on stderr (per-image errors, skipped images).
    pub fn eprintln(&self, line: impl Display) {
        self.clear();
        eprintln!("{line}");
        self.render(self.tty);
    }

    /// Wipe the counter and report the wall-clock total. Call before the
    /// command's own `done: …` summary.
    pub fn finish(&self) {
        if self.total == 0 {
            return;
        }
        {
            let mut st = self.state.borrow_mut();
            st.done = self.total;
            st.current = None;
        }
        self.clear();
        eprintln!(
            "{}: {} image(s) in {}",
            self.label,
            self.total,
            fmt_hms(self.started.elapsed()),
        );
    }

    /// Count the previous item as done and take `path` as the current one.
    fn advance(&self, path: &Path) {
        if self.total == 0 {
            return;
        }
        {
            let mut st = self.state.borrow_mut();
            if st.started_any {
                st.done += 1;
            } else {
                st.started_any = true;
            }
            st.current = Some(file_label(path));
        }
        self.render(false);
    }

    /// The wrapped iterator ran dry: everything handed out is done.
    fn all_done(&self) {
        let mut st = self.state.borrow_mut();
        if st.started_any {
            st.done = self.total;
            st.current = None;
        }
    }

    fn render(&self, force: bool) {
        if self.total == 0 {
            return;
        }
        let line = {
            let st = self.state.borrow();
            self.line(&st)
        };
        let mut st = self.state.borrow_mut();
        if self.tty {
            if !force && st.last_draw.is_some_and(|t| t.elapsed() < REDRAW_INTERVAL) {
                return;
            }
            let mut err = std::io::stderr();
            let _ = err.write_all(repaint(st.drawn, &line).as_bytes());
            let _ = err.flush();
            st.drawn = display_width(&line);
            st.last_draw = Some(Instant::now());
        } else {
            let due = st.done.is_multiple_of(self.step) && st.last_logged != Some(st.done);
            if !force && !due {
                return;
            }
            st.last_logged = Some(st.done);
            drop(st);
            eprintln!("{line}");
        }
    }

    /// Blank the counter line so something else can write to the terminal.
    fn clear(&self) {
        let mut st = self.state.borrow_mut();
        if !self.tty || st.drawn == 0 {
            return;
        }
        let mut err = std::io::stderr();
        let _ = err.write_all(blank(st.drawn).as_bytes());
        let _ = err.flush();
        st.drawn = 0;
    }

    fn line(&self, st: &State) -> String {
        let pct = st.done as f64 * 100.0 / self.total as f64;
        let w = self.total.to_string().len();
        let mut s = format!(
            "{} [ {:>w$}/{} ] {pct:>5.1}%  {} elapsed",
            self.label,
            st.done,
            self.total,
            fmt_hms(self.started.elapsed()),
        );
        if let Some(eta) = self.eta(st) {
            s.push_str(&format!("  ETA {}", fmt_hms(eta)));
        }
        if let Some(name) = &st.current {
            s.push_str("  ");
            s.push_str(name);
        }
        s
    }

    /// Remaining time from the mean per-image cost so far. `None` before the
    /// first item finishes (nothing to extrapolate from) and once every item
    /// has been handed out.
    fn eta(&self, st: &State) -> Option<Duration> {
        if st.done == 0 || st.done >= self.total {
            return None;
        }
        let per_item = self.started.elapsed().as_secs_f64() / st.done as f64;
        Duration::try_from_secs_f64(per_item * (self.total - st.done) as f64).ok()
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        // An error bubbling out of the loop skips `finish()`; don't leave a
        // half-written counter line under the error message.
        self.clear();
    }
}

pub struct ProgressIter<'a, I> {
    progress: &'a Progress,
    inner: I,
}

impl<I> Iterator for ProgressIter<'_, I>
where
    I: Iterator,
    I::Item: AsRef<Path>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            Some(item) => {
                self.progress.advance(item.as_ref());
                Some(item)
            }
            None => {
                self.progress.all_done();
                None
            }
        }
    }
}

/// Bytes that repaint the counter, given the width of what is on screen.
/// The trailing blanks cover the tail of a longer previous line: `\r` moves
/// the cursor but erases nothing, so a line that shrinks (a shorter file
/// name, an ETA that disappears) would otherwise leave its own debris.
fn repaint(drawn: usize, line: &str) -> String {
    let pad = drawn.saturating_sub(display_width(line));
    format!("\r{line}{:pad$}", "")
}

/// Bytes that blank the counter line and park the cursor at its start, so
/// whatever writes next starts from a clean line.
fn blank(drawn: usize) -> String {
    format!("\r{:drawn$}\r", "")
}

/// `m:ss`, or `h:mm:ss` past an hour.
fn fmt_hms(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// The file name, shortened to [`NAME_WIDTH`] columns. The tail is kept
/// (`…_0042.png`): dataset file names usually share a prefix and differ in
/// the index at the end.
fn file_label(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    if display_width(&name) <= NAME_WIDTH {
        return name;
    }
    let mut out = String::new();
    let mut width = 0;
    // Walk from the end, taking as much of the tail as fits beside the "…".
    for c in name.chars().rev() {
        let w = char_width(c);
        if width + w > NAME_WIDTH - 1 {
            break;
        }
        width += w;
        out.push(c);
    }
    let mut label = String::from("…");
    label.extend(out.chars().rev());
    label
}

/// Terminal columns a string occupies. Only the wide/narrow split matters
/// here — the counter has to blank exactly what it drew, and a dataset with
/// CJK file names would otherwise leave debris on screen after a redraw.
fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn char_width(c: char) -> usize {
    match c as u32 {
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE6F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1F300..=0x1F9FF
        | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hms_switches_to_hours() {
        assert_eq!(fmt_hms(Duration::from_secs(0)), "0:00");
        assert_eq!(fmt_hms(Duration::from_secs(72)), "1:12");
        assert_eq!(fmt_hms(Duration::from_secs(3600)), "1:00:00");
        assert_eq!(fmt_hms(Duration::from_secs(3671)), "1:01:11");
    }

    #[test]
    fn short_names_are_untouched() {
        assert_eq!(file_label(Path::new("/data/img_0042.png")), "img_0042.png");
    }

    #[test]
    fn long_names_keep_the_tail() {
        let label = file_label(Path::new("a_very_long_dataset_file_name_0042.png"));
        assert!(label.starts_with('…'), "{label}");
        assert!(label.ends_with("_0042.png"), "{label}");
        assert_eq!(display_width(&label), NAME_WIDTH);
    }

    #[test]
    fn wide_names_are_measured_in_columns() {
        // 12 CJK chars = 24 columns, over the budget on width but not on
        // char count — truncating by chars would overflow the line.
        let label = file_label(Path::new("あいうえおかきくけこさし.png"));
        assert!(display_width(&label) <= NAME_WIDTH, "{label}");
    }

    #[test]
    fn counter_line_reports_position_and_eta() {
        let p = Progress::new("tag", 340);
        let st = State {
            done: 12,
            ..State::default()
        };
        let line = p.line(&st);
        assert!(line.starts_with("tag [  12/340 ]   3.5%"), "{line}");
        assert!(line.contains("ETA"), "{line}");
    }

    #[test]
    fn repaint_covers_the_tail_of_a_longer_line() {
        assert_eq!(repaint(0, "ab"), "\rab");
        assert_eq!(repaint(2, "abcd"), "\rabcd");
        assert_eq!(repaint(5, "ab"), "\rab   ");
        // Widths are columns, not chars: two CJK chars cover four columns,
        // so nothing is left to erase from a 4-column predecessor.
        assert_eq!(repaint(4, "あい"), "\rあい");
    }

    #[test]
    fn blank_erases_and_rewinds() {
        assert_eq!(blank(3), "\r   \r");
        assert_eq!(blank(0), "\r\r");
    }

    #[test]
    fn no_eta_before_the_first_item_or_after_the_last() {
        let p = Progress::new("tag", 10);
        assert!(p.eta(&State::default()).is_none());
        let done = State {
            done: 10,
            ..State::default()
        };
        assert!(p.eta(&done).is_none());
    }
}
