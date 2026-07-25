//! CSI 2026 synchronized differential renderer.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::Result;
use crate::ansi::{OSC8_RESET, SGR_RESET, normalize_terminal_output, visible_width};
use crate::image::{delete_kitty_image, is_image_line};
use crate::line::ConstrainedLine;
use crate::terminal::Terminal;

const SYNC_START: &str = "\x1b[?2026h";
const SYNC_END: &str = "\x1b[?2026l";

/// Zero-width APC marker emitted by focused text controls for IME placement.
pub const CURSOR_MARKER: &str = "\x1b_pi:c\x07";

/// Hardware cursor destination in logical frame coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameCursor {
    /// Zero-based logical row.
    pub row: usize,
    /// Zero-based terminal column.
    pub column: usize,
    /// Whether to show the hardware cursor. Hidden positioning still helps
    /// terminals place IME candidate windows.
    pub visible: bool,
}

/// Complete render frame.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Frame {
    /// Vertically ordered lines.
    pub lines: Vec<ConstrainedLine>,
    /// Optional explicit hardware cursor.
    pub cursor: Option<FrameCursor>,
}

impl Frame {
    /// Creates a frame without an explicit cursor.
    pub fn new(lines: Vec<ConstrainedLine>) -> Self {
        Self {
            lines,
            cursor: None,
        }
    }

    /// Adds an explicit cursor.
    #[must_use]
    pub const fn with_cursor(mut self, cursor: FrameCursor) -> Self {
        self.cursor = Some(cursor);
        self
    }
}

/// Cumulative renderer counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderStats {
    /// Initial/forced/resize redraws.
    pub full_redraws: u64,
    /// In-place changed-range redraws.
    pub differential_redraws: u64,
    /// Frames whose text did not change.
    pub unchanged_frames: u64,
    /// Bytes written by frame updates (cursor visibility commands included).
    pub bytes_written: u64,
}

/// Stateful synchronized differential renderer.
#[derive(Debug)]
pub struct DifferentialRenderer<T> {
    terminal: T,
    previous_lines: Vec<String>,
    previous_width: u16,
    previous_height: u16,
    previous_viewport_top: usize,
    hardware_cursor_row: usize,
    max_lines_rendered: usize,
    previous_image_ids: BTreeSet<u32>,
    clear_on_shrink: bool,
    force_full: bool,
    started: bool,
    stats: RenderStats,
}

impl<T: Terminal> DifferentialRenderer<T> {
    /// Creates a renderer.
    pub fn new(terminal: T) -> Self {
        Self {
            terminal,
            previous_lines: Vec::new(),
            previous_width: 0,
            previous_height: 0,
            previous_viewport_top: 0,
            hardware_cursor_row: 0,
            max_lines_rendered: 0,
            previous_image_ids: BTreeSet::new(),
            clear_on_shrink: true,
            force_full: false,
            started: false,
            stats: RenderStats::default(),
        }
    }

    /// Starts the terminal and hides its cursor.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal mode setup, cursor hiding, or
    /// flushing fails.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        self.terminal.start()?;
        if let Err(error) = self
            .terminal
            .hide_cursor()
            .and_then(|()| self.terminal.flush())
        {
            let _ = self.terminal.stop();
            return Err(error.into());
        }
        self.started = true;
        Ok(())
    }

    /// Restores terminal modes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when restoring the cursor or terminal modes fails.
    pub fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        let cursor_result = self
            .terminal
            .show_cursor()
            .and_then(|()| self.terminal.flush());
        let stop_result = self.terminal.stop();
        self.started = false;
        cursor_result.and(stop_result)?;
        Ok(())
    }

    /// Chooses whether a shorter frame clears the historical work area.
    pub fn set_clear_on_shrink(&mut self, enabled: bool) {
        self.clear_on_shrink = enabled;
    }

    /// Forces the next frame through the safe full-redraw path.
    pub fn force_redraw(&mut self) {
        self.force_full = true;
    }

    /// Renders one frame atomically.
    ///
    /// # Errors
    ///
    /// Returns an error when terminal sizing or output fails, or when a line
    /// exceeds the current terminal width.
    pub fn render(&mut self, mut frame: Frame) -> Result<()> {
        let (width, height) = self.terminal.size()?;
        let width_usize = usize::from(width);
        let height_usize = usize::from(height.max(1));
        for line in &frame.lines {
            if line.width() > width_usize {
                return Err(crate::line::LineError::TooWide {
                    actual: line.width(),
                    maximum: width_usize,
                }
                .into());
            }
        }

        let marker_cursor = extract_cursor_marker(&mut frame.lines, height_usize);
        let cursor = frame.cursor.or(marker_cursor);
        let new_lines = frame
            .lines
            .into_iter()
            .map(ConstrainedLine::into_string)
            .map(|line| {
                if is_image_line(&line) {
                    line
                } else {
                    format!(
                        "{}{SGR_RESET}{OSC8_RESET}",
                        normalize_terminal_output(&line)
                    )
                }
            })
            .collect::<Vec<_>>();

        let resized = self.previous_width != 0
            && (self.previous_width != width || self.previous_height != height);
        let first = self.previous_width == 0;
        let shrunk = self.clear_on_shrink
            && new_lines.len() < self.max_lines_rendered
            && new_lines.len() != self.previous_lines.len();
        let image_changed = changed_image_region(&self.previous_lines, &new_lines);

        if first || resized || self.force_full || shrunk || image_changed {
            let clear = !first || self.force_full;
            self.full_render(&new_lines, cursor, width, height, clear)?;
            self.force_full = false;
            return Ok(());
        }

        let (first_changed, last_changed) = changed_range(&self.previous_lines, &new_lines);
        let Some(first_changed) = first_changed else {
            self.stats.unchanged_frames += 1;
            self.position_cursor(cursor, new_lines.len(), height_usize)?;
            self.previous_height = height;
            return Ok(());
        };
        let last_changed = last_changed.unwrap_or(first_changed);
        if first_changed < self.previous_viewport_top {
            self.full_render(&new_lines, cursor, width, height, true)?;
            return Ok(());
        }

        let appended = new_lines.len() > self.previous_lines.len()
            && first_changed == self.previous_lines.len()
            && first_changed > 0;
        let previous_viewport_top = self.previous_viewport_top;
        let current_screen_row = self
            .hardware_cursor_row
            .saturating_sub(previous_viewport_top)
            .min(height_usize - 1);
        let target_logical_row = if appended {
            first_changed - 1
        } else {
            first_changed.min(new_lines.len().saturating_sub(1))
        };
        let target_screen_row = target_logical_row
            .saturating_sub(previous_viewport_top)
            .min(height_usize - 1);

        let mut output = String::from(SYNC_START);
        append_vertical_move(&mut output, current_screen_row, target_screen_row);
        output.push_str(if appended { "\r\n" } else { "\r" });

        if new_lines.is_empty() {
            output.push_str("\x1b[2K");
        } else {
            let render_end = last_changed.min(new_lines.len() - 1);
            for (offset, line) in new_lines[first_changed..=render_end].iter().enumerate() {
                if offset > 0 {
                    output.push_str("\r\n");
                }
                output.push_str("\x1b[2K");
                output.push_str(line);
            }
        }

        if self.previous_lines.len() > new_lines.len() {
            let extra = self.previous_lines.len() - new_lines.len();
            for _ in 0..extra {
                output.push_str("\r\n\x1b[2K");
            }
            if extra > 0 {
                write!(output, "\x1b[{extra}A").expect("writing to a String cannot fail");
            }
        }
        output.push_str(SYNC_END);
        self.write_batch(&output)?;
        self.stats.differential_redraws += 1;

        let render_end = last_changed.min(new_lines.len().saturating_sub(1));
        self.hardware_cursor_row = render_end;
        self.previous_lines = new_lines;
        self.previous_width = width;
        self.previous_height = height;
        self.max_lines_rendered = self.max_lines_rendered.max(self.previous_lines.len());
        self.previous_viewport_top = self.previous_lines.len().saturating_sub(height_usize);
        self.previous_image_ids = collect_image_ids(&self.previous_lines);
        self.position_cursor(cursor, self.previous_lines.len(), height_usize)?;
        Ok(())
    }

    /// Borrows the backend.
    pub const fn terminal(&self) -> &T {
        &self.terminal
    }

    /// Mutably borrows the backend.
    pub fn terminal_mut(&mut self) -> &mut T {
        &mut self.terminal
    }

    /// Consumes the renderer.
    pub fn into_terminal(self) -> T {
        self.terminal
    }

    /// Cumulative statistics.
    pub const fn stats(&self) -> RenderStats {
        self.stats
    }

    fn full_render(
        &mut self,
        lines: &[String],
        cursor: Option<FrameCursor>,
        width: u16,
        height: u16,
        clear: bool,
    ) -> Result<()> {
        let mut output = String::from(SYNC_START);
        if clear {
            for id in &self.previous_image_ids {
                output.push_str(&delete_kitty_image(*id));
            }
            output.push_str("\x1b[2J\x1b[H\x1b[3J");
        }
        let height_usize = usize::from(height.max(1));
        let mut index = 0;
        while index < lines.len() {
            if index > 0 {
                output.push_str("\r\n");
            }
            let line = &lines[index];
            let reserved = kitty_reserved_rows(line)
                .min(lines.len().saturating_sub(index))
                .max(1);
            if is_image_line(line) && reserved > 1 && reserved <= height_usize {
                for _ in 1..reserved {
                    output.push_str("\r\n");
                }
                write!(output, "\x1b[{}A", reserved - 1).expect("writing to a String cannot fail");
                output.push_str(line);
                write!(output, "\x1b[{}B", reserved - 1).expect("writing to a String cannot fail");
                index += reserved;
            } else {
                output.push_str(line);
                index += 1;
            }
        }
        output.push_str(SYNC_END);
        self.write_batch(&output)?;
        self.stats.full_redraws += 1;
        self.hardware_cursor_row = lines.len().saturating_sub(1);
        self.previous_lines = lines.to_vec();
        self.previous_width = width;
        self.previous_height = height;
        self.max_lines_rendered = lines.len();
        self.previous_viewport_top = lines.len().saturating_sub(height_usize);
        self.previous_image_ids = collect_image_ids(lines);
        self.position_cursor(cursor, lines.len(), height_usize)?;
        Ok(())
    }

    fn position_cursor(
        &mut self,
        cursor: Option<FrameCursor>,
        total_lines: usize,
        height: usize,
    ) -> Result<()> {
        let Some(cursor) = cursor else {
            self.write_batch("\x1b[?25l")?;
            return Ok(());
        };
        if total_lines == 0 {
            self.write_batch("\x1b[?25l")?;
            return Ok(());
        }
        let target_row = cursor.row.min(total_lines - 1);
        let viewport_top = total_lines.saturating_sub(height);
        if target_row < viewport_top {
            self.write_batch("\x1b[?25l")?;
            return Ok(());
        }
        let current_screen = self
            .hardware_cursor_row
            .saturating_sub(viewport_top)
            .min(height - 1);
        let target_screen = target_row.saturating_sub(viewport_top).min(height - 1);
        let mut output = String::new();
        append_vertical_move(&mut output, current_screen, target_screen);
        write!(output, "\x1b[{}G", cursor.column + 1).expect("writing to a String cannot fail");
        output.push_str(if cursor.visible {
            "\x1b[?25h"
        } else {
            "\x1b[?25l"
        });
        self.write_batch(&output)?;
        self.hardware_cursor_row = target_row;
        Ok(())
    }

    fn write_batch(&mut self, output: &str) -> Result<()> {
        self.terminal.write(output)?;
        self.terminal.flush()?;
        self.stats.bytes_written = self.stats.bytes_written.saturating_add(output.len() as u64);
        Ok(())
    }
}

fn extract_cursor_marker(lines: &mut [ConstrainedLine], height: usize) -> Option<FrameCursor> {
    let viewport_top = lines.len().saturating_sub(height);
    for row in (viewport_top..lines.len()).rev() {
        let line = lines[row].as_str();
        let Some(marker) = line.find(CURSOR_MARKER) else {
            continue;
        };
        let column = visible_width(&line[..marker]);
        let mut stripped = line.to_owned();
        stripped.replace_range(marker..marker + CURSOR_MARKER.len(), "");
        let limit = lines[row].limit();
        let Ok(stripped) = ConstrainedLine::new(stripped, limit) else {
            continue;
        };
        lines[row] = stripped;
        return Some(FrameCursor {
            row,
            column,
            visible: false,
        });
    }
    None
}

fn changed_range(old: &[String], new: &[String]) -> (Option<usize>, Option<usize>) {
    let maximum = old.len().max(new.len());
    let mut first = None;
    let mut last = None;
    for index in 0..maximum {
        if old.get(index).map_or("", String::as_str) != new.get(index).map_or("", String::as_str) {
            first.get_or_insert(index);
            last = Some(index);
        }
    }
    (first, last)
}

fn append_vertical_move(output: &mut String, current: usize, target: usize) {
    if target > current {
        write!(output, "\x1b[{}B", target - current).expect("writing to a String cannot fail");
    } else if current > target {
        write!(output, "\x1b[{}A", current - target).expect("writing to a String cannot fail");
    }
}

fn collect_image_ids(lines: &[String]) -> BTreeSet<u32> {
    lines
        .iter()
        .flat_map(|line| kitty_parameters(line, 'i'))
        .collect()
}

fn kitty_reserved_rows(line: &str) -> usize {
    kitty_parameters(line, 'r')
        .next()
        .and_then(|rows| usize::try_from(rows).ok())
        .unwrap_or(1)
}

fn kitty_parameters(line: &str, key: char) -> impl Iterator<Item = u32> + '_ {
    line.match_indices("\x1b_G").filter_map(move |(start, _)| {
        let params = line.get(start + 3..)?.split_once(';')?.0;
        params.split(',').find_map(|parameter| {
            let (name, value) = parameter.split_once('=')?;
            (name == key.to_string())
                .then(|| value.parse().ok())
                .flatten()
        })
    })
}

fn changed_image_region(old: &[String], new: &[String]) -> bool {
    let (first, last) = changed_range(old, new);
    let (Some(first), Some(last)) = (first, last) else {
        return false;
    };
    (first..=last).any(|index| {
        old.get(index).is_some_and(|line| is_image_line(line))
            || new.get(index).is_some_and(|line| is_image_line(line))
    })
}

#[cfg(test)]
mod tests {
    use crate::VirtualTerminal;

    use super::*;

    fn frame(lines: &[&str], width: usize) -> Frame {
        Frame::new(
            lines
                .iter()
                .map(|line| ConstrainedLine::new(*line, width).unwrap())
                .collect(),
        )
    }

    #[test]
    fn updates_only_changed_range_inside_sync_frame() {
        let terminal = VirtualTerminal::new(20, 5);
        let mut renderer = DifferentialRenderer::new(terminal);
        renderer
            .render(frame(&["head", "old", "tail"], 20))
            .unwrap();
        renderer.terminal_mut().clear_writes();
        renderer
            .render(frame(&["head", "new", "tail"], 20))
            .unwrap();
        let output = renderer.terminal().output();
        assert!(output.starts_with(SYNC_START));
        assert!(output.ends_with("\x1b[?25l"));
        assert!(output.contains("new"));
        assert!(!output.contains("head"));
        assert!(!output.contains("\x1b[2J"));
        assert_eq!(renderer.terminal().viewport()[1], "new");
    }

    #[test]
    fn resize_forces_safe_full_redraw() {
        let terminal = VirtualTerminal::new(20, 5);
        let mut renderer = DifferentialRenderer::new(terminal);
        renderer.render(frame(&["hello"], 20)).unwrap();
        renderer.terminal_mut().resize(30, 6);
        renderer.render(frame(&["hello"], 30)).unwrap();
        assert!(renderer.terminal().output().contains("\x1b[2J"));
        assert_eq!(renderer.stats().full_redraws, 2);
    }
}
