//! Deterministic terminal emulator for renderer tests and headless consumers.

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use unicode_segmentation::UnicodeSegmentation;

use crate::ansi::grapheme_width;
use crate::terminal::{Terminal, TerminalEvent};

#[derive(Clone, Debug, Default)]
struct Cell {
    text: String,
    continuation: bool,
}

/// A small VT emulator implementing the library's terminal contract.
#[derive(Debug)]
pub struct VirtualTerminal {
    columns: u16,
    rows: u16,
    screen: Vec<Vec<Cell>>,
    scrollback: Vec<Vec<Cell>>,
    cursor_x: usize,
    cursor_y: usize,
    cursor_visible: bool,
    started: bool,
    writes: Vec<String>,
    events: VecDeque<TerminalEvent>,
}

impl VirtualTerminal {
    /// Creates an empty terminal.
    pub fn new(columns: u16, rows: u16) -> Self {
        let columns = columns.max(1);
        let rows = rows.max(1);
        Self {
            columns,
            rows,
            screen: blank_screen(columns, rows),
            scrollback: Vec::new(),
            cursor_x: 0,
            cursor_y: 0,
            cursor_visible: true,
            started: false,
            writes: Vec::new(),
            events: VecDeque::new(),
        }
    }

    /// Resizes and queues a resize event.
    pub fn resize(&mut self, columns: u16, rows: u16) {
        let columns = columns.max(1);
        let rows = rows.max(1);
        for line in &mut self.screen {
            line.resize(usize::from(columns), Cell::default());
        }
        self.screen.resize(
            usize::from(rows),
            vec![Cell::default(); usize::from(columns)],
        );
        self.columns = columns;
        self.rows = rows;
        self.cursor_x = self.cursor_x.min(usize::from(columns) - 1);
        self.cursor_y = self.cursor_y.min(usize::from(rows) - 1);
        self.events
            .push_back(TerminalEvent::Resize { columns, rows });
    }

    /// Queues input for `read_event`.
    pub fn send_event(&mut self, event: TerminalEvent) {
        self.events.push_back(event);
    }

    /// Returns visible rows with trailing blanks removed.
    pub fn viewport(&self) -> Vec<String> {
        self.screen.iter().map(line_to_string).collect()
    }

    /// Returns scrollback followed by the viewport.
    pub fn buffer(&self) -> Vec<String> {
        self.scrollback
            .iter()
            .chain(&self.screen)
            .map(line_to_string)
            .collect()
    }

    /// Returns all raw write batches.
    pub fn writes(&self) -> &[String] {
        &self.writes
    }

    /// Joins all raw write batches.
    pub fn output(&self) -> String {
        self.writes.concat()
    }

    /// Clears the write log without affecting terminal state.
    pub fn clear_writes(&mut self) {
        self.writes.clear();
    }

    /// Current zero-based cursor position.
    pub const fn cursor(&self) -> (usize, usize) {
        (self.cursor_x, self.cursor_y)
    }

    /// Whether the hardware cursor is visible.
    pub const fn cursor_visible(&self) -> bool {
        self.cursor_visible
    }

    /// Whether `start` has been called without a matching `stop`.
    pub const fn is_started(&self) -> bool {
        self.started
    }

    fn consume(&mut self, data: &str) {
        let mut offset = 0;
        while offset < data.len() {
            let bytes = data.as_bytes();
            let control = if bytes[offset] == 0x1b {
                control_sequence(data, offset)
            } else {
                None
            };
            if let Some((length, kind)) = control {
                let sequence = &data[offset..offset + length];
                self.apply_control(sequence, kind);
                offset += length;
                continue;
            }

            let character = data[offset..].chars().next().expect("valid string tail");
            match character {
                '\r' => self.cursor_x = 0,
                '\n' => self.line_feed(),
                '\x08' => self.cursor_x = self.cursor_x.saturating_sub(1),
                character if character.is_control() => {}
                _ => {
                    let grapheme = data[offset..]
                        .graphemes(true)
                        .next()
                        .expect("valid string tail");
                    self.print_grapheme(grapheme);
                    offset += grapheme.len();
                    continue;
                }
            }
            offset += character.len_utf8();
        }
    }

    fn apply_control(&mut self, sequence: &str, kind: ControlKind) {
        match kind {
            ControlKind::Csi => self.apply_csi(sequence),
            ControlKind::Osc | ControlKind::String | ControlKind::Single => {}
        }
    }

    fn apply_csi(&mut self, sequence: &str) {
        let Some(final_byte) = sequence.chars().last() else {
            return;
        };
        let body = &sequence[2..sequence.len() - final_byte.len_utf8()];
        let private = body.starts_with('?');
        let params = body.trim_start_matches('?');
        let first = params
            .split(';')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let amount = first.max(1);
        match final_byte {
            'A' => self.cursor_y = self.cursor_y.saturating_sub(amount),
            'B' => {
                self.cursor_y =
                    (self.cursor_y + amount).min(usize::from(self.rows).saturating_sub(1));
            }
            'C' => {
                self.cursor_x =
                    (self.cursor_x + amount).min(usize::from(self.columns).saturating_sub(1));
            }
            'D' => self.cursor_x = self.cursor_x.saturating_sub(amount),
            'G' => {
                self.cursor_x = amount
                    .saturating_sub(1)
                    .min(usize::from(self.columns).saturating_sub(1));
            }
            'H' | 'f' => {
                let mut values = params
                    .split(';')
                    .filter_map(|value| value.parse::<usize>().ok());
                self.cursor_y = values
                    .next()
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .min(usize::from(self.rows).saturating_sub(1));
                self.cursor_x = values
                    .next()
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .min(usize::from(self.columns).saturating_sub(1));
            }
            'J' if first == 2 => {
                self.screen = blank_screen(self.columns, self.rows);
            }
            'J' if first == 3 => self.scrollback.clear(),
            'J' => {
                self.clear_line_from(self.cursor_y, self.cursor_x);
                for row in self.cursor_y + 1..usize::from(self.rows) {
                    self.screen[row].fill(Cell::default());
                }
            }
            'K' if first == 2 => self.screen[self.cursor_y].fill(Cell::default()),
            'K' => self.clear_line_from(self.cursor_y, self.cursor_x),
            'h' if private && params == "25" => self.cursor_visible = true,
            'l' if private && params == "25" => self.cursor_visible = false,
            _ => {}
        }
    }

    fn clear_line_from(&mut self, row: usize, column: usize) {
        if let Some(line) = self.screen.get_mut(row) {
            for cell in line.iter_mut().skip(column) {
                *cell = Cell::default();
            }
        }
    }

    fn print_grapheme(&mut self, grapheme: &str) {
        let width = grapheme_width(grapheme);
        if width == 0 {
            if self.cursor_x > 0 {
                self.screen[self.cursor_y][self.cursor_x - 1]
                    .text
                    .push_str(grapheme);
            }
            return;
        }
        if self.cursor_x + width > usize::from(self.columns) {
            self.cursor_x = 0;
            self.line_feed();
        }
        let line = &mut self.screen[self.cursor_y];
        line[self.cursor_x] = Cell {
            text: grapheme.to_owned(),
            continuation: false,
        };
        for column in self.cursor_x + 1..(self.cursor_x + width).min(line.len()) {
            line[column] = Cell {
                text: String::new(),
                continuation: true,
            };
        }
        self.cursor_x += width;
        if self.cursor_x >= usize::from(self.columns) {
            // VT autowrap is pending until the next printable character. The
            // renderer always emits CRLF, so retaining the edge position is
            // sufficient and avoids an eager extra scroll.
            self.cursor_x = usize::from(self.columns);
        }
    }

    fn line_feed(&mut self) {
        if self.cursor_y + 1 < usize::from(self.rows) {
            self.cursor_y += 1;
            return;
        }
        let removed = self.screen.remove(0);
        self.scrollback.push(removed);
        self.screen
            .push(vec![Cell::default(); usize::from(self.columns)]);
        self.cursor_y = usize::from(self.rows) - 1;
    }
}

impl Terminal for VirtualTerminal {
    fn start(&mut self) -> io::Result<()> {
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) -> io::Result<()> {
        self.started = false;
        Ok(())
    }

    fn write(&mut self, data: &str) -> io::Result<()> {
        self.writes.push(data.to_owned());
        self.consume(data);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn size(&self) -> io::Result<(u16, u16)> {
        Ok((self.columns, self.rows))
    }

    fn read_event(&mut self, _timeout: Duration) -> io::Result<Option<TerminalEvent>> {
        Ok(self.events.pop_front())
    }

    fn kitty_protocol_active(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlKind {
    Csi,
    Osc,
    String,
    Single,
}

fn control_sequence(data: &str, offset: usize) -> Option<(usize, ControlKind)> {
    let bytes = data.as_bytes();
    let second = *bytes.get(offset + 1)?;
    match second {
        b'[' => {
            let mut index = offset + 2;
            while let Some(byte) = bytes.get(index) {
                if (0x40..=0x7e).contains(byte) {
                    return Some((index + 1 - offset, ControlKind::Csi));
                }
                index += 1;
            }
            Some((bytes.len() - offset, ControlKind::Csi))
        }
        b']' => Some((terminated_string(bytes, offset, true), ControlKind::Osc)),
        b'P' | b'_' | b'^' => Some((
            terminated_string(bytes, offset, second == b'_'),
            ControlKind::String,
        )),
        b'O' => Some((
            if bytes.get(offset + 2).is_some_and(u8::is_ascii) {
                3
            } else {
                2
            },
            ControlKind::Single,
        )),
        _ if second.is_ascii() => Some((2, ControlKind::Single)),
        _ => Some((1, ControlKind::Single)),
    }
}

fn terminated_string(bytes: &[u8], offset: usize, allow_bel: bool) -> usize {
    let mut index = offset + 2;
    while index < bytes.len() {
        if allow_bel && bytes[index] == 0x07 {
            return index + 1 - offset;
        }
        if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'\\') {
            return index + 2 - offset;
        }
        index += 1;
    }
    bytes.len() - offset
}

fn blank_screen(columns: u16, rows: u16) -> Vec<Vec<Cell>> {
    vec![vec![Cell::default(); usize::from(columns)]; usize::from(rows)]
}

fn line_to_string(line: &Vec<Cell>) -> String {
    let mut output = String::new();
    for cell in line {
        if !cell.continuation {
            if cell.text.is_empty() {
                output.push(' ');
            } else {
                output.push_str(&cell.text);
            }
        }
    }
    output.trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emulates_cursor_updates_and_wide_cells() {
        let mut terminal = VirtualTerminal::new(10, 3);
        terminal.write("A界\r\nsecond").unwrap();
        terminal.write("\x1b[1A\r\x1b[2Knew").unwrap();
        assert_eq!(terminal.viewport(), ["new", "second", ""]);
    }
}
