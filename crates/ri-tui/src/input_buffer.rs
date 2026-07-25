//! Framing for arbitrarily chunked terminal stdin.

use std::time::{Duration, Instant};

const ESC: u8 = 0x1b;
const PASTE_START: &[u8] = b"\x1b[200~";
const PASTE_END: &[u8] = b"\x1b[201~";

/// A complete stdin frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputFrame {
    /// One key, mouse, control-response, or plain grapheme sequence.
    Data(String),
    /// Contents of one bracketed paste, without framing markers.
    Paste(String),
}

/// Buffers partial input and emits complete terminal protocol frames.
#[derive(Debug)]
pub struct StdinFrameBuffer {
    buffer: Vec<u8>,
    paste_buffer: Vec<u8>,
    paste_mode: bool,
    timeout: Duration,
    deadline: Option<Instant>,
    pending_kitty_printable: Option<u32>,
}

impl Default for StdinFrameBuffer {
    fn default() -> Self {
        Self::new(Duration::from_millis(10))
    }
}

impl StdinFrameBuffer {
    /// Creates a buffer with an incomplete-sequence timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            buffer: Vec::new(),
            paste_buffer: Vec::new(),
            paste_mode: false,
            timeout,
            deadline: None,
            pending_kitty_printable: None,
        }
    }

    /// Feeds a chunk using the current clock.
    pub fn push(&mut self, data: impl AsRef<[u8]>) -> Vec<InputFrame> {
        self.push_at(data, Instant::now())
    }

    /// Feeds a chunk with an explicit timestamp for deterministic runtimes.
    pub fn push_at(&mut self, data: impl AsRef<[u8]>, now: Instant) -> Vec<InputFrame> {
        let data = data.as_ref();
        if data.is_empty() && self.buffer.is_empty() && !self.paste_mode {
            return vec![InputFrame::Data(String::new())];
        }
        self.deadline = None;
        self.buffer.extend_from_slice(data);

        let mut frames = Vec::new();
        self.extract_frames(&mut frames);
        if !self.buffer.is_empty() {
            self.deadline = now.checked_add(self.timeout);
        }
        frames
    }

    /// Flushes an incomplete sequence once its timeout has elapsed.
    pub fn flush_expired(&mut self, now: Instant) -> Vec<InputFrame> {
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            self.flush()
        } else {
            Vec::new()
        }
    }

    /// Flushes incomplete non-paste bytes as one data frame.
    pub fn flush(&mut self) -> Vec<InputFrame> {
        self.deadline = None;
        self.pending_kitty_printable = None;
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let bytes = std::mem::take(&mut self.buffer);
        vec![InputFrame::Data(
            String::from_utf8_lossy(&bytes).into_owned(),
        )]
    }

    /// Clears all state without emitting.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.paste_buffer.clear();
        self.paste_mode = false;
        self.deadline = None;
        self.pending_kitty_printable = None;
    }

    /// Returns bytes waiting for sequence completion.
    pub fn pending(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns whether a bracketed paste is currently open.
    pub fn is_pasting(&self) -> bool {
        self.paste_mode
    }

    fn extract_frames(&mut self, frames: &mut Vec<InputFrame>) {
        loop {
            if self.paste_mode {
                self.paste_buffer.append(&mut self.buffer);
                if let Some(end) = find_subslice(&self.paste_buffer, PASTE_END) {
                    let content = self.paste_buffer[..end].to_vec();
                    let remaining = self.paste_buffer[end + PASTE_END.len()..].to_vec();
                    self.paste_buffer.clear();
                    self.paste_mode = false;
                    self.pending_kitty_printable = None;
                    frames.push(InputFrame::Paste(
                        String::from_utf8_lossy(&content).into_owned(),
                    ));
                    self.buffer = remaining;
                    continue;
                }
                break;
            }

            if self.buffer.is_empty() {
                break;
            }
            if PASTE_START.starts_with(&self.buffer) && self.buffer.len() < PASTE_START.len() {
                break;
            }
            if self.buffer.starts_with(PASTE_START) {
                self.buffer.drain(..PASTE_START.len());
                self.paste_mode = true;
                self.pending_kitty_printable = None;
                continue;
            }

            match complete_frame_length(&self.buffer) {
                FrameStatus::Complete(length) => {
                    let bytes: Vec<u8> = self.buffer.drain(..length).collect();
                    let sequence = String::from_utf8_lossy(&bytes).into_owned();
                    self.emit_data(sequence, frames);
                }
                FrameStatus::Incomplete => break,
            }
        }
    }

    fn emit_data(&mut self, sequence: String, frames: &mut Vec<InputFrame>) {
        let raw = sequence
            .chars()
            .next()
            .filter(|_| sequence.chars().count() == 1);
        if raw
            .map(u32::from)
            .is_some_and(|raw| self.pending_kitty_printable == Some(raw))
        {
            self.pending_kitty_printable = None;
            return;
        }
        self.pending_kitty_printable = unmodified_kitty_printable(&sequence);
        frames.push(InputFrame::Data(sequence));
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameStatus {
    Complete(usize),
    Incomplete,
}

fn complete_frame_length(bytes: &[u8]) -> FrameStatus {
    if bytes[0] != ESC {
        return utf8_scalar_length(bytes);
    }
    if bytes.len() == 1 {
        return FrameStatus::Incomplete;
    }

    match bytes[1] {
        b'[' => complete_csi(bytes),
        b']' => complete_string_sequence(bytes, true),
        b'P' | b'_' | b'^' => complete_string_sequence(bytes, bytes[1] == b'_'),
        b'O' => {
            if bytes.len() >= 3 {
                FrameStatus::Complete(3)
            } else {
                FrameStatus::Incomplete
            }
        }
        ESC => {
            // WezTerm can concatenate raw Escape press with Kitty release.
            if bytes
                .get(2)
                .is_some_and(|byte| matches!(byte, b'[' | b']' | b'O' | b'P' | b'_'))
            {
                FrameStatus::Complete(1)
            } else {
                FrameStatus::Complete(2)
            }
        }
        _ => match utf8_scalar_length(&bytes[1..]) {
            FrameStatus::Complete(length) => FrameStatus::Complete(1 + length),
            FrameStatus::Incomplete => FrameStatus::Incomplete,
        },
    }
}

fn complete_csi(bytes: &[u8]) -> FrameStatus {
    if bytes.len() < 3 {
        return FrameStatus::Incomplete;
    }
    if bytes.starts_with(b"\x1b[M") {
        return if bytes.len() >= 6 {
            FrameStatus::Complete(6)
        } else {
            FrameStatus::Incomplete
        };
    }

    for (index, byte) in bytes.iter().enumerate().skip(2) {
        if (0x40..=0x7e).contains(byte) {
            if bytes.get(2) == Some(&b'<') && matches!(byte, b'M' | b'm') {
                let payload = &bytes[3..index];
                if payload.split(|value| *value == b';').count() != 3
                    || !payload
                        .iter()
                        .all(|value| value.is_ascii_digit() || *value == b';')
                {
                    return FrameStatus::Incomplete;
                }
            }
            return FrameStatus::Complete(index + 1);
        }
    }
    FrameStatus::Incomplete
}

fn complete_string_sequence(bytes: &[u8], allow_bel: bool) -> FrameStatus {
    let mut index = 2;
    while index < bytes.len() {
        if allow_bel && bytes[index] == 0x07 {
            return FrameStatus::Complete(index + 1);
        }
        if bytes[index] == ESC && bytes.get(index + 1) == Some(&b'\\') {
            return FrameStatus::Complete(index + 2);
        }
        index += 1;
    }
    FrameStatus::Incomplete
}

fn utf8_scalar_length(bytes: &[u8]) -> FrameStatus {
    let Some(first) = bytes.first().copied() else {
        return FrameStatus::Incomplete;
    };
    let length = match first {
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => 1,
    };
    if bytes.len() < length {
        return FrameStatus::Incomplete;
    }
    FrameStatus::Complete(length)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn unmodified_kitty_printable(sequence: &str) -> Option<u32> {
    let body = sequence.strip_prefix("\x1b[")?.strip_suffix('u')?;
    if body.contains(';') {
        return None;
    }
    let codepoint = body.split(':').next()?.parse().ok()?;
    (codepoint >= 32).then_some(codepoint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_split_sequences_and_splits_batches() {
        let mut buffer = StdinFrameBuffer::default();
        assert!(buffer.push("\x1b[<35").is_empty());
        assert_eq!(
            buffer.push(";20;5mab"),
            vec![
                InputFrame::Data("\x1b[<35;20;5m".to_owned()),
                InputFrame::Data("a".to_owned()),
                InputFrame::Data("b".to_owned()),
            ]
        );
    }

    #[test]
    fn frames_bracketed_paste_across_chunks() {
        let mut buffer = StdinFrameBuffer::default();
        assert!(buffer.push("\x1b[200~hello ").is_empty());
        assert_eq!(
            buffer.push("世界\x1b[201~x"),
            vec![
                InputFrame::Paste("hello 世界".to_owned()),
                InputFrame::Data("x".to_owned()),
            ]
        );
    }

    #[test]
    fn separates_wezterm_escape_press_and_release() {
        let mut buffer = StdinFrameBuffer::default();
        assert_eq!(
            buffer.push("\x1b\x1b[27;1:3u"),
            vec![
                InputFrame::Data("\x1b".to_owned()),
                InputFrame::Data("\x1b[27;1:3u".to_owned())
            ]
        );
    }
}
