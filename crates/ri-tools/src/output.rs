//! Bounded streaming output accumulation with spill-to-file.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

use uuid::Uuid;

use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationOptions, TruncationResult,
    truncate_tail,
};

/// Configuration for [`OutputAccumulator`].
#[derive(Clone, Debug)]
pub struct OutputAccumulatorOptions {
    /// Display line limit.
    pub max_lines: usize,
    /// Display byte limit.
    pub max_bytes: usize,
    /// Prefix for a spill file.
    pub temp_file_prefix: String,
}

impl Default for OutputAccumulatorOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
            temp_file_prefix: "ri-output".to_owned(),
        }
    }
}

/// One immutable output snapshot.
#[derive(Clone, Debug)]
pub struct OutputSnapshot {
    /// Tail selected for model display.
    pub content: String,
    /// Truncation metadata.
    pub truncation: TruncationResult,
    /// Full raw output file, when output spilled.
    pub full_output_path: Option<PathBuf>,
    /// Bytes in the current final line.
    pub last_line_bytes: usize,
}

/// Incremental output collector with bounded display memory.
#[derive(Debug)]
pub struct OutputAccumulator {
    options: OutputAccumulatorOptions,
    max_rolling_bytes: usize,
    prefix: Vec<u8>,
    tail: Vec<u8>,
    tail_starts_at_boundary: bool,
    spill_path: Option<PathBuf>,
    spill: Option<File>,
    total_bytes: usize,
    completed_lines: usize,
    has_open_line: bool,
    current_line_bytes: usize,
    finished: bool,
}

impl OutputAccumulator {
    /// Construct an empty accumulator.
    pub fn new(options: OutputAccumulatorOptions) -> Self {
        Self {
            max_rolling_bytes: options.max_bytes.saturating_mul(2).max(1),
            options,
            prefix: Vec::new(),
            tail: Vec::new(),
            tail_starts_at_boundary: true,
            spill_path: None,
            spill: None,
            total_bytes: 0,
            completed_lines: 0,
            has_open_line: false,
            current_line_bytes: 0,
            finished: false,
        }
    }

    /// Append one raw process-output chunk.
    ///
    /// # Errors
    ///
    /// Returns an error after finalization or when spill-file I/O fails.
    pub fn append(&mut self, data: &[u8]) -> io::Result<()> {
        if self.finished {
            return Err(io::Error::other(
                "cannot append to a finished output accumulator",
            ));
        }
        self.total_bytes = self.total_bytes.saturating_add(data.len());
        for byte in data {
            if *byte == b'\n' {
                self.completed_lines = self.completed_lines.saturating_add(1);
                self.has_open_line = false;
                self.current_line_bytes = 0;
            } else {
                self.has_open_line = true;
                self.current_line_bytes = self.current_line_bytes.saturating_add(1);
            }
        }

        self.tail.extend_from_slice(data);
        if self.tail.len() > self.max_rolling_bytes.saturating_mul(2) {
            self.trim_tail();
        }

        if let Some(spill) = &mut self.spill {
            spill.write_all(data)?;
        } else {
            self.prefix.extend_from_slice(data);
            if self.should_spill() {
                self.ensure_spill()?;
            }
        }
        Ok(())
    }

    /// Mark output complete and flush any spill file.
    ///
    /// # Errors
    ///
    /// Returns an error when spill-file creation or flushing fails.
    pub fn finish(&mut self) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        if self.should_spill() {
            self.ensure_spill()?;
        }
        if let Some(spill) = &mut self.spill {
            spill.flush()?;
        }
        Ok(())
    }

    /// Capture the current tail and optionally persist all output when truncated.
    ///
    /// # Errors
    ///
    /// Returns an error when persistence or flushing fails.
    pub fn snapshot(&mut self, persist_if_truncated: bool) -> io::Result<OutputSnapshot> {
        let snapshot_bytes = self.snapshot_bytes();
        let snapshot_text = String::from_utf8_lossy(snapshot_bytes);
        let mut truncation = truncate_tail(
            &snapshot_text,
            TruncationOptions {
                max_lines: self.options.max_lines,
                max_bytes: self.options.max_bytes,
            },
        );
        let total_lines = self.total_lines();
        let truncated =
            total_lines > self.options.max_lines || self.total_bytes > self.options.max_bytes;
        truncation.truncated = truncated;
        truncation.truncated_by = if truncated {
            truncation.truncated_by.or_else(|| {
                (self.total_bytes > self.options.max_bytes)
                    .then_some(TruncatedBy::Bytes)
                    .or(Some(TruncatedBy::Lines))
            })
        } else {
            None
        };
        truncation.total_lines = total_lines;
        truncation.total_bytes = self.total_bytes;
        if persist_if_truncated && truncated {
            self.ensure_spill()?;
            if let Some(spill) = &mut self.spill {
                spill.flush()?;
            }
        }
        Ok(OutputSnapshot {
            content: truncation.content.clone(),
            truncation,
            full_output_path: self.spill_path.clone(),
            last_line_bytes: self.current_line_bytes,
        })
    }

    fn total_lines(&self) -> usize {
        self.completed_lines + usize::from(self.has_open_line)
    }

    fn should_spill(&self) -> bool {
        self.total_bytes > self.options.max_bytes || self.total_lines() > self.options.max_lines
    }

    fn ensure_spill(&mut self) -> io::Result<()> {
        if self.spill_path.is_some() {
            return Ok(());
        }
        let path = std::env::temp_dir().join(format!(
            "{}-{}.log",
            self.options.temp_file_prefix,
            Uuid::new_v4()
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        file.write_all(&self.prefix)?;
        self.prefix.clear();
        self.spill_path = Some(path);
        self.spill = Some(file);
        Ok(())
    }

    fn trim_tail(&mut self) {
        if self.tail.len() <= self.max_rolling_bytes {
            return;
        }
        let mut start = self.tail.len() - self.max_rolling_bytes;
        while start < self.tail.len() && (self.tail[start] & 0xc0) == 0x80 {
            start += 1;
        }
        self.tail_starts_at_boundary = start == 0 && self.tail_starts_at_boundary
            || start > 0 && self.tail[start - 1] == b'\n';
        self.tail.drain(..start);
    }

    fn snapshot_bytes(&self) -> &[u8] {
        if self.tail_starts_at_boundary {
            return &self.tail;
        }
        self.tail
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(&self.tail, |newline| &self.tail[newline + 1..])
    }
}

impl Default for OutputAccumulator {
    fn default() -> Self {
        Self::new(OutputAccumulatorOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spills_on_line_only_truncation() {
        let mut accumulator = OutputAccumulator::new(OutputAccumulatorOptions {
            max_lines: 2,
            max_bytes: 1024,
            temp_file_prefix: "ri-tools-test".to_owned(),
        });
        accumulator.append(b"one\ntwo\nthree\n").unwrap();
        accumulator.finish().unwrap();
        let snapshot = accumulator.snapshot(true).unwrap();
        assert_eq!(snapshot.content, "two\nthree");
        let path = snapshot.full_output_path.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "one\ntwo\nthree\n");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn decodes_split_utf8_after_append() {
        let mut accumulator = OutputAccumulator::default();
        let euro = "€\n".as_bytes();
        accumulator.append(&euro[..1]).unwrap();
        accumulator.append(&euro[1..]).unwrap();
        accumulator.finish().unwrap();
        assert_eq!(accumulator.snapshot(false).unwrap().content, "€\n");
    }
}
