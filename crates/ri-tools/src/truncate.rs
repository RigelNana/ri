//! Shared line- and byte-aware truncation.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default maximum number of returned lines.
pub const DEFAULT_MAX_LINES: usize = 2_000;
/// Default maximum UTF-8 payload size.
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
/// Maximum number of characters retained from one grep line.
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// The limit that caused truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TruncatedBy {
    /// Line-count limit.
    Lines,
    /// UTF-8 byte limit.
    Bytes,
}

/// Truncation limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TruncationOptions {
    /// Maximum complete lines.
    pub max_lines: usize,
    /// Maximum UTF-8 bytes.
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Structured information about a truncation operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TruncationResult {
    /// Truncated or original content.
    pub content: String,
    /// Whether any content was omitted.
    pub truncated: bool,
    /// Limit that was reached first.
    #[serde(rename = "truncatedBy", skip_serializing_if = "Option::is_none")]
    pub truncated_by: Option<TruncatedBy>,
    /// Original line count. A final newline does not add an empty line.
    #[serde(rename = "totalLines")]
    pub total_lines: usize,
    /// Original UTF-8 byte count.
    #[serde(rename = "totalBytes")]
    pub total_bytes: usize,
    /// Number of complete output lines.
    #[serde(rename = "outputLines")]
    pub output_lines: usize,
    /// Output UTF-8 byte count.
    #[serde(rename = "outputBytes")]
    pub output_bytes: usize,
    /// Tail truncation retained only part of the final line.
    #[serde(rename = "lastLinePartial")]
    pub last_line_partial: bool,
    /// Head truncation could not retain the first line.
    #[serde(rename = "firstLineExceedsLimit")]
    pub first_line_exceeds_limit: bool,
    /// Applied line limit.
    #[serde(rename = "maxLines")]
    pub max_lines: usize,
    /// Applied byte limit.
    #[serde(rename = "maxBytes")]
    pub max_bytes: usize,
}

/// Truncate from the head without returning a partial line.
pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    let total_bytes = content.len();
    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged(content, total_lines, total_bytes, options);
    }

    if lines
        .first()
        .is_some_and(|line| line.len() > options.max_bytes)
    {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines: options.max_lines,
            max_bytes: options.max_bytes,
        };
    }

    let mut output = Vec::new();
    let mut bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    for (index, line) in lines.iter().take(options.max_lines).enumerate() {
        let line_bytes = line.len() + usize::from(index > 0);
        if bytes + line_bytes > options.max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output.push(*line);
        bytes += line_bytes;
    }
    if output.len() >= options.max_lines && bytes <= options.max_bytes {
        truncated_by = TruncatedBy::Lines;
    }
    let output_content = output.join("\n");
    TruncationResult {
        output_bytes: output_content.len(),
        output_lines: output.len(),
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Truncate from the tail, retaining a valid UTF-8 suffix of an oversized last line.
pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    let total_bytes = content.len();
    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return unchanged(content, total_lines, total_bytes, options);
    }

    let mut output = Vec::new();
    let mut bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;
    for line in lines.iter().rev().take(options.max_lines) {
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if bytes + line_bytes > options.max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if output.is_empty() {
                let suffix = utf8_suffix(line, options.max_bytes);
                bytes = suffix.len();
                output.push(suffix);
                last_line_partial = true;
            }
            break;
        }
        output.push(*line);
        bytes += line_bytes;
    }
    output.reverse();
    if output.len() >= options.max_lines && bytes <= options.max_bytes {
        truncated_by = TruncatedBy::Lines;
    }
    let output_content = output.join("\n");
    TruncationResult {
        output_bytes: output_content.len(),
        output_lines: output.len(),
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Truncate one grep line by Unicode scalar count.
pub fn truncate_line(line: &str, max_chars: usize) -> (String, bool) {
    if line.chars().count() <= max_chars {
        return (line.to_owned(), false);
    }
    let prefix: String = line.chars().take(max_chars).collect();
    (format!("{prefix}... [truncated]"), true)
}

/// Format a byte count for notices.
#[allow(clippy::cast_precision_loss)]
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<_> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

fn utf8_suffix(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut start = value.len().saturating_sub(max_bytes);
    while start < value.len() && !value.is_char_boundary(start) {
        start += 1;
    }
    &value[start..]
}

fn unchanged(
    content: &str,
    total_lines: usize,
    total_bytes: usize,
    options: TruncationOptions,
) -> TruncationResult {
    TruncationResult {
        content: content.to_owned(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes,
        output_lines: total_lines,
        output_bytes: total_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_preserves_complete_utf8_lines() {
        let result = truncate_head(
            "éé\nabc",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 4,
            },
        );
        assert_eq!(result.content, "éé");
        assert_eq!(result.truncated_by, Some(TruncatedBy::Bytes));
    }

    #[test]
    fn tail_keeps_valid_utf8_suffix() {
        let result = truncate_tail(
            "aé🙂b",
            TruncationOptions {
                max_lines: 10,
                max_bytes: 5,
            },
        );
        assert_eq!(result.content, "🙂b");
        assert!(result.last_line_partial);
    }

    #[test]
    fn trailing_newline_is_not_an_extra_line() {
        let result = truncate_head("one\ntwo\n", TruncationOptions::default());
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.content, "one\ntwo\n");
    }
}
