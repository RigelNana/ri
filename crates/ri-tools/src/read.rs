//! File reading tool.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{GenericImageView, ImageFormat};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::{Content, ToolResult};
use crate::error::{EnvError, ToolError};
use crate::paths::resolve_path;
use crate::truncate::{
    DEFAULT_MAX_BYTES, TruncatedBy, TruncationOptions, TruncationResult, format_size, truncate_head,
};

/// Input for [`read`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadInput {
    /// Relative or absolute file path.
    pub path: PathBuf,
    /// First line to return, using one-based numbering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// Maximum selected lines before normal output truncation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

impl ReadInput {
    /// Read a whole file from its first line.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            offset: None,
            limit: None,
        }
    }
}

/// Structured metadata returned by the read tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReadDetails {
    /// Output truncation details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
}

/// Result of the read tool.
pub type ReadResult = ToolResult<ReadDetails>;

/// Read a text file or supported image.
///
/// # Errors
///
/// Returns an error for missing files, invalid line ranges, cancellation, or
/// execution-environment failures.
pub async fn read(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: ReadInput,
    cancellation: &CancellationToken,
) -> Result<ReadResult, ToolError> {
    throw_if_cancelled(cancellation)?;
    let absolute = resolve_path(&input.path, cwd);
    let bytes = env
        .read_file(&absolute)
        .await
        .map_err(|error| map_path_error("read", &absolute, error))?;
    throw_if_cancelled(cancellation)?;

    if let Some(mime_type) = detect_image_mime(&bytes) {
        let (image_bytes, mime_type, hint) = prepare_image(bytes, mime_type);
        let encoded = BASE64.encode(&image_bytes);
        let note = hint.map_or_else(
            || format!("Read image file [{mime_type}]"),
            |hint| format!("Read image file [{mime_type}]\n{hint}"),
        );
        return Ok(ReadResult {
            content: vec![
                Content::text(note),
                Content::Image {
                    data: encoded,
                    mime_type: mime_type.to_owned(),
                },
            ],
            details: None,
        });
    }

    let text = String::from_utf8_lossy(&bytes);
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_file_lines = all_lines.len();
    let start = input.offset.unwrap_or(1).saturating_sub(1);
    if start >= total_file_lines {
        return Err(ToolError::InvalidInput(format!(
            "Offset {} is beyond end of file ({total_file_lines} lines total)",
            input.offset.unwrap_or(1)
        )));
    }
    let start_display = start + 1;
    let (selected, user_limited_lines) = if let Some(limit) = input.limit {
        let end = start.saturating_add(limit).min(total_file_lines);
        (all_lines[start..end].join("\n"), Some(end - start))
    } else {
        (all_lines[start..].join("\n"), None)
    };

    let truncation = truncate_head(&selected, TruncationOptions::default());
    let (output, details) = if truncation.first_line_exceeds_limit {
        let first_line_size = format_size(all_lines[start].len());
        (
            format!(
                "[Line {start_display} is {first_line_size}, exceeds {} limit. \
                 Use bash to inspect a byte-limited slice of {}]",
                format_size(DEFAULT_MAX_BYTES),
                input.path.display()
            ),
            Some(ReadDetails {
                truncation: Some(truncation),
            }),
        )
    } else if truncation.truncated {
        let end_display = start_display + truncation.output_lines.saturating_sub(1);
        let next_offset = end_display + 1;
        let notice = match truncation.truncated_by {
            Some(TruncatedBy::Lines) => format!(
                "[Showing lines {start_display}-{end_display} of {total_file_lines}. \
                 Use offset={next_offset} to continue.]"
            ),
            _ => format!(
                "[Showing lines {start_display}-{end_display} of {total_file_lines} \
                 ({} limit). Use offset={next_offset} to continue.]",
                format_size(DEFAULT_MAX_BYTES)
            ),
        };
        (
            format!("{}\n\n{notice}", truncation.content),
            Some(ReadDetails {
                truncation: Some(truncation),
            }),
        )
    } else if let Some(selected_lines) = user_limited_lines {
        if start + selected_lines < total_file_lines {
            let remaining = total_file_lines - (start + selected_lines);
            let next_offset = start + selected_lines + 1;
            (
                format!(
                    "{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]",
                    truncation.content
                ),
                None,
            )
        } else {
            (truncation.content, None)
        }
    } else {
        (truncation.content, None)
    };

    Ok(ReadResult::text(output, details))
}

fn prepare_image(
    bytes: Vec<u8>,
    mime_type: &'static str,
) -> (Vec<u8>, &'static str, Option<String>) {
    const MAX_DIMENSION: u32 = 2_000;
    if let Ok(decoded) = image::load_from_memory(&bytes) {
        let (width, height) = decoded.dimensions();
        let resize = width > MAX_DIMENSION || height > MAX_DIMENSION;
        if resize || mime_type == "image/bmp" {
            let output = if resize {
                decoded.thumbnail(MAX_DIMENSION, MAX_DIMENSION)
            } else {
                decoded
            };
            let (output_width, output_height) = output.dimensions();
            let mut encoded = Cursor::new(Vec::new());
            if output.write_to(&mut encoded, ImageFormat::Png).is_ok() {
                let hint = if resize {
                    format!(
                        "[Image resized from {width}x{height} to \
                         {output_width}x{output_height} and encoded as image/png.]"
                    )
                } else {
                    "[Image converted from image/bmp to image/png.]".to_owned()
                };
                return (encoded.into_inner(), "image/png", Some(hint));
            }
        }
    }
    (bytes, mime_type, None)
}

fn detect_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

fn throw_if_cancelled(cancellation: &CancellationToken) -> Result<(), ToolError> {
    if cancellation.is_cancelled() {
        Err(ToolError::Environment(EnvError::Cancelled))
    } else {
        Ok(())
    }
}

fn map_path_error(operation: &'static str, path: &Path, error: EnvError) -> ToolError {
    match error {
        EnvError::Io(source) => ToolError::io(operation, path, source),
        other => ToolError::Environment(other),
    }
}
