//! Piped input and `@file` prompt preparation.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use tokio::io::AsyncReadExt;

use crate::cli::InputArguments;
use crate::error::{CliError, Result};

/// Inline image prepared for the SDK adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageAttachment {
    /// Canonical source path.
    pub path: PathBuf,
    /// Base64-encoded bytes.
    pub data: String,
    /// Detected media type.
    pub mime_type: String,
}

/// Inputs sent sequentially through one shared session runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PreparedInput {
    /// First prompt, merged with piped stdin and file context.
    pub initial: Option<String>,
    /// Images attached to the first prompt.
    pub images: Vec<ImageAttachment>,
    /// Remaining command-line messages.
    pub follow_ups: Vec<String>,
}

impl PreparedInput {
    /// True when no prompt text or images were supplied.
    pub fn is_empty(&self) -> bool {
        self.initial.is_none() && self.images.is_empty() && self.follow_ups.is_empty()
    }
}

/// Read redirected standard input completely.
///
/// # Errors
///
/// Returns an I/O error if redirected stdin cannot be read.
pub async fn read_piped_stdin() -> Result<Option<String>> {
    let mut content = String::new();
    tokio::io::stdin()
        .read_to_string(&mut content)
        .await
        .map_err(|source| CliError::Io {
            operation: "read stdin",
            source,
        })?;
    Ok((!content.trim().is_empty()).then_some(content))
}

/// Read explicit files and merge them with stdin and the first message.
///
/// Like Pi, stdin comes first, followed by XML-delimited file content and then
/// the first message. Remaining messages are separate prompts.
///
/// # Errors
///
/// Returns an I/O or argument error if an attachment cannot be resolved,
/// decoded, or represented as a supported file type.
pub async fn prepare(arguments: InputArguments, stdin: Option<String>) -> Result<PreparedInput> {
    let processed = read_files(&arguments.files).await?;
    let mut messages = arguments.messages.into_iter();
    let mut initial_parts = Vec::new();
    if let Some(stdin) = stdin.filter(|value| !value.is_empty()) {
        initial_parts.push(stdin);
    }
    if !processed.text.is_empty() {
        initial_parts.push(processed.text);
    }
    if let Some(first) = messages.next() {
        initial_parts.push(first);
    }

    Ok(PreparedInput {
        initial: (!initial_parts.is_empty()).then(|| initial_parts.concat()),
        images: processed.images,
        follow_ups: messages.collect(),
    })
}

#[derive(Debug, Default)]
struct ProcessedFiles {
    text: String,
    images: Vec<ImageAttachment>,
}

async fn read_files(paths: &[PathBuf]) -> Result<ProcessedFiles> {
    let mut output = ProcessedFiles::default();
    for requested in paths {
        let canonical =
            tokio::fs::canonicalize(requested)
                .await
                .map_err(|source| CliError::Io {
                    operation: "resolve attached file",
                    source,
                })?;
        let bytes = tokio::fs::read(&canonical)
            .await
            .map_err(|source| CliError::Io {
                operation: "read attached file",
                source,
            })?;
        if bytes.is_empty() {
            continue;
        }
        if let Some(mime_type) = detect_image(&bytes) {
            writeln!(
                output.text,
                "<file name=\"{}\"></file>",
                escape_attribute(&canonical)
            )
            .expect("writing to a String cannot fail");
            output.images.push(ImageAttachment {
                path: canonical,
                data: BASE64.encode(bytes),
                mime_type: mime_type.to_owned(),
            });
            continue;
        }
        let content = String::from_utf8(bytes).map_err(|_| {
            CliError::InvalidArguments(format!(
                "attached file `{}` is neither UTF-8 text nor a supported image",
                canonical.display()
            ))
        })?;
        write!(
            output.text,
            "<file name=\"{}\">\n{content}\n</file>\n",
            escape_attribute(&canonical)
        )
        .expect("writing to a String cannot fail");
    }
    Ok(output)
}

fn detect_image(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        Some("image/webp")
    } else {
        None
    }
}

fn escape_attribute(path: &Path) -> String {
    path.to_string_lossy()
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn first_message_merges_and_followups_remain_separate() {
        let prepared = prepare(
            InputArguments {
                files: Vec::new(),
                messages: vec!["question".to_owned(), "follow up".to_owned()],
            },
            Some("piped\n".to_owned()),
        )
        .await
        .unwrap();
        assert_eq!(prepared.initial.as_deref(), Some("piped\nquestion"));
        assert_eq!(prepared.follow_ups, ["follow up"]);
    }

    #[test]
    fn recognizes_supported_image_signatures() {
        assert_eq!(detect_image(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(detect_image(b"\xff\xd8\xffrest"), Some("image/jpeg"));
        assert_eq!(detect_image(b"GIF89arest"), Some("image/gif"));
        assert_eq!(detect_image(b"RIFFxxxxWEBPrest"), Some("image/webp"));
        assert_eq!(detect_image(b"plain text"), None);
    }

    #[test]
    fn escapes_file_name_attributes() {
        assert_eq!(
            escape_attribute(Path::new("a&\"<b>")),
            "a&amp;&quot;&lt;b&gt;"
        );
    }
}
