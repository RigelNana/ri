//! Serialized process output with a clean stdout/stderr split.

use std::sync::Arc;

use serde::Serialize;
use tokio::io::{AsyncWriteExt, Stderr, Stdout};
use tokio::sync::Mutex;

use crate::error::{CliError, Result};

/// Cloneable process output handles.
///
/// Runtime diagnostics and failures always go to stderr. Text responses,
/// metadata results, JSON events, and RPC records exclusively use stdout.
#[derive(Clone, Debug)]
pub struct Output {
    stdout: Arc<Mutex<Stdout>>,
    stderr: Arc<Mutex<Stderr>>,
}

impl Default for Output {
    fn default() -> Self {
        Self::stdio()
    }
}

impl Output {
    /// Bind to process stdout and stderr.
    pub fn stdio() -> Self {
        Self {
            stdout: Arc::new(Mutex::new(tokio::io::stdout())),
            stderr: Arc::new(Mutex::new(tokio::io::stderr())),
        }
    }

    /// Write response text exactly as supplied.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if stdout cannot be written or flushed.
    pub async fn stdout(&self, text: &str) -> Result<()> {
        let mut stdout = self.stdout.lock().await;
        stdout
            .write_all(text.as_bytes())
            .await
            .map_err(|source| CliError::Io {
                operation: "write stdout",
                source,
            })?;
        stdout.flush().await.map_err(|source| CliError::Io {
            operation: "flush stdout",
            source,
        })
    }

    /// Write one newline-terminated response line.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if stdout cannot be written or flushed.
    pub async fn stdout_line(&self, text: &str) -> Result<()> {
        self.stdout(&terminated(text)).await
    }

    /// Serialize one compact LF-terminated JSON record.
    ///
    /// # Errors
    ///
    /// Returns a JSON serialization or stdout I/O error.
    pub async fn json<T: Serialize + ?Sized>(&self, value: &T) -> Result<()> {
        let record = json_record(value)?;
        self.stdout(&record).await
    }

    /// Write one diagnostic line to stderr.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if stderr cannot be written or flushed.
    pub async fn stderr_line(&self, text: &str) -> Result<()> {
        let mut stderr = self.stderr.lock().await;
        stderr
            .write_all(terminated(text).as_bytes())
            .await
            .map_err(|source| CliError::Io {
                operation: "write stderr",
                source,
            })?;
        stderr.flush().await.map_err(|source| CliError::Io {
            operation: "flush stderr",
            source,
        })
    }
}

fn terminated(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_owned()
    } else {
        format!("{text}\n")
    }
}

fn json_record<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let mut record = serde_json::to_string(value).map_err(|source| CliError::Json {
        operation: "encoding an output record",
        source,
    })?;
    record.push('\n');
    Ok(record)
}

/// Whether a failure came from a downstream pipe closing normally.
pub fn is_broken_pipe(error: &CliError) -> bool {
    matches!(
        error,
        CliError::Io { source, .. } if source.kind() == std::io::ErrorKind::BrokenPipe
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn text_gets_exactly_one_terminating_newline() {
        assert_eq!(terminated("answer"), "answer\n");
        assert_eq!(terminated("answer\n"), "answer\n");
    }

    #[test]
    fn json_records_are_compact_and_lf_delimited() {
        assert_eq!(
            json_record(&json!({"type": "event", "text": "x"})).unwrap(),
            "{\"type\":\"event\",\"text\":\"x\"}\n"
        );
    }
}
