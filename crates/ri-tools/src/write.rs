//! UTF-8 file writing tool.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::error::{EnvError, ToolError};
use crate::mutation::with_file_mutation;
use crate::paths::resolve_path;

/// Input for [`write`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteInput {
    /// Relative or absolute destination.
    pub path: PathBuf,
    /// UTF-8 content that replaces the destination.
    pub content: String,
}

/// Structured write metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WriteDetails {
    /// Number of UTF-8 bytes written.
    #[serde(rename = "bytesWritten")]
    pub bytes_written: usize,
}

/// Result of the write tool.
pub type WriteResult = ToolResult<WriteDetails>;

/// Create or replace a UTF-8 file, creating missing parent directories.
///
/// # Errors
///
/// Returns an error for cancellation, parent creation failures, write failures,
/// or canonicalization failures.
pub async fn write(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: WriteInput,
    cancellation: &CancellationToken,
) -> Result<WriteResult, ToolError> {
    let absolute = resolve_path(&input.path, cwd);
    let bytes_written = input.content.len();
    let parent = absolute
        .parent()
        .map_or_else(|| cwd.to_owned(), Path::to_owned);
    let content = input.content.into_bytes();

    with_file_mutation(env, &absolute, || async {
        check_cancelled(cancellation)?;
        env.create_dir_all(&parent).await?;
        check_cancelled(cancellation)?;
        env.write_file(&absolute, &content).await?;
        // Keep the canonical-path lock until an in-flight write has settled.
        check_cancelled(cancellation)?;
        Ok(())
    })
    .await?;

    Ok(WriteResult::text(
        format!(
            "Successfully wrote {bytes_written} bytes to {}",
            input.path.display()
        ),
        Some(WriteDetails { bytes_written }),
    ))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), EnvError> {
    if cancellation.is_cancelled() {
        Err(EnvError::Cancelled)
    } else {
        Ok(())
    }
}
