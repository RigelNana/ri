//! Directory listing tool.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::error::{EnvError, ToolError};
use crate::paths::resolve_path;
use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationOptions, TruncationResult,
    format_size, truncate_head,
};

const DEFAULT_LIMIT: usize = 500;

/// Input for [`ls`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct LsInput {
    /// Directory to list. Defaults to the working directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Entry limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Structured listing metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LsDetails {
    /// Byte truncation details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    /// Entry limit that stopped collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_limit_reached: Option<usize>,
}

/// Result of the listing tool.
pub type LsResult = ToolResult<LsDetails>;

/// List one directory, including dotfiles.
///
/// # Errors
///
/// Returns an error when the target is missing or not a directory, when the
/// operation is cancelled, or when the execution environment fails.
pub async fn ls(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: LsInput,
    cancellation: &CancellationToken,
) -> Result<LsResult, ToolError> {
    if cancellation.is_cancelled() {
        return Err(ToolError::Environment(EnvError::Cancelled));
    }
    let directory = resolve_path(input.path.as_deref().unwrap_or_else(|| Path::new(".")), cwd);
    let metadata = env
        .metadata(&directory)
        .await
        .map_err(|error| match error {
            EnvError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
                ToolError::PathNotFound(directory.clone())
            }
            other => ToolError::Environment(other),
        })?;
    if !metadata.is_dir {
        return Err(ToolError::NotDirectory(directory));
    }

    let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);
    let mut entries = env.read_dir(&directory).await?;
    entries.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.name.cmp(&right.name))
    });

    let mut results = Vec::new();
    let mut limit_reached = false;
    for entry in entries {
        if cancellation.is_cancelled() {
            return Err(ToolError::Environment(EnvError::Cancelled));
        }
        if results.len() >= limit {
            limit_reached = true;
            break;
        }
        let is_dir = match entry.is_dir {
            Some(value) => value,
            None => match env.metadata(&entry.path).await {
                Ok(metadata) => metadata.is_dir,
                Err(_) => continue,
            },
        };
        results.push(if is_dir {
            format!("{}/", entry.name)
        } else {
            entry.name
        });
    }
    if results.is_empty() {
        return Ok(LsResult::text("(empty directory)", None));
    }

    let truncation = truncate_head(&results.join("\n"), TruncationOptions::default());
    let mut output = truncation.content.clone();
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} entries limit reached. Use limit={} for more",
            limit.saturating_mul(2)
        ));
    }
    if truncation.truncated {
        notices.push(match truncation.truncated_by {
            Some(TruncatedBy::Lines) => format!("{DEFAULT_MAX_LINES} line limit reached"),
            _ => format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)),
        });
    }
    if !notices.is_empty() {
        let _ = write!(output, "\n\n[{}]", notices.join(". "));
    }
    Ok(LsResult::text(
        output,
        Some(LsDetails {
            truncation: truncation.truncated.then_some(truncation),
            entry_limit_reached: limit_reached.then_some(limit),
        }),
    ))
}
