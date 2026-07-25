//! Error types shared by the coding tools and execution environments.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

/// Failures produced by an [`ExecutionEnv`](crate::ExecutionEnv).
#[derive(Debug, Error)]
pub enum EnvError {
    /// A local or remote I/O operation failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The operation was cancelled.
    #[error("operation cancelled")]
    Cancelled,
    /// The operation exceeded its configured timeout.
    #[error("operation timed out after {0:?}")]
    TimedOut(Duration),
    /// The requested executable could not be resolved.
    #[error("executable not found: {0}")]
    ExecutableNotFound(String),
    /// The environment does not support an operation.
    #[error("execution environment does not support {0}")]
    Unsupported(&'static str),
    /// An environment-specific failure that has no more precise representation.
    #[error("{0}")]
    Other(String),
}

/// Failures produced by a built-in coding tool.
#[derive(Debug, Error)]
pub enum ToolError {
    /// An execution-environment operation failed.
    #[error(transparent)]
    Environment(#[from] EnvError),
    /// A filesystem operation failed at a known path.
    #[error("{operation} failed for {path}: {source}")]
    Io {
        /// Short name of the attempted operation.
        operation: &'static str,
        /// Target path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// Tool input was invalid.
    #[error("{0}")]
    InvalidInput(String),
    /// A path did not exist.
    #[error("path not found: {0}")]
    PathNotFound(PathBuf),
    /// A path was expected to be a directory.
    #[error("not a directory: {0}")]
    NotDirectory(PathBuf),
    /// Text input was not valid UTF-8.
    #[error("{0} is not valid UTF-8")]
    InvalidUtf8(PathBuf),
    /// A regular expression was invalid.
    #[error("invalid regular expression: {0}")]
    InvalidRegex(String),
    /// A glob expression was invalid.
    #[error("error parsing glob: {0}")]
    InvalidGlob(String),
    /// An edit could not be applied safely.
    #[error("{0}")]
    Edit(String),
    /// A shell command returned a non-zero status.
    #[error("{output}\n\nCommand exited with code {code}")]
    CommandFailed {
        /// Process exit code.
        code: i32,
        /// Captured, possibly truncated output.
        output: String,
    },
    /// A shell command was cancelled.
    #[error("{output}\n\nCommand aborted")]
    CommandCancelled {
        /// Captured, possibly truncated output.
        output: String,
    },
    /// A shell command timed out.
    #[error("{output}\n\nCommand timed out after {seconds} seconds")]
    CommandTimedOut {
        /// Configured timeout in seconds.
        seconds: f64,
        /// Captured, possibly truncated output.
        output: String,
    },
}

impl ToolError {
    /// Attach path and operation context to an I/O error.
    pub(crate) fn io(operation: &'static str, path: impl Into<PathBuf>, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.into(),
            source,
        }
    }
}
