//! Session error types.

use std::path::PathBuf;

/// Result type used by the session subsystem.
pub type Result<T> = std::result::Result<T, Error>;

/// A typed failure produced by session models or repositories.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A requested session or entry does not exist.
    #[error("{0}")]
    NotFound(String),

    /// A session header, tree, or materialized state is inconsistent.
    #[error("invalid session: {0}")]
    InvalidSession(String),

    /// An individual append-only entry is invalid.
    #[error("invalid session entry: {0}")]
    InvalidEntry(String),

    /// A fork target cannot be used with the requested fork position.
    #[error("invalid fork target: {0}")]
    InvalidForkTarget(String),

    /// A unique session or entry identifier already exists.
    #[error("session conflict: {0}")]
    Conflict(String),

    /// A filesystem operation failed.
    #[error("session I/O failed for {path}: {source}")]
    Io {
        /// Path associated with the operation.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// JSON serialization or deserialization failed.
    #[error("session JSON failed: {0}")]
    Json(#[from] serde_json::Error),

    /// A storage backend failed while preserving session state.
    #[error("session storage failed: {0}")]
    Storage(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
