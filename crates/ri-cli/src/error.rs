//! Typed command-line failures.

use std::io;

/// Result returned by CLI operations.
pub type Result<T> = std::result::Result<T, CliError>;

/// Failures surfaced by the `ri` binary.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// Command-line arguments describe an invalid operation.
    #[error("{0}")]
    InvalidArguments(String),

    /// A persisted setting cannot be decoded or applied.
    #[error("invalid configuration: {message}")]
    InvalidConfig {
        /// Actionable configuration failure.
        message: String,
    },

    /// A mode was requested without its compile-time feature.
    #[error("{mode} mode is unavailable in this build; rebuild ri with the `{feature}` feature")]
    FeatureUnavailable {
        /// Human-readable mode or operation.
        mode: &'static str,
        /// Cargo feature that enables it.
        feature: &'static str,
    },

    /// A prompt-taking mode had no input.
    #[error("no prompt was provided (pass a message argument or pipe text on stdin)")]
    MissingPrompt,

    /// A requested resource could not be found.
    #[error("{kind} `{name}` was not found")]
    NotFound {
        /// Resource kind.
        kind: &'static str,
        /// User-provided identifier.
        name: String,
    },

    /// An operation is unsupported by the selected provider/runtime.
    #[error("{operation} is not supported{detail}")]
    Unsupported {
        /// Operation name.
        operation: &'static str,
        /// Optional explanatory suffix, including punctuation.
        detail: String,
    },

    /// Process I/O failed.
    #[error("failed to {operation}: {source}")]
    Io {
        /// I/O operation.
        operation: &'static str,
        /// Underlying failure.
        #[source]
        source: io::Error,
    },

    /// JSON encoding or decoding failed.
    #[error("invalid JSON while {operation}: {source}")]
    Json {
        /// JSON operation.
        operation: &'static str,
        /// Underlying failure.
        #[source]
        source: serde_json::Error,
    },

    /// Interactive terminal operation failed.
    #[cfg(feature = "interactive")]
    #[error(transparent)]
    Tui(#[from] ri_tui::Error),

    /// RPC transport or server operation failed.
    #[cfg(feature = "rpc")]
    #[error("RPC server failed: {0}")]
    Rpc(String),

    /// A bounded event subscriber could not keep up with the runtime.
    #[error("runtime event stream lost {0} event(s)")]
    EventLagged(u64),

    /// The user interrupted the active operation.
    #[error("operation interrupted")]
    Interrupted,

    /// The shared SDK runtime rejected an operation.
    #[error("{operation} failed: {source}")]
    Runtime {
        /// Runtime operation.
        operation: &'static str,
        /// Concrete SDK/provider/storage error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl CliError {
    /// Wrap a runtime error while preserving its source chain.
    pub fn runtime(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Runtime {
            operation,
            source: Box::new(source),
        }
    }

    /// Create an unsupported-operation error with optional context.
    pub fn unsupported(operation: &'static str, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        Self::Unsupported {
            operation,
            detail: if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            },
        }
    }
}
