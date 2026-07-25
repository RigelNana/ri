//! Stable errors exposed by the high-level runtime.

use std::time::Duration;

use thiserror::Error;

/// Result type used by the harness.
pub type Result<T> = std::result::Result<T, Error>;

/// Stable high-level error classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    /// Another structural operation is active.
    Busy,
    /// A public operation is invalid for the current lifecycle phase.
    InvalidState,
    /// Caller-supplied data is invalid.
    InvalidArgument,
    /// The selected model is unavailable or unauthenticated.
    Model,
    /// Session persistence or projection failed.
    Session,
    /// A runtime hook failed.
    Hook,
    /// The low-level agent runtime failed.
    Agent,
    /// Context compaction failed.
    Compaction,
    /// Branch summarization or navigation failed.
    BranchSummary,
    /// The operation was cancelled.
    Aborted,
}

/// Error returned by the unified harness.
#[derive(Debug, Error)]
pub enum Error {
    /// Another structural operation is active.
    #[error("harness is busy in phase {phase}")]
    Busy {
        /// Human-readable active phase.
        phase: &'static str,
    },
    /// The operation is invalid for the current state.
    #[error("invalid harness state: {0}")]
    InvalidState(String),
    /// An argument failed validation.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Model lookup or authentication failed.
    #[error("model runtime error: {0}")]
    Model(String),
    /// Session persistence or projection failed.
    #[error("session error: {0}")]
    Session(String),
    /// A hook or observer failed.
    #[error("hook error: {0}")]
    Hook(String),
    /// The low-level agent runtime failed.
    #[error("agent runtime error: {0}")]
    Agent(String),
    /// Context compaction failed.
    #[error("compaction error: {0}")]
    Compaction(String),
    /// Branch summarization failed.
    #[error("branch summary error: {0}")]
    BranchSummary(String),
    /// Cancellation was requested.
    #[error("operation aborted")]
    Aborted,
}

impl Error {
    /// Stable category for programmatic handling.
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Busy { .. } => ErrorCode::Busy,
            Self::InvalidState(_) => ErrorCode::InvalidState,
            Self::InvalidArgument(_) => ErrorCode::InvalidArgument,
            Self::Model(_) => ErrorCode::Model,
            Self::Session(_) => ErrorCode::Session,
            Self::Hook(_) => ErrorCode::Hook,
            Self::Agent(_) => ErrorCode::Agent,
            Self::Compaction(_) => ErrorCode::Compaction,
            Self::BranchSummary(_) => ErrorCode::BranchSummary,
            Self::Aborted => ErrorCode::Aborted,
        }
    }
}

impl From<ri_session::Error> for Error {
    fn from(value: ri_session::Error) -> Self {
        Self::Session(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Session(value.to_string())
    }
}

/// Failure returned by the narrow low-level runtime adapter.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct BackendError {
    /// Retry and recovery classification.
    pub kind: BackendErrorKind,
    /// Redacted human-readable failure.
    pub message: String,
    /// Provider-requested minimum delay, when present.
    pub retry_after: Option<Duration>,
}

impl BackendError {
    /// Creates a backend failure.
    pub fn new(kind: BackendErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retry_after: None,
        }
    }

    /// Attaches a provider-requested retry delay.
    #[must_use]
    pub const fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    /// Whether retrying the same operation may succeed.
    pub const fn is_retryable(&self) -> bool {
        matches!(self.kind, BackendErrorKind::Transient)
    }
}

/// Runtime failure classification used by retry and overflow recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendErrorKind {
    /// Rate limit, overload, timeout, or other transient failure.
    Transient,
    /// Provider explicitly rejected an oversized context.
    ContextOverflow,
    /// Authentication or model configuration is invalid.
    Model,
    /// Caller cancellation.
    Aborted,
    /// Non-retryable runtime failure.
    Fatal,
}
