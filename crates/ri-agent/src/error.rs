//! Typed failures produced by the agent runtime.

use thiserror::Error;

/// Failures that prevent an agent run from being started or completed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AgentError {
    /// Another prompt or continuation is already active.
    #[error("agent is already processing a run")]
    Busy,
    /// A continuation was requested without any transcript messages.
    #[error("cannot continue: no messages in context")]
    EmptyContinuation,
    /// A continuation was requested from an assistant message.
    #[error("cannot continue from an assistant message")]
    AssistantContinuation,
    /// No explicit or process-wide stream function was configured.
    #[error("no stream function configured; pass one explicitly or call set_default_stream_fn")]
    MissingStreamFunction,
    /// Projection or context transformation failed.
    #[error("message projection failed: {0}")]
    Projection(String),
    /// A provider-neutral assistant stream failed.
    #[error(transparent)]
    Ai(#[from] ri_ai::AiError),
    /// An agent event consumer disappeared before the run completed.
    #[error("agent event stream closed before the run completed")]
    EventStreamClosed,
    /// A user-supplied callback failed.
    #[error("agent callback failed: {0}")]
    Callback(String),
}

/// Failures returned while preparing or executing a tool.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ToolError {
    /// Tool-specific execution failure.
    #[error("{0}")]
    Message(String),
    /// Typed argument deserialization failed.
    #[error("invalid typed tool arguments: {0}")]
    Arguments(String),
    /// A typed tool schema could not be represented as JSON.
    #[error("tool schema serialization failed: {0}")]
    Schema(String),
}

impl ToolError {
    /// Creates a tool failure from displayable text.
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}
