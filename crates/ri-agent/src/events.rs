//! Agent and tool lifecycle events.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolResult;

/// Stable event discriminator useful for compact traces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEventKind {
    /// A run began.
    AgentStart,
    /// A run emitted its final event.
    AgentEnd,
    /// An assistant turn began.
    TurnStart,
    /// An assistant turn ended.
    TurnEnd,
    /// A transcript message began.
    MessageStart,
    /// A streaming assistant snapshot changed.
    MessageUpdate,
    /// A transcript message completed.
    MessageEnd,
    /// Tool preflight began.
    ToolExecutionStart,
    /// A tool emitted a partial result.
    ToolExecutionUpdate,
    /// A tool result was finalized.
    ToolExecutionEnd,
}

/// Events emitted by low-level loops and stateful agents.
#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent<M> {
    /// The run has started.
    AgentStart,
    /// The final event of a run.
    AgentEnd {
        /// Messages produced by this loop invocation.
        messages: Vec<M>,
    },
    /// A provider turn has started.
    TurnStart,
    /// One assistant response and its tool batch have completed.
    TurnEnd {
        /// Final assistant message, wrapped as an application message.
        message: M,
        /// Tool result messages in assistant source order.
        tool_results: Vec<ri_ai::ToolResultMessage>,
    },
    /// A transcript message has started.
    MessageStart {
        /// Initial message snapshot.
        message: M,
    },
    /// A streamed assistant message changed.
    MessageUpdate {
        /// Current complete assistant snapshot.
        message: M,
        /// Incremental provider-neutral stream event.
        assistant_event: ri_ai::AssistantMessageEvent,
    },
    /// A transcript message has completed.
    MessageEnd {
        /// Final message.
        message: M,
    },
    /// Tool preflight has started.
    ToolExecutionStart {
        /// Provider tool-call id.
        tool_call_id: String,
        /// Requested tool name.
        tool_name: String,
        /// Raw, pre-rewrite arguments.
        arguments: Value,
    },
    /// A running tool emitted a partial result.
    ToolExecutionUpdate {
        /// Provider tool-call id.
        tool_call_id: String,
        /// Requested tool name.
        tool_name: String,
        /// Raw, pre-rewrite arguments.
        arguments: Value,
        /// Partial tool result.
        partial_result: ToolResult,
    },
    /// A tool call was finalized.
    ToolExecutionEnd {
        /// Provider tool-call id.
        tool_call_id: String,
        /// Requested tool name.
        tool_name: String,
        /// Final tool result.
        result: ToolResult,
        /// Whether the invocation is represented as an error.
        is_error: bool,
    },
}

impl<M> AgentEvent<M> {
    /// Returns the stable discriminator for this event.
    pub const fn kind(&self) -> AgentEventKind {
        match self {
            Self::AgentStart => AgentEventKind::AgentStart,
            Self::AgentEnd { .. } => AgentEventKind::AgentEnd,
            Self::TurnStart => AgentEventKind::TurnStart,
            Self::TurnEnd { .. } => AgentEventKind::TurnEnd,
            Self::MessageStart { .. } => AgentEventKind::MessageStart,
            Self::MessageUpdate { .. } => AgentEventKind::MessageUpdate,
            Self::MessageEnd { .. } => AgentEventKind::MessageEnd,
            Self::ToolExecutionStart { .. } => AgentEventKind::ToolExecutionStart,
            Self::ToolExecutionUpdate { .. } => AgentEventKind::ToolExecutionUpdate,
            Self::ToolExecutionEnd { .. } => AgentEventKind::ToolExecutionEnd,
        }
    }
}
