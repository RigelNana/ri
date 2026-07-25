//! Event-driven, provider-neutral agent loop and tool scheduler.
//!
//! The crate keeps application messages extensible, projects them to
//! `ri-ai` messages only at provider boundaries, and owns the observable
//! ordering contracts around streaming, tools, queues, cancellation, and
//! awaited event listeners.

mod agent;
mod agent_loop;
mod error;
mod events;
mod message;
mod stream_fn;
mod tool;

#[cfg(test)]
mod tests;

pub use agent::{Agent, AgentEventListener, AgentOptions, AgentState, ListenerId, QueueMode};
pub use agent_loop::{
    AfterToolCallContext, AfterToolCallResult, AgentContext, AgentEventSink, AgentEventStream,
    AgentLoopConfig, AgentLoopTurnUpdate, BeforeToolCallContext, BeforeToolCallResult,
    CompletedTurn, SharedEventSink, agent_loop, agent_loop_continue, run_agent_loop,
    run_agent_loop_continue,
};
pub use error::{AgentError, ToolError};
pub use events::{AgentEvent, AgentEventKind};
pub use message::{AgentMessage, Prompt, StandardAgentMessage};
pub use stream_fn::{StreamFn, StreamOptions, default_stream_fn, set_default_stream_fn};
pub use tool::{FnTool, Tool, ToolCallContext, ToolExecutionMode, ToolResult, ToolUpdateSink};
