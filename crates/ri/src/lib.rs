//! Ri is the compact facade for the native Pi-compatible agent SDK.
//!
//! Applications normally construct an [`Agent`], explicitly inject its tools
//! and resources, and subscribe to typed streaming events. [`SessionBuilder`]
//! remains available when storage and lifecycle components must be supplied
//! directly. Lower-level crates are namespaced below.

pub use ri_macros::{extension, tool};
pub use ri_sdk::{
    Agent, AgentBuilder, AgentEvent, AgentEvents, ApiKey, BuiltinProvider, CompactionResult,
    CompactionSettings, ExpandedResource, ExtensionRuntime, FrontendMode, HarnessEvent,
    HarnessObserver, InvalidApiKey, ModelRuntime, NavigateOptions, NavigateResult, PromptOptions,
    PromptOutcome, PromptSource, PromptTemplate, QueueMode, RequestOptions, ResourceExpansion,
    ResourceRuntime, Resources, RetryPolicy, SessionBuilder, SessionFrontend, SessionRuntime,
    Skill, StreamingBehavior, Url, local_tools,
};

/// Low-level agent loop and tool contracts.
pub use ri_agent as agent;
/// Provider-neutral AI types and APIs.
pub use ri_ai as ai;
/// Native extension contracts and resource loading.
pub use ri_ext as ext;
/// Durable high-level orchestration.
pub use ri_harness as harness;
/// High-level SDK builders and runtime.
pub use ri_sdk as sdk;
/// Append-only session model.
pub use ri_session as session;
/// Built-in coding tools.
pub use ri_tools as tools;

/// Dependencies used by procedural macro expansions.
#[doc(hidden)]
pub mod __private {
    pub use async_trait::async_trait;
}
