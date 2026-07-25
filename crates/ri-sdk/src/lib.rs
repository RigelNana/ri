//! User-facing construction and shared runtime facade.
//!
//! [`Agent`] is the concise, strongly typed base for ordinary applications. It
//! starts empty and accepts explicit tools, skills, prompt templates, context,
//! and streaming subscribers. [`SessionBuilder`] exposes the underlying model,
//! repository, hooks, and harness composition for advanced frontends.

mod builder;
mod builtin_tools;
mod coding_agent;
mod error;
mod extension_runtime;
mod model_runtime;
mod resource_runtime;
mod runtime;

pub use builder::SessionBuilder;
pub use builtin_tools::local_tools;
pub use coding_agent::{
    Agent, AgentBuilder, AgentEvent, AgentEvents, ApiKey, BuiltinProvider, InvalidApiKey,
};
pub use error::{Error, Result};
pub use extension_runtime::ExtensionRuntime;
pub use model_runtime::{ModelRuntime, resolve_model_auth};
pub use resource_runtime::ResourceRuntime;
pub use runtime::{FrontendMode, SessionFrontend, SessionRuntime};
pub use url::Url;

pub use ri_harness::{
    CompactionResult, CompactionSettings, ExpandedResource, HarnessEvent, HarnessObserver,
    NavigateOptions, NavigateResult, PromptOptions, PromptOutcome, PromptSource, PromptTemplate,
    QueueMode, RequestOptions, ResourceExpansion, Resources, RetryPolicy, Skill, StreamingBehavior,
};
