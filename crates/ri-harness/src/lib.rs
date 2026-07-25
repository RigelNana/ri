//! Durable high-level orchestration shared by every Ri frontend.
//!
//! [`Harness`] is the sole owner of prompt preprocessing, immutable provider
//! turn snapshots, save-point persistence, queue continuation, retry,
//! compaction, branch navigation, session replacement, and settlement.
//! Concrete provider, agent-loop, and extension implementations enter through
//! the narrow traits in [`backend`].

pub mod agent_backend;
pub mod backend;
pub mod compaction;
pub mod error;
mod harness;
pub mod projection;
pub mod prompt;
pub mod types;

pub use agent_backend::{AgentBackend, AgentBackendHooks, ModelAccess};
pub use backend::{
    BeforeAgentStart, BeforeAgentStartResult, BeforeCompactionResult, BeforeNavigation,
    BeforeNavigationResult, BranchSummaryOverride, CompactionOverride, HarnessBackend,
    HarnessHooks, HarnessObserver, HookContext, InputAction, InputEvent,
};
pub use compaction::{
    BranchPreparation, CompactionPreparation, CompactionResult, CompactionSettings,
    ContextEstimate, CutPoint, FileLists, FileOperations, append_file_lists, branch_request_text,
    collect_abandoned_branch, combine_usage, compaction_request_text, context_tokens,
    estimate_context_tokens, estimate_message_tokens, find_cut_point, prepare_branch,
    prepare_compaction, serialize_conversation, should_compact,
};
pub use error::{BackendError, BackendErrorKind, Error, ErrorCode, Result};
pub use harness::Harness;
pub use projection::{assistant_text, message_value, project_session, project_values, user_text};
pub use prompt::{
    ExpandedResource, ResourceExpansion, expand_resources, format_skill, format_template,
};
pub use types::{
    CompactionReason, HarnessConfig, HarnessEvent, HarnessStatus, NavigateOptions, NavigateResult,
    Phase, PromptOptions, PromptOutcome, PromptSource, PromptTemplate, QueueLengths, QueueMode,
    RequestOptions, Resources, RetryOperation, RetryPolicy, SessionWrite, Skill, StreamingBehavior,
    SummaryKind, SummaryRequest, SummaryResponse, TurnOutput, TurnRequest, TurnSnapshot,
};
