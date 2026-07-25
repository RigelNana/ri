//! Public configuration, snapshots, events, and operation results.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

pub use ri_agent::QueueMode;
use ri_ai::{
    AssistantMessage, CacheRetention, Context, ImageContent, Message, Model, ThinkingLevel, Tool,
};
pub use ri_protocol_core::CompactionReason;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompt::ExpandedResource;

/// High-level harness phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    /// No structural operation is active.
    #[default]
    Idle,
    /// Prompt preprocessing or an agent turn is active.
    Turn,
    /// Manual, threshold, or overflow compaction is active.
    Compaction,
    /// Branch summarization and tree navigation are active.
    BranchSummary,
    /// An agent or summarization retry is waiting or starting.
    Retry,
    /// The current session is being shut down and replaced.
    ReplacingSession,
    /// Final callbacks and accepted writes are being drained.
    Settling,
}

impl Phase {
    /// Stable phase spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Turn => "turn",
            Self::Compaction => "compaction",
            Self::BranchSummary => "branch_summary",
            Self::Retry => "retry",
            Self::ReplacingSession => "replacing_session",
            Self::Settling => "settling",
        }
    }
}

/// How a prompt submitted during an active run should be delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingBehavior {
    /// Deliver at the next post-turn steering point.
    Steer,
    /// Deliver after the current agent run would otherwise stop.
    FollowUp,
}

/// Origin of a prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptSource {
    /// Interactive editor or terminal input.
    #[default]
    Interactive,
    /// Print-mode input.
    Print,
    /// JSON event-mode input.
    Json,
    /// RPC request input.
    Rpc,
    /// In-process SDK caller.
    Sdk,
    /// Input submitted by an extension action.
    Extension,
}

/// Options applied by the canonical prompt pipeline.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PromptOptions {
    /// Optional image blocks.
    pub images: Vec<ImageContent>,
    /// Input origin.
    pub source: PromptSource,
    /// Required delivery mode when another turn is active.
    pub streaming_behavior: Option<StreamingBehavior>,
    /// Whether slash skills and prompt templates are expanded.
    pub expand_resources: bool,
}

impl PromptOptions {
    /// Options for normal resource-expanding input.
    pub fn interactive() -> Self {
        Self {
            expand_resources: true,
            ..Self::default()
        }
    }
}

/// Result of accepting a prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum PromptOutcome {
    /// An extension command or input hook handled the prompt immediately.
    Handled,
    /// The prompt was queued into an active run.
    Queued(StreamingBehavior),
    /// The prompt ran to a terminal assistant response.
    Completed(Box<AssistantMessage>),
}

/// A loaded skill.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Skill {
    /// Stable invocation name.
    pub name: String,
    /// Model-visible purpose.
    pub description: String,
    /// Full instructions.
    pub content: String,
    /// Source path or resource identifier.
    pub source: String,
    /// Hide from model-visible listings while preserving explicit invocation.
    #[serde(default)]
    pub disable_model_invocation: bool,
}

impl Skill {
    /// Creates an enabled skill from explicit metadata and instructions.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            content: content.into(),
            source: source.into(),
            disable_model_invocation: false,
        }
    }

    /// Hides the skill from the model-visible catalog while preserving direct invocation.
    #[must_use]
    pub const fn hidden_from_model(mut self) -> Self {
        self.disable_model_invocation = true;
        self
    }
}

/// A loaded prompt template.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplate {
    /// Stable invocation name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Template body.
    pub content: String,
    /// Source path or resource identifier.
    pub source: String,
}

impl PromptTemplate {
    /// Creates a prompt template with no description.
    pub fn new(
        name: impl Into<String>,
        content: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            content: content.into(),
            source: source.into(),
        }
    }

    /// Adds a human-readable template description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Immutable resources captured by a turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    /// Available skills.
    pub skills: Arc<[Skill]>,
    /// Available prompt templates.
    pub prompt_templates: Arc<[PromptTemplate]>,
    /// Additional model-visible context fragments.
    pub context: Arc<[String]>,
}

impl Resources {
    /// Creates a shallowly immutable resource snapshot.
    pub fn new(
        skills: impl Into<Vec<Skill>>,
        prompt_templates: impl Into<Vec<PromptTemplate>>,
        context: impl Into<Vec<String>>,
    ) -> Self {
        Self {
            skills: skills.into().into(),
            prompt_templates: prompt_templates.into().into(),
            context: context.into().into(),
        }
    }
}

/// Curated request options snapshotted for each provider turn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestOptions {
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Transport-layer retry count.
    pub transport_retries: Option<u32>,
    /// Maximum transport retry delay.
    pub max_transport_retry_delay: Option<Duration>,
    /// Request header overlay.
    pub headers: BTreeMap<String, String>,
    /// Provider metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Prompt-cache retention preference.
    pub cache_retention: Option<CacheRetention>,
}

/// Agent and summarization retry settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Enable high-level retries.
    pub enabled: bool,
    /// Number of retries after the initial attempt.
    pub max_retries: u32,
    /// Initial backoff.
    pub base_delay: Duration,
    /// Maximum exponential delay.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    // `Duration::from_mins` is newer than the workspace's declared MSRV.
    #[allow(clippy::duration_suboptimal_units)]
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(60),
        }
    }
}

impl RetryPolicy {
    /// Computes a saturating exponential delay for a one-based retry attempt.
    pub fn delay(self, attempt: u32, requested: Option<Duration>) -> Duration {
        let exponent = attempt.saturating_sub(1).min(31);
        let factor = 1_u32 << exponent;
        let exponential = self
            .base_delay
            .checked_mul(factor)
            .unwrap_or(self.max_delay)
            .min(self.max_delay);
        requested.map_or(exponential, |delay| {
            delay.max(exponential).min(self.max_delay)
        })
    }
}

/// Complete mutable runtime configuration. Individual turns receive an immutable
/// [`TurnSnapshot`] derived from this value.
#[derive(Clone, Debug)]
pub struct HarnessConfig {
    /// Selected model.
    pub model: Arc<Model>,
    /// Requested reasoning level.
    pub thinking_level: ThinkingLevel,
    /// Model-visible system prompt.
    pub system_prompt: String,
    /// All registered tool schemas.
    pub tools: Arc<[Tool]>,
    /// Active tool names in presentation order.
    pub active_tool_names: Arc<[String]>,
    /// Loaded resources.
    pub resources: Resources,
    /// Provider request options.
    pub request_options: RequestOptions,
    /// Steering queue drain behavior.
    pub steering_mode: QueueMode,
    /// Follow-up queue drain behavior.
    pub follow_up_mode: QueueMode,
    /// High-level retry policy.
    pub retry: RetryPolicy,
    /// Context compaction settings.
    pub compaction: crate::compaction::CompactionSettings,
}

/// Concrete immutable state used by one provider request.
#[derive(Clone, Debug)]
pub struct TurnSnapshot {
    /// Monotonically increasing session binding generation.
    pub generation: u64,
    /// Session identifier used for provider affinity.
    pub session_id: Arc<str>,
    /// Context projected at this save point.
    pub context: Context,
    /// Selected model.
    pub model: Arc<Model>,
    /// Requested reasoning level.
    pub thinking_level: ThinkingLevel,
    /// All tool schemas known to the runtime.
    pub tools: Arc<[Tool]>,
    /// Active tool names.
    pub active_tool_names: Arc<[String]>,
    /// Resource snapshot.
    pub resources: Resources,
    /// Resolved system prompt.
    pub system_prompt: Arc<str>,
    /// Provider options.
    pub request_options: RequestOptions,
}

/// Input passed to the low-level agent adapter for exactly one save-point turn.
#[derive(Clone, Debug)]
pub struct TurnRequest {
    /// Immutable request state.
    pub snapshot: TurnSnapshot,
    /// Whether this starts a user-initiated run or continues existing context.
    pub continuation: bool,
}

/// Output from one low-level assistant response and its tool-result batch.
#[derive(Clone, Debug)]
pub struct TurnOutput {
    /// New assistant and tool-result messages in source order.
    pub messages: Vec<Message>,
    /// Whether tool execution requires another provider request.
    pub continue_after_tools: bool,
}

impl TurnOutput {
    /// Returns the last assistant message produced by the turn.
    pub fn assistant(&self) -> Option<&AssistantMessage> {
        self.messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant(message) => Some(message),
                Message::User(_) | Message::ToolResult(_) => None,
            })
    }
}

/// A session mutation accepted while an operation is active.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionWrite {
    /// Append a provider-neutral message.
    Message(Message),
    /// Append an application-owned JSON message.
    RawMessage(Value),
    /// Change selected model.
    Model {
        /// Provider id.
        provider: String,
        /// Model id.
        model_id: String,
    },
    /// Change reasoning level.
    Thinking(ThinkingLevel),
    /// Change active tools.
    ActiveTools(Vec<String>),
    /// Persist context-free extension state.
    Custom {
        /// State namespace.
        kind: String,
        /// JSON payload.
        data: Option<Value>,
    },
    /// Persist a model-visible extension message.
    CustomMessage {
        /// Message namespace.
        kind: String,
        /// String or content-block array.
        content: Value,
        /// Whether interactive clients should display it.
        display: bool,
        /// Non-context metadata.
        details: Option<Value>,
    },
}

/// Summary operation kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryKind {
    /// Full historical compaction summary.
    Compaction,
    /// Prefix of a turn split by compaction.
    TurnPrefix,
    /// Abandoned branch summary.
    Branch,
}

/// High-level operation being retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryOperation {
    /// Main agent turn.
    Agent,
    /// Full compaction summary.
    Compaction,
    /// Prefix of a split turn.
    TurnPrefix,
    /// Branch summary.
    BranchSummary,
}

/// One standalone summarization request.
#[derive(Clone, Debug)]
pub struct SummaryRequest {
    /// Summary purpose.
    pub kind: SummaryKind,
    /// Model used for the request.
    pub model: Arc<Model>,
    /// System instruction.
    pub system_prompt: String,
    /// Serialized conversation and summary instruction.
    pub prompt: String,
    /// Maximum generated tokens.
    pub max_tokens: u64,
    /// Reasoning level when supported.
    pub thinking_level: ThinkingLevel,
    /// Fresh request id; summary calls never reuse the conversation affinity id.
    pub request_id: String,
    /// Provider request options with cache writes disabled.
    pub request_options: RequestOptions,
}

/// Successful standalone summary response.
#[derive(Clone, Debug)]
pub struct SummaryResponse {
    /// Generated summary text.
    pub text: String,
    /// Provider usage.
    pub usage: ri_ai::Usage,
}

/// Tree navigation options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NavigateOptions {
    /// Generate a summary of the abandoned branch.
    pub summarize: bool,
    /// Additional summary focus.
    pub custom_instructions: Option<String>,
    /// Replace rather than append the default summary instructions.
    pub replace_instructions: bool,
    /// Optional label applied to the navigation target.
    pub label: Option<String>,
}

/// Tree navigation outcome.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NavigateResult {
    /// A hook cancelled the navigation.
    pub cancelled: bool,
    /// User text restored when navigating before a user message.
    pub editor_text: Option<String>,
    /// Persisted branch-summary entry id.
    pub summary_entry_id: Option<String>,
}

/// Queue sizes exposed in events and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueLengths {
    /// Steering messages.
    pub steer: usize,
    /// Follow-up messages.
    pub follow_up: usize,
    /// Messages held for the next user-initiated turn.
    pub next_turn: usize,
}

/// Observable high-level lifecycle event.
#[derive(Clone, Debug)]
pub enum HarnessEvent {
    /// A slash skill or prompt template was expanded for the model.
    ResourceExpanded {
        /// Selected resource.
        resource: ExpandedResource,
        /// Complete model-visible expansion result.
        text: String,
    },
    /// A prompt was accepted for execution.
    PromptAccepted {
        /// Operation sequence.
        operation: u64,
    },
    /// Queue contents changed.
    QueueUpdated(QueueLengths),
    /// An agent-emitted message became durable.
    MessagePersisted {
        /// Session entry id.
        entry_id: String,
        /// Message role.
        role: &'static str,
    },
    /// Agent messages and accepted pending writes reached a save point.
    SavePoint {
        /// Operation sequence.
        operation: u64,
        /// Whether pending writes were flushed.
        had_pending_writes: bool,
    },
    /// A retry was scheduled.
    RetryScheduled {
        /// Retry operation.
        operation: RetryOperation,
        /// One-based retry number.
        attempt: u32,
        /// Maximum number of retries after the initial attempt.
        max_attempts: u32,
        /// Backoff delay.
        delay: Duration,
        /// Redacted error.
        error: String,
    },
    /// A summarization retry finished waiting and is starting its request.
    RetryAttemptStarted {
        /// Summary request kind.
        kind: SummaryKind,
        /// Compaction trigger when the operation belongs to compaction.
        reason: Option<CompactionReason>,
    },
    /// A retry sequence ended through success, exhaustion, or cancellation.
    RetryFinished {
        /// Retry operation.
        operation: RetryOperation,
        /// Whether a later attempt recovered.
        success: bool,
        /// Last one-based retry number.
        attempt: u32,
        /// Final error for an unsuccessful agent retry.
        final_error: Option<String>,
    },
    /// Compaction started.
    CompactionStarted {
        /// Trigger.
        reason: CompactionReason,
    },
    /// Compaction completed.
    CompactionFinished {
        /// Trigger.
        reason: CompactionReason,
        /// Successful compaction result.
        result: Option<Box<crate::compaction::CompactionResult>>,
        /// Whether cancellation stopped compaction.
        aborted: bool,
        /// Whether an overflow continuation will follow.
        will_retry: bool,
        /// Failure text for a non-cancellation error.
        error_message: Option<String>,
    },
    /// Branch navigation completed.
    BranchNavigated {
        /// Previous leaf.
        old_leaf: Option<String>,
        /// New selected leaf.
        new_leaf: Option<String>,
        /// Optional branch summary entry.
        summary_entry: Option<String>,
    },
    /// The old session is about to be invalidated.
    SessionReplacing {
        /// Old session id.
        old_session_id: String,
    },
    /// A new session generation was bound.
    SessionReplaced {
        /// New session id.
        session_id: String,
        /// Binding generation.
        generation: u64,
    },
    /// All continuations, callbacks, and accepted writes for an operation settled.
    Settled {
        /// Completed operation sequence.
        operation: u64,
        /// Messages retained for a future user turn.
        next_turn: usize,
    },
}

/// Snapshot returned by diagnostics without exposing mutable internals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessStatus {
    /// Current lifecycle phase.
    pub phase: Phase,
    /// Active operation sequence.
    pub operation: Option<u64>,
    /// Last completely settled operation.
    pub settled_operation: u64,
    /// Session binding generation.
    pub generation: u64,
    /// Queue lengths.
    pub queues: QueueLengths,
}
