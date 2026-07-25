//! Shared, serializable RPC payload types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use ri_protocol_core::{CompactionReason, QueueMode, ThinkingLevel};

/// A JSON object at an explicitly extensible protocol boundary.
pub type JsonObject = Map<String, Value>;

// Serde's `skip_serializing_if` predicate ABI requires a shared reference.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TextMarker {
    #[serde(rename = "text")]
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ImageMarker {
    #[serde(rename = "image")]
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ThinkingMarker {
    #[serde(rename = "thinking")]
    Thinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ToolCallMarker {
    #[serde(rename = "toolCall")]
    ToolCall,
}

/// How a prompt received during streaming is queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingBehavior {
    /// Deliver after the current assistant turn's tool calls.
    #[serde(rename = "steer")]
    Steer,
    /// Deliver after the complete run settles.
    #[serde(rename = "followUp")]
    FollowUp,
}

/// Text supplied to or returned by a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    #[serde(rename = "type")]
    marker: TextMarker,
    /// Text payload.
    pub text: String,
    /// Provider-specific replay signature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    /// Construct a plain text block.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            marker: TextMarker::Text,
            text: text.into(),
            text_signature: None,
        }
    }
}

/// An inline base64 image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    #[serde(rename = "type")]
    marker: ImageMarker,
    /// Base64-encoded image bytes.
    pub data: String,
    /// Image media type.
    pub mime_type: String,
}

impl ImageContent {
    /// Construct an inline image block.
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            marker: ImageMarker::Image,
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

/// A model reasoning block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    #[serde(rename = "type")]
    marker: ThinkingMarker,
    /// Human-readable or encrypted-placeholder reasoning text.
    pub thinking: String,
    /// Provider-specific replay signature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// Whether the block was redacted by the provider.
    #[serde(default, skip_serializing_if = "is_false")]
    pub redacted: bool,
}

impl ThinkingContent {
    /// Construct a reasoning block.
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            marker: ThinkingMarker::Thinking,
            thinking: thinking.into(),
            thinking_signature: None,
            redacted: false,
        }
    }
}

/// A model-requested tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    #[serde(rename = "type")]
    marker: ToolCallMarker,
    /// Provider tool-call identifier.
    pub id: String,
    /// Registered tool name.
    pub name: String,
    /// Tool-defined arguments.
    pub arguments: JsonObject,
    /// Google-compatible thought replay signature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl ToolCall {
    /// Construct a typed tool-call block.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: JsonObject) -> Self {
        Self {
            marker: ToolCallMarker::ToolCall,
            id: id.into(),
            name: name.into(),
            arguments,
            thought_signature: None,
        }
    }
}

/// Content accepted in user and custom messages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    /// Plain text.
    Text(TextContent),
    /// Inline image.
    Image(ImageContent),
}

/// Content returned in an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    /// Plain text.
    Text(TextContent),
    /// Model reasoning.
    Thinking(ThinkingContent),
    /// Tool invocation.
    ToolCall(ToolCall),
}

/// String or block-array message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string form.
    Text(String),
    /// Structured content form.
    Blocks(Vec<InputContent>),
}

/// Cost values in US dollars.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    /// Input-token cost.
    pub input: f64,
    /// Output-token cost.
    pub output: f64,
    /// Cache-read cost.
    pub cache_read: f64,
    /// Cache-write cost.
    pub cache_write: f64,
    /// Total cost.
    pub total: f64,
}

/// Token usage attached to messages and summarization operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Uncached input tokens.
    pub input: u64,
    /// Output tokens, including reasoning tokens when reported.
    pub output: u64,
    /// Cache-read tokens.
    #[serde(default)]
    pub cache_read: u64,
    /// Cache-write tokens.
    #[serde(default)]
    pub cache_write: u64,
    /// One-hour cache writes, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write1h: Option<u64>,
    /// Reasoning-token subset, when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    /// Provider total.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Calculated monetary cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<UsageCost>,
}

/// Why assistant generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    /// Natural stop.
    #[serde(rename = "stop")]
    Stop,
    /// Output limit.
    #[serde(rename = "length")]
    Length,
    /// Tool use.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// Provider or runtime failure.
    #[serde(rename = "error")]
    Error,
    /// User cancellation.
    #[serde(rename = "aborted")]
    Aborted,
}

/// Messages emitted by an agent session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum AgentMessage {
    /// User input.
    #[serde(rename = "user", rename_all = "camelCase")]
    User {
        /// User content.
        content: MessageContent,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Assistant output.
    #[serde(rename = "assistant", rename_all = "camelCase")]
    Assistant {
        /// Assistant content.
        content: Vec<AssistantContent>,
        /// Wire API identifier.
        api: String,
        /// Provider identifier.
        provider: String,
        /// Requested model identifier.
        model: String,
        /// Concrete response model, if different.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_model: Option<String>,
        /// Provider response identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
        /// Redacted provider/runtime diagnostics.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        diagnostics: Vec<Value>,
        /// Token and cost usage.
        usage: Usage,
        /// Completion reason.
        stop_reason: StopReason,
        /// Human-readable error.
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Result of executing a tool.
    #[serde(rename = "toolResult", rename_all = "camelCase")]
    ToolResult {
        /// Correlated tool-call identifier.
        tool_call_id: String,
        /// Tool name.
        tool_name: String,
        /// Text and image result blocks.
        content: Vec<InputContent>,
        /// Tool-defined details.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        /// Nested LLM usage incurred by the tool.
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// Deferred tools made available at this point.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        added_tool_names: Vec<String>,
        /// Whether execution failed.
        is_error: bool,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Output of a direct RPC bash command.
    #[serde(rename = "bashExecution", rename_all = "camelCase")]
    BashExecution {
        /// Executed command.
        command: String,
        /// Combined output.
        output: String,
        /// Process exit code.
        #[serde(skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        /// Whether execution was cancelled.
        cancelled: bool,
        /// Whether inline output was truncated.
        truncated: bool,
        /// Path containing complete output.
        #[serde(skip_serializing_if = "Option::is_none")]
        full_output_path: Option<String>,
        /// Whether to omit this result from model context.
        #[serde(default, skip_serializing_if = "is_false")]
        exclude_from_context: bool,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Extension-injected message.
    #[serde(rename = "custom", rename_all = "camelCase")]
    Custom {
        /// Extension-defined message type.
        custom_type: String,
        /// Message content.
        content: MessageContent,
        /// Whether an interactive UI should display the message.
        display: bool,
        /// Extension-owned details.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Summary of an abandoned branch.
    #[serde(rename = "branchSummary", rename_all = "camelCase")]
    BranchSummary {
        /// Summary text.
        summary: String,
        /// Entry from which navigation departed.
        from_id: String,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Summary replacing compacted context.
    #[serde(rename = "compactionSummary", rename_all = "camelCase")]
    CompactionSummary {
        /// Summary text.
        summary: String,
        /// Estimated tokens before compaction.
        tokens_before: u64,
        /// Unix timestamp in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
}

/// Model input modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelInput {
    /// Text input.
    Text,
    /// Image input.
    Image,
}

/// Complete alternate model rates above an input-token threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// Threshold after which this tier applies.
    pub input_tokens_above: u64,
    /// Input rate per million tokens.
    pub input: f64,
    /// Output rate per million tokens.
    pub output: f64,
    /// Cache-read rate per million tokens.
    pub cache_read: f64,
    /// Cache-write rate per million tokens.
    pub cache_write: f64,
}

/// Model rates per million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    /// Input rate.
    pub input: f64,
    /// Output rate.
    pub output: f64,
    /// Cache-read rate.
    pub cache_read: f64,
    /// Cache-write rate.
    pub cache_write: f64,
    /// Optional request-wide pricing tiers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

/// Full model descriptor carried by RPC responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// Provider model identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Wire API identifier.
    pub api: String,
    /// Provider identifier.
    pub provider: String,
    /// API base URL.
    pub base_url: String,
    /// Whether reasoning is supported.
    pub reasoning: bool,
    /// Supported input modalities.
    pub input: Vec<ModelInput>,
    /// Context-window size.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Token rates.
    pub cost: ModelCost,
    /// Model-specific reasoning-level mapping.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub thinking_level_map: BTreeMap<ThinkingLevel, Option<String>>,
    /// Model-specific HTTP headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Provider/API-specific compatibility settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

/// Current RPC-visible session state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionState {
    /// Active model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
    /// Active reasoning level.
    pub thinking_level: ThinkingLevel,
    /// Whether a run is active.
    pub is_streaming: bool,
    /// Whether compaction is active.
    pub is_compacting: bool,
    /// Steering delivery mode.
    pub steering_mode: QueueMode,
    /// Follow-up delivery mode.
    pub follow_up_mode: QueueMode,
    /// Persisted session path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    /// Session identifier.
    pub session_id: String,
    /// User-visible session name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    /// Whether threshold compaction is enabled.
    pub auto_compaction_enabled: bool,
    /// Current conversation message count.
    pub message_count: usize,
    /// Number of queued messages.
    pub pending_message_count: usize,
}

/// Final result of a direct bash command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashResult {
    /// Combined sanitized output.
    pub output: String,
    /// Exit status, absent when killed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Whether execution was cancelled.
    pub cancelled: bool,
    /// Whether inline output was truncated.
    pub truncated: bool,
    /// File containing full output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

/// Result of a compaction operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionResult {
    /// Generated summary.
    pub summary: String,
    /// First pre-compaction entry retained in context.
    pub first_kept_entry_id: String,
    /// Estimated tokens before compaction.
    pub tokens_before: u64,
    /// Estimated rebuilt-context tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_tokens_after: Option<u64>,
    /// Summarization usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Extension-owned compaction metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// Base fields common to every session tree entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntryBase {
    /// Stable entry identifier.
    pub id: String,
    /// Parent entry, or `None` for a root.
    pub parent_id: Option<String>,
    /// ISO-8601 timestamp.
    pub timestamp: String,
}

/// An append-only Pi-compatible session entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    /// Conversation message.
    #[serde(rename = "message", rename_all = "camelCase")]
    Message {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Stored message.
        message: AgentMessage,
    },
    /// Active model change.
    #[serde(rename = "model_change", rename_all = "camelCase")]
    ModelChange {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Provider identifier.
        provider: String,
        /// Model identifier.
        model_id: String,
    },
    /// Reasoning-level change.
    #[serde(rename = "thinking_level_change", rename_all = "camelCase")]
    ThinkingLevelChange {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// New level. Older sessions may contain nonstandard strings.
        thinking_level: String,
    },
    /// Compaction checkpoint.
    #[serde(rename = "compaction", rename_all = "camelCase")]
    Compaction {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Generated summary.
        summary: String,
        /// Legacy retained-entry pointer.
        #[serde(skip_serializing_if = "Option::is_none")]
        first_kept_entry_id: Option<String>,
        /// Estimated tokens before compaction.
        tokens_before: u64,
        /// Materialized retained context.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        retained_tail: Vec<AgentMessage>,
        /// Summarization usage.
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// Extension-owned metadata.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        /// Whether an extension generated the entry.
        #[serde(default, skip_serializing_if = "is_false")]
        from_hook: bool,
    },
    /// Summary of an abandoned branch.
    #[serde(rename = "branch_summary", rename_all = "camelCase")]
    BranchSummary {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Entry from which navigation departed.
        from_id: String,
        /// Generated summary.
        summary: String,
        /// Summarization usage.
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// Extension-owned metadata.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        /// Whether an extension generated the entry.
        #[serde(default, skip_serializing_if = "is_false")]
        from_hook: bool,
    },
    /// Extension state not included in model context.
    #[serde(rename = "custom", rename_all = "camelCase")]
    Custom {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Extension-defined type.
        custom_type: String,
        /// Extension-owned data.
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    /// Extension message included in model context.
    #[serde(rename = "custom_message", rename_all = "camelCase")]
    CustomMessage {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Extension-defined type.
        custom_type: String,
        /// Injected content.
        content: MessageContent,
        /// Extension-owned details.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        /// Whether a UI should display this entry.
        display: bool,
    },
    /// Label mutation.
    #[serde(rename = "label", rename_all = "camelCase")]
    Label {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// Entry whose label changes.
        target_id: String,
        /// New label; absent clears it.
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// Session metadata mutation.
    #[serde(rename = "session_info", rename_all = "camelCase")]
    SessionInfo {
        /// Stable entry identifier.
        id: String,
        /// Parent entry.
        parent_id: Option<String>,
        /// ISO-8601 timestamp.
        timestamp: String,
        /// New display name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl SessionEntry {
    /// Return the stable entry identifier.
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::ModelChange { id, .. }
            | Self::ThinkingLevelChange { id, .. }
            | Self::Compaction { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::Custom { id, .. }
            | Self::CustomMessage { id, .. }
            | Self::Label { id, .. }
            | Self::SessionInfo { id, .. } => id,
        }
    }

    /// Return the parent identifier.
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::ModelChange { parent_id, .. }
            | Self::ThinkingLevelChange { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::Custom { parent_id, .. }
            | Self::CustomMessage { parent_id, .. }
            | Self::Label { parent_id, .. }
            | Self::SessionInfo { parent_id, .. } => parent_id.as_deref(),
        }
    }
}

/// Session tree node returned by `get_tree`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTreeNode {
    /// Entry at this node.
    pub entry: SessionEntry,
    /// Child nodes.
    pub children: Vec<Self>,
    /// Latest resolved label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Timestamp of the latest label mutation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_timestamp: Option<String>,
}

/// Aggregate token totals for session statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionTokenTotals {
    /// Uncached input.
    pub input: u64,
    /// Output.
    pub output: u64,
    /// Cache reads.
    pub cache_read: u64,
    /// Cache writes.
    pub cache_write: u64,
    /// Sum of all token categories.
    pub total: u64,
}

/// Current context-window use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsage {
    /// Current token estimate, null immediately after compaction.
    pub tokens: Option<u64>,
    /// Model context-window size.
    pub context_window: u64,
    /// Percentage used, null immediately after compaction.
    pub percent: Option<f64>,
}

/// Aggregate statistics for a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    /// Persisted session path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_file: Option<String>,
    /// Session identifier.
    pub session_id: String,
    /// User-message count.
    pub user_messages: usize,
    /// Assistant-message count.
    pub assistant_messages: usize,
    /// Tool-call count.
    pub tool_calls: usize,
    /// Tool-result count.
    pub tool_results: usize,
    /// All message count.
    pub total_messages: usize,
    /// Aggregate tokens.
    pub tokens: SessionTokenTotals,
    /// Aggregate cost.
    pub cost: f64,
    /// Current context use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_usage: Option<ContextUsage>,
}

/// Origin scope for an invokable resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceScope {
    /// User-level resource.
    User,
    /// Project-level resource.
    Project,
    /// Runtime-only resource.
    Temporary,
}

/// How a resource entered the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceOrigin {
    /// Package-owned resource.
    Package,
    /// Explicit top-level resource.
    TopLevel,
}

/// Source metadata for a registered resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    /// Absolute source path.
    pub path: String,
    /// Source identifier.
    pub source: String,
    /// Loading scope.
    pub scope: SourceScope,
    /// Loading origin.
    pub origin: SourceOrigin,
    /// Package base directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_dir: Option<String>,
}

/// Kind of slash command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlashCommandSource {
    /// Registered extension command.
    Extension,
    /// Prompt template.
    Prompt,
    /// Agent skill.
    Skill,
}

/// A command available for invocation through `prompt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlashCommand {
    /// Invocation name without `/`.
    pub name: String,
    /// Optional human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Command kind.
    pub source: SlashCommandSource,
    /// Resource source metadata.
    pub source_info: SourceInfo,
}

/// Final or partial tool execution result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolExecutionResult {
    /// Text and image result content.
    pub content: Vec<InputContent>,
    /// Tool-owned details.
    pub details: Value,
    /// Nested LLM usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Deferred tools loaded by this result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_tool_names: Vec<String>,
    /// Hint to terminate after the current batch.
    #[serde(default, skip_serializing_if = "is_false")]
    pub terminate: bool,
}
