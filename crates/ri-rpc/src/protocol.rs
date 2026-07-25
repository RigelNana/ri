//! Typed commands, responses, events, and extension-UI messages.

use std::fmt;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::types::{
    AgentMessage, BashResult, CompactionReason, CompactionResult, ImageContent, JsonObject, Model,
    QueueMode, SessionEntry, SessionState, SessionStats, SessionTreeNode, SlashCommand,
    StreamingBehavior, ThinkingLevel, ToolCall, ToolExecutionResult,
};

/// Caller-selected request identifier.
///
/// Pi treats this as an opaque string and echoes even an empty string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(String);

impl RequestId {
    /// Create an opaque request identifier.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Require a non-empty identifier for protocols that mandate one.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidRequestId`] when the supplied string is empty.
    pub fn try_non_empty(value: impl Into<String>) -> Result<Self, InvalidRequestId> {
        let id = Self::new(value);
        if id.0.is_empty() {
            Err(InvalidRequestId)
        } else {
            Ok(id)
        }
    }

    /// Borrow the wire value.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the identifier.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for RequestId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RequestId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self)
    }
}

/// Error returned for an empty request identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("request id must not be empty")]
pub struct InvalidRequestId;

/// Non-empty identifier used by the extension-UI sub-protocol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UiRequestId(RequestId);

impl UiRequestId {
    /// Create an extension-UI request identifier.
    pub fn new(value: impl Into<String>) -> Self {
        Self(RequestId::new(value))
    }

    /// Require a non-empty extension-UI identifier.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidRequestId`] when the supplied string is empty.
    pub fn try_non_empty(value: impl Into<String>) -> Result<Self, InvalidRequestId> {
        RequestId::try_non_empty(value).map(Self)
    }

    /// Borrow the wire value.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for UiRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ImageMarker {
    #[serde(rename = "image")]
    Image,
}

/// Image object used by prompt, steer, and follow-up commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandImage {
    #[serde(rename = "type")]
    marker: ImageMarker,
    /// Base64-encoded bytes.
    pub data: String,
    /// Image media type.
    pub mime_type: String,
}

impl CommandImage {
    /// Construct a command image.
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            marker: ImageMarker::Image,
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }
}

impl From<ImageContent> for CommandImage {
    fn from(value: ImageContent) -> Self {
        Self::new(value.data, value.mime_type)
    }
}

impl From<CommandImage> for ImageContent {
    fn from(value: CommandImage) -> Self {
        Self::new(value.data, value.mime_type)
    }
}

/// Every command accepted by RPC mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Command {
    /// Submit a user prompt.
    Prompt {
        /// Prompt text.
        message: String,
        /// Optional inline images.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<CommandImage>,
        /// Required queue behavior when a run is active.
        #[serde(skip_serializing_if = "Option::is_none")]
        streaming_behavior: Option<StreamingBehavior>,
    },
    /// Queue a steering message.
    Steer {
        /// Message text.
        message: String,
        /// Optional inline images.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<CommandImage>,
    },
    /// Queue a follow-up message.
    FollowUp {
        /// Message text.
        message: String,
        /// Optional inline images.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<CommandImage>,
    },
    /// Abort the active agent operation.
    Abort,
    /// Start a fresh session.
    NewSession {
        /// Optional lineage path.
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_session: Option<String>,
    },
    /// Read current state.
    GetState,
    /// Select a model.
    SetModel {
        /// Provider identifier.
        provider: String,
        /// Model identifier.
        model_id: String,
    },
    /// Cycle to the next configured model.
    CycleModel,
    /// List available models.
    GetAvailableModels,
    /// Select a reasoning level.
    SetThinkingLevel {
        /// New level.
        level: ThinkingLevel,
    },
    /// Cycle to the next reasoning level.
    CycleThinkingLevel,
    /// List reasoning levels available for the active model.
    GetAvailableThinkingLevels,
    /// Configure steering delivery.
    SetSteeringMode {
        /// Delivery mode.
        mode: QueueMode,
    },
    /// Configure follow-up delivery.
    SetFollowUpMode {
        /// Delivery mode.
        mode: QueueMode,
    },
    /// Compact current context.
    Compact {
        /// Optional summarization instructions.
        #[serde(skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    /// Enable or disable automatic compaction.
    SetAutoCompaction {
        /// New setting.
        enabled: bool,
    },
    /// Enable or disable automatic retry.
    SetAutoRetry {
        /// New setting.
        enabled: bool,
    },
    /// Abort a retry delay.
    AbortRetry,
    /// Execute a direct shell command.
    Bash {
        /// Command text.
        command: String,
        /// Omit the result from future model context.
        #[serde(default, skip_serializing_if = "crate::types::is_false")]
        exclude_from_context: bool,
    },
    /// Abort a direct shell command.
    AbortBash,
    /// Read aggregate session statistics.
    GetSessionStats,
    /// Export the current session as HTML.
    ExportHtml {
        /// Optional destination path.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_path: Option<String>,
    },
    /// Switch to another session file.
    SwitchSession {
        /// Session JSONL path.
        session_path: String,
    },
    /// Fork from a user message.
    Fork {
        /// Entry identifier.
        entry_id: String,
    },
    /// Clone the current active branch.
    Clone,
    /// List messages eligible as fork points.
    GetForkMessages,
    /// Read append-order entries.
    GetEntries {
        /// Return only entries strictly after this identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        since: Option<String>,
    },
    /// Read the complete session tree.
    GetTree,
    /// Read the latest assistant text.
    GetLastAssistantText,
    /// Set the current session display name.
    SetSessionName {
        /// Nonblank display name.
        name: String,
    },
    /// Read projected conversation messages.
    GetMessages,
    /// List commands invokable through `prompt`.
    GetCommands,
}

impl Command {
    /// Return this command's stable wire name.
    pub const fn name(&self) -> CommandName {
        match self {
            Self::Prompt { .. } => CommandName::Prompt,
            Self::Steer { .. } => CommandName::Steer,
            Self::FollowUp { .. } => CommandName::FollowUp,
            Self::Abort => CommandName::Abort,
            Self::NewSession { .. } => CommandName::NewSession,
            Self::GetState => CommandName::GetState,
            Self::SetModel { .. } => CommandName::SetModel,
            Self::CycleModel => CommandName::CycleModel,
            Self::GetAvailableModels => CommandName::GetAvailableModels,
            Self::SetThinkingLevel { .. } => CommandName::SetThinkingLevel,
            Self::CycleThinkingLevel => CommandName::CycleThinkingLevel,
            Self::GetAvailableThinkingLevels => CommandName::GetAvailableThinkingLevels,
            Self::SetSteeringMode { .. } => CommandName::SetSteeringMode,
            Self::SetFollowUpMode { .. } => CommandName::SetFollowUpMode,
            Self::Compact { .. } => CommandName::Compact,
            Self::SetAutoCompaction { .. } => CommandName::SetAutoCompaction,
            Self::SetAutoRetry { .. } => CommandName::SetAutoRetry,
            Self::AbortRetry => CommandName::AbortRetry,
            Self::Bash { .. } => CommandName::Bash,
            Self::AbortBash => CommandName::AbortBash,
            Self::GetSessionStats => CommandName::GetSessionStats,
            Self::ExportHtml { .. } => CommandName::ExportHtml,
            Self::SwitchSession { .. } => CommandName::SwitchSession,
            Self::Fork { .. } => CommandName::Fork,
            Self::Clone => CommandName::Clone,
            Self::GetForkMessages => CommandName::GetForkMessages,
            Self::GetEntries { .. } => CommandName::GetEntries,
            Self::GetTree => CommandName::GetTree,
            Self::GetLastAssistantText => CommandName::GetLastAssistantText,
            Self::SetSessionName { .. } => CommandName::SetSessionName,
            Self::GetMessages => CommandName::GetMessages,
            Self::GetCommands => CommandName::GetCommands,
        }
    }
}

/// Closed set of command wire names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandName {
    /// `prompt`.
    Prompt,
    /// `steer`.
    Steer,
    /// `follow_up`.
    FollowUp,
    /// `abort`.
    Abort,
    /// `new_session`.
    NewSession,
    /// `get_state`.
    GetState,
    /// `set_model`.
    SetModel,
    /// `cycle_model`.
    CycleModel,
    /// `get_available_models`.
    GetAvailableModels,
    /// `set_thinking_level`.
    SetThinkingLevel,
    /// `cycle_thinking_level`.
    CycleThinkingLevel,
    /// `get_available_thinking_levels`.
    GetAvailableThinkingLevels,
    /// `set_steering_mode`.
    SetSteeringMode,
    /// `set_follow_up_mode`.
    SetFollowUpMode,
    /// `compact`.
    Compact,
    /// `set_auto_compaction`.
    SetAutoCompaction,
    /// `set_auto_retry`.
    SetAutoRetry,
    /// `abort_retry`.
    AbortRetry,
    /// `bash`.
    Bash,
    /// `abort_bash`.
    AbortBash,
    /// `get_session_stats`.
    GetSessionStats,
    /// `export_html`.
    ExportHtml,
    /// `switch_session`.
    SwitchSession,
    /// `fork`.
    Fork,
    /// `clone`.
    Clone,
    /// `get_fork_messages`.
    GetForkMessages,
    /// `get_entries`.
    GetEntries,
    /// `get_tree`.
    GetTree,
    /// `get_last_assistant_text`.
    GetLastAssistantText,
    /// `set_session_name`.
    SetSessionName,
    /// `get_messages`.
    GetMessages,
    /// `get_commands`.
    GetCommands,
}

impl CommandName {
    /// Return the JSON spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Steer => "steer",
            Self::FollowUp => "follow_up",
            Self::Abort => "abort",
            Self::NewSession => "new_session",
            Self::GetState => "get_state",
            Self::SetModel => "set_model",
            Self::CycleModel => "cycle_model",
            Self::GetAvailableModels => "get_available_models",
            Self::SetThinkingLevel => "set_thinking_level",
            Self::CycleThinkingLevel => "cycle_thinking_level",
            Self::GetAvailableThinkingLevels => "get_available_thinking_levels",
            Self::SetSteeringMode => "set_steering_mode",
            Self::SetFollowUpMode => "set_follow_up_mode",
            Self::Compact => "compact",
            Self::SetAutoCompaction => "set_auto_compaction",
            Self::SetAutoRetry => "set_auto_retry",
            Self::AbortRetry => "abort_retry",
            Self::Bash => "bash",
            Self::AbortBash => "abort_bash",
            Self::GetSessionStats => "get_session_stats",
            Self::ExportHtml => "export_html",
            Self::SwitchSession => "switch_session",
            Self::Fork => "fork",
            Self::Clone => "clone",
            Self::GetForkMessages => "get_fork_messages",
            Self::GetEntries => "get_entries",
            Self::GetTree => "get_tree",
            Self::GetLastAssistantText => "get_last_assistant_text",
            Self::SetSessionName => "set_session_name",
            Self::GetMessages => "get_messages",
            Self::GetCommands => "get_commands",
        }
    }
}

impl fmt::Display for CommandName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A command and its optional response-correlation identifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// Correlation identifier copied to exactly one response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    /// Typed command body.
    #[serde(flatten)]
    pub command: Command,
}

impl Request {
    /// Construct an uncorrelated request.
    pub const fn notification(command: Command) -> Self {
        Self { id: None, command }
    }

    /// Construct a correlated request.
    pub const fn correlated(id: RequestId, command: Command) -> Self {
        Self {
            id: Some(id),
            command,
        }
    }
}

/// Whether a session-switching operation was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelledData {
    /// Cancellation state.
    pub cancelled: bool,
}

/// Result of cycling models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CycleModelData {
    /// Newly selected model.
    pub model: Model,
    /// Newly selected reasoning level.
    pub thinking_level: ThinkingLevel,
    /// Whether cycling is constrained by a scoped model list.
    pub is_scoped: bool,
}

/// Result of cycling reasoning levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingLevelData {
    /// Newly selected level.
    pub level: ThinkingLevel,
}

/// A user message that can be selected as a fork point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForkMessage {
    /// Session entry identifier.
    pub entry_id: String,
    /// User-visible message text.
    pub text: String,
}

/// Successful fork result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkData {
    /// Selected message text.
    pub text: String,
    /// Whether an extension cancelled the operation.
    pub cancelled: bool,
}

/// Successful payloads, one variant for every command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "command",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ResponsePayload {
    /// Prompt accepted, queued, or handled.
    Prompt,
    /// Steering message queued.
    Steer,
    /// Follow-up message queued.
    FollowUp,
    /// Abort acknowledged.
    Abort,
    /// New-session result.
    NewSession {
        /// Operation data.
        data: CancelledData,
    },
    /// Current state.
    GetState {
        /// Operation data.
        data: SessionState,
    },
    /// Selected model.
    SetModel {
        /// Operation data.
        data: Model,
    },
    /// Cycled model, or null when cycling is unavailable.
    CycleModel {
        /// Operation data.
        data: Option<CycleModelData>,
    },
    /// Available models.
    GetAvailableModels {
        /// Operation data.
        data: AvailableModelsData,
    },
    /// Reasoning level set.
    SetThinkingLevel,
    /// Cycled level, or null when unsupported.
    CycleThinkingLevel {
        /// Operation data.
        data: Option<ThinkingLevelData>,
    },
    /// Available reasoning levels.
    GetAvailableThinkingLevels {
        /// Operation data.
        data: AvailableThinkingLevelsData,
    },
    /// Steering mode set.
    SetSteeringMode,
    /// Follow-up mode set.
    SetFollowUpMode,
    /// Compaction result.
    Compact {
        /// Operation data.
        data: CompactionResult,
    },
    /// Automatic compaction setting updated.
    SetAutoCompaction,
    /// Automatic retry setting updated.
    SetAutoRetry,
    /// Retry aborted.
    AbortRetry,
    /// Direct shell result.
    Bash {
        /// Operation data.
        data: BashResult,
    },
    /// Direct shell abort acknowledged.
    AbortBash,
    /// Session statistics.
    GetSessionStats {
        /// Operation data.
        data: SessionStats,
    },
    /// HTML export destination.
    ExportHtml {
        /// Operation data.
        data: ExportHtmlData,
    },
    /// Session switch result.
    SwitchSession {
        /// Operation data.
        data: CancelledData,
    },
    /// Fork result.
    Fork {
        /// Operation data.
        data: ForkData,
    },
    /// Clone result.
    Clone {
        /// Operation data.
        data: CancelledData,
    },
    /// Forkable messages.
    GetForkMessages {
        /// Operation data.
        data: ForkMessagesData,
    },
    /// Append-order entries.
    GetEntries {
        /// Operation data.
        data: EntriesData,
    },
    /// Session tree.
    GetTree {
        /// Operation data.
        data: TreeData,
    },
    /// Latest assistant text.
    GetLastAssistantText {
        /// Operation data.
        data: LastAssistantTextData,
    },
    /// Session name set.
    SetSessionName,
    /// Projected conversation messages.
    GetMessages {
        /// Operation data.
        data: MessagesData,
    },
    /// Invokable commands.
    GetCommands {
        /// Operation data.
        data: CommandsData,
    },
}

impl ResponsePayload {
    /// Return the corresponding command name.
    pub const fn command(&self) -> CommandName {
        match self {
            Self::Prompt => CommandName::Prompt,
            Self::Steer => CommandName::Steer,
            Self::FollowUp => CommandName::FollowUp,
            Self::Abort => CommandName::Abort,
            Self::NewSession { .. } => CommandName::NewSession,
            Self::GetState { .. } => CommandName::GetState,
            Self::SetModel { .. } => CommandName::SetModel,
            Self::CycleModel { .. } => CommandName::CycleModel,
            Self::GetAvailableModels { .. } => CommandName::GetAvailableModels,
            Self::SetThinkingLevel => CommandName::SetThinkingLevel,
            Self::CycleThinkingLevel { .. } => CommandName::CycleThinkingLevel,
            Self::GetAvailableThinkingLevels { .. } => CommandName::GetAvailableThinkingLevels,
            Self::SetSteeringMode => CommandName::SetSteeringMode,
            Self::SetFollowUpMode => CommandName::SetFollowUpMode,
            Self::Compact { .. } => CommandName::Compact,
            Self::SetAutoCompaction => CommandName::SetAutoCompaction,
            Self::SetAutoRetry => CommandName::SetAutoRetry,
            Self::AbortRetry => CommandName::AbortRetry,
            Self::Bash { .. } => CommandName::Bash,
            Self::AbortBash => CommandName::AbortBash,
            Self::GetSessionStats { .. } => CommandName::GetSessionStats,
            Self::ExportHtml { .. } => CommandName::ExportHtml,
            Self::SwitchSession { .. } => CommandName::SwitchSession,
            Self::Fork { .. } => CommandName::Fork,
            Self::Clone { .. } => CommandName::Clone,
            Self::GetForkMessages { .. } => CommandName::GetForkMessages,
            Self::GetEntries { .. } => CommandName::GetEntries,
            Self::GetTree { .. } => CommandName::GetTree,
            Self::GetLastAssistantText { .. } => CommandName::GetLastAssistantText,
            Self::SetSessionName => CommandName::SetSessionName,
            Self::GetMessages { .. } => CommandName::GetMessages,
            Self::GetCommands { .. } => CommandName::GetCommands,
        }
    }
}

/// Models result wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvailableModelsData {
    /// Available models.
    pub models: Vec<Model>,
}

/// Reasoning-level result wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableThinkingLevelsData {
    /// Supported levels.
    pub levels: Vec<ThinkingLevel>,
}

/// HTML-export result wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportHtmlData {
    /// Exported path.
    pub path: String,
}

/// Fork-message result wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkMessagesData {
    /// Forkable messages.
    pub messages: Vec<ForkMessage>,
}

/// Entry-query result wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntriesData {
    /// Append-order entries.
    pub entries: Vec<SessionEntry>,
    /// Current leaf identifier.
    pub leaf_id: Option<String>,
}

/// Tree-query result wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeData {
    /// Session roots.
    pub tree: Vec<SessionTreeNode>,
    /// Current leaf identifier.
    pub leaf_id: Option<String>,
}

/// Last-assistant-text result wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastAssistantTextData {
    /// Last text, or null when absent.
    pub text: Option<String>,
}

/// Message-query result wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessagesData {
    /// Current projected messages.
    pub messages: Vec<AgentMessage>,
}

/// Slash-command result wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandsData {
    /// Available commands.
    pub commands: Vec<SlashCommand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ResponseMarker {
    #[serde(rename = "response")]
    Response,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrueMarker;

impl Serialize for TrueMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for TrueMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if bool::deserialize(deserializer)? {
            Ok(Self)
        } else {
            Err(D::Error::custom("expected success: true"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FalseMarker;

impl Serialize for FalseMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(false)
    }
}

impl<'de> Deserialize<'de> for FalseMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if bool::deserialize(deserializer)? {
            Err(D::Error::custom("expected success: false"))
        } else {
            Ok(Self)
        }
    }
}

/// A successful command response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuccessResponse {
    /// Correlated request identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    #[serde(rename = "type")]
    marker: ResponseMarker,
    success: TrueMarker,
    /// Command-specific successful payload.
    #[serde(flatten)]
    pub payload: ResponsePayload,
}

impl SuccessResponse {
    /// Construct a successful response.
    pub const fn new(id: Option<RequestId>, payload: ResponsePayload) -> Self {
        Self {
            id,
            marker: ResponseMarker::Response,
            success: TrueMarker,
            payload,
        }
    }
}

/// A failed command response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Correlated request identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    #[serde(rename = "type")]
    marker: ResponseMarker,
    /// Known or unknown command name.
    pub command: String,
    success: FalseMarker,
    /// Human-readable failure.
    pub error: String,
}

impl ErrorResponse {
    /// Construct an error response.
    pub fn new(
        id: Option<RequestId>,
        command: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            id,
            marker: ResponseMarker::Response,
            command: command.into(),
            success: FalseMarker,
            error: error.into(),
        }
    }
}

/// Successful or failed command response.
// Boxing either public wire variant would change construction and matching APIs.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Response {
    /// Successful response.
    Success(SuccessResponse),
    /// Failed response.
    Error(ErrorResponse),
}

impl Response {
    /// Construct a successful response.
    pub const fn success(id: Option<RequestId>, payload: ResponsePayload) -> Self {
        Self::Success(SuccessResponse::new(id, payload))
    }

    /// Construct a failed response.
    pub fn error(
        id: Option<RequestId>,
        command: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::Error(ErrorResponse::new(id, command, error))
    }

    /// Return the correlated request identifier.
    pub fn request_id(&self) -> Option<&RequestId> {
        match self {
            Self::Success(response) => response.id.as_ref(),
            Self::Error(response) => response.id.as_ref(),
        }
    }
}

/// Streaming assistant-message delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum AssistantMessageEvent {
    /// Generation started.
    Start {
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Text block started.
    TextStart {
        /// Content-array index.
        content_index: usize,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Text delta.
    TextDelta {
        /// Content-array index.
        content_index: usize,
        /// New text.
        delta: String,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Text block ended.
    TextEnd {
        /// Content-array index.
        content_index: usize,
        /// Complete block text.
        content: String,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Reasoning block started.
    ThinkingStart {
        /// Content-array index.
        content_index: usize,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Reasoning delta.
    ThinkingDelta {
        /// Content-array index.
        content_index: usize,
        /// New reasoning text.
        delta: String,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Reasoning block ended.
    ThinkingEnd {
        /// Content-array index.
        content_index: usize,
        /// Complete reasoning text.
        content: String,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Tool call started.
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        /// Content-array index.
        content_index: usize,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Tool-call argument delta.
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        /// Content-array index.
        content_index: usize,
        /// New JSON text.
        delta: String,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Tool call ended.
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        /// Content-array index.
        content_index: usize,
        /// Complete tool call.
        tool_call: ToolCall,
        /// Partial assistant message.
        partial: AgentMessage,
    },
    /// Generation completed successfully.
    Done {
        /// Successful stop reason.
        reason: SuccessfulStopReason,
        /// Final assistant message.
        message: AgentMessage,
    },
    /// Generation failed or was aborted.
    Error {
        /// Failure reason.
        reason: FailedStopReason,
        /// Final error assistant message.
        error: AgentMessage,
    },
}

/// Successful assistant-stream terminal reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuccessfulStopReason {
    /// Natural stop.
    #[serde(rename = "stop")]
    Stop,
    /// Output length limit.
    #[serde(rename = "length")]
    Length,
    /// Tool use.
    #[serde(rename = "toolUse")]
    ToolUse,
}

/// Failed assistant-stream terminal reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FailedStopReason {
    /// User cancellation.
    Aborted,
    /// Provider or runtime error.
    Error,
}

/// Summarization operation that is being retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SummarizationSource {
    /// Branch-summary generation.
    #[serde(rename = "branchSummary")]
    BranchSummary,
    /// Context compaction.
    #[serde(rename = "compaction")]
    Compaction,
}

/// Full event stream emitted by an RPC session.
// Boxing public event fields would change the typed Pi wire DTO API.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum Event {
    /// Agent run started.
    AgentStart,
    /// One low-level agent run ended.
    AgentEnd {
        /// Messages generated by the run.
        messages: Vec<AgentMessage>,
        /// Whether automatic retry will follow.
        #[serde(default)]
        will_retry: bool,
    },
    /// The complete session-level run settled.
    AgentSettled,
    /// Assistant turn started.
    TurnStart,
    /// Assistant turn ended.
    TurnEnd {
        /// Assistant message.
        message: AgentMessage,
        /// Tool results produced by the turn.
        tool_results: Vec<AgentMessage>,
    },
    /// Message started.
    MessageStart {
        /// Message snapshot.
        message: AgentMessage,
    },
    /// Assistant message updated.
    MessageUpdate {
        /// Partial message.
        message: AgentMessage,
        /// Fine-grained stream delta.
        assistant_message_event: AssistantMessageEvent,
    },
    /// Message ended.
    MessageEnd {
        /// Final message.
        message: AgentMessage,
    },
    /// Tool execution started.
    ToolExecutionStart {
        /// Tool-call identifier.
        tool_call_id: String,
        /// Tool name.
        tool_name: String,
        /// Tool-defined arguments.
        args: JsonObject,
    },
    /// Tool execution progress.
    ToolExecutionUpdate {
        /// Tool-call identifier.
        tool_call_id: String,
        /// Tool name.
        tool_name: String,
        /// Tool-defined arguments.
        args: JsonObject,
        /// Accumulated result.
        partial_result: ToolExecutionResult,
    },
    /// Tool execution ended.
    ToolExecutionEnd {
        /// Tool-call identifier.
        tool_call_id: String,
        /// Tool name.
        tool_name: String,
        /// Final result.
        result: ToolExecutionResult,
        /// Whether execution failed.
        is_error: bool,
    },
    /// Direct RPC bash output.
    BashExecutionUpdate {
        /// Originating command identifier.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<RequestId>,
        /// New output text.
        delta: String,
    },
    /// Queued messages changed.
    QueueUpdate {
        /// Pending steering messages.
        steering: Vec<String>,
        /// Pending follow-up messages.
        follow_up: Vec<String>,
    },
    /// Compaction started.
    CompactionStart {
        /// Trigger.
        reason: CompactionReason,
    },
    /// Compaction ended.
    CompactionEnd {
        /// Trigger.
        reason: CompactionReason,
        /// Successful result.
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<CompactionResult>,
        /// Whether it was aborted.
        aborted: bool,
        /// Whether an overflowed turn will retry.
        will_retry: bool,
        /// Failure text.
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
    },
    /// Assistant retry delay started.
    AutoRetryStart {
        /// One-based attempt.
        attempt: u32,
        /// Maximum attempts.
        max_attempts: u32,
        /// Delay before this attempt.
        delay_ms: u64,
        /// Triggering failure.
        error_message: String,
    },
    /// Assistant retry loop ended.
    AutoRetryEnd {
        /// Whether retry recovered.
        success: bool,
        /// Last attempt.
        attempt: u32,
        /// Final failure.
        #[serde(skip_serializing_if = "Option::is_none")]
        final_error: Option<String>,
    },
    /// Summarization retry scheduled.
    SummarizationRetryScheduled {
        /// One-based attempt.
        attempt: u32,
        /// Maximum attempts.
        max_attempts: u32,
        /// Delay before this attempt.
        delay_ms: u64,
        /// Triggering failure.
        error_message: String,
    },
    /// Summarization retry attempt started.
    SummarizationRetryAttemptStart {
        /// Operation kind.
        source: SummarizationSource,
        /// Compaction trigger, only for compaction.
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<CompactionReason>,
    },
    /// Summarization retry loop ended.
    SummarizationRetryFinished,
    /// A session entry was durably appended.
    EntryAppended {
        /// Appended entry.
        entry: SessionEntry,
    },
    /// Session display name changed.
    SessionInfoChanged {
        /// New name; absent means cleared.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Active reasoning level changed.
    ThinkingLevelChanged {
        /// New level.
        level: ThinkingLevel,
    },
    /// Extension callback failed.
    ExtensionError {
        /// Extension source path.
        extension_path: String,
        /// Extension event name.
        event: String,
        /// Error text.
        error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ExtensionUiRequestMarker {
    #[serde(rename = "extension_ui_request")]
    Request,
}

/// Notification severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NotifyType {
    /// Informational notification.
    Info,
    /// Warning notification.
    Warning,
    /// Error notification.
    Error,
}

/// Widget placement around the editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WidgetPlacement {
    /// Above the editor.
    #[serde(rename = "aboveEditor")]
    AboveEditor,
    /// Below the editor.
    #[serde(rename = "belowEditor")]
    BelowEditor,
}

/// Typed extension-UI action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all_fields = "camelCase")]
pub enum ExtensionUiAction {
    /// Choose one string.
    #[serde(rename = "select")]
    Select {
        /// Dialog title.
        title: String,
        /// Candidate strings.
        options: Vec<String>,
        /// Agent-side timeout in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout: Option<u64>,
    },
    /// Confirm or reject.
    #[serde(rename = "confirm")]
    Confirm {
        /// Dialog title.
        title: String,
        /// Dialog body.
        message: String,
        /// Agent-side timeout in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout: Option<u64>,
    },
    /// Enter one line.
    #[serde(rename = "input")]
    Input {
        /// Dialog title.
        title: String,
        /// Placeholder text.
        #[serde(skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
        /// Agent-side timeout in milliseconds.
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout: Option<u64>,
    },
    /// Edit multiline text.
    #[serde(rename = "editor")]
    Editor {
        /// Dialog title.
        title: String,
        /// Initial content.
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill: Option<String>,
    },
    /// Display a notification.
    #[serde(rename = "notify")]
    Notify {
        /// Notification text.
        message: String,
        /// Severity; omitted means informational.
        #[serde(skip_serializing_if = "Option::is_none")]
        notify_type: Option<NotifyType>,
    },
    /// Set or clear a status item.
    #[serde(rename = "setStatus")]
    SetStatus {
        /// Extension-owned key.
        status_key: String,
        /// New text; absent clears it.
        #[serde(skip_serializing_if = "Option::is_none")]
        status_text: Option<String>,
    },
    /// Set or clear a widget.
    #[serde(rename = "setWidget")]
    SetWidget {
        /// Extension-owned key.
        widget_key: String,
        /// New lines; absent clears it.
        #[serde(skip_serializing_if = "Option::is_none")]
        widget_lines: Option<Vec<String>>,
        /// Placement; omitted means above the editor.
        #[serde(skip_serializing_if = "Option::is_none")]
        widget_placement: Option<WidgetPlacement>,
    },
    /// Set terminal title.
    #[serde(rename = "setTitle")]
    SetTitle {
        /// New title.
        title: String,
    },
    /// Replace editor text.
    #[serde(rename = "set_editor_text")]
    SetEditorText {
        /// New text.
        text: String,
    },
}

impl ExtensionUiAction {
    /// Whether this action requires a client response.
    pub const fn expects_response(&self) -> bool {
        matches!(
            self,
            Self::Select { .. } | Self::Confirm { .. } | Self::Input { .. } | Self::Editor { .. }
        )
    }

    /// Agent-side timeout declared by this action.
    pub const fn timeout_ms(&self) -> Option<u64> {
        match self {
            Self::Select { timeout, .. }
            | Self::Confirm { timeout, .. }
            | Self::Input { timeout, .. } => *timeout,
            _ => None,
        }
    }
}

/// Request from an extension to an RPC client UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionUiRequest {
    #[serde(rename = "type")]
    marker: ExtensionUiRequestMarker,
    /// Unique sub-protocol identifier.
    pub id: UiRequestId,
    /// Requested UI operation.
    #[serde(flatten)]
    pub action: ExtensionUiAction,
}

impl ExtensionUiRequest {
    /// Construct a request.
    pub const fn new(id: UiRequestId, action: ExtensionUiAction) -> Self {
        Self {
            marker: ExtensionUiRequestMarker::Request,
            id,
            action,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ExtensionUiResponseMarker {
    #[serde(rename = "extension_ui_response")]
    Response,
}

/// Singleton marker serialized as JSON boolean `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelledMarker;

impl Serialize for CancelledMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for CancelledMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if bool::deserialize(deserializer)? {
            Ok(Self)
        } else {
            Err(D::Error::custom("expected cancelled: true"))
        }
    }
}

/// Dialog result sent by an RPC client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtensionUiResult {
    /// Selected or entered string.
    Value {
        /// Returned value.
        value: String,
    },
    /// Confirmation state.
    Confirmation {
        /// Whether the user confirmed.
        confirmed: bool,
    },
    /// Dialog cancellation.
    Cancelled {
        /// Always true on the wire.
        cancelled: CancelledMarker,
    },
}

impl ExtensionUiResult {
    /// Construct a cancellation result.
    pub const fn cancelled() -> Self {
        Self::Cancelled {
            cancelled: CancelledMarker,
        }
    }
}

/// Response to an extension-UI dialog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionUiResponse {
    #[serde(rename = "type")]
    marker: ExtensionUiResponseMarker,
    /// Matching UI request identifier.
    pub id: UiRequestId,
    /// Typed dialog result.
    #[serde(flatten)]
    pub result: ExtensionUiResult,
}

impl ExtensionUiResponse {
    /// Construct a response.
    pub const fn new(id: UiRequestId, result: ExtensionUiResult) -> Self {
        Self {
            marker: ExtensionUiResponseMarker::Response,
            id,
            result,
        }
    }
}

/// A client record that could not be interpreted as a valid command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidClientMessage {
    /// Correlation identifier recovered before validation failed.
    pub id: Option<RequestId>,
    /// Recovered command name, or `parse`.
    pub command: String,
    /// Error returned to the client.
    pub error: String,
}

/// Any stdin record accepted by an RPC server.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientFrame {
    /// Normal command request.
    Request(Request),
    /// Extension dialog response.
    ExtensionUiResponse(ExtensionUiResponse),
    /// Syntactically valid JSON with an invalid command shape.
    Invalid(InvalidClientMessage),
}

impl Serialize for ClientFrame {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Request(request) => request.serialize(serializer),
            Self::ExtensionUiResponse(response) => response.serialize(serializer),
            Self::Invalid(_) => Err(serde::ser::Error::custom(
                "invalid client messages cannot be serialized",
            )),
        }
    }
}

impl<'de> Deserialize<'de> for ClientFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| D::Error::custom("RPC record must be a JSON object"))?;
        let kind = object.get("type").and_then(Value::as_str);

        if kind == Some("extension_ui_response") {
            return serde_json::from_value(value)
                .map(Self::ExtensionUiResponse)
                .map_err(D::Error::custom);
        }

        let id = object.get("id").and_then(Value::as_str).map(RequestId::new);
        let Some(command) = kind.map(str::to_owned) else {
            return Ok(Self::Invalid(InvalidClientMessage {
                id,
                command: "parse".to_owned(),
                error: "Failed to parse command: missing string field `type`".to_owned(),
            }));
        };

        let known = serde_json::from_value::<CommandName>(Value::String(command.clone())).is_ok();
        if !known {
            return Ok(Self::Invalid(InvalidClientMessage {
                id,
                command: command.clone(),
                error: format!("Unknown command: {command}"),
            }));
        }

        match serde_json::from_value::<Request>(value) {
            Ok(request) => Ok(Self::Request(request)),
            Err(error) => Ok(Self::Invalid(InvalidClientMessage {
                id,
                command,
                error: format!("Failed to parse command: {error}"),
            })),
        }
    }
}

/// Any stdout record emitted by an RPC server.
// Boxing public frame variants would change construction and matching APIs.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ServerFrame {
    /// Command response.
    Response(Response),
    /// Agent session event.
    Event(Event),
    /// Extension UI request.
    ExtensionUiRequest(ExtensionUiRequest),
}
