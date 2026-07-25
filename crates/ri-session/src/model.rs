//! Serializable session records and repository options.

use std::collections::BTreeMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current native JSONL session format.
pub const CURRENT_SESSION_VERSION: u32 = 3;

fn legacy_version() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum SessionRecordType {
    #[serde(rename = "session")]
    Session,
}

/// The first JSONL record and immutable metadata for one session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    #[serde(rename = "type")]
    record_type: SessionRecordType,
    /// On-disk format version.
    #[serde(default = "legacy_version")]
    pub version: u32,
    /// Stable repository-wide session identifier.
    pub id: String,
    /// Session creation time.
    #[serde(with = "timestamp")]
    pub timestamp: DateTime<Utc>,
    /// Working directory in which the session began.
    pub cwd: String,
    /// Source session identifier or path for a fork.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
    /// Application-owned header metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Value>>,
}

impl SessionHeader {
    /// Create a current-version session header.
    pub fn new(id: impl Into<String>, cwd: impl Into<String>) -> Self {
        Self {
            record_type: SessionRecordType::Session,
            version: CURRENT_SESSION_VERSION,
            id: id.into(),
            timestamp: Utc::now(),
            cwd: cwd.into(),
            parent_session: None,
            metadata: None,
        }
    }

    /// Replace the record version after a successful migration.
    pub(crate) fn make_current(&mut self) {
        self.version = CURRENT_SESSION_VERSION;
    }
}

/// Fields shared by every append-only tree entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryBase {
    /// Stable entry identifier within a session.
    pub id: String,
    /// Parent entry, or `None` for a tree root.
    pub parent_id: Option<String>,
    /// Time at which the entry was appended.
    #[serde(with = "timestamp")]
    pub timestamp: DateTime<Utc>,
}

impl EntryBase {
    /// Construct common entry fields.
    pub fn new(id: impl Into<String>, parent_id: Option<String>) -> Self {
        Self {
            id: id.into(),
            parent_id,
            timestamp: Utc::now(),
        }
    }
}

/// A stored application message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Application message payload.  Its `role` and provider fields remain
    /// application-defined and forward compatible.
    pub message: Value,
}

/// A selected model change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChangeEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Provider identifier.
    pub provider: String,
    /// Provider model identifier.
    pub model_id: String,
}

/// A reasoning-level change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingLevelChangeEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Application-defined reasoning level.
    pub thinking_level: String,
}

/// A change to the set of tools exposed to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveToolsChangeEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Active tool names in presentation order.
    pub active_tool_names: Vec<String>,
}

/// Monetary usage components for an LLM request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    /// Input-token cost.
    #[serde(default)]
    pub input: f64,
    /// Output-token cost.
    #[serde(default)]
    pub output: f64,
    /// Cache-read cost.
    #[serde(default)]
    pub cache_read: f64,
    /// Cache-write cost.
    #[serde(default)]
    pub cache_write: f64,
    /// Total request cost.
    #[serde(default)]
    pub total: f64,
}

/// Token and cost accounting attached to generated output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Uncached input tokens.
    #[serde(default)]
    pub input: u64,
    /// Generated output tokens.
    #[serde(default)]
    pub output: u64,
    /// Tokens read from a provider cache.
    #[serde(default)]
    pub cache_read: u64,
    /// Tokens written to a provider cache.
    #[serde(default)]
    pub cache_write: u64,
    /// Provider-reported total, when present.
    #[serde(default)]
    pub total_tokens: u64,
    /// Monetary accounting.
    #[serde(default)]
    pub cost: UsageCost,
}

/// A checkpoint replacing older context with a summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Human-readable summary of replaced context.
    pub summary: String,
    /// First historical entry retained by legacy compactions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_kept_entry_id: Option<String>,
    /// Context token count before compaction.
    pub tokens_before: u64,
    /// Self-contained retained message tail for current compactions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retained_tail: Option<Vec<Value>>,
    /// Application-specific summary metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage spent creating the summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Whether an extension supplied the summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

/// A summary carried from an abandoned branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Leaf of the branch being summarized.
    pub from_id: String,
    /// Human-readable branch summary.
    pub summary: String,
    /// Application-specific summary metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage spent creating the summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Whether an extension supplied the summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

/// Application state that is persisted but omitted from model context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Namespace used by the owning application or extension.
    pub custom_type: String,
    /// Application-owned state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Application content that participates in model context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessageEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Namespace used by the owning application or extension.
    pub custom_type: String,
    /// String or content-block array.
    pub content: Value,
    /// Whether a UI should render the message.
    pub display: bool,
    /// Application-owned metadata not sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// A label update for another entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Entry receiving or losing the label.
    pub target_id: String,
    /// New label; `None` or a blank string clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Session-level display metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Display name; `None` or blank clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Durable navigation marker for the active tree leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeafEntry {
    /// Common tree fields.
    #[serde(flatten)]
    pub base: EntryBase,
    /// Selected entry, or `None` for the position before all roots.
    pub target_id: Option<String>,
}

/// Every typed append-only session entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    /// Application message.
    Message(MessageEntry),
    /// Selected model changed.
    ModelChange(ModelChangeEntry),
    /// Reasoning level changed.
    ThinkingLevelChange(ThinkingLevelChangeEntry),
    /// Active tools changed.
    ActiveToolsChange(ActiveToolsChangeEntry),
    /// Context compaction checkpoint.
    Compaction(CompactionEntry),
    /// Summary of an abandoned branch.
    BranchSummary(BranchSummaryEntry),
    /// Context-free application state.
    Custom(CustomEntry),
    /// Application message included in context.
    CustomMessage(CustomMessageEntry),
    /// Label update.
    Label(LabelEntry),
    /// Session display metadata.
    SessionInfo(SessionInfoEntry),
    /// Durable active-leaf update.
    Leaf(LeafEntry),
}

impl SessionEntry {
    /// Common fields for this entry.
    pub fn base(&self) -> &EntryBase {
        match self {
            Self::Message(entry) => &entry.base,
            Self::ModelChange(entry) => &entry.base,
            Self::ThinkingLevelChange(entry) => &entry.base,
            Self::ActiveToolsChange(entry) => &entry.base,
            Self::Compaction(entry) => &entry.base,
            Self::BranchSummary(entry) => &entry.base,
            Self::Custom(entry) => &entry.base,
            Self::CustomMessage(entry) => &entry.base,
            Self::Label(entry) => &entry.base,
            Self::SessionInfo(entry) => &entry.base,
            Self::Leaf(entry) => &entry.base,
        }
    }

    /// Mutable common fields, primarily useful to compatibility migrations.
    pub fn base_mut(&mut self) -> &mut EntryBase {
        match self {
            Self::Message(entry) => &mut entry.base,
            Self::ModelChange(entry) => &mut entry.base,
            Self::ThinkingLevelChange(entry) => &mut entry.base,
            Self::ActiveToolsChange(entry) => &mut entry.base,
            Self::Compaction(entry) => &mut entry.base,
            Self::BranchSummary(entry) => &mut entry.base,
            Self::Custom(entry) => &mut entry.base,
            Self::CustomMessage(entry) => &mut entry.base,
            Self::Label(entry) => &mut entry.base,
            Self::SessionInfo(entry) => &mut entry.base,
            Self::Leaf(entry) => &mut entry.base,
        }
    }

    /// Stable snake-case type name used by storage backends.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Message(_) => "message",
            Self::ModelChange(_) => "model_change",
            Self::ThinkingLevelChange(_) => "thinking_level_change",
            Self::ActiveToolsChange(_) => "active_tools_change",
            Self::Compaction(_) => "compaction",
            Self::BranchSummary(_) => "branch_summary",
            Self::Custom(_) => "custom",
            Self::CustomMessage(_) => "custom_message",
            Self::Label(_) => "label",
            Self::SessionInfo(_) => "session_info",
            Self::Leaf(_) => "leaf",
        }
    }

    /// Entry identifier.
    pub fn id(&self) -> &str {
        &self.base().id
    }

    /// Parent identifier.
    pub fn parent_id(&self) -> Option<&str> {
        self.base().parent_id.as_deref()
    }
}

/// An entry paired with its backend-assigned append sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SequencedEntry {
    /// Strictly increasing sequence within the session.
    pub sequence: u64,
    /// Typed immutable entry.
    pub entry: SessionEntry,
}

/// Model selected by the active branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSelection {
    /// Provider identifier.
    pub provider: String,
    /// Provider model identifier.
    pub model_id: String,
}

/// Context projected from one active session branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionContext {
    /// Messages to send to the model, in order.
    pub messages: Vec<Value>,
    /// Most recent reasoning level on the full branch.
    pub thinking_level: String,
    /// Most recent explicit or assistant-message model.
    pub model: Option<ModelSelection>,
    /// Most recent active-tool selection.
    pub active_tool_names: Option<Vec<String>>,
}

/// Aggregate accounting over committed session entries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    /// Number of stored message entries.
    pub message_count: u64,
    /// Cache-read tokens.
    pub cached_tokens: u64,
    /// Uncached input plus cache-write tokens.
    pub uncached_tokens: u64,
    /// Input, output, cache-read, and cache-write tokens.
    pub total_tokens: u64,
    /// Total monetary cost.
    pub cost_total: f64,
}

/// Backend-neutral metadata returned by repository listing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMetadata {
    /// Session identifier.
    pub id: String,
    /// Creation time.
    #[serde(with = "timestamp")]
    pub created_at: DateTime<Utc>,
    /// Working directory.
    pub cwd: String,
    /// Parent session identifier or source path.
    pub parent_session: Option<String>,
    /// Application-owned metadata.
    pub metadata: Option<BTreeMap<String, Value>>,
    /// Backend file or database path, when meaningful.
    pub path: Option<PathBuf>,
}

impl SessionMetadata {
    /// Build metadata from a header and optional backend path.
    pub fn from_header(header: &SessionHeader, path: Option<PathBuf>) -> Self {
        Self {
            id: header.id.clone(),
            created_at: header.timestamp,
            cwd: header.cwd.clone(),
            parent_session: header.parent_session.clone(),
            metadata: header.metadata.clone(),
            path,
        }
    }
}

/// Options for creating an empty session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CreateOptions {
    /// Optional caller-selected identifier.
    pub id: Option<String>,
    /// Session working directory.
    pub cwd: String,
    /// Source session identifier or path.
    pub parent_session: Option<String>,
    /// Application-owned header metadata.
    pub metadata: Option<BTreeMap<String, Value>>,
}

impl CreateOptions {
    /// Create options for a working directory.
    pub fn new(cwd: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            ..Self::default()
        }
    }
}

/// Position of a selected entry in a fork.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ForkPosition {
    /// Fork immediately before a user message.
    #[default]
    Before,
    /// Include the selected entry in the fork.
    At,
}

/// Options for copying session history into a new session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ForkOptions {
    /// Optional caller-selected destination identifier.
    pub id: Option<String>,
    /// Destination working directory; the source directory is inherited when absent.
    pub cwd: Option<String>,
    /// Optional entry delimiting the copied path.
    pub entry_id: Option<String>,
    /// Whether the selected entry is included.
    pub position: ForkPosition,
    /// Destination application metadata; source metadata is inherited when absent.
    pub metadata: Option<BTreeMap<String, Value>>,
}

/// Repository listing filter.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ListOptions {
    /// Return only sessions created for this working directory.
    pub cwd: Option<String>,
}

pub(crate) mod timestamp {
    use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
    use serde::{Deserialize, Deserializer, Serializer, de};

    pub(crate) fn serialize<S>(value: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_rfc3339_opts(SecondsFormat::Millis, true))
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum WireTimestamp {
            Text(String),
            Milliseconds(i64),
        }

        match WireTimestamp::deserialize(deserializer)? {
            WireTimestamp::Text(text) => DateTime::parse_from_rfc3339(&text)
                .map(|value| value.with_timezone(&Utc))
                .map_err(de::Error::custom),
            WireTimestamp::Milliseconds(value) => Utc
                .timestamp_millis_opt(value)
                .single()
                .ok_or_else(|| de::Error::custom("timestamp is outside the supported range")),
        }
    }
}
