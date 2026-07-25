//! Provider-neutral conversation and streaming event types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Unix time in milliseconds.
pub type Timestamp = i64;

/// Returns the current Unix timestamp in milliseconds.
pub fn now_millis() -> Timestamp {
    chrono::Utc::now().timestamp_millis()
}

/// A plain text content block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextContent {
    /// UTF-8 text.
    pub text: String,
    /// Provider-owned replay metadata, such as an `OpenAI` message id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_signature: Option<String>,
}

impl TextContent {
    /// Creates an unsigned text block.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            text_signature: None,
        }
    }
}

/// A provider reasoning/thinking content block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingContent {
    /// Human-readable reasoning or reasoning summary.
    pub thinking: String,
    /// Opaque provider metadata needed to replay the reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_signature: Option<String>,
    /// Whether the text was safety-redacted and the signature is opaque data.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub redacted: bool,
}

/// A base64-encoded image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64 data without a data-URL prefix.
    pub data: String,
    /// MIME type, for example `image/png`.
    pub mime_type: String,
}

/// A model-requested tool invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Provider tool-call identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed JSON arguments. During streaming this is a best-effort partial value.
    #[serde(default = "empty_object")]
    pub arguments: Value,
    /// Google thought signature or compatible opaque replay metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

/// Content emitted by an assistant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text.
    #[serde(rename = "text")]
    Text(TextContent),
    /// Reasoning content.
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    /// Tool invocation.
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
}

impl ContentBlock {
    /// Returns this block's text-like payload, if any.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(&block.text),
            Self::Thinking(block) => Some(&block.thinking),
            Self::ToolCall(_) => None,
        }
    }
}

/// Content accepted in user and tool-result messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputContent {
    /// Plain text.
    #[serde(rename = "text")]
    Text(TextContent),
    /// Base64 image.
    #[serde(rename = "image")]
    Image(ImageContent),
}

/// A user message is either a string or an ordered list of text/image blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    /// Compact string representation.
    Text(String),
    /// Multimodal representation.
    Blocks(Vec<InputContent>),
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

/// Monetary cost, in US dollars.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    /// Uncached input cost.
    pub input: f64,
    /// Output cost.
    pub output: f64,
    /// Cache-read cost.
    pub cache_read: f64,
    /// Cache-write cost.
    pub cache_write: f64,
    /// Sum of all cost components.
    pub total: f64,
}

/// Token usage and derived cost.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Uncached input tokens.
    pub input: u64,
    /// Output tokens, including reasoning when the provider reports it that way.
    pub output: u64,
    /// Prompt-cache read tokens.
    pub cache_read: u64,
    /// Prompt-cache write tokens.
    pub cache_write: u64,
    /// Portion of `cache_write` stored at Anthropic's one-hour retention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    /// Reasoning tokens, a subset of output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    /// Provider-reported or normalized total token count.
    pub total_tokens: u64,
    /// Derived monetary cost.
    #[serde(default)]
    pub cost: UsageCost,
}

impl Usage {
    /// Creates usage with a normalized total and zero cost.
    pub fn from_parts(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
            total_tokens: input
                .saturating_add(output)
                .saturating_add(cache_read)
                .saturating_add(cache_write),
            ..Self::default()
        }
    }

    /// Total prompt tokens, including cache reads and writes.
    pub fn prompt_tokens(&self) -> u64 {
        self.input
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }
}

/// Why assistant generation stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    /// Normal model completion.
    #[default]
    #[serde(rename = "stop")]
    Stop,
    /// Output limit reached.
    #[serde(rename = "length")]
    Length,
    /// The model requested one or more tools.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// Provider or protocol failure.
    #[serde(rename = "error")]
    Error,
    /// Caller cancellation.
    #[serde(rename = "aborted")]
    Aborted,
}

impl StopReason {
    /// Whether this is a terminal failure rather than a successful completion.
    pub fn is_error(self) -> bool {
        matches!(self, Self::Error | Self::Aborted)
    }
}

/// User-authored message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    /// User content.
    pub content: UserContent,
    /// Creation time in Unix milliseconds.
    pub timestamp: Timestamp,
}

impl UserMessage {
    /// Creates a user message stamped with the current time.
    pub fn new(content: impl Into<UserContent>) -> Self {
        Self {
            content: content.into(),
            timestamp: now_millis(),
        }
    }
}

/// Model-authored message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    /// Ordered text, thinking, and tool-call blocks.
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    /// Wire API used to produce this message.
    pub api: String,
    /// Provider id.
    pub provider: String,
    /// Requested model id.
    pub model: String,
    /// Concrete routed model, when different from `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    /// Provider response/message identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Redacted diagnostic records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Value>,
    /// Token accounting.
    #[serde(default)]
    pub usage: Usage,
    /// Terminal reason.
    pub stop_reason: StopReason,
    /// Human-readable provider/protocol error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Creation time in Unix milliseconds.
    pub timestamp: Timestamp,
}

impl AssistantMessage {
    /// Creates an empty in-progress message for a model.
    pub fn empty(
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            content: Vec::new(),
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: now_millis(),
        }
    }

    /// Concatenates model-visible text blocks, excluding thinking and tool calls.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.as_str()),
                ContentBlock::Thinking(_) | ContentBlock::ToolCall(_) => None,
            })
            .collect()
    }
}

/// Result returned by an application tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    /// Matching tool-call id.
    pub tool_call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Text and image result blocks.
    #[serde(default)]
    pub content: Vec<InputContent>,
    /// Application-specific details that are not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage incurred by the tool itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Tools made available after this result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_tool_names: Vec<String>,
    /// Whether tool execution failed.
    pub is_error: bool,
    /// Creation time in Unix milliseconds.
    pub timestamp: Timestamp,
}

/// A conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    /// User message.
    #[serde(rename = "user")]
    User(UserMessage),
    /// Assistant message.
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    /// Tool result.
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultMessage),
}

impl Message {
    /// Message timestamp.
    pub fn timestamp(&self) -> Timestamp {
        match self {
            Self::User(message) => message.timestamp,
            Self::Assistant(message) => message.timestamp,
            Self::ToolResult(message) => message.timestamp,
        }
    }
}

/// Conversation input for a model request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    /// Optional provider system/developer instruction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Ordered transcript.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Tools currently known to the client.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<crate::tool::Tool>,
}

/// Event protocol produced by text providers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum AssistantMessageEvent {
    /// Stream accepted and partial state initialized.
    Start {
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A text block began.
    TextStart {
        /// Content block index.
        content_index: usize,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// Text was appended.
    TextDelta {
        /// Content block index.
        content_index: usize,
        /// Append-only text.
        delta: String,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A text block ended.
    TextEnd {
        /// Content block index.
        content_index: usize,
        /// Final block text.
        content: String,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A thinking block began.
    ThinkingStart {
        /// Content block index.
        content_index: usize,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// Thinking text was appended.
    ThinkingDelta {
        /// Content block index.
        content_index: usize,
        /// Append-only text.
        delta: String,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A thinking block ended.
    ThinkingEnd {
        /// Content block index.
        content_index: usize,
        /// Final thinking text.
        content: String,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A tool call began.
    ToolcallStart {
        /// Content block index.
        content_index: usize,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// Tool argument JSON was appended.
    ToolcallDelta {
        /// Content block index.
        content_index: usize,
        /// Append-only JSON.
        delta: String,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// A tool call ended.
    ToolcallEnd {
        /// Content block index.
        content_index: usize,
        /// Final parsed call.
        tool_call: ToolCall,
        /// Current message snapshot.
        partial: AssistantMessage,
    },
    /// Successful terminal event.
    Done {
        /// `stop`, `length`, or `toolUse`.
        reason: StopReason,
        /// Final message.
        message: AssistantMessage,
    },
    /// Failed terminal event.
    Error {
        /// `error` or `aborted`.
        reason: StopReason,
        /// Final partial message.
        error: AssistantMessage,
    },
}

impl AssistantMessageEvent {
    /// Returns the final message for terminal events.
    pub fn final_message(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Done { message, .. } => Some(message),
            Self::Error { error, .. } => Some(error),
            _ => None,
        }
    }

    /// Returns the partial snapshot for non-terminal events.
    pub fn partial(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Start { partial }
            | Self::TextStart { partial, .. }
            | Self::TextDelta { partial, .. }
            | Self::TextEnd { partial, .. }
            | Self::ThinkingStart { partial, .. }
            | Self::ThinkingDelta { partial, .. }
            | Self::ThinkingEnd { partial, .. }
            | Self::ToolcallStart { partial, .. }
            | Self::ToolcallDelta { partial, .. }
            | Self::ToolcallEnd { partial, .. } => Some(partial),
            Self::Done { .. } | Self::Error { .. } => None,
        }
    }
}

/// Input for image generation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImagesContext {
    /// Ordered prompt and reference images.
    #[serde(default)]
    pub input: Vec<InputContent>,
}

/// Why image generation stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ImagesStopReason {
    /// Normal completion.
    #[default]
    Stop,
    /// Provider failure.
    Error,
    /// Caller cancellation.
    Aborted,
}

/// Image-generation result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    /// Wire API id.
    pub api: String,
    /// Provider id.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Generated text and images.
    #[serde(default)]
    pub output: Vec<InputContent>,
    /// Provider response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Token usage when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Terminal reason.
    pub stop_reason: ImagesStopReason,
    /// Error text for failed generations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Creation time in Unix milliseconds.
    pub timestamp: Timestamp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_json_matches_pi_tags() {
        let message = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent::new("hello")),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "a"}),
                    thought_signature: None,
                }),
            ],
            api: "openai-responses".into(),
            provider: "openai".into(),
            model: "gpt-test".into(),
            response_model: None,
            response_id: None,
            diagnostics: Vec::new(),
            usage: Usage::from_parts(1, 2, 3, 4),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 7,
        });

        let json = serde_json::to_value(&message).expect("serialize message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "toolCall");
        assert_eq!(json["stopReason"], "toolUse");
        assert_eq!(
            serde_json::from_value::<Message>(json).expect("round trip"),
            message
        );
    }

    #[test]
    fn usage_prompt_count_includes_cache_classes() {
        let usage = Usage::from_parts(10, 5, 20, 3);
        assert_eq!(usage.prompt_tokens(), 33);
        assert_eq!(usage.total_tokens, 38);
    }

    #[test]
    fn event_json_uses_pi_event_and_field_names() {
        let event = AssistantMessageEvent::ToolcallDelta {
            content_index: 2,
            delta: "{}".into(),
            partial: AssistantMessage::empty("api", "provider", "model"),
        };
        let value = serde_json::to_value(event).expect("serialize event");
        assert_eq!(value["type"], "toolcall_delta");
        assert_eq!(value["contentIndex"], 2);
        assert!(value.get("content_index").is_none());
    }
}
