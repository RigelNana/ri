//! Session-to-model context projection.

use chrono::DateTime;
use ri_ai::{ContentBlock, Message, TextContent, ToolResultMessage, UserContent, UserMessage};
use serde_json::Value;

use crate::error::{Error, Result};

/// Projects the compaction-aware active branch into provider-neutral messages.
///
/// # Errors
/// Returns an error when the session cannot be read or contains invalid message data.
pub async fn project_session(session: &ri_session::Session) -> Result<Vec<Message>> {
    let context = session.context().await?;
    project_values(&context.messages)
}

/// Projects serialized session messages into provider-neutral messages.
///
/// Standard user, assistant, and tool-result records are deserialized exactly.
/// Harness-only summary/custom messages become user context markers. Unknown
/// application roles are omitted instead of being sent with an invalid wire role.
///
/// # Errors
/// Returns an error when a message has no role or a standard message cannot be deserialized.
pub fn project_values(values: &[Value]) -> Result<Vec<Message>> {
    let mut projected = Vec::new();
    for value in values {
        let Some(role) = value.get("role").and_then(Value::as_str) else {
            return Err(Error::Session(
                "session context contains a message without a role".to_owned(),
            ));
        };
        match role {
            "user" | "assistant" | "toolResult" => {
                projected.push(serde_json::from_value(value.clone()).map_err(|error| {
                    Error::Session(format!(
                        "invalid {role} message in session context: {error}"
                    ))
                })?);
            }
            "compactionSummary" => {
                let summary = value
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let tokens = value
                    .get("tokensBefore")
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
                projected.push(context_message(
                    format!(
                        "The conversation history was compacted at approximately {tokens} tokens.\n\n{summary}"
                    ),
                    timestamp(value),
                ));
            }
            "branchSummary" => {
                let summary = value
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                projected.push(context_message(
                    format!(
                        "The user explored a different conversation branch before returning here.\n\
                         Summary of that exploration:\n\n{summary}"
                    ),
                    timestamp(value),
                ));
            }
            "custom" => {
                if let Some(text) = text_from_value(value.get("content")) {
                    projected.push(context_message(text, timestamp(value)));
                }
            }
            _ => {}
        }
    }
    Ok(projected)
}

/// Serializes a provider-neutral message for session persistence.
///
/// # Errors
/// Returns an error when the message cannot be represented as JSON.
pub fn message_value(message: &Message) -> Result<Value> {
    serde_json::to_value(message).map_err(Error::from)
}

/// Returns the user-facing text from a user message.
pub fn user_text(message: &UserMessage) -> String {
    match &message.content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| {
                let value = serde_json::to_value(block).ok()?;
                value.get("text")?.as_str().map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Returns all text-like assistant content.
pub fn assistant_text(message: &ri_ai::AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(ContentBlock::text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns all text blocks from a tool result.
pub fn tool_result_text(message: &ToolResultMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|block| {
            let value = serde_json::to_value(block).ok()?;
            value.get("text")?.as_str().map(str::to_owned)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn context_message(text: String, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text),
        timestamp,
    })
}

fn timestamp(value: &Value) -> i64 {
    value
        .get("timestamp")
        .and_then(|timestamp| {
            timestamp.as_i64().or_else(|| {
                timestamp
                    .as_str()
                    .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
                    .map(|value| value.timestamp_millis())
            })
        })
        .unwrap_or_default()
}

fn text_from_value(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) => Some(text.clone()),
        Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

/// Creates a text-only user message with an explicit timestamp.
pub fn user_message(text: impl Into<String>, timestamp: i64) -> Message {
    Message::User(UserMessage {
        content: UserContent::Text(text.into()),
        timestamp,
    })
}

/// Creates a text block for callers constructing result messages.
pub fn text(text: impl Into<String>) -> TextContent {
    TextContent::new(text)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn projects_summary_and_ignores_unknown_roles() {
        let values = vec![
            json!({
                "role": "compactionSummary",
                "summary": "kept facts",
                "tokensBefore": 12,
                "timestamp": "2025-01-01T00:00:00.000Z"
            }),
            json!({"role": "uiOnly", "content": "hidden"}),
        ];
        let messages = project_values(&values).expect("project");
        assert_eq!(messages.len(), 1);
        let Message::User(message) = &messages[0] else {
            panic!("summary projects to user context");
        };
        assert!(user_text(message).contains("kept facts"));
        assert_eq!(message.timestamp, 1_735_689_600_000);
    }

    #[test]
    fn malformed_standard_message_is_not_silently_dropped() {
        let error = project_values(&[json!({"role": "assistant"})]).expect_err("invalid");
        assert!(error.to_string().contains("invalid assistant message"));
    }
}
