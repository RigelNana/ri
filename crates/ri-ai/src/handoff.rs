//! Cross-provider transcript normalization.

use std::collections::{HashMap, HashSet};

use crate::{
    message::{
        AssistantMessage, ContentBlock, ImageContent, InputContent, Message, StopReason,
        TextContent, ToolCall, ToolResultMessage, UserContent, now_millis,
    },
    model::Model,
};

/// Placeholder used when user images are replayed to a text-only model.
pub const NON_VISION_USER_IMAGE_PLACEHOLDER: &str =
    "(image omitted: model does not support images)";
/// Placeholder used when tool images are replayed to a text-only model.
pub const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

/// Cross-provider tool-call id normalization callback.
pub type ToolIdNormalizer<'a> = dyn FnMut(&str, &Model, &AssistantMessage) -> String + Send + 'a;

/// Normalizes a transcript for replay by `target`.
///
/// This applies image fallback, converts foreign thinking to ordinary text,
/// strips foreign signatures, rewrites tool ids and matching results, removes
/// errored/aborted assistant turns, and synthesizes missing tool results.
pub fn transform_messages(
    messages: &[Message],
    target: &Model,
    mut normalize_tool_call_id: Option<&mut ToolIdNormalizer<'_>>,
) -> Vec<Message> {
    let image_aware = downgrade_unsupported_images(messages, target);
    let mut id_map = HashMap::<String, String>::new();
    let mut transformed = Vec::with_capacity(image_aware.len());

    for message in image_aware {
        match message {
            Message::User(message) => transformed.push(Message::User(message)),
            Message::ToolResult(mut message) => {
                if let Some(normalized) = id_map.get(&message.tool_call_id) {
                    message.tool_call_id.clone_from(normalized);
                }
                transformed.push(Message::ToolResult(message));
            }
            Message::Assistant(mut message) => {
                let same_model = message.provider == target.provider
                    && message.api == target.api
                    && message.model == target.id;
                let source = message.clone();
                message.content = message
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        ContentBlock::Thinking(thinking) if thinking.redacted => {
                            same_model.then_some(ContentBlock::Thinking(thinking))
                        }
                        ContentBlock::Thinking(thinking) => {
                            if same_model && thinking.thinking_signature.is_some() {
                                return Some(ContentBlock::Thinking(thinking));
                            }
                            if thinking.thinking.trim().is_empty() {
                                return None;
                            }
                            if same_model {
                                Some(ContentBlock::Thinking(thinking))
                            } else {
                                Some(ContentBlock::Text(TextContent::new(thinking.thinking)))
                            }
                        }
                        ContentBlock::Text(mut text) => {
                            if !same_model {
                                text.text_signature = None;
                            }
                            Some(ContentBlock::Text(text))
                        }
                        ContentBlock::ToolCall(mut call) => {
                            if !same_model {
                                call.thought_signature = None;
                                if let Some(normalizer) = normalize_tool_call_id.as_deref_mut() {
                                    let normalized = normalizer(&call.id, target, &source);
                                    if normalized != call.id {
                                        id_map.insert(call.id.clone(), normalized.clone());
                                        call.id = normalized;
                                    }
                                }
                            }
                            Some(ContentBlock::ToolCall(call))
                        }
                    })
                    .collect();
                transformed.push(Message::Assistant(message));
            }
        }
    }

    insert_orphan_results(transformed)
}

fn downgrade_unsupported_images(messages: &[Message], target: &Model) -> Vec<Message> {
    if target.supports_images() {
        return messages.to_vec();
    }
    messages
        .iter()
        .cloned()
        .map(|message| match message {
            Message::User(mut message) => {
                if let UserContent::Blocks(content) = message.content {
                    message.content = UserContent::Blocks(replace_images(
                        content,
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    ));
                }
                Message::User(message)
            }
            Message::ToolResult(mut message) => {
                message.content =
                    replace_images(message.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER);
                Message::ToolResult(message)
            }
            Message::Assistant(message) => Message::Assistant(message),
        })
        .collect()
}

fn replace_images(content: Vec<InputContent>, placeholder: &str) -> Vec<InputContent> {
    let mut output = Vec::with_capacity(content.len());
    let mut previous_was_placeholder = false;
    for block in content {
        match block {
            InputContent::Image(ImageContent { .. }) => {
                if !previous_was_placeholder {
                    output.push(InputContent::Text(TextContent::new(placeholder)));
                }
                previous_was_placeholder = true;
            }
            InputContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                output.push(InputContent::Text(text));
            }
        }
    }
    output
}

fn insert_orphan_results(messages: Vec<Message>) -> Vec<Message> {
    let mut output = Vec::with_capacity(messages.len());
    let mut pending_tool_calls = Vec::<ToolCall>::new();
    let mut existing_result_ids = HashSet::<String>::new();

    let flush =
        |output: &mut Vec<Message>, pending: &mut Vec<ToolCall>, existing: &mut HashSet<String>| {
            for call in pending.drain(..) {
                if !existing.contains(&call.id) {
                    output.push(Message::ToolResult(ToolResultMessage {
                        tool_call_id: call.id,
                        tool_name: call.name,
                        content: vec![InputContent::Text(TextContent::new("No result provided"))],
                        details: None,
                        usage: None,
                        added_tool_names: Vec::new(),
                        is_error: true,
                        timestamp: now_millis(),
                    }));
                }
            }
            existing.clear();
        };

    for message in messages {
        match message {
            Message::Assistant(message) => {
                flush(
                    &mut output,
                    &mut pending_tool_calls,
                    &mut existing_result_ids,
                );
                if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                    continue;
                }
                pending_tool_calls = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(call.clone()),
                        ContentBlock::Text(_) | ContentBlock::Thinking(_) => None,
                    })
                    .collect();
                output.push(Message::Assistant(message));
            }
            Message::ToolResult(message) => {
                existing_result_ids.insert(message.tool_call_id.clone());
                output.push(Message::ToolResult(message));
            }
            Message::User(message) => {
                flush(
                    &mut output,
                    &mut pending_tool_calls,
                    &mut existing_result_ids,
                );
                output.push(Message::User(message));
            }
        }
    }
    flush(
        &mut output,
        &mut pending_tool_calls,
        &mut existing_result_ids,
    );
    output
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        message::{
            AssistantMessage, ImageContent, InputContent, ToolResultMessage, Usage, UserMessage,
        },
        model::{Model, ModelInput},
    };

    fn target() -> Model {
        Model::new(
            "anthropic",
            "claude-test",
            "anthropic-messages",
            "https://example.test",
        )
    }

    fn source_assistant(content: Vec<ContentBlock>, reason: StopReason) -> Message {
        let mut message = AssistantMessage::empty("openai-responses", "openai", "gpt-test");
        message.content = content;
        message.stop_reason = reason;
        message.usage = Usage::default();
        Message::Assistant(message)
    }

    #[test]
    fn converts_foreign_thinking_and_strips_signatures() {
        let messages = vec![source_assistant(
            vec![
                ContentBlock::Thinking(crate::message::ThinkingContent {
                    thinking: "reasoning".into(),
                    thinking_signature: Some("opaque".into()),
                    redacted: false,
                }),
                ContentBlock::Text(TextContent {
                    text: "answer".into(),
                    text_signature: Some("message-id".into()),
                }),
                ContentBlock::Thinking(crate::message::ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some("redacted".into()),
                    redacted: true,
                }),
            ],
            StopReason::Stop,
        )];
        let output = transform_messages(&messages, &target(), None);
        let Message::Assistant(message) = &output[0] else {
            unreachable!("assistant expected")
        };
        assert_eq!(
            message.content,
            vec![
                ContentBlock::Text(TextContent::new("reasoning")),
                ContentBlock::Text(TextContent::new("answer")),
            ]
        );
    }

    #[test]
    fn rewrites_tool_calls_and_results_together() {
        let call = ToolCall {
            id: "call|bad/id".into(),
            name: "read".into(),
            arguments: json!({"path": "a"}),
            thought_signature: Some("foreign".into()),
        };
        let messages = vec![
            source_assistant(vec![ContentBlock::ToolCall(call)], StopReason::ToolUse),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call|bad/id".into(),
                tool_name: "read".into(),
                content: vec![InputContent::Text(TextContent::new("ok"))],
                details: None,
                usage: None,
                added_tool_names: Vec::new(),
                is_error: false,
                timestamp: 1,
            }),
        ];
        let mut normalizer =
            |id: &str, _: &Model, _: &AssistantMessage| id.replace(['|', '/'], "_");
        let output = transform_messages(&messages, &target(), Some(&mut normalizer));
        let Message::Assistant(assistant) = &output[0] else {
            unreachable!("assistant expected")
        };
        let ContentBlock::ToolCall(call) = &assistant.content[0] else {
            unreachable!("tool call expected")
        };
        assert_eq!(call.id, "call_bad_id");
        assert_eq!(call.thought_signature, None);
        let Message::ToolResult(result) = &output[1] else {
            unreachable!("result expected")
        };
        assert_eq!(result.tool_call_id, "call_bad_id");
    }

    #[test]
    fn fills_orphaned_calls_before_interruption_and_at_end() {
        let call = |id: &str| {
            ContentBlock::ToolCall(ToolCall {
                id: id.into(),
                name: "tool".into(),
                arguments: json!({}),
                thought_signature: None,
            })
        };
        let messages = vec![
            source_assistant(vec![call("one"), call("two")], StopReason::ToolUse),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "one".into(),
                tool_name: "tool".into(),
                content: vec![],
                details: None,
                usage: None,
                added_tool_names: Vec::new(),
                is_error: false,
                timestamp: 1,
            }),
            Message::User(UserMessage::new("continue")),
            source_assistant(vec![call("three")], StopReason::ToolUse),
        ];
        let output = transform_messages(&messages, &target(), None);
        let synthetic = output
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) if result.is_error => {
                    Some(result.tool_call_id.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(synthetic, vec!["two", "three"]);
    }

    #[test]
    fn drops_error_and_aborted_assistant_replay() {
        let messages = vec![
            source_assistant(vec![], StopReason::Error),
            source_assistant(vec![], StopReason::Aborted),
            Message::User(UserMessage::new("retry")),
        ];
        assert_eq!(
            transform_messages(&messages, &target(), None),
            vec![Message::User(UserMessage {
                content: UserContent::Text("retry".into()),
                timestamp: match &messages[2] {
                    Message::User(message) => message.timestamp,
                    _ => 0,
                },
            })]
        );
    }

    #[test]
    fn collapses_adjacent_images_for_text_only_target() {
        let user = Message::User(UserMessage {
            content: UserContent::Blocks(vec![
                InputContent::Image(ImageContent {
                    data: "a".into(),
                    mime_type: "image/png".into(),
                }),
                InputContent::Image(ImageContent {
                    data: "b".into(),
                    mime_type: "image/png".into(),
                }),
                InputContent::Text(TextContent::new("caption")),
            ]),
            timestamp: 1,
        });
        let output = transform_messages(&[user], &target(), None);
        let Message::User(user) = &output[0] else {
            unreachable!("user expected")
        };
        assert_eq!(
            user.content,
            UserContent::Blocks(vec![
                InputContent::Text(TextContent::new(NON_VISION_USER_IMAGE_PLACEHOLDER)),
                InputContent::Text(TextContent::new("caption")),
            ])
        );

        let mut vision = target();
        vision.input.push(ModelInput::Image);
        let image = Message::User(UserMessage {
            content: UserContent::Blocks(vec![InputContent::Image(ImageContent {
                data: "a".into(),
                mime_type: "image/png".into(),
            })]),
            timestamp: 1,
        });
        assert_eq!(
            transform_messages(std::slice::from_ref(&image), &vision, None),
            vec![image]
        );
    }
}
