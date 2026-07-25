//! Native pi-messages protocol adapter.

use serde_json::{Map, Value, json};
use url::Url;

use crate::{
    error::AiError,
    message::{
        AssistantMessage, ContentBlock, Context, InputContent, Message, StopReason, Usage,
        UserContent,
    },
    model::{CacheRetention, Model},
    transport::{HttpRequest, SseFrame},
};

use super::{
    WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request, merge_wire_headers,
    resolve_optional_cache_retention,
};

/// Adapter for pi's provider-neutral SSE protocol.
#[derive(Clone, Copy, Debug, Default)]
pub struct PiMessagesAdapter;

impl WireAdapter for PiMessagesAdapter {
    fn api(&self) -> &'static str {
        "pi-messages"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let base = model.base_url.trim_end_matches('/');
        let endpoint = if base.ends_with("/messages") {
            base.to_owned()
        } else {
            format!("{base}/messages")
        };
        let endpoint =
            Url::parse(&endpoint).map_err(|error| AiError::Validation(error.to_string()))?;
        let mut defaults = vec![
            ("accept".into(), "text/event-stream".into()),
            ("content-type".into(), "application/json".into()),
        ];
        if let Some(api_key) = &options.api_key {
            defaults.push(("authorization".into(), format!("Bearer {api_key}")));
        }
        let headers = merge_wire_headers(defaults, model, &options.headers);
        if !headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("authorization"))
        {
            return Err(AiError::Auth(format!(
                "no API key for provider {}",
                model.provider
            )));
        }
        let body = build_pi_messages_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_pi_messages_frame(frame, state)
    }
}

fn build_pi_messages_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
    let mut wire_options = Map::new();
    if let Some(temperature) = options.temperature {
        wire_options.insert(
            "temperature".into(),
            Value::Number(
                serde_json::Number::from_f64(temperature)
                    .ok_or_else(|| AiError::Validation("temperature must be finite".into()))?,
            ),
        );
    }
    if let Some(max_tokens) = options.max_tokens {
        wire_options.insert("maxTokens".into(), Value::Number(max_tokens.into()));
    }
    if let Some(reasoning) = options.reasoning {
        wire_options.insert("reasoning".into(), Value::String(reasoning.as_str().into()));
    }
    if let Some(retention) = resolve_optional_cache_retention(options) {
        wire_options.insert(
            "cacheRetention".into(),
            Value::String(
                match retention {
                    CacheRetention::None => "none",
                    CacheRetention::Short => "short",
                    CacheRetention::Long => "long",
                }
                .into(),
            ),
        );
    }
    if let Some(session_id) = &options.session_id {
        wire_options.insert("sessionId".into(), Value::String(session_id.clone()));
    }
    if let Some(tool_choice) = &options.tool_choice {
        wire_options.insert("toolChoice".into(), Value::String(tool_choice.clone()));
    }
    Ok(json!({
        "model": model.id,
        "context": context_to_wire(context),
        "options": wire_options
    }))
}

fn context_to_wire(context: &Context) -> Value {
    let mut value = Map::from_iter([
        (
            "messages".into(),
            Value::Array(context.messages.iter().map(message_to_wire).collect()),
        ),
        (
            "tools".into(),
            Value::Array(
                context
                    .tools
                    .iter()
                    .map(|tool| {
                        let mut value = json!({
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        });
                        if let Some(constraint) = &tool.constrained_sampling
                            && let Ok(constraint) = serde_json::to_value(constraint)
                        {
                            value["constrainedSampling"] = constraint;
                        }
                        value
                    })
                    .collect(),
            ),
        ),
    ]);
    if let Some(system) = &context.system_prompt {
        value.insert("systemPrompt".into(), Value::String(system.clone()));
    }
    Value::Object(value)
}

fn message_to_wire(message: &Message) -> Value {
    match message {
        Message::User(message) => json!({
            "role": "user",
            "content": match &message.content {
                UserContent::Text(text) => Value::String(text.clone()),
                UserContent::Blocks(blocks) => Value::Array(blocks.iter().map(input_to_wire).collect())
            },
            "timestamp": message.timestamp
        }),
        Message::Assistant(message) => assistant_to_wire(message),
        Message::ToolResult(message) => {
            let mut value = json!({
                "role": "toolResult",
                "toolCallId": message.tool_call_id,
                "toolName": message.tool_name,
                "content": message.content.iter().map(input_to_wire).collect::<Vec<_>>(),
                "isError": message.is_error,
                "timestamp": message.timestamp,
                "addedToolNames": message.added_tool_names
            });
            if let Some(details) = &message.details {
                value["details"] = details.clone();
            }
            if let Some(usage) = &message.usage {
                value["usage"] = usage_to_wire(usage);
            }
            value
        }
    }
}

fn assistant_to_wire(message: &AssistantMessage) -> Value {
    let mut value = json!({
        "role": "assistant",
        "content": message.content.iter().map(content_to_wire).collect::<Vec<_>>(),
        "api": message.api,
        "provider": message.provider,
        "model": message.model,
        "usage": usage_to_wire(&message.usage),
        "stopReason": stop_reason_name(message.stop_reason),
        "timestamp": message.timestamp
    });
    if let Some(error) = &message.error_message {
        value["errorMessage"] = Value::String(error.clone());
    }
    if let Some(id) = &message.response_id {
        value["responseId"] = Value::String(id.clone());
    }
    if let Some(model) = &message.response_model {
        value["responseModel"] = Value::String(model.clone());
    }
    if !message.diagnostics.is_empty() {
        value["diagnostics"] = Value::Array(message.diagnostics.clone());
    }
    value
}

fn input_to_wire(content: &InputContent) -> Value {
    match content {
        InputContent::Text(text) => json!({"type": "text", "text": text.text}),
        InputContent::Image(image) => json!({
            "type": "image",
            "data": image.data,
            "mimeType": image.mime_type
        }),
    }
}

fn content_to_wire(content: &ContentBlock) -> Value {
    match content {
        ContentBlock::Text(text) => {
            let mut value = json!({"type": "text", "text": text.text});
            if let Some(signature) = &text.text_signature {
                value["textSignature"] = Value::String(signature.clone());
            }
            value
        }
        ContentBlock::Thinking(thinking) => {
            let mut value = json!({
                "type": "thinking",
                "thinking": thinking.thinking,
                "redacted": thinking.redacted
            });
            if let Some(signature) = &thinking.thinking_signature {
                value["thinkingSignature"] = Value::String(signature.clone());
            }
            value
        }
        ContentBlock::ToolCall(call) => {
            let mut value = json!({
                "type": "toolCall",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments
            });
            if let Some(signature) = &call.thought_signature {
                value["thoughtSignature"] = Value::String(signature.clone());
            }
            value
        }
    }
}

fn usage_to_wire(usage: &Usage) -> Value {
    let mut value = json!({
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
        "totalTokens": usage.total_tokens,
        "cost": {
            "input": usage.cost.input,
            "output": usage.cost.output,
            "cacheRead": usage.cost.cache_read,
            "cacheWrite": usage.cost.cache_write,
            "total": usage.cost.total
        }
    });
    if let Some(cache_write_1h) = usage.cache_write_1h {
        value["cacheWrite1h"] = json!(cache_write_1h);
    }
    if let Some(reasoning) = usage.reasoning {
        value["reasoning"] = json!(reasoning);
    }
    value
}

fn decode_pi_messages_frame(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    if frame.data.trim().is_empty() || frame.data.trim() == "[DONE]" {
        return Ok(Vec::new());
    }
    let event: Value = serde_json::from_str(&frame.data)
        .map_err(|error| AiError::Stream(format!("invalid pi-messages event: {error}")))?;
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let remote_index = event
        .get("contentIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let index = state.slot(format!("pi:{remote_index}"));
    let output = match kind {
        "text_start" => vec![WireDelta::TextStart { index }],
        "text_delta" => vec![WireDelta::TextDelta {
            index,
            delta: event
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        }],
        "text_end" => vec![WireDelta::TextEnd {
            index,
            content: event
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_owned),
            signature: event
                .get("contentSignature")
                .and_then(Value::as_str)
                .map(str::to_owned),
        }],
        "thinking_start" => vec![WireDelta::ThinkingStart { index }],
        "thinking_delta" => vec![WireDelta::ThinkingDelta {
            index,
            delta: event
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        }],
        "thinking_end" => vec![WireDelta::ThinkingEnd {
            index,
            content: event
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_owned),
            signature: event
                .get("contentSignature")
                .and_then(Value::as_str)
                .map(str::to_owned),
            redacted: event
                .get("redacted")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }],
        "toolcall_start" => vec![WireDelta::ToolStart {
            index,
            id: event
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            name: event
                .get("toolName")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            thought_signature: None,
        }],
        "toolcall_delta" => vec![WireDelta::ToolDelta {
            index,
            delta: event
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        }],
        "toolcall_end" => {
            let mut output = Vec::new();
            if event.pointer("/toolCall/thoughtSignature").is_some() {
                output.push(WireDelta::ToolSignature {
                    index,
                    signature: event
                        .pointer("/toolCall/thoughtSignature")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                });
            }
            output.push(WireDelta::ToolEnd {
                index,
                arguments: event.pointer("/toolCall/arguments").cloned(),
            });
            output
        }
        "done" => {
            let mut output = terminal_metadata(&event);
            output.push(WireDelta::Done(parse_stop_reason(
                event
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("stop"),
            )));
            output
        }
        "error" => {
            let mut output = terminal_metadata(&event);
            let aborted = event.get("reason").and_then(Value::as_str) == Some("aborted");
            let message = event
                .get("errorMessage")
                .and_then(Value::as_str)
                .unwrap_or(if aborted {
                    "request was aborted"
                } else {
                    "pi-messages backend failed"
                })
                .to_owned();
            output.push(if aborted {
                WireDelta::Aborted(message)
            } else {
                WireDelta::Error(message)
            });
            output
        }
        _ => Vec::new(),
    };
    Ok(output)
}

fn terminal_metadata(event: &Value) -> Vec<WireDelta> {
    let mut output = Vec::new();
    if let Some(usage) = event.get("usage") {
        output.push(WireDelta::Usage(parse_pi_usage(usage)));
    }
    if let Some(id) = event.get("responseId").and_then(Value::as_str) {
        output.push(WireDelta::ResponseId(id.to_owned()));
    }
    if let Some(rewrite) = event.get("rewrite") {
        output.push(WireDelta::Diagnostic(json!({
            "type": "pi_messages_rewrite",
            "details": rewrite
        })));
    }
    output
}

fn parse_pi_usage(value: &Value) -> Usage {
    Usage {
        input: value.get("input").and_then(Value::as_u64).unwrap_or(0),
        output: value.get("output").and_then(Value::as_u64).unwrap_or(0),
        cache_read: value.get("cacheRead").and_then(Value::as_u64).unwrap_or(0),
        cache_write: value.get("cacheWrite").and_then(Value::as_u64).unwrap_or(0),
        cache_write_1h: value.get("cacheWrite1h").and_then(Value::as_u64),
        reasoning: value.get("reasoning").and_then(Value::as_u64),
        total_tokens: value
            .get("totalTokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cost: crate::message::UsageCost {
            input: value
                .pointer("/cost/input")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            output: value
                .pointer("/cost/output")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            cache_read: value
                .pointer("/cost/cacheRead")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            cache_write: value
                .pointer("/cost/cacheWrite")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            total: value
                .pointer("/cost/total")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
        },
    }
}

fn parse_stop_reason(reason: &str) -> StopReason {
    match reason {
        "length" => StopReason::Length,
        "toolUse" => StopReason::ToolUse,
        "error" => StopReason::Error,
        "aborted" => StopReason::Aborted,
        _ => StopReason::Stop,
    }
}

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Stop => "stop",
        StopReason::Length => "length",
        StopReason::ToolUse => "toolUse",
        StopReason::Error => "error",
        StopReason::Aborted => "aborted",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, UserMessage};

    #[test]
    fn payload_uses_pi_camel_case_protocol() {
        let model = Model::new("radius", "model", "pi-messages", "https://example.test");
        let context = Context {
            system_prompt: Some("system".into()),
            messages: vec![Message::User(UserMessage::new("hello"))],
            tools: Vec::new(),
        };
        let value = build_pi_messages_body(
            &model,
            &context,
            &WireRequestOptions {
                max_tokens: Some(20),
                session_id: Some("session".into()),
                ..WireRequestOptions::default()
            },
        )
        .expect("payload");
        assert_eq!(value["context"]["systemPrompt"], "system");
        assert_eq!(value["options"]["maxTokens"], 20);
        assert_eq!(value["options"]["sessionId"], "session");
    }

    #[test]
    fn explicit_cache_disable_is_forwarded_to_backend() {
        let model = Model::new("radius", "model", "pi-messages", "https://example.test");
        let value = build_pi_messages_body(
            &model,
            &Context::default(),
            &WireRequestOptions {
                cache_retention: Some(CacheRetention::None),
                ..WireRequestOptions::default()
            },
        )
        .expect("payload");
        assert_eq!(value["options"]["cacheRetention"], "none");
    }

    #[test]
    fn terminal_done_keeps_usage_and_response_id() {
        let frame = SseFrame {
            data: json!({
                "type": "done",
                "reason": "toolUse",
                "usage": {
                    "input": 4,
                    "output": 2,
                    "cacheRead": 1,
                    "cacheWrite": 0,
                    "totalTokens": 7,
                    "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3}
                },
                "responseId": "response"
            })
            .to_string(),
            ..SseFrame::default()
        };
        let deltas =
            decode_pi_messages_frame(&frame, &mut WireDecodeState::default()).expect("decode");
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            WireDelta::ResponseId(id) if id == "response"
        )));
        assert_eq!(deltas.last(), Some(&WireDelta::Done(StopReason::ToolUse)));
    }

    #[test]
    fn aborted_error_remains_distinct_from_provider_failure() {
        let frame = SseFrame {
            data: json!({"type": "error", "reason": "aborted"}).to_string(),
            ..SseFrame::default()
        };
        let deltas =
            decode_pi_messages_frame(&frame, &mut WireDecodeState::default()).expect("decode");
        assert!(matches!(
            deltas.last(),
            Some(WireDelta::Aborted(message)) if message == "request was aborted"
        ));
    }
}
