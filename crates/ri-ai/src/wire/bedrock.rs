//! Amazon Bedrock `ConverseStream` adapter.
//!
//! The production transport supports Bedrock's binary AWS event-stream
//! framing. Authentication can use Bedrock bearer tokens directly; callers
//! using `SigV4` may inject a pre-signed `authorization` header.

use serde_json::{Map, Value, json};
use url::Url;

use crate::{
    error::AiError,
    handoff::transform_messages,
    message::{
        ContentBlock, Context, InputContent, Message, StopReason, Usage, UsageCost, UserContent,
    },
    model::{CacheRetention, Model, ThinkingLevel},
    tool::resolve_json_schema_strict,
    transport::{HttpRequest, SseFrame},
};

use super::{
    StreamEncoding, WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request,
    merge_wire_headers, resolve_cache_retention,
};

/// Bedrock `ConverseStream` adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct BedrockConverseStreamAdapter;

impl WireAdapter for BedrockConverseStreamAdapter {
    fn api(&self) -> &'static str {
        "bedrock-converse-stream"
    }

    fn stream_encoding(&self) -> StreamEncoding {
        StreamEncoding::AwsEventStream
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = bedrock_endpoint(model)?;
        let mut defaults = vec![
            ("accept".into(), "application/vnd.amazon.eventstream".into()),
            ("content-type".into(), "application/json".into()),
        ];
        if let Some(token) = &options.api_key {
            defaults.push(("authorization".into(), format!("Bearer {token}")));
        }
        let headers = merge_wire_headers(defaults, model, &options.headers);
        if !headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("authorization"))
        {
            return Err(AiError::Auth(
                "Bedrock requires a bearer token or a SigV4 authorization header".into(),
            ));
        }
        let body = build_bedrock_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_bedrock_event(frame, state)
    }
}

fn bedrock_endpoint(model: &Model) -> Result<Url, AiError> {
    let base = model.base_url.trim_end_matches('/');
    if base.ends_with("/converse-stream") {
        return Url::parse(base).map_err(|error| AiError::Validation(error.to_string()));
    }
    let mut url = Url::parse(base).map_err(|error| AiError::Validation(error.to_string()))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| AiError::Validation("Bedrock base URL cannot be a base".into()))?;
        segments.pop_if_empty();
        segments.push("model");
        segments.push(&model.id);
        segments.push("converse-stream");
    }
    Ok(url)
}

fn build_bedrock_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
    let retention = resolve_cache_retention(options);
    let caching = retention != CacheRetention::None && supports_cache(model);
    let reasoning_level = options
        .reasoning
        .filter(|level| *level != ThinkingLevel::Off);
    let inference_max_tokens = if is_claude(model) {
        if let Some(level) = reasoning_level.filter(|_| !adaptive_thinking(model)) {
            Some(
                options
                    .max_tokens
                    .map_or(model.max_tokens, |output_tokens| {
                        output_tokens
                            .saturating_add(bedrock_thinking_budget(level))
                            .min(model.max_tokens)
                    }),
            )
        } else {
            Some(
                options
                    .max_tokens
                    .unwrap_or(model.max_tokens)
                    .min(model.max_tokens),
            )
        }
    } else {
        options.max_tokens
    };
    let mut body = Map::from_iter([(
        "messages".into(),
        Value::Array(convert_bedrock_messages(model, context, retention)),
    )]);
    if let Some(system) = &context.system_prompt {
        let mut blocks = vec![json!({"text": non_blank(system)})];
        if caching {
            blocks.push(cache_point(retention));
        }
        body.insert("system".into(), Value::Array(blocks));
    }
    let mut inference = Map::new();
    if let Some(max_tokens) = inference_max_tokens {
        inference.insert("maxTokens".into(), Value::Number(max_tokens.into()));
    }
    if let Some(temperature) = options.temperature {
        inference.insert(
            "temperature".into(),
            Value::Number(
                serde_json::Number::from_f64(temperature)
                    .ok_or_else(|| AiError::Validation("temperature must be finite".into()))?,
            ),
        );
    }
    if !inference.is_empty() {
        body.insert("inferenceConfig".into(), Value::Object(inference));
    }
    if !context.tools.is_empty() && options.tool_choice.as_deref() != Some("none") {
        let strict_supported = model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_strict_mode)
            .unwrap_or(false);
        let tools = context
            .tools
            .iter()
            .map(|tool| -> Result<Value, AiError> {
                let strict = resolve_json_schema_strict(tool, strict_supported)
                    .map_err(|error| AiError::Validation(error.to_string()))?
                    == Some(true);
                let mut spec = json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters}
                });
                if strict {
                    spec["strict"] = Value::Bool(true);
                }
                Ok(json!({"toolSpec": spec}))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let choice = match options.tool_choice.as_deref() {
            Some("any" | "required") => Some(json!({"any": {}})),
            Some("auto") => Some(json!({"auto": {}})),
            Some(name) if !name.is_empty() => Some(json!({"tool": {"name": name}})),
            _ => None,
        };
        let mut config = json!({"tools": tools});
        if let Some(choice) = choice {
            config["toolChoice"] = choice;
        }
        body.insert("toolConfig".into(), config);
    }
    if model.reasoning
        && let Some(level) = reasoning_level
    {
        let reasoning = bedrock_reasoning(
            model,
            level,
            inference_max_tokens.unwrap_or(model.max_tokens),
        );
        if reasoning
            .as_object()
            .is_some_and(|fields| !fields.is_empty())
        {
            body.insert("additionalModelRequestFields".into(), reasoning);
        }
    }
    Ok(Value::Object(body))
}

fn convert_bedrock_messages(
    model: &Model,
    context: &Context,
    retention: CacheRetention,
) -> Vec<Value> {
    let mut normalize = |id: &str, _: &Model, _: &crate::message::AssistantMessage| {
        id.chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                    character
                } else {
                    '_'
                }
            })
            .take(64)
            .collect()
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalize));
    let mut output: Vec<Value> = Vec::new();
    let mut index = 0usize;
    while index < messages.len() {
        match &messages[index] {
            Message::User(message) => {
                let blocks = match &message.content {
                    UserContent::Text(text) => vec![json!({"text": non_blank(text)})],
                    UserContent::Blocks(input_blocks) => {
                        let blocks = input_blocks
                            .iter()
                            .filter_map(|block| match block {
                                InputContent::Text(text) if !text.text.trim().is_empty() => {
                                    Some(json!({"text": text.text}))
                                }
                                InputContent::Image(image) => Some(json!({
                                    "image": {
                                        "format": image_format(&image.mime_type),
                                        "source": {"bytes": image.data}
                                    }
                                })),
                                InputContent::Text(_) => None,
                            })
                            .collect::<Vec<_>>();
                        if blocks.is_empty() {
                            vec![json!({"text": "<empty>"})]
                        } else {
                            blocks
                        }
                    }
                };
                output.push(json!({"role": "user", "content": blocks}));
            }
            Message::Assistant(message) => {
                let blocks = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                            Some(json!({"text": text.text}))
                        }
                        ContentBlock::ToolCall(call) => Some(json!({
                            "toolUse": {
                                "toolUseId": call.id,
                                "name": call.name,
                                "input": call.arguments
                            }
                        })),
                        ContentBlock::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            if is_claude(model) {
                                thinking.thinking_signature.as_ref().map_or_else(
                                    || Some(json!({"text": thinking.thinking})),
                                    |signature| {
                                        Some(json!({
                                            "reasoningContent": {
                                                "reasoningText": {
                                                    "text": thinking.thinking,
                                                    "signature": signature
                                                }
                                            }
                                        }))
                                    },
                                )
                            } else {
                                Some(json!({
                                    "reasoningContent": {
                                        "reasoningText": {"text": thinking.thinking}
                                    }
                                }))
                            }
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !blocks.is_empty() {
                    output.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            Message::ToolResult(_) => {
                let mut blocks = Vec::new();
                while index < messages.len() {
                    let Message::ToolResult(message) = &messages[index] else {
                        break;
                    };
                    let result_content = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            InputContent::Text(text) if !text.text.trim().is_empty() => {
                                Some(json!({"text": text.text}))
                            }
                            InputContent::Image(image) => Some(json!({
                                "image": {
                                    "format": image_format(&image.mime_type),
                                    "source": {"bytes": image.data}
                                }
                            })),
                            InputContent::Text(_) => None,
                        })
                        .collect::<Vec<_>>();
                    blocks.push(json!({
                        "toolResult": {
                            "toolUseId": message.tool_call_id,
                            "content": if result_content.is_empty() {
                                vec![json!({"text": "<empty>"})]
                            } else {
                                result_content
                            },
                            "status": if message.is_error { "error" } else { "success" }
                        }
                    }));
                    index += 1;
                }
                output.push(json!({"role": "user", "content": blocks}));
                continue;
            }
        }
        index += 1;
    }
    if retention != CacheRetention::None
        && supports_cache(model)
        && let Some(last) = output.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        content.push(cache_point(retention));
    }
    output
}

fn cache_point(retention: CacheRetention) -> Value {
    let mut value = json!({"cachePoint": {"type": "default"}});
    if retention == CacheRetention::Long {
        value["cachePoint"]["ttl"] = Value::String("1h".into());
    }
    value
}

fn supports_cache(model: &Model) -> bool {
    let candidate = format!("{} {}", model.id, model.name).to_ascii_lowercase();
    candidate.contains("claude")
        && (candidate.contains("-4-")
            || candidate.contains("claude-3-7-sonnet")
            || candidate.contains("claude-3-5-haiku")
            || candidate.contains("fable-5")
            || candidate.contains("opus-5")
            || candidate.contains("sonnet-5"))
}

fn is_claude(model: &Model) -> bool {
    format!("{} {}", model.id, model.name)
        .to_ascii_lowercase()
        .contains("claude")
}

fn adaptive_thinking(model: &Model) -> bool {
    let candidate = format!("{} {}", model.id, model.name)
        .to_ascii_lowercase()
        .replace([' ', '_', '.', ':'], "-");
    [
        "opus-4-6",
        "opus-4-7",
        "opus-4-8",
        "opus-5",
        "sonnet-4-6",
        "sonnet-5",
        "fable-5",
    ]
    .iter()
    .any(|name| candidate.contains(name))
}

fn bedrock_reasoning(model: &Model, level: ThinkingLevel, max_tokens: u64) -> Value {
    if !is_claude(model) {
        return json!({});
    }
    let effort = model
        .thinking_level_map
        .get(&level)
        .and_then(Clone::clone)
        .unwrap_or_else(|| match level {
            ThinkingLevel::Off | ThinkingLevel::Minimal | ThinkingLevel::Low => "low".into(),
            ThinkingLevel::Medium => "medium".into(),
            ThinkingLevel::High => "high".into(),
            ThinkingLevel::Xhigh => "xhigh".into(),
            ThinkingLevel::Max => "max".into(),
        });
    if adaptive_thinking(model) {
        json!({
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": effort}
        })
    } else {
        let budget = bedrock_thinking_budget(level).min(max_tokens.saturating_sub(1_024));
        json!({
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
                "display": "summarized"
            },
            "anthropic_beta": ["interleaved-thinking-2025-05-14"]
        })
    }
}

const fn bedrock_thinking_budget(level: ThinkingLevel) -> u64 {
    match level {
        ThinkingLevel::Minimal => 1_024,
        ThinkingLevel::Low => 2_048,
        ThinkingLevel::Medium => 8_192,
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => 16_384,
        ThinkingLevel::Off => 0,
    }
}

fn non_blank(text: &str) -> &str {
    if text.trim().is_empty() {
        "<empty>"
    } else {
        text
    }
}

fn image_format(mime: &str) -> &'static str {
    match mime {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    }
}

fn decode_bedrock_event(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    let event: Value = if frame.data.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&frame.data)
            .map_err(|error| AiError::Stream(format!("invalid Bedrock event: {error}")))?
    };
    let event_name = frame
        .event
        .as_deref()
        .or_else(|| event.as_object()?.keys().next().map(String::as_str))
        .unwrap_or_default();
    if let Some(exception) = event_name.strip_prefix("exception:") {
        let message = event
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or(exception);
        return Ok(vec![WireDelta::Error(format!(
            "{}: {message}",
            bedrock_error_prefix(exception)
        ))]);
    }
    let payload = event.get(event_name).unwrap_or(&event);
    let mut output = Vec::new();
    match event_name {
        "contentBlockStart" => {
            let wire_index = payload
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if let Some(tool) = payload.pointer("/start/toolUse") {
                let index = state.slot(format!("bedrock:{wire_index}"));
                state.set(
                    format!("bedrock:kind:{wire_index}"),
                    Value::String("tool".into()),
                );
                output.push(WireDelta::ToolStart {
                    index,
                    id: tool
                        .get("toolUseId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name: tool
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    thought_signature: None,
                });
            }
        }
        "contentBlockDelta" => {
            let wire_index = payload
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let index = state.slot(format!("bedrock:{wire_index}"));
            let delta = payload.get("delta").unwrap_or(&Value::Null);
            if let Some(text) = delta.get("text").and_then(Value::as_str) {
                ensure_bedrock_block(wire_index, "text", index, state, &mut output);
                output.push(WireDelta::TextDelta {
                    index,
                    delta: text.to_owned(),
                });
            } else if let Some(tool) = delta.get("toolUse") {
                output.push(WireDelta::ToolDelta {
                    index,
                    delta: tool
                        .get("input")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            } else if let Some(reasoning) = delta.get("reasoningContent") {
                ensure_bedrock_block(wire_index, "thinking", index, state, &mut output);
                if let Some(text) = reasoning.get("text").and_then(Value::as_str) {
                    output.push(WireDelta::ThinkingDelta {
                        index,
                        delta: text.to_owned(),
                    });
                }
                if let Some(signature) = reasoning.get("signature").and_then(Value::as_str) {
                    let key = format!("bedrock:signature:{wire_index}");
                    let mut current = state
                        .take(&key)
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .unwrap_or_default();
                    current.push_str(signature);
                    state.set(key, Value::String(current));
                }
            }
        }
        "contentBlockStop" => {
            let wire_index = payload
                .get("contentBlockIndex")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let index = state.slot(format!("bedrock:{wire_index}"));
            let kind = state
                .get(&format!("bedrock:kind:{wire_index}"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            match kind.as_deref() {
                Some("text") => output.push(WireDelta::TextEnd {
                    index,
                    content: None,
                    signature: None,
                }),
                Some("thinking") => output.push(WireDelta::ThinkingEnd {
                    index,
                    content: None,
                    signature: state
                        .take(&format!("bedrock:signature:{wire_index}"))
                        .and_then(|value| value.as_str().map(str::to_owned)),
                    redacted: false,
                }),
                Some("tool") => output.push(WireDelta::ToolEnd {
                    index,
                    arguments: None,
                }),
                _ => {}
            }
        }
        "messageStop" => {
            state.pending_stop_reason = Some(map_bedrock_stop(
                payload
                    .get("stopReason")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ));
        }
        "metadata" => {
            if let Some(usage) = payload.get("usage") {
                output.push(WireDelta::Usage(parse_bedrock_usage(usage)));
            }
            output.push(WireDelta::Done(
                state.pending_stop_reason.unwrap_or(StopReason::Stop),
            ));
        }
        "internalServerException"
        | "modelStreamErrorException"
        | "validationException"
        | "throttlingException"
        | "serviceUnavailableException" => {
            output.push(WireDelta::Error(format!(
                "{}: {}",
                bedrock_error_prefix(event_name),
                payload
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Bedrock stream failed")
            )));
        }
        _ => {}
    }
    Ok(output)
}

fn ensure_bedrock_block(
    wire_index: u64,
    kind: &str,
    index: usize,
    state: &mut WireDecodeState,
    output: &mut Vec<WireDelta>,
) {
    let key = format!("bedrock:kind:{wire_index}");
    if state.get(&key).is_none() {
        state.set(key, Value::String(kind.to_owned()));
        output.push(if kind == "thinking" {
            WireDelta::ThinkingStart { index }
        } else {
            WireDelta::TextStart { index }
        });
    }
}

fn parse_bedrock_usage(value: &Value) -> Usage {
    let input = value
        .get("inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = value
        .get("outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = value
        .get("cacheReadInputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = value
        .get("cacheWriteInputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: value
            .get("totalTokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                input
                    .saturating_add(output)
                    .saturating_add(cache_read)
                    .saturating_add(cache_write)
            }),
        cost: UsageCost::default(),
    }
}

fn map_bedrock_stop(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

fn bedrock_error_prefix(kind: &str) -> &str {
    match kind {
        "internalServerException" | "InternalServerException" => "Internal server error",
        "modelStreamErrorException" | "ModelStreamErrorException" => "Model stream error",
        "validationException" | "ValidationException" => "Validation error",
        "throttlingException" | "ThrottlingException" => "Throttling error",
        "serviceUnavailableException" | "ServiceUnavailableException" => "Service unavailable",
        _ => kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ConstrainedSampling, JsonSchemaStrictness, Tool};

    #[test]
    fn endpoint_percent_encodes_model_arns() {
        let model = Model::new(
            "amazon-bedrock",
            "arn:aws:bedrock:us-east-1:123:application-inference-profile/profile",
            "bedrock-converse-stream",
            "https://bedrock-runtime.us-east-1.amazonaws.com",
        );
        let url = bedrock_endpoint(&model).expect("url");
        assert!(
            url.as_str()
                .contains("application-inference-profile%2Fprofile")
        );
        assert!(url.as_str().ends_with("/converse-stream"));
    }

    #[test]
    fn required_strict_tools_fail_for_unsupported_models() {
        let model = Model::new(
            "amazon-bedrock",
            "model",
            "bedrock-converse-stream",
            "https://example.test",
        );
        let mut tool = Tool::new("lookup", "Lookup", json!({"type": "object"}));
        tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: JsonSchemaStrictness::Require,
        });
        let context = Context {
            tools: vec![tool],
            ..Context::default()
        };
        assert!(build_bedrock_body(&model, &context, &WireRequestOptions::default()).is_err());
    }

    #[test]
    fn claude_budget_thinking_preserves_requested_output_capacity() {
        let mut model = Model::new(
            "amazon-bedrock",
            "anthropic.claude-3-7-sonnet",
            "bedrock-converse-stream",
            "https://example.test",
        );
        model.name = "Claude 3.7 Sonnet".into();
        model.reasoning = true;
        model.max_tokens = 64_000;
        let body = build_bedrock_body(
            &model,
            &Context::default(),
            &WireRequestOptions {
                max_tokens: Some(2_000),
                reasoning: Some(ThinkingLevel::High),
                ..WireRequestOptions::default()
            },
        )
        .expect("body");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 18_384);
        assert_eq!(
            body["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            16_384
        );
    }

    #[test]
    fn decoder_handles_reasoning_signature_and_metadata() {
        let mut state = WireDecodeState::default();
        let delta = SseFrame {
            event: Some("contentBlockDelta".into()),
            data: json!({
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "think", "signature": "sig"}}
            })
            .to_string(),
            ..SseFrame::default()
        };
        let events = decode_bedrock_event(&delta, &mut state).expect("delta");
        assert!(events.iter().any(|event| matches!(
            event,
            WireDelta::ThinkingDelta { delta, .. } if delta == "think"
        )));
        let stop = SseFrame {
            event: Some("contentBlockStop".into()),
            data: json!({"contentBlockIndex": 0}).to_string(),
            ..SseFrame::default()
        };
        let events = decode_bedrock_event(&stop, &mut state).expect("stop");
        assert!(events.iter().any(|event| matches!(
            event,
            WireDelta::ThinkingEnd { signature: Some(signature), .. } if signature == "sig"
        )));
    }

    #[test]
    fn usage_keeps_cache_tokens_separate() {
        let usage = parse_bedrock_usage(&json!({
            "inputTokens": 10,
            "outputTokens": 3,
            "cacheReadInputTokens": 8,
            "cacheWriteInputTokens": 2,
            "totalTokens": 23
        }));
        assert_eq!(usage.input, 10);
        assert_eq!(usage.cache_read, 8);
        assert_eq!(usage.cache_write, 2);
        assert_eq!(usage.total_tokens, 23);
    }
}
