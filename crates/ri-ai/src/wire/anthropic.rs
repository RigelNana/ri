//! Anthropic Messages request and SSE adapter.

use std::collections::BTreeSet;

use serde_json::{Value, json};
use url::Url;

use crate::{
    error::AiError,
    handoff::transform_messages,
    message::{ContentBlock, Context, InputContent, Message, StopReason, Usage},
    model::{CacheRetention, Model, ThinkingLevel},
    tool::{ToolDescriptor, describe_tool, split_deferred_tools},
    transport::{HttpRequest, SseFrame},
};

use super::{
    WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request, merge_wire_headers,
    resolve_cache_retention,
};

/// Anthropic Messages protocol adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessagesAdapter;

impl WireAdapter for AnthropicMessagesAdapter {
    fn api(&self) -> &'static str {
        "anthropic-messages"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = anthropic_endpoint(&model.base_url)?;
        let mut defaults = vec![
            ("accept".into(), "text/event-stream".into()),
            ("content-type".into(), "application/json".into()),
            ("anthropic-version".into(), "2023-06-01".into()),
        ];
        if let Some(api_key) = &options.api_key {
            if api_key.contains("sk-ant-oat") {
                defaults.push(("authorization".into(), format!("Bearer {api_key}")));
                defaults.push((
                    "anthropic-beta".into(),
                    "claude-code-20250219,oauth-2025-04-20".into(),
                ));
            } else {
                defaults.push(("x-api-key".into(), api_key.clone()));
            }
        }
        let compat = model.compat.as_ref();
        if resolve_cache_retention(options) != CacheRetention::None
            && compat.and_then(|compat| compat.send_session_affinity_headers) == Some(true)
            && let Some(session_id) = &options.session_id
        {
            defaults.push(("x-session-affinity".into(), session_id.clone()));
        }
        let headers = merge_wire_headers(defaults, model, &options.headers);
        if !headers.keys().any(|name| {
            name.eq_ignore_ascii_case("authorization")
                || name.eq_ignore_ascii_case("x-api-key")
                || name.eq_ignore_ascii_case("cf-aig-authorization")
        }) {
            return Err(AiError::Auth(format!(
                "no API key for provider {}",
                model.provider
            )));
        }

        let body = build_anthropic_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        if frame.data.trim().is_empty() {
            return Ok(Vec::new());
        }
        let event: Value = serde_json::from_str(&frame.data).map_err(|error| {
            AiError::Stream(format!(
                "invalid Anthropic SSE event {:?}: {error}; data={}",
                frame.event, frame.data
            ))
        })?;
        Ok(decode_anthropic_event(&event, state))
    }
}

fn anthropic_endpoint(base_url: &str) -> Result<Url, AiError> {
    let base = base_url.trim_end_matches('/');
    let endpoint = if base.ends_with("/v1") {
        format!("{base}/messages")
    } else {
        format!("{base}/v1/messages")
    };
    Url::parse(&endpoint).map_err(|error| AiError::Validation(error.to_string()))
}

fn build_anthropic_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
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
            .collect::<String>()
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalize));
    let cache_control = cache_control(model, resolve_cache_retention(options));
    let adaptive_thinking = model
        .compat
        .as_ref()
        .and_then(|compat| compat.force_adaptive_thinking)
        == Some(true);
    let reasoning_level = options
        .reasoning
        .filter(|level| *level != ThinkingLevel::Off);
    let max_tokens = if let Some(level) = reasoning_level.filter(|_| !adaptive_thinking) {
        options
            .max_tokens
            .map_or(model.max_tokens, |output_tokens| {
                output_tokens
                    .saturating_add(thinking_budget(level))
                    .min(model.max_tokens)
            })
    } else {
        options
            .max_tokens
            .unwrap_or(model.max_tokens)
            .min(model.max_tokens)
    };
    let deferred_enabled = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_tool_references)
        .unwrap_or_else(|| {
            model.provider == "anthropic" && !model.id.contains("haiku") && model.id.contains("4-5")
        });
    let transformed_context = Context {
        system_prompt: context.system_prompt.clone(),
        messages: messages.clone(),
        tools: context.tools.clone(),
    };
    let mut placement = split_deferred_tools(&transformed_context, deferred_enabled, str::to_owned);
    if placement.immediate.is_empty() && !placement.deferred.is_empty() {
        placement
            .immediate
            .extend(placement.deferred.drain(..).map(|(_, tool)| tool));
    }
    let deferred_names = placement.deferred.keys().cloned().collect::<Vec<_>>();
    let mut body = serde_json::Map::from_iter([
        ("model".into(), Value::String(model.id.clone())),
        ("stream".into(), Value::Bool(true)),
        ("max_tokens".into(), Value::Number(max_tokens.into())),
        (
            "messages".into(),
            convert_anthropic_messages(&messages, &deferred_names),
        ),
    ]);

    if let Some(system_prompt) = &context.system_prompt {
        let mut system = json!({"type": "text", "text": system_prompt});
        if let Some(cache_control) = &cache_control {
            system["cache_control"] = cache_control.clone();
        }
        body.insert("system".into(), Value::Array(vec![system]));
    }

    let supports_strict = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_strict_tools)
        .unwrap_or(model.provider == "anthropic");
    let mut tools = Vec::new();
    for tool in &placement.immediate {
        tools.push(anthropic_tool(
            describe_tool(tool, supports_strict, false, false)
                .map_err(|error| AiError::Validation(error.to_string()))?,
            cache_control.as_ref(),
        ));
    }
    for tool in placement.deferred.values() {
        tools.push(anthropic_tool(
            describe_tool(tool, supports_strict, false, true)
                .map_err(|error| AiError::Validation(error.to_string()))?,
            None,
        ));
    }
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }

    if let Some(temperature) = options.temperature
        && matches!(options.reasoning, None | Some(ThinkingLevel::Off))
        && model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_temperature)
            != Some(false)
    {
        body.insert(
            "temperature".into(),
            Value::Number(
                serde_json::Number::from_f64(temperature)
                    .ok_or_else(|| AiError::Validation("temperature must be finite".into()))?,
            ),
        );
    }

    if model.reasoning {
        match options.reasoning {
            Some(ThinkingLevel::Off) | None => {
                body.insert("thinking".into(), json!({"type": "disabled"}));
            }
            Some(level) => {
                if adaptive_thinking {
                    body.insert(
                        "thinking".into(),
                        json!({"type": "adaptive", "display": "summarized"}),
                    );
                    let effort = model
                        .thinking_level_map
                        .get(&level)
                        .and_then(Clone::clone)
                        .unwrap_or_else(|| level.as_str().to_owned());
                    body.insert("output_config".into(), json!({"effort": effort}));
                } else {
                    body.insert(
                        "thinking".into(),
                        json!({
                            "type": "enabled",
                            "budget_tokens": thinking_budget(level).min(max_tokens.saturating_sub(1_024)),
                            "display": "summarized"
                        }),
                    );
                }
            }
        }
    }
    if let Some(choice) = &options.tool_choice {
        let choice = if choice == "required" { "any" } else { choice };
        body.insert("tool_choice".into(), json!({"type": choice}));
    }
    Ok(Value::Object(body))
}

fn cache_control(model: &Model, retention: CacheRetention) -> Option<Value> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(json!({"type": "ephemeral"})),
        CacheRetention::Long => {
            if model
                .compat
                .as_ref()
                .and_then(|compat| compat.supports_long_cache_retention)
                == Some(false)
            {
                Some(json!({"type": "ephemeral"}))
            } else {
                Some(json!({"type": "ephemeral", "ttl": "1h"}))
            }
        }
    }
}

fn thinking_budget(level: ThinkingLevel) -> u64 {
    match level {
        ThinkingLevel::Off => 0,
        ThinkingLevel::Minimal => 1_024,
        ThinkingLevel::Low => 2_048,
        ThinkingLevel::Medium => 8_192,
        ThinkingLevel::High => 16_384,
        ThinkingLevel::Xhigh => 32_768,
        ThinkingLevel::Max => 64_000,
    }
}

fn anthropic_tool(descriptor: ToolDescriptor, cache_control: Option<&Value>) -> Value {
    match descriptor {
        ToolDescriptor::Function {
            name,
            description,
            parameters,
            strict,
            defer_loading,
        } => {
            let mut value = json!({
                "name": name,
                "description": description,
                "input_schema": parameters,
                "eager_input_streaming": true
            });
            if let Some(strict) = strict {
                value["strict"] = Value::Bool(strict);
            }
            if defer_loading {
                value["defer_loading"] = Value::Bool(true);
            }
            if let Some(cache_control) = cache_control {
                value["cache_control"] = cache_control.clone();
            }
            value
        }
        ToolDescriptor::Custom { .. } => Value::Null,
    }
}

fn convert_anthropic_messages(messages: &[Message], deferred_names: &[String]) -> Value {
    let mut output = Vec::<Value>::new();
    let mut loaded = BTreeSet::<String>::new();
    for message in messages {
        match message {
            Message::User(message) => {
                let content = match &message.content {
                    crate::message::UserContent::Text(text) => {
                        Value::Array(vec![json!({"type": "text", "text": text})])
                    }
                    crate::message::UserContent::Blocks(blocks) => {
                        Value::Array(blocks.iter().map(anthropic_input_block).collect())
                    }
                };
                output.push(json!({"role": "user", "content": content}));
            }
            Message::Assistant(message) => {
                let content = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) if !text.text.is_empty() => {
                            Some(json!({"type": "text", "text": text.text}))
                        }
                        ContentBlock::Thinking(thinking) if thinking.redacted => Some(json!({
                            "type": "redacted_thinking",
                            "data": thinking.thinking_signature
                        })),
                        ContentBlock::Thinking(thinking)
                            if thinking.thinking_signature.is_some() =>
                        {
                            Some(json!({
                                "type": "thinking",
                                "thinking": thinking.thinking,
                                "signature": thinking.thinking_signature
                            }))
                        }
                        ContentBlock::ToolCall(call) => Some(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments
                        })),
                        ContentBlock::Text(_) | ContentBlock::Thinking(_) => None,
                    })
                    .collect::<Vec<_>>();
                if !content.is_empty() {
                    output.push(json!({"role": "assistant", "content": content}));
                }
            }
            Message::ToolResult(message) => {
                let mut references = Vec::new();
                for name in &message.added_tool_names {
                    if deferred_names.contains(name) && !loaded.contains(name) {
                        loaded.insert(name.clone());
                        references.push(json!({"type": "tool_reference", "tool_name": name}));
                    }
                }
                let ordinary = message
                    .content
                    .iter()
                    .map(anthropic_input_block)
                    .collect::<Vec<_>>();
                let tool_content = if references.is_empty() {
                    ordinary
                } else {
                    references
                };
                let mut user_content = vec![json!({
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": tool_content,
                    "is_error": message.is_error
                })];
                if !message.added_tool_names.is_empty() {
                    user_content.extend(message.content.iter().map(anthropic_input_block));
                }
                output.push(json!({"role": "user", "content": user_content}));
            }
        }
    }
    Value::Array(output)
}

fn anthropic_input_block(block: &InputContent) -> Value {
    match block {
        InputContent::Text(text) => json!({"type": "text", "text": text.text}),
        InputContent::Image(image) => json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": image.data
            }
        }),
    }
}

fn decode_anthropic_event(event: &Value, state: &mut WireDecodeState) -> Vec<WireDelta> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut deltas = Vec::new();
    match event_type {
        "message_start" => {
            if let Some(id) = event.pointer("/message/id").and_then(Value::as_str) {
                deltas.push(WireDelta::ResponseId(id.to_owned()));
            }
            if let Some(model) = event.pointer("/message/model").and_then(Value::as_str) {
                deltas.push(WireDelta::ResponseModel(model.to_owned()));
            }
            let usage = anthropic_usage(event.pointer("/message/usage"), None);
            state.set("usage", serde_json::to_value(&usage).unwrap_or_default());
            deltas.push(WireDelta::Usage(usage));
        }
        "content_block_start" => {
            let wire_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
            let key = format!("anthropic:{wire_index}");
            let index = state.slot(key);
            let block = event.get("content_block").unwrap_or(&Value::Null);
            let kind = block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            state.set(format!("kind:{index}"), Value::String(kind.to_owned()));
            match kind {
                "text" => {
                    deltas.push(WireDelta::TextStart { index });
                    if let Some(text) = block.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        deltas.push(WireDelta::TextDelta {
                            index,
                            delta: text.to_owned(),
                        });
                    }
                }
                "thinking" | "redacted_thinking" => {
                    deltas.push(WireDelta::ThinkingStart { index });
                    if let Some(text) = block.get("thinking").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        deltas.push(WireDelta::ThinkingDelta {
                            index,
                            delta: text.to_owned(),
                        });
                    }
                    if let Some(signature) = block
                        .get("signature")
                        .or_else(|| block.get("data"))
                        .and_then(Value::as_str)
                    {
                        state.set(
                            format!("signature:{index}"),
                            Value::String(signature.to_owned()),
                        );
                    }
                }
                "tool_use" => {
                    deltas.push(WireDelta::ToolStart {
                        index,
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        thought_signature: None,
                    });
                    if block.get("input").is_some_and(|input| input != &json!({})) {
                        deltas.push(WireDelta::ToolEnd {
                            index,
                            arguments: block.get("input").cloned(),
                        });
                    }
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let wire_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
            let index = state.slot(format!("anthropic:{wire_index}"));
            let delta = event.get("delta").unwrap_or(&Value::Null);
            match delta
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "text_delta" => deltas.push(WireDelta::TextDelta {
                    index,
                    delta: delta
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }),
                "thinking_delta" => deltas.push(WireDelta::ThinkingDelta {
                    index,
                    delta: delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }),
                "signature_delta" => {
                    if let Some(signature) = delta.get("signature").and_then(Value::as_str) {
                        let key = format!("signature:{index}");
                        let mut current = state
                            .take(&key)
                            .and_then(|value| value.as_str().map(str::to_owned))
                            .unwrap_or_default();
                        current.push_str(signature);
                        state.set(key, Value::String(current));
                    }
                }
                "input_json_delta" => deltas.push(WireDelta::ToolDelta {
                    index,
                    delta: delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                }),
                _ => {}
            }
        }
        "content_block_stop" => {
            let wire_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
            let index = state.slot(format!("anthropic:{wire_index}"));
            let kind = state
                .get(&format!("kind:{index}"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let signature = state
                .take(&format!("signature:{index}"))
                .and_then(|value| value.as_str().map(str::to_owned));
            match kind.as_str() {
                "text" => deltas.push(WireDelta::TextEnd {
                    index,
                    content: None,
                    signature,
                }),
                "thinking" | "redacted_thinking" => deltas.push(WireDelta::ThinkingEnd {
                    index,
                    content: None,
                    signature,
                    redacted: kind == "redacted_thinking",
                }),
                "tool_use" => deltas.push(WireDelta::ToolEnd {
                    index,
                    arguments: None,
                }),
                _ => {}
            }
        }
        "message_delta" => {
            if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                state.pending_stop_reason = Some(map_anthropic_stop(reason));
            }
            let current = state
                .get("usage")
                .cloned()
                .and_then(|value| serde_json::from_value::<Usage>(value).ok())
                .unwrap_or_default();
            let usage = anthropic_usage(event.get("usage"), Some(current));
            state.set("usage", serde_json::to_value(&usage).unwrap_or_default());
            deltas.push(WireDelta::Usage(usage));
        }
        "message_stop" => deltas.push(WireDelta::Done(
            state.pending_stop_reason.unwrap_or(StopReason::Stop),
        )),
        "error" => deltas.push(WireDelta::Error(
            event
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Anthropic stream error")
                .to_owned(),
        )),
        _ => {}
    }
    deltas
}

fn anthropic_usage(value: Option<&Value>, previous: Option<Usage>) -> Usage {
    let mut usage = previous.unwrap_or_default();
    let Some(value) = value else {
        return usage;
    };
    let input = value.get("input_tokens").and_then(Value::as_u64);
    let cache_read = value.get("cache_read_input_tokens").and_then(Value::as_u64);
    let short_write = value
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64);
    let long_write = value
        .pointer("/cache_creation/ephemeral_1h_input_tokens")
        .and_then(Value::as_u64);
    if let Some(input) = input {
        usage.input = input;
    }
    if let Some(cache_read) = cache_read {
        usage.cache_read = cache_read;
    }
    if let Some(cache_write) = short_write {
        usage.cache_write = cache_write;
    }
    if let Some(long_write) = long_write {
        usage.cache_write_1h = Some(long_write);
    }
    if let Some(output) = value.get("output_tokens").and_then(Value::as_u64) {
        usage.output = output;
    }
    usage.total_tokens = usage
        .input
        .saturating_add(usage.output)
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
    usage
}

fn map_anthropic_stop(reason: &str) -> StopReason {
    match reason {
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "end_turn" | "stop_sequence" | "pause_turn" | "refusal" => StopReason::Stop,
        _ => StopReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::{Message, UserMessage},
        model::{ModelCompatibility, ModelCostRates},
    };

    fn model() -> Model {
        let mut model = Model::new(
            "anthropic",
            "claude-test",
            "anthropic-messages",
            "https://api.anthropic.com",
        );
        model.reasoning = true;
        model.cost.rates = ModelCostRates::default();
        model.compat = Some(ModelCompatibility {
            force_adaptive_thinking: Some(true),
            supports_long_cache_retention: Some(true),
            ..ModelCompatibility::default()
        });
        model
    }

    #[test]
    fn builds_cache_and_adaptive_thinking_payload() {
        let context = Context {
            system_prompt: Some("system".into()),
            messages: vec![Message::User(UserMessage::new("hello"))],
            tools: Vec::new(),
        };
        let body = build_anthropic_body(
            &model(),
            &context,
            &WireRequestOptions {
                reasoning: Some(ThinkingLevel::High),
                cache_retention: Some(CacheRetention::Long),
                ..WireRequestOptions::default()
            },
        )
        .expect("payload");
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn explicit_reasoning_off_keeps_temperature_and_maps_required_tools() {
        let body = build_anthropic_body(
            &model(),
            &Context::default(),
            &WireRequestOptions {
                temperature: Some(0.5),
                reasoning: Some(ThinkingLevel::Off),
                tool_choice: Some("required".into()),
                ..WireRequestOptions::default()
            },
        )
        .expect("payload");
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["tool_choice"]["type"], "any");
    }

    #[test]
    fn budget_thinking_adds_reasoning_to_requested_output_cap() {
        let mut budget_model = model();
        budget_model.max_tokens = 64_000;
        budget_model
            .compat
            .as_mut()
            .expect("compatibility")
            .force_adaptive_thinking = Some(false);
        let body = build_anthropic_body(
            &budget_model,
            &Context::default(),
            &WireRequestOptions {
                max_tokens: Some(2_000),
                reasoning: Some(ThinkingLevel::High),
                ..WireRequestOptions::default()
            },
        )
        .expect("payload");
        assert_eq!(body["max_tokens"], 18_384);
        assert_eq!(body["thinking"]["budget_tokens"], 16_384);
    }

    #[test]
    fn decodes_interleaved_text_tool_and_usage() {
        let adapter = AnthropicMessagesAdapter;
        let mut state = WireDecodeState::default();
        let frames = [
            json!({"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":2}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
            json!({"type":"message_stop"}),
        ];
        let deltas = frames
            .into_iter()
            .flat_map(|value| {
                adapter
                    .decode_frame(
                        &SseFrame {
                            event: None,
                            data: value.to_string(),
                            ..SseFrame::default()
                        },
                        &mut state,
                    )
                    .expect("decode")
            })
            .collect::<Vec<_>>();
        assert!(deltas.contains(&WireDelta::TextDelta {
            index: 0,
            delta: "hi".into()
        }));
        assert!(deltas.contains(&WireDelta::Done(StopReason::Stop)));
    }
}
