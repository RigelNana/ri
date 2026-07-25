//! Mistral Conversations (`chat/completions`) wire adapter.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
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
    WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request, merge_wire_headers,
    resolve_cache_retention,
};

/// Adapter for Mistral's streaming conversations API.
#[derive(Clone, Copy, Debug, Default)]
pub struct MistralConversationsAdapter;

impl WireAdapter for MistralConversationsAdapter {
    fn api(&self) -> &'static str {
        "mistral-conversations"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let base = model.base_url.trim_end_matches('/');
        let endpoint = if base.ends_with("chat/completions") {
            base.to_owned()
        } else if base.ends_with("/v1") {
            format!("{base}/chat/completions")
        } else {
            format!("{base}/v1/chat/completions")
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
        if resolve_cache_retention(options) != CacheRetention::None
            && let Some(session_id) = &options.session_id
        {
            defaults.push(("x-affinity".into(), session_id.clone()));
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
        let body = build_mistral_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_mistral_frame(frame, state)
    }
}

fn build_mistral_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
    let mut id_map = indexmap::IndexMap::<String, String>::new();
    let mut reverse = std::collections::HashSet::<String>::new();
    let mut normalizer = |id: &str, _: &Model, _: &crate::message::AssistantMessage| {
        if let Some(mapped) = id_map.get(id) {
            return mapped.clone();
        }
        let mut attempt = 0usize;
        loop {
            let mapped = derive_mistral_tool_id(id, attempt);
            if reverse.insert(mapped.clone()) {
                id_map.insert(id.to_owned(), mapped.clone());
                return mapped;
            }
            attempt += 1;
        }
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalizer));
    let mut converted = convert_mistral_messages(model, messages);
    if let Some(system) = &context.system_prompt {
        converted.insert(0, json!({"role": "system", "content": system}));
    }
    let mut body = Map::from_iter([
        ("model".into(), Value::String(model.id.clone())),
        ("stream".into(), Value::Bool(true)),
        ("messages".into(), Value::Array(converted)),
    ]);
    if let Some(max_tokens) = options.max_tokens {
        body.insert("max_tokens".into(), Value::Number(max_tokens.into()));
    }
    if let Some(temperature) = options.temperature {
        body.insert(
            "temperature".into(),
            Value::Number(
                serde_json::Number::from_f64(temperature)
                    .ok_or_else(|| AiError::Validation("temperature must be finite".into()))?,
            ),
        );
    }
    if let Some(tool_choice) = &options.tool_choice {
        body.insert("tool_choice".into(), Value::String(tool_choice.clone()));
    }
    if resolve_cache_retention(options) != CacheRetention::None
        && let Some(session_id) = &options.session_id
    {
        body.insert("prompt_cache_key".into(), Value::String(session_id.clone()));
    }
    if model.reasoning {
        let reasoning = options.reasoning.unwrap_or(ThinkingLevel::Off);
        if reasoning != ThinkingLevel::Off {
            if uses_reasoning_effort(&model.id) {
                let mapped = model
                    .thinking_level_map
                    .get(&reasoning)
                    .and_then(Clone::clone)
                    .unwrap_or_else(|| "high".into());
                body.insert("reasoning_effort".into(), Value::String(mapped));
            } else {
                body.insert("prompt_mode".into(), Value::String("reasoning".into()));
            }
        }
    }
    if !context.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(
                context
                    .tools
                    .iter()
                    .map(|tool| {
                        let strict =
                            resolve_json_schema_strict(tool, true).ok().flatten() == Some(true);
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                                "strict": strict
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    Ok(Value::Object(body))
}

fn convert_mistral_messages(model: &Model, messages: Vec<Message>) -> Vec<Value> {
    let mut output = Vec::new();
    for message in messages {
        match message {
            Message::User(message) => match message.content {
                UserContent::Text(text) => {
                    output.push(json!({"role": "user", "content": text}));
                }
                UserContent::Blocks(blocks) => {
                    let had_images = blocks
                        .iter()
                        .any(|block| matches!(block, InputContent::Image(_)));
                    let content = blocks
                        .into_iter()
                        .filter_map(|block| match block {
                            InputContent::Text(text) => {
                                Some(json!({"type": "text", "text": text.text}))
                            }
                            InputContent::Image(image) if model.supports_images() => {
                                Some(json!({
                                    "type": "image_url",
                                    "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
                                }))
                            }
                            InputContent::Image(_) => None,
                        })
                        .collect::<Vec<_>>();
                    if !content.is_empty() {
                        output.push(json!({"role": "user", "content": content}));
                    } else if had_images {
                        output.push(json!({
                            "role": "user",
                            "content": "(image omitted: model does not support images)"
                        }));
                    }
                }
            },
            Message::Assistant(message) => {
                let mut content = Vec::new();
                let mut tools = Vec::new();
                for block in message.content {
                    match block {
                        ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                            content.push(json!({"type": "text", "text": text.text}));
                        }
                        ContentBlock::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            content.push(json!({
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": thinking.thinking}]
                            }));
                        }
                        ContentBlock::ToolCall(call) => tools.push(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into())
                            }
                        })),
                        _ => {}
                    }
                }
                if !content.is_empty() || !tools.is_empty() {
                    let mut value = json!({"role": "assistant"});
                    if !content.is_empty() {
                        value["content"] = Value::Array(content);
                    }
                    if !tools.is_empty() {
                        value["tool_calls"] = Value::Array(tools);
                    }
                    output.push(value);
                }
            }
            Message::ToolResult(message) => {
                let text = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        InputContent::Text(text) => Some(text.text.as_str()),
                        InputContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = message
                    .content
                    .iter()
                    .any(|block| matches!(block, InputContent::Image(_)));
                let tool_text =
                    mistral_tool_text(&text, has_images, model.supports_images(), message.is_error);
                let mut content = vec![json!({"type": "text", "text": tool_text})];
                if model.supports_images() {
                    content.extend(message.content.iter().filter_map(|block| match block {
                        InputContent::Image(image) => Some(json!({
                            "type": "image_url",
                            "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
                        })),
                        InputContent::Text(_) => None,
                    }));
                }
                output.push(json!({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.tool_name,
                    "content": content
                }));
            }
        }
    }
    output
}

fn mistral_tool_text(
    text: &str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> String {
    let prefix = if is_error { "[tool error] " } else { "" };
    let text = text.trim();
    if !text.is_empty() {
        let suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{prefix}{text}{suffix}");
    }
    match (has_images, supports_images) {
        (true, true) => format!("{prefix}(see attached image)"),
        (true, false) => format!("{prefix}(image omitted: model does not support images)"),
        (false, _) => format!("{prefix}(no tool output)"),
    }
}

fn derive_mistral_tool_id(id: &str, attempt: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let normalized = id
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>();
    if attempt == 0 && normalized.len() == 9 {
        return normalized;
    }
    let seed = if attempt == 0 {
        if normalized.is_empty() {
            id.to_owned()
        } else {
            normalized
        }
    } else {
        format!(
            "{}:{attempt}",
            if normalized.is_empty() {
                id
            } else {
                &normalized
            }
        )
    };
    let digest = Sha256::digest(seed.as_bytes());
    let mut output = String::with_capacity(10);
    for byte in &digest[..5] {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output.truncate(9);
    output
}

fn uses_reasoning_effort(model: &str) -> bool {
    matches!(
        model,
        "mistral-small-2603" | "mistral-small-latest" | "mistral-medium-3.5"
    )
}

fn decode_mistral_frame(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    if frame.data.trim() == "[DONE]" {
        let mut output = Vec::new();
        finish_mistral_content(state, &mut output);
        finish_mistral_tools(state, &mut output);
        output.push(WireDelta::Done(
            state.pending_stop_reason.unwrap_or(StopReason::Stop),
        ));
        return Ok(output);
    }
    if frame.data.trim().is_empty() {
        return Ok(Vec::new());
    }
    let event: Value = serde_json::from_str(&frame.data)
        .map_err(|error| AiError::Stream(format!("invalid Mistral event: {error}")))?;
    let chunk = event.get("data").unwrap_or(&event);
    if let Some(error) = chunk.pointer("/error/message").and_then(Value::as_str) {
        return Ok(vec![WireDelta::Error(error.to_owned())]);
    }
    let mut output = Vec::new();
    if let Some(id) = chunk.get("id").and_then(Value::as_str) {
        output.push(WireDelta::ResponseId(id.to_owned()));
    }
    if let Some(usage) = chunk.get("usage") {
        output.push(WireDelta::Usage(parse_mistral_usage(usage)));
    }
    let Some(choice) = chunk.pointer("/choices/0") else {
        return Ok(output);
    };
    if let Some(reason) = choice
        .get("finish_reason")
        .or_else(|| choice.get("finishReason"))
        .and_then(Value::as_str)
    {
        state.pending_stop_reason = Some(map_mistral_stop(reason));
    }
    let delta = choice.get("delta").unwrap_or(&Value::Null);
    if let Some(content) = delta.get("content") {
        let items = content
            .as_array()
            .cloned()
            .unwrap_or_else(|| vec![content.clone()]);
        for item in items {
            if let Some(text) = item.as_str() {
                mistral_content_delta("text", text, state, &mut output);
            } else {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => mistral_content_delta(
                        "text",
                        item.get("text").and_then(Value::as_str).unwrap_or_default(),
                        state,
                        &mut output,
                    ),
                    Some("thinking") => {
                        let text = item
                            .get("thinking")
                            .and_then(Value::as_array)
                            .map(|parts| {
                                parts
                                    .iter()
                                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                                    .collect::<String>()
                            })
                            .unwrap_or_default();
                        if !text.is_empty() {
                            mistral_content_delta("thinking", &text, state, &mut output);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(tool_calls) = delta
        .get("tool_calls")
        .or_else(|| delta.get("toolCalls"))
        .and_then(Value::as_array)
    {
        finish_mistral_content(state, &mut output);
        for call in tool_calls {
            let wire_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let existing_key = state
                .get(&format!("toolkey:{wire_index}"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| *id != "null")
                .map(str::to_owned)
                .or_else(|| {
                    state
                        .get(&format!("toolid:{wire_index}"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| derive_mistral_tool_id(&format!("toolcall:{wire_index}"), 0));
            let key = existing_key.unwrap_or_else(|| format!("mistral:tool:{id}:{wire_index}"));
            let index = state.slot(&key);
            if state.get(&format!("started:{key}")).is_none() {
                state.set(format!("started:{key}"), Value::Bool(true));
                state.set(format!("toolkey:{wire_index}"), Value::String(key.clone()));
                state.set(format!("toolid:{wire_index}"), Value::String(id.clone()));
                output.push(WireDelta::ToolStart {
                    index,
                    id,
                    name: call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    thought_signature: None,
                });
            }
            let arguments = call
                .pointer("/function/arguments")
                .map(|arguments| {
                    arguments.as_str().map_or_else(
                        || serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into()),
                        str::to_owned,
                    )
                })
                .unwrap_or_default();
            if !arguments.is_empty() {
                let raw_key = format!("raw:{key}");
                let mut raw = state
                    .take(&raw_key)
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_default();
                raw.push_str(&arguments);
                state.set(raw_key, Value::String(raw));
                output.push(WireDelta::ToolDelta {
                    index,
                    delta: arguments,
                });
            }
            let count = state
                .get("mistral:tool_count")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            state.set(
                "mistral:tool_count",
                Value::Number(count.max(wire_index + 1).into()),
            );
        }
    }
    if choice.get("finish_reason").is_some() || choice.get("finishReason").is_some() {
        finish_mistral_content(state, &mut output);
        finish_mistral_tools(state, &mut output);
    }
    Ok(output)
}

fn finish_mistral_tools(state: &mut WireDecodeState, output: &mut Vec<WireDelta>) {
    let count = state
        .get("mistral:tool_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    for wire_index in 0..count {
        let Some(key) = state
            .get(&format!("toolkey:{wire_index}"))
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let ended_key = format!("ended:{key}");
        if state.get(&ended_key).is_some() {
            continue;
        }
        state.set(ended_key, Value::Bool(true));
        let index = state.slot(&key);
        let raw = state
            .take(&format!("raw:{key}"))
            .and_then(|value| value.as_str().map(str::to_owned));
        output.push(WireDelta::ToolEnd {
            index,
            arguments: raw
                .as_deref()
                .map(|raw| crate::tool::partial_json::parse_streaming_json(Some(raw))),
        });
    }
}

fn mistral_content_delta(
    kind: &str,
    text: &str,
    state: &mut WireDecodeState,
    output: &mut Vec<WireDelta>,
) {
    let changed = state
        .get("mistral:current")
        .and_then(Value::as_str)
        .is_some_and(|current| current != kind);
    if changed {
        finish_mistral_content(state, output);
    }
    if state.get("mistral:current").is_none() {
        let block = state
            .get("mistral:block_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        state.set("mistral:block_count", Value::Number((block + 1).into()));
        let key = format!("mistral:block:{block}");
        state.set("mistral:current", Value::String(kind.to_owned()));
        state.set("mistral:current_key", Value::String(key.clone()));
        let index = state.slot(key);
        output.push(if kind == "thinking" {
            WireDelta::ThinkingStart { index }
        } else {
            WireDelta::TextStart { index }
        });
    }
    let key = state
        .get("mistral:current_key")
        .and_then(Value::as_str)
        .unwrap_or("mistral:block:0")
        .to_owned();
    let index = state.slot(key);
    output.push(if kind == "thinking" {
        WireDelta::ThinkingDelta {
            index,
            delta: text.to_owned(),
        }
    } else {
        WireDelta::TextDelta {
            index,
            delta: text.to_owned(),
        }
    });
}

fn finish_mistral_content(state: &mut WireDecodeState, output: &mut Vec<WireDelta>) {
    let Some(kind) = state
        .take("mistral:current")
        .and_then(|value| value.as_str().map(str::to_owned))
    else {
        return;
    };
    let key = state
        .take("mistral:current_key")
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "mistral:block:unknown".into());
    let index = state.slot(key);
    output.push(if kind == "thinking" {
        WireDelta::ThinkingEnd {
            index,
            content: None,
            signature: None,
            redacted: false,
        }
    } else {
        WireDelta::TextEnd {
            index,
            content: None,
            signature: None,
        }
    });
}

fn parse_mistral_usage(value: &Value) -> Usage {
    let prompt = value
        .get("prompt_tokens")
        .or_else(|| value.get("promptTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = value
        .pointer("/prompt_tokens_details/cached_tokens")
        .or_else(|| value.pointer("/promptTokensDetails/cachedTokens"))
        .or_else(|| value.get("num_cached_tokens"))
        .or_else(|| value.get("numCachedTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(prompt);
    let output = value
        .get("completion_tokens")
        .or_else(|| value.get("completionTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input: prompt.saturating_sub(cached),
        output,
        cache_read: cached,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: value
            .get("total_tokens")
            .or_else(|| value.get("totalTokens"))
            .and_then(Value::as_u64)
            .unwrap_or(prompt.saturating_add(output)),
        cost: UsageCost::default(),
    }
}

fn map_mistral_stop(reason: &str) -> StopReason {
    match reason {
        "length" | "model_length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        "error" => StopReason::Error,
        _ => StopReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_ids_are_exactly_nine_alphanumeric_characters() {
        let id = derive_mistral_tool_id("call_123/with punctuation", 0);
        assert_eq!(id.len(), 9);
        assert!(
            id.chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
        assert_eq!(derive_mistral_tool_id("Abc123xyz", 0), "Abc123xyz");
    }

    #[test]
    fn request_targets_mistral_v1_chat_endpoint() {
        let model = Model::new(
            "mistral",
            "model",
            "mistral-conversations",
            "https://api.mistral.ai",
        );
        let request = MistralConversationsAdapter
            .build_request(
                &model,
                &Context::default(),
                &WireRequestOptions {
                    api_key: Some("key".into()),
                    ..WireRequestOptions::default()
                },
            )
            .expect("request");
        assert_eq!(
            request.url.as_str(),
            "https://api.mistral.ai/v1/chat/completions"
        );
    }

    #[test]
    fn tool_images_have_explicit_fallback() {
        assert_eq!(
            mistral_tool_text("", true, false, true),
            "[tool error] (image omitted: model does not support images)"
        );
    }

    #[test]
    fn usage_recognizes_common_cache_fields() {
        let usage = parse_mistral_usage(&json!({
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 8}
        }));
        assert_eq!(usage.input, 12);
        assert_eq!(usage.cache_read, 8);
    }
}
