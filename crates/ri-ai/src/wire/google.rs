//! Google Gemini Generative Language and Vertex AI adapters.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};
use url::Url;

use crate::{
    error::AiError,
    handoff::transform_messages,
    message::{
        ContentBlock, Context, InputContent, Message, StopReason, Usage, UsageCost, UserContent,
    },
    model::{Model, ThinkingLevel},
    tool::resolve_json_schema_strict,
    transport::{HttpRequest, SseFrame},
};

use super::{
    WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request, merge_wire_headers,
};

/// Gemini API adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoogleGenerativeAiAdapter;

/// Vertex AI Gemini adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoogleVertexAdapter;

impl WireAdapter for GoogleGenerativeAiAdapter {
    fn api(&self) -> &'static str {
        "google-generative-ai"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = google_endpoint(model, false)?;
        let headers = google_headers(model, options, false)?;
        let body = build_google_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_google_frame(frame, state)
    }
}

impl WireAdapter for GoogleVertexAdapter {
    fn api(&self) -> &'static str {
        "google-vertex"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = google_endpoint(model, true)?;
        let headers = google_headers(model, options, true)?;
        let body = build_google_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_google_frame(frame, state)
    }
}

fn google_endpoint(model: &Model, vertex: bool) -> Result<Url, AiError> {
    let base = model.base_url.trim_end_matches('/');
    let model_path = format!("models/{}", model.id);
    let url = if base.contains(":streamGenerateContent") {
        base.to_owned()
    } else if base.ends_with(&model_path) {
        format!("{base}:streamGenerateContent?alt=sse")
    } else {
        format!("{base}/{model_path}:streamGenerateContent?alt=sse")
    };
    let mut url = Url::parse(&url).map_err(|error| AiError::Validation(error.to_string()))?;
    if vertex && !url.query_pairs().any(|(key, _)| key == "alt") {
        url.query_pairs_mut().append_pair("alt", "sse");
    }
    Ok(url)
}

fn google_headers(
    model: &Model,
    options: &WireRequestOptions,
    vertex: bool,
) -> Result<crate::transport::HttpHeaders, AiError> {
    let mut defaults = vec![
        ("accept".into(), "text/event-stream".into()),
        ("content-type".into(), "application/json".into()),
    ];
    if let Some(api_key) = &options.api_key {
        let vertex_api_key = vertex
            && (api_key.starts_with("AIza")
                || options.extra.get("vertexApiKey").and_then(Value::as_bool) == Some(true));
        defaults.push(if vertex && !vertex_api_key {
            ("authorization".into(), format!("Bearer {api_key}"))
        } else {
            ("x-goog-api-key".into(), api_key.clone())
        });
    }
    let headers = merge_wire_headers(defaults, model, &options.headers);
    let has_auth = if vertex {
        headers.keys().any(|name| {
            name.eq_ignore_ascii_case("authorization")
                || name.eq_ignore_ascii_case("x-goog-api-key")
        })
    } else {
        headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("x-goog-api-key"))
    };
    if !has_auth {
        return Err(AiError::Auth(format!(
            "no API key for provider {}",
            model.provider
        )));
    }
    Ok(headers)
}

fn build_google_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
    let mut body = Map::from_iter([(
        "contents".into(),
        Value::Array(convert_google_messages(model, context)),
    )]);
    if let Some(system) = &context.system_prompt {
        body.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": system}]}),
        );
    }
    let mut generation = Map::new();
    if let Some(max_tokens) = options.max_tokens {
        generation.insert("maxOutputTokens".into(), Value::Number(max_tokens.into()));
    }
    if let Some(temperature) = options.temperature {
        generation.insert(
            "temperature".into(),
            Value::Number(
                serde_json::Number::from_f64(temperature)
                    .ok_or_else(|| AiError::Validation("temperature must be finite".into()))?,
            ),
        );
    }
    if model.reasoning {
        generation.insert(
            "thinkingConfig".into(),
            google_thinking_config(model, options),
        );
    }
    if !generation.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation));
    }
    if !context.tools.is_empty() {
        body.insert(
            "tools".into(),
            json!([{
                "functionDeclarations": context.tools.iter().map(|tool| json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parametersJsonSchema": tool.parameters
                })).collect::<Vec<_>>()
            }]),
        );
        let supports_validated = gemini_major(&model.id).is_some_and(|major| major >= 3);
        let strict = context.tools.iter().try_fold(false, |strict, tool| {
            resolve_json_schema_strict(tool, supports_validated)
                .map(|resolved| strict || resolved == Some(true))
                .map_err(|error| AiError::Validation(error.to_string()))
        })?;
        let (mode, allowed_name) = match options.tool_choice.as_deref() {
            Some("none") => (Some("NONE"), None),
            Some("any" | "required") => (Some("ANY"), None),
            Some(name) if name != "auto" && !name.is_empty() => (Some("ANY"), Some(name)),
            _ if strict => (Some("VALIDATED"), None),
            Some("auto") => (Some("AUTO"), None),
            _ => (None, None),
        };
        if let Some(mode) = mode {
            let mut function_calling = json!({"mode": mode});
            if let Some(name) = allowed_name {
                function_calling["allowedFunctionNames"] = json!([name]);
            }
            body.insert(
                "toolConfig".into(),
                json!({"functionCallingConfig": function_calling}),
            );
        }
    }
    Ok(Value::Object(body))
}

fn google_thinking_config(model: &Model, options: &WireRequestOptions) -> Value {
    let level = options.reasoning.unwrap_or(ThinkingLevel::Off);
    if level == ThinkingLevel::Off {
        let id = model.id.to_ascii_lowercase();
        if id.contains("gemini-3") || id.contains("gemma-4") || id.contains("gemma4") {
            let minimum = if id.contains("pro") { "LOW" } else { "MINIMAL" };
            return json!({"thinkingLevel": minimum});
        }
        return json!({"thinkingBudget": 0});
    }
    let mapped = model
        .thinking_level_map
        .get(&level)
        .and_then(Clone::clone)
        .unwrap_or_else(|| match level {
            ThinkingLevel::Minimal => "MINIMAL".into(),
            ThinkingLevel::Low => "LOW".into(),
            ThinkingLevel::Medium => "MEDIUM".into(),
            ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => "HIGH".into(),
            ThinkingLevel::Off => "THINKING_LEVEL_UNSPECIFIED".into(),
        });
    json!({"includeThoughts": true, "thinkingLevel": mapped})
}

fn convert_google_messages(model: &Model, context: &Context) -> Vec<Value> {
    let needs_id = requires_tool_call_id(&model.id);
    let mut normalizer = |id: &str, _: &Model, _: &crate::message::AssistantMessage| {
        if needs_id {
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
        } else {
            id.to_owned()
        }
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalizer));
    let mut contents: Vec<Value> = Vec::new();
    for message in messages {
        match message {
            Message::User(message) => {
                let parts = match message.content {
                    UserContent::Text(text) => vec![json!({"text": text})],
                    UserContent::Blocks(blocks) => blocks
                        .into_iter()
                        .map(|block| match block {
                            InputContent::Text(text) => json!({"text": text.text}),
                            InputContent::Image(image) => json!({
                                "inlineData": {
                                    "mimeType": image.mime_type,
                                    "data": image.data
                                }
                            }),
                        })
                        .collect::<Vec<_>>(),
                };
                if !parts.is_empty() {
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
            Message::Assistant(message) => {
                let same_model = message.provider == model.provider && message.model == model.id;
                let parts = message
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                            let signature = resolve_signature(same_model, text.text_signature);
                            let mut part = json!({"text": text.text});
                            if let Some(signature) = signature {
                                part["thoughtSignature"] = Value::String(signature);
                            }
                            Some(part)
                        }
                        ContentBlock::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            let signature =
                                resolve_signature(same_model, thinking.thinking_signature);
                            let mut part = json!({"text": thinking.thinking});
                            if same_model {
                                part["thought"] = Value::Bool(true);
                            }
                            if let Some(signature) = signature {
                                part["thoughtSignature"] = Value::String(signature);
                            }
                            Some(part)
                        }
                        ContentBlock::ToolCall(call) => {
                            let mut function = json!({
                                "name": call.name,
                                "args": call.arguments
                            });
                            if needs_id {
                                function["id"] = Value::String(call.id);
                            }
                            let mut part = json!({"functionCall": function});
                            if let Some(signature) =
                                resolve_signature(same_model, call.thought_signature)
                            {
                                part["thoughtSignature"] = Value::String(signature);
                            }
                            Some(part)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
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
                let images = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        InputContent::Image(image) if model.supports_images() => Some(image),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let response = if message.is_error {
                    json!({"error": if text.is_empty() { "(see attached image)" } else { &text }})
                } else {
                    json!({"output": if text.is_empty() && !images.is_empty() { "(see attached image)" } else { &text }})
                };
                let multimodal = supports_multimodal_function_response(&model.id);
                let mut function = json!({
                    "name": message.tool_name,
                    "response": response
                });
                if needs_id {
                    function["id"] = Value::String(message.tool_call_id.clone());
                }
                if multimodal && !images.is_empty() {
                    function["parts"] = Value::Array(
                        images
                            .iter()
                            .map(|image| {
                                json!({"inlineData": {
                                    "mimeType": image.mime_type,
                                    "data": image.data
                                }})
                            })
                            .collect(),
                    );
                }
                let response_part = json!({"functionResponse": function});
                if let Some(last) = contents.last_mut()
                    && last.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(parts) = last.get_mut("parts").and_then(Value::as_array_mut)
                    && parts
                        .iter()
                        .any(|part| part.get("functionResponse").is_some())
                {
                    parts.push(response_part);
                } else {
                    contents.push(json!({"role": "user", "parts": [response_part]}));
                }
                if !multimodal && !images.is_empty() {
                    let mut parts = vec![json!({"text": "Tool result image:"})];
                    parts.extend(images.into_iter().map(|image| {
                        json!({"inlineData": {
                            "mimeType": image.mime_type,
                            "data": image.data
                        }})
                    }));
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
        }
    }
    contents
}

fn resolve_signature(same_model: bool, signature: Option<String>) -> Option<String> {
    if !same_model {
        return None;
    }
    signature.filter(|signature| {
        signature.len() % 4 == 0
            && signature.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '+' | '/' | '=')
            })
            && STANDARD.decode(signature).is_ok()
    })
}

fn requires_tool_call_id(model: &str) -> bool {
    model.starts_with("claude-") || model.starts_with("gpt-oss-")
}

fn gemini_major(model: &str) -> Option<u32> {
    let suffix = model
        .to_ascii_lowercase()
        .strip_prefix("gemini-")?
        .to_owned();
    suffix.split(['.', '-']).next()?.parse().ok()
}

fn supports_multimodal_function_response(model: &str) -> bool {
    gemini_major(model).is_none_or(|major| major >= 3)
}

fn decode_google_frame(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    if frame.data.trim().is_empty() || frame.data.trim() == "[DONE]" {
        return Ok(Vec::new());
    }
    let chunk: Value = serde_json::from_str(&frame.data)
        .map_err(|error| AiError::Stream(format!("invalid Google event: {error}")))?;
    if let Some(error) = chunk.pointer("/error/message").and_then(Value::as_str) {
        return Ok(vec![WireDelta::Error(error.to_owned())]);
    }
    let mut output = Vec::new();
    if let Some(id) = chunk.get("responseId").and_then(Value::as_str) {
        output.push(WireDelta::ResponseId(id.to_owned()));
    }
    if let Some(model) = chunk.get("modelVersion").and_then(Value::as_str) {
        output.push(WireDelta::ResponseModel(model.to_owned()));
    }
    if let Some(usage) = chunk.get("usageMetadata") {
        output.push(WireDelta::Usage(parse_google_usage(usage)));
    }
    let parts = chunk
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut saw_tool = state
        .get("google:saw_tool")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            let kind = if part.get("thought").and_then(Value::as_bool) == Some(true) {
                "thinking"
            } else {
                "text"
            };
            close_google_block_if_changed(kind, state, &mut output);
            if state.get("google:current").is_none() {
                let block = state
                    .get("google:block_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                state.set("google:block_count", Value::Number((block + 1).into()));
                let key = format!("google:block:{block}");
                state.set("google:current", Value::String(kind.into()));
                state.set("google:current_key", Value::String(key.clone()));
                let index = state.slot(key);
                output.push(if kind == "thinking" {
                    WireDelta::ThinkingStart { index }
                } else {
                    WireDelta::TextStart { index }
                });
            }
            let key = state
                .get("google:current_key")
                .and_then(Value::as_str)
                .unwrap_or("google:block:0")
                .to_owned();
            let index = state.slot(&key);
            if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
                state.set(
                    format!("google:signature:{key}"),
                    Value::String(signature.to_owned()),
                );
            }
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
        if let Some(function) = part.get("functionCall") {
            close_google_block(state, &mut output);
            let count = state
                .get("google:tool_count")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            state.set("google:tool_count", Value::Number((count + 1).into()));
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = function
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("{name}_{}", count + 1), str::to_owned);
            let index = state.slot(format!("google:tool:{count}"));
            let arguments = function.get("args").cloned().unwrap_or_else(|| json!({}));
            output.push(WireDelta::ToolStart {
                index,
                id,
                name: name.to_owned(),
                thought_signature: part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
            output.push(WireDelta::ToolDelta {
                index,
                delta: serde_json::to_string(&arguments)
                    .map_err(|error| AiError::Stream(error.to_string()))?,
            });
            output.push(WireDelta::ToolEnd {
                index,
                arguments: Some(arguments),
            });
            saw_tool = true;
            state.set("google:saw_tool", Value::Bool(true));
        }
    }
    if let Some(reason) = chunk
        .pointer("/candidates/0/finishReason")
        .and_then(Value::as_str)
    {
        close_google_block(state, &mut output);
        output.push(WireDelta::Done(if saw_tool {
            StopReason::ToolUse
        } else {
            map_google_stop(reason)
        }));
    }
    Ok(output)
}

fn close_google_block_if_changed(
    next: &str,
    state: &mut WireDecodeState,
    output: &mut Vec<WireDelta>,
) {
    let changed = state
        .get("google:current")
        .and_then(Value::as_str)
        .is_some_and(|current| current != next);
    if changed {
        close_google_block(state, output);
    }
}

fn close_google_block(state: &mut WireDecodeState, output: &mut Vec<WireDelta>) {
    let Some(kind) = state
        .take("google:current")
        .and_then(|value| value.as_str().map(str::to_owned))
    else {
        return;
    };
    let key = state
        .take("google:current_key")
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "google:block:unknown".into());
    let index = state.slot(&key);
    let signature = state
        .take(&format!("google:signature:{key}"))
        .and_then(|value| value.as_str().map(str::to_owned));
    output.push(if kind == "thinking" {
        WireDelta::ThinkingEnd {
            index,
            content: None,
            signature,
            redacted: false,
        }
    } else {
        WireDelta::TextEnd {
            index,
            content: None,
            signature,
        }
    });
}

fn parse_google_usage(value: &Value) -> Usage {
    let prompt = value
        .get("promptTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = value
        .get("cachedContentTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let candidates = value
        .get("candidatesTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = value
        .get("thoughtsTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input: prompt.saturating_sub(cached),
        output: candidates.saturating_add(reasoning),
        cache_read: cached,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: Some(reasoning),
        total_tokens: value
            .get("totalTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or(prompt.saturating_add(candidates).saturating_add(reasoning)),
        cost: UsageCost::default(),
    }
}

fn map_google_stop(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ConstrainedSampling, JsonSchemaStrictness, Tool};

    #[test]
    fn signatures_require_same_model_and_valid_base64() {
        assert_eq!(
            resolve_signature(true, Some("dGhvdWdodA==".into())),
            Some("dGhvdWdodA==".into())
        );
        assert_eq!(resolve_signature(false, Some("dGhvdWdodA==".into())), None);
        assert_eq!(
            resolve_signature(true, Some("not a signature".into())),
            None
        );
    }

    #[test]
    fn google_usage_counts_reasoning_as_output() {
        let usage = parse_google_usage(&json!({
            "promptTokenCount": 10,
            "cachedContentTokenCount": 4,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 15
        }));
        assert_eq!(usage.input, 6);
        assert_eq!(usage.output, 5);
        assert_eq!(usage.cache_read, 4);
        assert_eq!(usage.reasoning, Some(2));
    }

    #[test]
    fn tool_choice_and_required_strictness_follow_model_capabilities() {
        let mut model = Model::new(
            "google",
            "gemini-2.5-pro",
            "google-generative-ai",
            "https://example.test",
        );
        let mut tool = Tool::new(
            "lookup",
            "Lookup",
            json!({
                "type": "object",
                "$defs": {"query": {"type": "string"}},
                "properties": {"query": {"$ref": "#/$defs/query"}}
            }),
        );
        tool.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: JsonSchemaStrictness::Require,
        });
        let context = Context {
            tools: vec![tool],
            ..Context::default()
        };
        assert!(build_google_body(&model, &context, &WireRequestOptions::default()).is_err());

        model.id = "gemini-3-pro".into();
        let options = WireRequestOptions {
            tool_choice: Some("lookup".into()),
            ..WireRequestOptions::default()
        };
        let body = build_google_body(&model, &context, &options).expect("supported tool choice");
        assert_eq!(
            body.pointer("/toolConfig/functionCallingConfig"),
            Some(&json!({
                "mode": "ANY",
                "allowedFunctionNames": ["lookup"]
            }))
        );
        assert_eq!(
            body.pointer("/tools/0/functionDeclarations/0/parametersJsonSchema/$defs/query/type"),
            Some(&json!("string"))
        );

        let body = build_google_body(
            &model,
            &context,
            &WireRequestOptions {
                tool_choice: Some("auto".into()),
                ..WireRequestOptions::default()
            },
        )
        .expect("strict auto mode");
        assert_eq!(
            body.pointer("/toolConfig/functionCallingConfig/mode"),
            Some(&json!("VALIDATED"))
        );
    }

    #[test]
    fn decoder_preserves_non_thought_signature_on_text() {
        let mut state = WireDecodeState::default();
        let first = SseFrame {
            data: json!({
                "responseId": "resp",
                "candidates": [{"content": {"parts": [{
                    "text": "answer",
                    "thoughtSignature": "dGhvdWdodA=="
                }]}}]
            })
            .to_string(),
            ..SseFrame::default()
        };
        let deltas = decode_google_frame(&first, &mut state).expect("decode");
        assert!(
            deltas
                .iter()
                .any(|delta| matches!(delta, WireDelta::TextDelta { .. }))
        );
        let finish = SseFrame {
            data: json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
            ..SseFrame::default()
        };
        let deltas = decode_google_frame(&finish, &mut state).expect("finish");
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            WireDelta::TextEnd { signature: Some(signature), .. }
                if signature == "dGhvdWdodA=="
        )));
    }
}
