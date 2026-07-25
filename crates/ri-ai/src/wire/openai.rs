//! `OpenAI` Chat Completions, Responses, Codex, and Azure Responses adapters.

use std::collections::HashSet;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use url::Url;

use crate::{
    error::AiError,
    handoff::transform_messages,
    message::{
        ContentBlock, Context, InputContent, Message, StopReason, ToolResultMessage, Usage,
        UsageCost, UserContent,
    },
    model::{CacheRetention, Model, SessionAffinityFormat, ThinkingLevel},
    tool::{ToolDescriptor, describe_tool, resolve_grammar_constraint, split_deferred_tools},
    transport::{HttpRequest, SseFrame},
};

use super::{
    WireAdapter, WireDecodeState, WireDelta, WireRequestOptions, json_request, merge_wire_headers,
    resolve_cache_retention,
};

/// OpenAI-compatible Chat Completions adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiCompletionsAdapter;

/// `OpenAI` Responses adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiResponsesAdapter;

/// `ChatGPT` Codex Responses adapter.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiCodexAdapter;

/// Azure `OpenAI` Responses adapter.
#[derive(Clone, Debug)]
pub struct AzureOpenAiResponsesAdapter {
    /// Azure API version query value.
    pub api_version: String,
}

impl Default for AzureOpenAiResponsesAdapter {
    fn default() -> Self {
        Self {
            api_version: "v1".into(),
        }
    }
}

impl WireAdapter for OpenAiCompletionsAdapter {
    fn api(&self) -> &'static str {
        "openai-completions"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = endpoint(&model.base_url, "chat/completions")?;
        let headers = openai_headers(model, options, false, false)?;
        let body = build_completions_body(model, context, options)?;
        json_request(endpoint, &body, headers, options)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_completions_frame(frame, state)
    }
}

impl WireAdapter for OpenAiResponsesAdapter {
    fn api(&self) -> &'static str {
        "openai-responses"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = endpoint(&model.base_url, "responses")?;
        let headers = openai_headers(model, options, false, false)?;
        let body = build_responses_body(model, context, options, true)?;
        json_request(endpoint, &body, headers, options)
    }

    fn initial_decode_state(
        &self,
        model: &Model,
        context: &Context,
        _options: &WireRequestOptions,
    ) -> Result<WireDecodeState, AiError> {
        responses_decode_state(model, context)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_responses_frame(frame, state)
    }
}

impl WireAdapter for OpenAiCodexAdapter {
    fn api(&self) -> &'static str {
        "openai-codex-responses"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let endpoint = endpoint(&model.base_url, "codex/responses")?;
        let mut headers = openai_headers(model, options, false, true)?;
        headers
            .entry("openai-beta".into())
            .or_insert_with(|| "responses=experimental".into());
        headers
            .entry("originator".into())
            .or_insert_with(|| "pi".into());
        if let Some(session_id) = &options.session_id {
            headers
                .entry("session-id".into())
                .or_insert_with(|| session_id.clone());
            headers
                .entry("x-client-request-id".into())
                .or_insert_with(|| session_id.clone());
        }
        let mut body = build_responses_body(model, context, options, true)?;
        if let Value::Object(object) = &mut body {
            object.insert(
                "instructions".into(),
                Value::String(
                    context
                        .system_prompt
                        .clone()
                        .unwrap_or_else(|| "You are a helpful assistant.".into()),
                ),
            );
            object.insert("text".into(), json!({"verbosity": "low"}));
            object
                .entry("tool_choice")
                .or_insert_with(|| Value::String("auto".into()));
            object.insert("parallel_tool_calls".into(), Value::Bool(true));
            if let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) {
                input.retain(|item| {
                    !matches!(
                        item.get("role").and_then(Value::as_str),
                        Some("system" | "developer")
                    )
                });
            }
        }
        json_request(endpoint, &body, headers, options)
    }

    fn initial_decode_state(
        &self,
        model: &Model,
        context: &Context,
        _options: &WireRequestOptions,
    ) -> Result<WireDecodeState, AiError> {
        responses_decode_state(model, context)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_responses_frame(frame, state)
    }
}

impl WireAdapter for AzureOpenAiResponsesAdapter {
    fn api(&self) -> &'static str {
        "azure-openai-responses"
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError> {
        let mut endpoint = azure_endpoint(&model.base_url)?;
        endpoint
            .query_pairs_mut()
            .append_pair("api-version", &self.api_version);
        let headers = openai_headers(model, options, true, false)?;
        let body = build_responses_body(model, context, options, true)?;
        json_request(endpoint, &body, headers, options)
    }

    fn initial_decode_state(
        &self,
        model: &Model,
        context: &Context,
        _options: &WireRequestOptions,
    ) -> Result<WireDecodeState, AiError> {
        responses_decode_state(model, context)
    }

    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError> {
        decode_responses_frame(frame, state)
    }
}

fn endpoint(base_url: &str, suffix: &str) -> Result<Url, AiError> {
    let base = base_url.trim_end_matches('/');
    let suffix = suffix.trim_start_matches('/');
    let url = if base.ends_with(suffix) {
        base.to_owned()
    } else {
        format!("{base}/{suffix}")
    };
    Url::parse(&url).map_err(|error| AiError::Validation(error.to_string()))
}

fn azure_endpoint(base_url: &str) -> Result<Url, AiError> {
    let mut base =
        Url::parse(base_url.trim()).map_err(|error| AiError::Validation(error.to_string()))?;
    let azure_host = base.host_str().is_some_and(|host| {
        host.ends_with(".openai.azure.com")
            || host.ends_with(".cognitiveservices.azure.com")
            || host.ends_with(".ai.azure.com")
    });
    let path = base.path().trim_end_matches('/');
    if azure_host && matches!(path, "" | "/" | "/openai" | "/openai/v1/responses") {
        base.set_path("/openai/v1");
        base.set_query(None);
    }
    let normalized = base.as_str().trim_end_matches('/');
    if normalized.ends_with("/responses") {
        Url::parse(normalized).map_err(|error| AiError::Validation(error.to_string()))
    } else {
        Url::parse(&format!("{normalized}/responses"))
            .map_err(|error| AiError::Validation(error.to_string()))
    }
}

fn openai_headers(
    model: &Model,
    options: &WireRequestOptions,
    azure: bool,
    codex: bool,
) -> Result<crate::transport::HttpHeaders, AiError> {
    let mut defaults = vec![
        ("accept".into(), "text/event-stream".into()),
        ("content-type".into(), "application/json".into()),
    ];
    if let Some(api_key) = &options.api_key {
        defaults.push(if azure {
            ("api-key".into(), api_key.clone())
        } else {
            ("authorization".into(), format!("Bearer {api_key}"))
        });
    }
    let compat = model.compat.as_ref();
    let send_affinity = codex
        || model.api != "openai-completions"
        || compat.and_then(|compat| compat.send_session_affinity_headers) == Some(true);
    if resolve_cache_retention(options) != CacheRetention::None
        && send_affinity
        && let Some(session_id) = &options.session_id
    {
        match compat
            .and_then(|compat| compat.session_affinity_format)
            .unwrap_or_else(|| {
                if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
                    SessionAffinityFormat::Openrouter
                } else {
                    SessionAffinityFormat::Openai
                }
            }) {
            SessionAffinityFormat::Openrouter => {
                defaults.push(("x-session-id".into(), session_id.clone()));
            }
            SessionAffinityFormat::Openai => {
                defaults.push(("session_id".into(), session_id.clone()));
                defaults.push(("x-client-request-id".into(), session_id.clone()));
                if model.api == "openai-completions" {
                    defaults.push(("x-session-affinity".into(), session_id.clone()));
                }
            }
            SessionAffinityFormat::OpenaiNosession => {
                defaults.push(("x-client-request-id".into(), session_id.clone()));
                if model.api == "openai-completions" {
                    defaults.push(("x-session-affinity".into(), session_id.clone()));
                }
            }
        }
    }
    let headers = merge_wire_headers(defaults, model, &options.headers);
    if !headers.keys().any(|name| {
        name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("api-key")
    }) {
        return Err(AiError::Auth(format!(
            "no API key for provider {}",
            model.provider
        )));
    }
    Ok(headers)
}

fn build_completions_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
) -> Result<Value, AiError> {
    let messages = convert_completions_messages(model, context);
    let mut body = Map::from_iter([
        ("model".into(), Value::String(model.id.clone())),
        ("messages".into(), Value::Array(messages)),
        ("stream".into(), Value::Bool(true)),
    ]);
    if model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_usage_in_streaming)
        != Some(false)
    {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_store)
        .unwrap_or(model.provider == "openai")
    {
        body.insert("store".into(), Value::Bool(false));
    }
    if let Some(max_tokens) = options.max_tokens {
        let field = model
            .compat
            .as_ref()
            .and_then(|compat| compat.max_tokens_field)
            .map_or(
                "max_completion_tokens",
                crate::model::MaxTokensField::as_str,
            );
        body.insert(field.into(), Value::Number(max_tokens.into()));
    }
    insert_finite(&mut body, "temperature", options.temperature)?;
    apply_openai_cache(&mut body, model, options, false);
    apply_reasoning(&mut body, model, options, false);

    if !context.tools.is_empty() {
        let supports_strict = model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_strict_mode)
            .unwrap_or(true);
        let supports_grammar = model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_openai_grammar_tools)
            .unwrap_or(false);
        let tools = context
            .tools
            .iter()
            .map(|tool| {
                describe_tool(tool, supports_strict, supports_grammar, false)
                    .map(completions_tool)
                    .map_err(|error| AiError::Validation(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = &options.tool_choice {
        body.insert(
            "tool_choice".into(),
            Value::String(openai_tool_choice(tool_choice).into()),
        );
    }
    Ok(Value::Object(body))
}

fn completions_tool(descriptor: ToolDescriptor) -> Value {
    match descriptor {
        ToolDescriptor::Function {
            name,
            description,
            parameters,
            strict,
            ..
        } => {
            let mut function = json!({
                "name": name,
                "description": description,
                "parameters": parameters
            });
            if let Some(strict) = strict {
                function["strict"] = Value::Bool(strict);
            }
            json!({"type": "function", "function": function})
        }
        ToolDescriptor::Custom {
            name,
            description,
            format,
            definition,
            ..
        } => json!({
            "type": "custom",
            "custom": {
                "name": name,
                "description": description,
                "format": {
                    "type": "grammar",
                    "syntax": format,
                    "definition": definition
                }
            }
        }),
    }
}

fn convert_completions_messages(model: &Model, context: &Context) -> Vec<Value> {
    let mut normalizer = |id: &str, _: &Model, _: &crate::message::AssistantMessage| {
        normalize_completions_tool_id(id, model.provider == "openai")
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalizer));
    let mut output = Vec::new();
    if let Some(system) = &context.system_prompt {
        let role = if model.reasoning
            && model
                .compat
                .as_ref()
                .and_then(|compat| compat.supports_developer_role)
                != Some(false)
        {
            "developer"
        } else {
            "system"
        };
        output.push(json!({"role": role, "content": system}));
    }
    for message in messages {
        match message {
            Message::User(message) => output.push(json!({
                "role": "user",
                "content": openai_user_content(&message.content)
            })),
            Message::Assistant(message) => {
                let mut object =
                    Map::from_iter([("role".into(), Value::String("assistant".into()))]);
                let texts = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) if !text.text.is_empty() => {
                            Some(text.text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<String>();
                let thinking = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Thinking(thinking) if !thinking.thinking.is_empty() => {
                            Some(thinking.thinking.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if texts.is_empty() {
                    object.insert("content".into(), Value::Null);
                } else {
                    object.insert("content".into(), Value::String(texts));
                }
                if !thinking.is_empty() {
                    let thinking_as_text = model
                        .compat
                        .as_ref()
                        .and_then(|compat| compat.requires_thinking_as_text)
                        == Some(true);
                    if thinking_as_text {
                        let existing = object
                            .get("content")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        object.insert(
                            "content".into(),
                            Value::String(format!("{thinking}{existing}")),
                        );
                    } else {
                        object.insert("reasoning_content".into(), Value::String(thinking));
                    }
                }
                let calls = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into())
                            }
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !calls.is_empty() {
                    object.insert("tool_calls".into(), Value::Array(calls));
                }
                let has_content = object
                    .get("content")
                    .is_some_and(|content| !content.is_null() && content.as_str() != Some(""));
                if has_content || object.contains_key("tool_calls") {
                    output.push(Value::Object(object));
                }
            }
            Message::ToolResult(message) => {
                output.extend(completions_tool_result(model, &message));
            }
        }
    }
    output
}

fn completions_tool_result(model: &Model, message: &ToolResultMessage) -> Vec<Value> {
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
            InputContent::Image(image) => Some(image),
            InputContent::Text(_) => None,
        })
        .collect::<Vec<_>>();
    let content = if !text.is_empty() {
        text
    } else if images.is_empty() {
        "(no tool output)".into()
    } else {
        "(see attached image)".into()
    };
    let mut tool = json!({
        "role": "tool",
        "tool_call_id": message.tool_call_id,
        "content": content
    });
    if model
        .compat
        .as_ref()
        .and_then(|compat| compat.requires_tool_result_name)
        == Some(true)
    {
        tool["name"] = Value::String(message.tool_name.clone());
    }
    let mut output = vec![tool];
    if model.supports_images() && !images.is_empty() {
        let mut content = vec![json!({"type": "text", "text": "Tool result image:"})];
        content.extend(images.into_iter().map(|image| {
            json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{};base64,{}", image.mime_type, image.data)}
            })
        }));
        output.push(json!({"role": "user", "content": content}));
    }
    output
}

fn openai_user_content(content: &UserContent) -> Value {
    match content {
        UserContent::Text(text) => Value::String(text.clone()),
        UserContent::Blocks(blocks) => Value::Array(
            blocks
                .iter()
                .map(|block| match block {
                    InputContent::Text(text) => json!({"type": "text", "text": text.text}),
                    InputContent::Image(image) => json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:{};base64,{}", image.mime_type, image.data)}
                    }),
                })
                .collect(),
        ),
    }
}

fn build_responses_body(
    model: &Model,
    context: &Context,
    options: &WireRequestOptions,
    include_model: bool,
) -> Result<Value, AiError> {
    let supports_deferred = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_tool_search)
        .unwrap_or(false);
    let placement = split_deferred_tools(context, supports_deferred, str::to_owned);
    let input = convert_responses_input(model, context, &placement.deferred);
    let mut body = Map::from_iter([
        ("input".into(), Value::Array(input)),
        ("stream".into(), Value::Bool(true)),
        ("store".into(), Value::Bool(false)),
    ]);
    if include_model {
        body.insert("model".into(), Value::String(model.id.clone()));
    }
    if let Some(max_tokens) = options.max_tokens {
        body.insert(
            "max_output_tokens".into(),
            Value::Number(max_tokens.max(16).into()),
        );
    }
    insert_finite(&mut body, "temperature", options.temperature)?;
    apply_openai_cache(&mut body, model, options, true);
    apply_reasoning(&mut body, model, options, true);

    let supports_strict = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_strict_mode)
        .unwrap_or(true);
    let supports_grammar = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_openai_grammar_tools)
        .unwrap_or(false);
    let tools = placement
        .immediate
        .iter()
        .map(|tool| {
            describe_tool(tool, supports_strict, supports_grammar, false)
                .map(responses_tool)
                .map_err(|error| AiError::Validation(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = &options.tool_choice {
        body.insert(
            "tool_choice".into(),
            Value::String(openai_tool_choice(tool_choice).into()),
        );
    }
    Ok(Value::Object(body))
}

fn openai_tool_choice(choice: &str) -> &str {
    if choice == "any" { "required" } else { choice }
}

fn responses_decode_state(model: &Model, context: &Context) -> Result<WireDecodeState, AiError> {
    let supports_grammar = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_openai_grammar_tools)
        .unwrap_or(false);
    let mut state = WireDecodeState::default();
    for tool in &context.tools {
        if let Some((_, _, property)) = resolve_grammar_constraint(tool, supports_grammar)
            .map_err(|error| AiError::Validation(error.to_string()))?
        {
            state.set(
                format!("responses:grammar:{}", tool.name),
                Value::String(property),
            );
        }
    }
    Ok(state)
}

fn responses_tool(descriptor: ToolDescriptor) -> Value {
    match descriptor {
        ToolDescriptor::Function {
            name,
            description,
            parameters,
            strict,
            defer_loading,
        } => {
            let mut value = json!({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters
            });
            if let Some(strict) = strict {
                value["strict"] = Value::Bool(strict);
            }
            if defer_loading {
                value["defer_loading"] = Value::Bool(true);
            }
            value
        }
        ToolDescriptor::Custom {
            name,
            description,
            format,
            definition,
            defer_loading,
            ..
        } => {
            let mut value = json!({
                "type": "custom",
                "name": name,
                "description": description,
                "format": {
                    "type": "grammar",
                    "syntax": format,
                    "definition": definition
                }
            });
            if defer_loading {
                value["defer_loading"] = Value::Bool(true);
            }
            value
        }
    }
}

fn convert_responses_input(
    model: &Model,
    context: &Context,
    deferred: &indexmap::IndexMap<String, crate::tool::Tool>,
) -> Vec<Value> {
    let allowed = ["openai", "openai-codex", "opencode"]
        .into_iter()
        .collect::<HashSet<_>>();
    let mut normalize = |id: &str, _: &Model, source: &crate::message::AssistantMessage| {
        normalize_responses_tool_id(id, source, model, &allowed)
    };
    let messages = transform_messages(&context.messages, model, Some(&mut normalize));
    let mut output = Vec::new();
    if let Some(system) = &context.system_prompt {
        let role = if model.reasoning
            && model
                .compat
                .as_ref()
                .and_then(|compat| compat.supports_developer_role)
                != Some(false)
        {
            "developer"
        } else {
            "system"
        };
        output.push(json!({"role": role, "content": system}));
    }
    let mut loaded = HashSet::<String>::new();
    for (message_index, message) in messages.into_iter().enumerate() {
        match message {
            Message::User(message) => {
                let input_content = match message.content {
                    UserContent::Text(text) => vec![json!({"type": "input_text", "text": text})],
                    UserContent::Blocks(blocks) => blocks
                        .into_iter()
                        .map(|block| match block {
                            InputContent::Text(text) => {
                                json!({"type": "input_text", "text": text.text})
                            }
                            InputContent::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
                            }),
                        })
                        .collect(),
                };
                if !input_content.is_empty() {
                    output.push(json!({"role": "user", "content": input_content}));
                }
            }
            Message::Assistant(message) => {
                let different_model = message.provider == model.provider
                    && message.api == model.api
                    && message.model != model.id;
                let mut text_index = 0usize;
                for block in message.content {
                    match block {
                        ContentBlock::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature
                                && let Ok(item) = serde_json::from_str::<Value>(&signature)
                            {
                                output.push(item);
                            }
                        }
                        ContentBlock::Text(text) => {
                            let fallback = if text_index == 0 {
                                format!("msg_pi_{message_index}")
                            } else {
                                format!("msg_pi_{message_index}_{text_index}")
                            };
                            text_index += 1;
                            let id = text
                                .text_signature
                                .as_deref()
                                .and_then(parse_text_signature_id)
                                .map_or(fallback, normalize_message_id);
                            output.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "id": id,
                                "content": [{
                                    "type": "output_text",
                                    "text": text.text,
                                    "annotations": []
                                }]
                            }));
                        }
                        ContentBlock::ToolCall(call) => {
                            let (call_id, item_id) = split_tool_id(&call.id);
                            let item_id =
                                item_id.filter(|id| !different_model && id.starts_with("fc_"));
                            let mut value = json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into())
                            });
                            if let Some(item_id) = item_id {
                                value["id"] = Value::String(item_id.to_owned());
                            }
                            output.push(value);
                        }
                    }
                }
            }
            Message::ToolResult(message) => {
                let (call_id, _) = split_tool_id(&message.tool_call_id);
                output.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": responses_tool_output(model, &message)
                }));
                let newly_loaded = message
                    .added_tool_names
                    .iter()
                    .filter_map(|name| {
                        if loaded.insert(name.clone()) {
                            deferred.get(name)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if !newly_loaded.is_empty() {
                    let names = newly_loaded
                        .iter()
                        .map(|tool| tool.name.as_str())
                        .collect::<Vec<_>>()
                        .join(",");
                    let search_id = format!(
                        "pi_tool_load_{}",
                        short_hash(&format!("{}:{names}", message.tool_call_id))
                    );
                    output.push(json!({
                        "type": "tool_search_call",
                        "call_id": search_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": names, "limit": newly_loaded.len()}
                    }));
                    let tools = newly_loaded
                        .into_iter()
                        .filter_map(|tool| {
                            describe_tool(tool, true, false, true)
                                .ok()
                                .map(responses_tool)
                        })
                        .collect::<Vec<_>>();
                    output.push(json!({
                        "type": "tool_search_output",
                        "call_id": search_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": tools
                    }));
                }
            }
        }
    }
    output
}

fn responses_tool_output(model: &Model, message: &ToolResultMessage) -> Value {
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
            InputContent::Image(image) => Some(image),
            InputContent::Text(_) => None,
        })
        .collect::<Vec<_>>();
    if model.supports_images() && !images.is_empty() {
        let mut content = vec![json!({
            "type": "input_text",
            "text": if text.is_empty() { "(see attached image)" } else { &text }
        })];
        content.extend(images.into_iter().map(|image| {
            json!({
                "type": "input_image",
                "detail": "auto",
                "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
            })
        }));
        Value::Array(content)
    } else if text.is_empty() {
        Value::String(if message.content.is_empty() {
            "(no tool output)".into()
        } else {
            "(tool image omitted: model does not support images)".into()
        })
    } else {
        Value::String(text)
    }
}

fn apply_openai_cache(
    body: &mut Map<String, Value>,
    model: &Model,
    options: &WireRequestOptions,
    responses: bool,
) {
    let retention = resolve_cache_retention(options);
    let configured_long = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_long_cache_retention);
    let supports_long = if responses {
        configured_long != Some(false)
    } else {
        configured_long.unwrap_or_else(|| {
            model.provider == "openai" || model.base_url.contains("api.openai.com")
        })
    };
    let use_cache_key = if responses {
        retention != CacheRetention::None
    } else {
        (model.base_url.contains("api.openai.com") && retention != CacheRetention::None)
            || (retention == CacheRetention::Long && supports_long)
    };
    if use_cache_key && let Some(session_id) = &options.session_id {
        body.insert(
            "prompt_cache_key".into(),
            Value::String(clamp_cache_key(session_id)),
        );
    }
    if retention == CacheRetention::Long && supports_long {
        body.insert("prompt_cache_retention".into(), Value::String("24h".into()));
    }
    if responses
        && retention == CacheRetention::None
        && model
            .compat
            .as_ref()
            .and_then(|compat| compat.supports_explicit_prompt_cache_mode)
            == Some(true)
    {
        body.insert("prompt_cache_options".into(), json!({"mode": "explicit"}));
    }
}

fn apply_reasoning(
    body: &mut Map<String, Value>,
    model: &Model,
    options: &WireRequestOptions,
    responses: bool,
) {
    if !model.reasoning {
        return;
    }
    let Some(level) = options
        .reasoning
        .filter(|level| *level != ThinkingLevel::Off)
    else {
        return;
    };
    let effort = model
        .thinking_level_map
        .get(&level)
        .and_then(Clone::clone)
        .unwrap_or_else(|| level.as_str().to_owned());
    if responses {
        body.insert(
            "reasoning".into(),
            json!({"effort": effort, "summary": "auto"}),
        );
        body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    } else {
        body.insert("reasoning_effort".into(), Value::String(effort));
    }
}

fn insert_finite(
    body: &mut Map<String, Value>,
    key: &str,
    value: Option<f64>,
) -> Result<(), AiError> {
    if let Some(value) = value {
        let value = serde_json::Number::from_f64(value)
            .ok_or_else(|| AiError::Validation(format!("{key} must be finite")))?;
        body.insert(key.into(), Value::Number(value));
    }
    Ok(())
}

fn clamp_cache_key(session_id: &str) -> String {
    if session_id.len() <= 64 {
        session_id.to_owned()
    } else {
        format!("pi_{}", short_hash(session_id))
    }
}

fn normalize_completions_tool_id(id: &str, first_party: bool) -> String {
    if id.contains('|') {
        let (call_id, item_id) = split_tool_id(id);
        let call_id = sanitize_id(call_id);
        let item_id = item_id.map(sanitize_id).unwrap_or_default();
        let combined = if item_id.is_empty() {
            call_id.clone()
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.len() <= 40 {
            combined
        } else {
            let hash = &short_hash(id)[..8];
            let prefix_len = 40usize.saturating_sub(hash.len() + 1).max(1);
            format!("{}_{}", &call_id[..call_id.len().min(prefix_len)], hash)
        }
    } else if first_party {
        id.chars().take(40).collect()
    } else {
        id.to_owned()
    }
}

fn normalize_responses_tool_id(
    id: &str,
    source: &crate::message::AssistantMessage,
    target: &Model,
    allowed: &HashSet<&str>,
) -> String {
    if !allowed.contains(target.provider.as_str()) {
        return sanitize_id(id).chars().take(64).collect();
    }
    let (call_id, item_id) = split_tool_id(id);
    let call_id = sanitize_id(call_id);
    let foreign = source.provider != target.provider || source.api != target.api;
    let mut item = item_id.map_or_else(
        || format!("fc_{}", short_hash(id)),
        |item| {
            if foreign {
                format!("fc_{}", short_hash(item))
            } else {
                sanitize_id(item)
            }
        },
    );
    if !item.starts_with("fc_") {
        item = format!("fc_{item}");
    }
    item.truncate(64);
    format!("{call_id}|{item}")
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_end_matches('_')
        .to_owned()
}

fn split_tool_id(id: &str) -> (&str, Option<&str>) {
    id.split_once('|')
        .map_or((id, None), |(call, item)| (call, Some(item)))
}

fn short_hash(input: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let digest = Sha256::digest(input.as_bytes());
    let mut output = String::with_capacity(24);
    for byte in &digest[..12] {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn parse_text_signature_id(signature: &str) -> Option<&str> {
    if signature.starts_with('{') {
        return None;
    }
    (!signature.is_empty()).then_some(signature)
}

fn normalize_message_id(id: &str) -> String {
    if id.len() <= 64 {
        id.to_owned()
    } else {
        format!("msg_{}", short_hash(id))
    }
}

fn decode_completions_frame(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    if frame.data.trim() == "[DONE]" {
        return Ok(vec![WireDelta::Done(
            state.pending_stop_reason.unwrap_or(StopReason::Stop),
        )]);
    }
    if frame.data.trim().is_empty() {
        return Ok(Vec::new());
    }
    let chunk: Value = serde_json::from_str(&frame.data)
        .map_err(|error| AiError::Stream(format!("invalid completions event: {error}")))?;
    if let Some(error) = chunk.pointer("/error/message").and_then(Value::as_str) {
        return Ok(vec![WireDelta::Error(error.to_owned())]);
    }
    let mut output = Vec::new();
    if let Some(id) = chunk.get("id").and_then(Value::as_str) {
        output.push(WireDelta::ResponseId(id.to_owned()));
    }
    if let Some(model) = chunk.get("model").and_then(Value::as_str) {
        output.push(WireDelta::ResponseModel(model.to_owned()));
    }
    if let Some(usage) = chunk.get("usage") {
        output.push(WireDelta::Usage(parse_openai_usage(usage)));
    }
    let Some(choice) = chunk.pointer("/choices/0") else {
        return Ok(output);
    };
    let delta = choice.get("delta").unwrap_or(&Value::Null);
    if let Some(reasoning) = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(Value::as_str)
        && !reasoning.is_empty()
    {
        let index = state.slot("completions:thinking");
        if state.get("started:thinking").is_none() {
            state.set("started:thinking", Value::Bool(true));
            output.push(WireDelta::ThinkingStart { index });
        }
        output.push(WireDelta::ThinkingDelta {
            index,
            delta: reasoning.to_owned(),
        });
    }
    if let Some(content) = delta.get("content").and_then(Value::as_str)
        && !content.is_empty()
    {
        let index = state.slot("completions:text");
        if state.get("started:text").is_none() {
            state.set("started:text", Value::Bool(true));
            output.push(WireDelta::TextStart { index });
        }
        output.push(WireDelta::TextDelta {
            index,
            delta: content.to_owned(),
        });
    }
    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let wire_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let key = format!("completions:tool:{wire_index}");
            let index = state.slot(&key);
            let started_key = format!("started:{key}");
            if state.get(&started_key).is_none() {
                state.set(started_key, Value::Bool(true));
                output.push(WireDelta::ToolStart {
                    index,
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name: call
                        .pointer("/function/name")
                        .or_else(|| call.pointer("/custom/name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    thought_signature: None,
                });
            }
            if let Some(arguments) = call
                .pointer("/function/arguments")
                .or_else(|| call.pointer("/custom/input"))
                .and_then(Value::as_str)
                && !arguments.is_empty()
            {
                output.push(WireDelta::ToolDelta {
                    index,
                    delta: arguments.to_owned(),
                });
            }
        }
    }
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        state.pending_stop_reason = Some(map_openai_stop(reason));
        if state.get("started:text").is_some() {
            output.push(WireDelta::TextEnd {
                index: state.slot("completions:text"),
                content: None,
                signature: None,
            });
            state.take("started:text");
        }
        if state.get("started:thinking").is_some() {
            output.push(WireDelta::ThinkingEnd {
                index: state.slot("completions:thinking"),
                content: None,
                signature: None,
                redacted: false,
            });
            state.take("started:thinking");
        }
        let tool_count = state
            .get("tool_count")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| {
                delta
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map_or(0, |calls| calls.len() as u64)
            });
        for wire_index in 0..tool_count {
            let key = format!("completions:tool:{wire_index}");
            if state.get(&format!("started:{key}")).is_some() {
                output.push(WireDelta::ToolEnd {
                    index: state.slot(key),
                    arguments: None,
                });
            }
        }
    }
    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        let max = tool_calls
            .iter()
            .filter_map(|call| call.get("index").and_then(Value::as_u64))
            .max()
            .map_or(0, |index| index + 1);
        let current = state.get("tool_count").and_then(Value::as_u64).unwrap_or(0);
        state.set("tool_count", Value::Number(current.max(max).into()));
    }
    Ok(output)
}

fn decode_responses_frame(
    frame: &SseFrame,
    state: &mut WireDecodeState,
) -> Result<Vec<WireDelta>, AiError> {
    if frame.data.trim() == "[DONE]" || frame.data.trim().is_empty() {
        return Ok(Vec::new());
    }
    let event: Value = serde_json::from_str(&frame.data)
        .map_err(|error| AiError::Stream(format!("invalid Responses event: {error}")))?;
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .or(frame.event.as_deref())
        .unwrap_or_default();
    let mut output = Vec::new();
    match event_type {
        "response.created" | "response.in_progress" => {
            if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                output.push(WireDelta::ResponseId(id.to_owned()));
            }
            if let Some(model) = event.pointer("/response/model").and_then(Value::as_str) {
                output.push(WireDelta::ResponseModel(model.to_owned()));
            }
        }
        "response.output_item.added" => {
            let item = event.get("item").unwrap_or(&Value::Null);
            let output_index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("output:{output_index}"), str::to_owned);
            let index = state.slot(format!("responses:{item_id}"));
            state.set(
                format!("responses:item:{output_index}"),
                Value::String(item_id.clone()),
            );
            let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
            state.set(format!("kind:{item_id}"), Value::String(kind.to_owned()));
            match kind {
                "message" => {
                    output.push(WireDelta::TextStart { index });
                    state.set(format!("text_signature:{item_id}"), Value::String(item_id));
                }
                "reasoning" => output.push(WireDelta::ThinkingStart { index }),
                "function_call" => {
                    output.push(WireDelta::ToolStart {
                        index,
                        id: format!(
                            "{}|{}",
                            item.get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                            item.get("id").and_then(Value::as_str).unwrap_or_default()
                        ),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        thought_signature: None,
                    });
                    if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
                        && !arguments.is_empty()
                    {
                        output.push(WireDelta::ToolDelta {
                            index,
                            delta: arguments.to_owned(),
                        });
                    }
                }
                "custom_tool_call" => {
                    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                    output.push(WireDelta::ToolStart {
                        index,
                        id: format!(
                            "{}|{}",
                            item.get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                            item.get("id").and_then(Value::as_str).unwrap_or_default()
                        ),
                        name: name.to_owned(),
                        thought_signature: None,
                    });
                    if let Some(property) = state
                        .get(&format!("responses:grammar:{name}"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                    {
                        state.set(
                            format!("responses:custom_property:{item_id}"),
                            Value::String(property),
                        );
                    }
                    if let Some(input) = item.get("input").and_then(Value::as_str)
                        && !input.is_empty()
                        && let Some(delta) =
                            append_custom_tool_input(state, &item_id, input, false)?
                    {
                        output.push(WireDelta::ToolDelta { index, delta });
                    }
                }
                _ => {}
            }
        }
        "response.output_text.delta" => {
            let index = responses_index(&event, state);
            output.push(WireDelta::TextDelta {
                index,
                delta: event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }
        "response.output_text.done" => {
            let index = responses_index(&event, state);
            let signature = responses_item_id(&event, state).and_then(|item| {
                state
                    .take(&format!("text_signature:{item}"))
                    .and_then(|value| value.as_str().map(str::to_owned))
            });
            output.push(WireDelta::TextEnd {
                index,
                content: event.get("text").and_then(Value::as_str).map(str::to_owned),
                signature,
            });
        }
        "response.reasoning_summary_part.added" => {
            let index = responses_index(&event, state);
            output.push(WireDelta::ThinkingStart { index });
        }
        "response.reasoning_summary_text.delta" => {
            let index = responses_index(&event, state);
            output.push(WireDelta::ThinkingDelta {
                index,
                delta: event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }
        "response.function_call_arguments.delta" => {
            let index = responses_index(&event, state);
            output.push(WireDelta::ToolDelta {
                index,
                delta: event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            });
        }
        "response.custom_tool_call_input.delta" => {
            let index = responses_index(&event, state);
            let item_id = responses_item_id(&event, state).unwrap_or_else(|| {
                format!(
                    "output:{}",
                    event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                )
            });
            let input = event
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(delta) = append_custom_tool_input(state, &item_id, input, false)? {
                output.push(WireDelta::ToolDelta { index, delta });
            }
        }
        "response.function_call_arguments.done" => {
            let index = responses_index(&event, state);
            let raw = event.get("arguments").and_then(Value::as_str);
            output.push(WireDelta::ToolEnd {
                index,
                arguments: raw
                    .map(|raw| crate::tool::partial_json::parse_streaming_json(Some(raw))),
            });
            if let Some(item_id) = responses_item_id(&event, state) {
                state.set(format!("responses:ended:{item_id}"), Value::Bool(true));
            }
        }
        "response.custom_tool_call_input.done" => {
            let index = responses_index(&event, state);
            let item_id = responses_item_id(&event, state).unwrap_or_else(|| {
                format!(
                    "output:{}",
                    event
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                )
            });
            let input = event
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(delta) = append_custom_tool_input(state, &item_id, input, true)? {
                output.push(WireDelta::ToolDelta { index, delta });
            }
            let property = custom_property(state, &item_id);
            output.push(WireDelta::ToolEnd {
                index,
                arguments: Some(json!({(property): input})),
            });
            state.set(format!("responses:ended:{item_id}"), Value::Bool(true));
        }
        "response.output_item.done" => {
            let item = event.get("item").unwrap_or(&Value::Null);
            let output_index = event
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("output:{output_index}"), str::to_owned);
            let index = state.slot(format!("responses:{item_id}"));
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "reasoning" => output.push(WireDelta::ThinkingEnd {
                    index,
                    content: None,
                    signature: serde_json::to_string(item).ok(),
                    redacted: false,
                }),
                "function_call" | "custom_tool_call"
                    if state.get(&format!("responses:ended:{item_id}")).is_none() =>
                {
                    let custom =
                        item.get("type").and_then(Value::as_str) == Some("custom_tool_call");
                    let raw = item
                        .get("arguments")
                        .or_else(|| item.get("input"))
                        .and_then(Value::as_str);
                    if custom
                        && let Some(input) = raw
                        && let Some(delta) = append_custom_tool_input(state, &item_id, input, true)?
                    {
                        output.push(WireDelta::ToolDelta { index, delta });
                    }
                    let arguments = if custom {
                        raw.map(|input| json!({(custom_property(state, &item_id)): input}))
                    } else {
                        raw.map(|raw| crate::tool::partial_json::parse_streaming_json(Some(raw)))
                    };
                    output.push(WireDelta::ToolEnd { index, arguments });
                    state.set(format!("responses:ended:{item_id}"), Value::Bool(true));
                }
                "message" if state.get(&format!("text_signature:{item_id}")).is_some() => {
                    output.push(WireDelta::TextEnd {
                        index,
                        content: item
                            .pointer("/content/0/text")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        signature: Some(item_id),
                    });
                }
                _ => {}
            }
        }
        "response.completed" => {
            let response = event.get("response").unwrap_or(&Value::Null);
            if let Some(usage) = response.get("usage") {
                output.push(WireDelta::Usage(parse_responses_usage(usage)));
            }
            if let Some(id) = response.get("id").and_then(Value::as_str) {
                output.push(WireDelta::ResponseId(id.to_owned()));
            }
            output.push(WireDelta::Done(StopReason::Stop));
        }
        "response.incomplete" => {
            let response = event.get("response").unwrap_or(&Value::Null);
            if let Some(usage) = response.get("usage") {
                output.push(WireDelta::Usage(parse_responses_usage(usage)));
            }
            output.push(WireDelta::Done(StopReason::Length));
        }
        "response.failed" | "error" => {
            output.push(WireDelta::Error(
                event
                    .pointer("/response/error/message")
                    .or_else(|| event.pointer("/error/message"))
                    .or_else(|| event.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("OpenAI Responses stream failed")
                    .to_owned(),
            ));
        }
        _ => {}
    }
    Ok(output)
}

fn responses_item_id(event: &Value, state: &WireDecodeState) -> Option<String> {
    event
        .get("item_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            let index = event.get("output_index").and_then(Value::as_u64)?;
            state
                .get(&format!("responses:item:{index}"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn custom_property(state: &WireDecodeState, item_id: &str) -> String {
    state
        .get(&format!("responses:custom_property:{item_id}"))
        .and_then(Value::as_str)
        .unwrap_or("input")
        .to_owned()
}

fn append_custom_tool_input(
    state: &mut WireDecodeState,
    item_id: &str,
    incoming: &str,
    close: bool,
) -> Result<Option<String>, AiError> {
    let raw_key = format!("responses:custom_raw:{item_id}");
    let mut raw = state
        .take(&raw_key)
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default();
    let fragment = if close {
        incoming.strip_prefix(&raw).unwrap_or_default()
    } else {
        incoming
    };
    if close {
        incoming.clone_into(&mut raw);
    } else {
        raw.push_str(incoming);
    }
    state.set(raw_key, Value::String(raw));
    let started_key = format!("responses:custom_started:{item_id}");
    let started = state.get(&started_key).is_some();
    if fragment.is_empty() && (!close || started) {
        return Ok(close.then(|| "\"}".into()));
    }
    let encoded = serde_json::to_string(fragment)
        .map_err(|error| AiError::Stream(format!("custom tool input encoding failed: {error}")))?;
    let escaped = encoded
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(encoded.as_str());
    let mut delta = String::new();
    if !started {
        let property =
            serde_json::to_string(&custom_property(state, item_id)).map_err(|error| {
                AiError::Stream(format!("custom tool property encoding failed: {error}"))
            })?;
        delta.push('{');
        delta.push_str(&property);
        delta.push_str(":\"");
        state.set(started_key, Value::Bool(true));
    }
    delta.push_str(escaped);
    if close {
        delta.push_str("\"}");
    }
    Ok(Some(delta))
}

fn responses_index(event: &Value, state: &mut WireDecodeState) -> usize {
    let output_index = event
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let item_id = event
        .get("item_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            state
                .get(&format!("responses:item:{output_index}"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("output:{output_index}"));
    state.slot(format!("responses:{item_id}"))
}

fn parse_openai_usage(value: &Value) -> Usage {
    let prompt = value
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = value
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = value
        .pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = cached.saturating_sub(cache_write);
    let output = value
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning = value
        .pointer("/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64);
    Usage {
        input: prompt
            .saturating_sub(cache_read)
            .saturating_sub(cache_write),
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning,
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(prompt.saturating_add(output)),
        cost: UsageCost::default(),
    }
}

fn parse_responses_usage(value: &Value) -> Usage {
    let prompt = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = value
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input: prompt.saturating_sub(cached).saturating_sub(cache_write),
        output,
        cache_read: cached,
        cache_write,
        cache_write_1h: None,
        reasoning: value
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64),
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(prompt.saturating_add(output)),
        cost: UsageCost::default(),
    }
}

fn map_openai_stop(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::{Message, UserMessage},
        model::ModelCompatibility,
    };

    fn responses_model() -> Model {
        let mut model = Model::new(
            "openai",
            "gpt-test",
            "openai-responses",
            "https://api.openai.com/v1",
        );
        model.reasoning = true;
        model.compat = Some(ModelCompatibility {
            supports_explicit_prompt_cache_mode: Some(true),
            supports_strict_mode: Some(true),
            ..ModelCompatibility::default()
        });
        model
    }

    #[test]
    fn responses_none_cache_uses_explicit_mode_without_key() {
        let body = build_responses_body(
            &responses_model(),
            &Context {
                system_prompt: None,
                messages: vec![Message::User(UserMessage::new("hello"))],
                tools: Vec::new(),
            },
            &WireRequestOptions {
                cache_retention: Some(CacheRetention::None),
                session_id: Some("session".into()),
                ..WireRequestOptions::default()
            },
            true,
        )
        .expect("payload");
        assert!(body.get("prompt_cache_key").is_none());
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
    }

    #[test]
    fn session_affinity_respects_api_compatibility() {
        let options = WireRequestOptions {
            api_key: Some("key".into()),
            cache_retention: Some(CacheRetention::Short),
            session_id: Some("session".into()),
            ..WireRequestOptions::default()
        };
        let mut completions = Model::new(
            "compatible",
            "model",
            "openai-completions",
            "https://example.test",
        );
        let headers =
            openai_headers(&completions, &options, false, false).expect("completion headers");
        assert!(!headers.contains_key("x-session-affinity"));
        let body = build_completions_body(&completions, &Context::default(), &options)
            .expect("completion body");
        assert!(body.get("prompt_cache_key").is_none());

        completions.compat = Some(ModelCompatibility {
            send_session_affinity_headers: Some(true),
            ..ModelCompatibility::default()
        });
        let headers =
            openai_headers(&completions, &options, false, false).expect("affinity headers");
        assert_eq!(
            headers.get("x-session-affinity").map(String::as_str),
            Some("session")
        );

        let responses = responses_model();
        let headers = openai_headers(&responses, &options, false, false).expect("response headers");
        assert_eq!(
            headers.get("x-client-request-id").map(String::as_str),
            Some("session")
        );
    }

    #[test]
    fn foreign_responses_ids_are_rewritten_and_linked() {
        let source =
            crate::message::AssistantMessage::empty("openai-responses", "other", "gpt-other");
        let id = normalize_responses_tool_id(
            "call/one|long+item/id",
            &source,
            &responses_model(),
            &["openai"].into_iter().collect(),
        );
        assert!(id.starts_with("call_one|fc_"));
        assert!(!id.contains(['/', '+']));
    }

    #[test]
    fn completions_decodes_usage_and_tool_deltas() {
        let mut state = WireDecodeState::default();
        let frame = SseFrame {
            data: json!({
                "id": "chatcmpl_1",
                "model": "routed",
                "choices": [{
                    "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "read", "arguments": "{\"path\":"}
                    }]},
                    "finish_reason": null
                }]
            })
            .to_string(),
            ..SseFrame::default()
        };
        let deltas = decode_completions_frame(&frame, &mut state).expect("decode");
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            WireDelta::ToolStart { id, name, .. } if id == "call_1" && name == "read"
        )));
        assert!(deltas.iter().any(|delta| matches!(
            delta,
            WireDelta::ToolDelta { delta, .. } if delta == "{\"path\":"
        )));
    }

    #[test]
    fn responses_terminal_event_carries_usage() {
        let mut state = WireDecodeState::default();
        let frame = SseFrame {
            data: json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 4},
                        "output_tokens": 3,
                        "output_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 13
                    }
                }
            })
            .to_string(),
            ..SseFrame::default()
        };
        let deltas = decode_responses_frame(&frame, &mut state).expect("decode");
        assert!(matches!(
            &deltas[0],
            WireDelta::Usage(Usage {
                input: 6,
                cache_read: 4,
                reasoning: Some(2),
                ..
            })
        ));
        assert_eq!(deltas.last(), Some(&WireDelta::Done(StopReason::Stop)));
    }
}
