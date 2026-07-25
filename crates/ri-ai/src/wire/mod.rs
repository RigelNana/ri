//! Provider wire adapters and generic streaming execution.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{
    auth::{ProviderEnv, ProviderHeaders},
    error::AiError,
    message::{
        AssistantMessage, AssistantMessageEvent, ContentBlock, Context, StopReason, TextContent,
        ThinkingContent, ToolCall, Usage,
    },
    model::{CacheRetention, Model, ThinkingLevel, calculate_cost, clamp_thinking_level},
    stream::{AssistantEventSender, AssistantEventStream, create_assistant_message_event_stream},
    tool::partial_json::parse_streaming_json,
    transport::{DynHttpTransport, HttpHeaders, HttpRequest, SseFrame},
};

pub mod anthropic;
pub mod bedrock;
pub mod google;
pub mod mistral;
pub mod openai;
pub mod openrouter_images;
pub mod pi_messages;

/// Byte framing used by a provider's streaming response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamEncoding {
    /// UTF-8 Server-Sent Events.
    #[default]
    Sse,
    /// AWS binary event-stream messages.
    AwsEventStream,
}

/// Common request options consumed by wire adapters.
#[derive(Clone, Debug, Default)]
pub struct WireRequestOptions {
    /// Resolved API key/bearer token.
    pub api_key: Option<String>,
    /// Caller/auth headers. `None` suppresses a model/adapter default.
    pub headers: ProviderHeaders,
    /// Provider-scoped environment/configuration.
    pub env: ProviderEnv,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Output token cap.
    pub max_tokens: Option<u64>,
    /// Provider-neutral reasoning effort.
    pub reasoning: Option<ThinkingLevel>,
    /// Cache retention preference.
    pub cache_retention: Option<CacheRetention>,
    /// Session/cache affinity id.
    pub session_id: Option<String>,
    /// Provider-neutral tool choice (`auto`, `none`, `any`, or `required`).
    pub tool_choice: Option<String>,
    /// Whole-request timeout.
    pub timeout: Option<Duration>,
    /// Cooperative cancellation.
    pub cancellation: Option<CancellationToken>,
    /// API-specific options, intentionally explicit rather than silently
    /// forwarding unknown top-level request properties.
    pub extra: BTreeMap<String, Value>,
}

impl WireRequestOptions {
    /// Clamps model-dependent options before provider dispatch.
    pub fn normalize_for_model(&mut self, model: &Model) {
        if let Some(requested) = self.reasoning {
            self.reasoning = Some(clamp_thinking_level(model, requested));
        }
    }
}

/// Provider-neutral delta decoded from a wire event.
#[derive(Clone, Debug, PartialEq)]
pub enum WireDelta {
    /// Provider response id.
    ResponseId(String),
    /// Concrete routed model.
    ResponseModel(String),
    /// Provider/gateway diagnostic metadata safe to persist.
    Diagnostic(Value),
    /// Begin a text block.
    TextStart {
        /// Dense provider-neutral content index.
        index: usize,
    },
    /// Append text.
    TextDelta {
        /// Target content index.
        index: usize,
        /// Append-only UTF-8 text.
        delta: String,
    },
    /// Finish text and attach replay metadata.
    TextEnd {
        /// Target content index.
        index: usize,
        /// Authoritative final text, when present in the end event.
        content: Option<String>,
        /// Opaque provider replay signature.
        signature: Option<String>,
    },
    /// Begin a thinking block.
    ThinkingStart {
        /// Dense provider-neutral content index.
        index: usize,
    },
    /// Append thinking text.
    ThinkingDelta {
        /// Target content index.
        index: usize,
        /// Append-only reasoning text.
        delta: String,
    },
    /// Finish thinking and attach replay metadata.
    ThinkingEnd {
        /// Target content index.
        index: usize,
        /// Authoritative final reasoning text, when present in the end event.
        content: Option<String>,
        /// Opaque provider replay signature.
        signature: Option<String>,
        /// Whether the thinking payload was redacted.
        redacted: bool,
    },
    /// Begin a tool call.
    ToolStart {
        /// Dense provider-neutral content index.
        index: usize,
        /// Provider tool-call identifier.
        id: String,
        /// Requested tool name.
        name: String,
        /// Opaque provider reasoning signature.
        thought_signature: Option<String>,
    },
    /// Append tool argument JSON.
    ToolDelta {
        /// Target content index.
        index: usize,
        /// Append-only JSON or grammar-input encoding.
        delta: String,
    },
    /// Attach provider replay metadata to a tool call.
    ToolSignature {
        /// Target content index.
        index: usize,
        /// Opaque provider reasoning signature.
        signature: Option<String>,
    },
    /// Finish a tool call, optionally replacing its parsed arguments.
    ToolEnd {
        /// Target content index.
        index: usize,
        /// Final parsed arguments, when supplied by the provider.
        arguments: Option<Value>,
    },
    /// Replace current usage.
    Usage(Usage),
    /// Successful terminal state.
    Done(StopReason),
    /// Provider-reported failure.
    Error(String),
    /// Provider-reported cancellation.
    Aborted(String),
}

/// Mutable adapter state for mapping provider item ids to content indices.
#[derive(Clone, Debug, Default)]
pub struct WireDecodeState {
    slots: HashMap<String, usize>,
    values: HashMap<String, Value>,
    next_index: usize,
    /// Last finish reason observed before a `[DONE]` sentinel.
    pub pending_stop_reason: Option<StopReason>,
}

impl WireDecodeState {
    /// Returns a stable content index for a provider item key.
    pub fn slot(&mut self, key: impl Into<String>) -> usize {
        let key = key.into();
        if let Some(index) = self.slots.get(&key) {
            return *index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.slots.insert(key, index);
        index
    }

    /// Stores adapter-specific JSON state.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }

    /// Reads adapter-specific JSON state.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Removes adapter-specific JSON state.
    pub fn take(&mut self, key: &str) -> Option<Value> {
        self.values.remove(key)
    }
}

/// Text-generation wire protocol adapter.
pub trait WireAdapter: Send + Sync + std::fmt::Debug {
    /// API identifier handled by this adapter.
    fn api(&self) -> &'static str;
    /// Streaming response framing used by this API.
    fn stream_encoding(&self) -> StreamEncoding {
        StreamEncoding::Sse
    }
    /// Converts provider-neutral input to a concrete HTTP request.
    ///
    /// # Errors
    ///
    /// Returns an authentication or validation error when the request cannot
    /// be represented for this provider.
    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &WireRequestOptions,
    ) -> Result<HttpRequest, AiError>;
    /// Seeds request-specific decode state (for example grammar tool metadata).
    ///
    /// # Errors
    ///
    /// Returns a validation error when request-specific tool or sampling state
    /// cannot be prepared.
    fn initial_decode_state(
        &self,
        _model: &Model,
        _context: &Context,
        _options: &WireRequestOptions,
    ) -> Result<WireDecodeState, AiError> {
        Ok(WireDecodeState::default())
    }
    /// Converts one SSE frame to provider-neutral deltas.
    ///
    /// # Errors
    ///
    /// Returns a stream error when the provider frame is malformed or cannot
    /// be represented as provider-neutral deltas.
    fn decode_frame(
        &self,
        frame: &SseFrame,
        state: &mut WireDecodeState,
    ) -> Result<Vec<WireDelta>, AiError>;
}

/// Runs a wire adapter using an injectable HTTP transport.
pub fn execute_text_stream(
    transport: DynHttpTransport,
    adapter: Arc<dyn WireAdapter>,
    model: Model,
    context: Context,
    mut options: WireRequestOptions,
) -> AssistantEventStream {
    options.normalize_for_model(&model);
    let (sender, stream) = create_assistant_message_event_stream();
    tokio::spawn(async move {
        let mut assembler = StreamAssembler::new(model.clone(), sender);
        let result = async {
            if adapter.api() != model.api {
                return Err(AiError::Provider(format!(
                    "adapter {} cannot stream model API {}",
                    adapter.api(),
                    model.api
                )));
            }
            let request = adapter.build_request(&model, &context, &options)?;
            let mut response = match adapter.stream_encoding() {
                StreamEncoding::Sse => transport.execute_sse(request).await?,
                StreamEncoding::AwsEventStream => {
                    transport.execute_aws_event_stream(request).await?
                }
            };
            assembler.start();
            let mut decode_state = adapter.initial_decode_state(&model, &context, &options)?;
            while let Some(frame) = response.events.next().await {
                let frame = frame?;
                for delta in adapter.decode_frame(&frame, &mut decode_state)? {
                    if assembler.apply(delta)? {
                        return Ok(());
                    }
                }
            }
            Err(AiError::Stream(format!(
                "{} stream ended without a terminal event",
                model.provider
            )))
        }
        .await;
        if let Err(error) = result {
            assembler.fail(
                &error,
                options
                    .cancellation
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled),
            );
        }
        assembler.close();
    });
    stream
}

struct StreamAssembler {
    model: Model,
    partial: AssistantMessage,
    tool_json: HashMap<usize, String>,
    sender: Option<AssistantEventSender>,
}

impl StreamAssembler {
    fn new(model: Model, sender: AssistantEventSender) -> Self {
        Self {
            partial: AssistantMessage::empty(&model.api, &model.provider, &model.id),
            model,
            tool_json: HashMap::new(),
            sender: Some(sender),
        }
    }

    fn send(&mut self, event: AssistantMessageEvent) {
        if let Some(sender) = &mut self.sender {
            sender.send(event);
        }
    }

    fn start(&mut self) {
        self.send(AssistantMessageEvent::Start {
            partial: self.partial.clone(),
        });
    }

    fn apply(&mut self, delta: WireDelta) -> Result<bool, AiError> {
        match delta {
            WireDelta::ResponseId(id) => self.partial.response_id = Some(id),
            WireDelta::ResponseModel(model) => self.partial.response_model = Some(model),
            WireDelta::Diagnostic(diagnostic) => self.partial.diagnostics.push(diagnostic),
            WireDelta::TextStart { index } => self.start_text(index),
            WireDelta::TextDelta { index, delta } => {
                self.start_text(index);
                let Some(ContentBlock::Text(block)) = self.partial.content.get_mut(index) else {
                    return Err(AiError::Stream(format!(
                        "text delta targeted non-text block {index}"
                    )));
                };
                block.text.push_str(&delta);
                self.send(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::TextEnd {
                index,
                content,
                signature,
            } => {
                let Some(ContentBlock::Text(block)) = self.partial.content.get_mut(index) else {
                    return Err(AiError::Stream(format!(
                        "text end targeted missing block {index}"
                    )));
                };
                if let Some(content) = content {
                    block.text = content;
                }
                block.text_signature = signature;
                let content = block.text.clone();
                self.send(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::ThinkingStart { index } => self.start_thinking(index),
            WireDelta::ThinkingDelta { index, delta } => {
                self.start_thinking(index);
                let Some(ContentBlock::Thinking(block)) = self.partial.content.get_mut(index)
                else {
                    return Err(AiError::Stream(format!(
                        "thinking delta targeted non-thinking block {index}"
                    )));
                };
                block.thinking.push_str(&delta);
                self.send(AssistantMessageEvent::ThinkingDelta {
                    content_index: index,
                    delta,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::ThinkingEnd {
                index,
                content,
                signature,
                redacted,
            } => {
                let Some(ContentBlock::Thinking(block)) = self.partial.content.get_mut(index)
                else {
                    return Err(AiError::Stream(format!(
                        "thinking end targeted missing block {index}"
                    )));
                };
                if let Some(content) = content {
                    block.thinking = content;
                }
                block.thinking_signature = signature;
                block.redacted = redacted;
                let content = block.thinking.clone();
                self.send(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::ToolStart {
                index,
                id,
                name,
                thought_signature,
            } => {
                self.ensure_index(index)?;
                self.partial.content[index] = ContentBlock::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: serde_json::json!({}),
                    thought_signature,
                });
                self.tool_json.insert(index, String::new());
                self.send(AssistantMessageEvent::ToolcallStart {
                    content_index: index,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::ToolDelta { index, delta } => {
                let json = self.tool_json.entry(index).or_default();
                json.push_str(&delta);
                let arguments = parse_streaming_json(Some(json));
                let Some(ContentBlock::ToolCall(call)) = self.partial.content.get_mut(index) else {
                    return Err(AiError::Stream(format!(
                        "tool delta targeted missing block {index}"
                    )));
                };
                call.arguments = arguments;
                self.send(AssistantMessageEvent::ToolcallDelta {
                    content_index: index,
                    delta,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::ToolSignature { index, signature } => {
                let Some(ContentBlock::ToolCall(call)) = self.partial.content.get_mut(index) else {
                    return Err(AiError::Stream(format!(
                        "tool signature targeted missing block {index}"
                    )));
                };
                call.thought_signature = signature;
            }
            WireDelta::ToolEnd { index, arguments } => {
                let parsed = arguments.unwrap_or_else(|| {
                    parse_streaming_json(self.tool_json.get(&index).map(String::as_str))
                });
                let Some(ContentBlock::ToolCall(call)) = self.partial.content.get_mut(index) else {
                    return Err(AiError::Stream(format!(
                        "tool end targeted missing block {index}"
                    )));
                };
                call.arguments = parsed;
                let call = call.clone();
                self.tool_json.remove(&index);
                self.send(AssistantMessageEvent::ToolcallEnd {
                    content_index: index,
                    tool_call: call,
                    partial: self.partial.clone(),
                });
            }
            WireDelta::Usage(usage) => self.partial.usage = usage,
            WireDelta::Done(mut reason) => {
                if self
                    .partial
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolCall(_)))
                {
                    reason = StopReason::ToolUse;
                }
                self.partial.stop_reason = reason;
                self.finalize_cost();
                if reason.is_error() {
                    self.partial.error_message.get_or_insert_with(|| {
                        if reason == StopReason::Aborted {
                            "request was aborted".into()
                        } else {
                            "provider stopped with an error".into()
                        }
                    });
                    self.send(AssistantMessageEvent::Error {
                        reason,
                        error: self.partial.clone(),
                    });
                } else {
                    self.send(AssistantMessageEvent::Done {
                        reason,
                        message: self.partial.clone(),
                    });
                }
                return Ok(true);
            }
            WireDelta::Error(message) => {
                self.partial.stop_reason = StopReason::Error;
                self.partial.error_message = Some(message);
                self.finalize_cost();
                self.send(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: self.partial.clone(),
                });
                return Ok(true);
            }
            WireDelta::Aborted(message) => {
                self.partial.stop_reason = StopReason::Aborted;
                self.partial.error_message = Some(message);
                self.finalize_cost();
                self.send(AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error: self.partial.clone(),
                });
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn ensure_index(&mut self, index: usize) -> Result<(), AiError> {
        if index > self.partial.content.len() {
            return Err(AiError::Stream(format!(
                "provider emitted sparse content index {index}"
            )));
        }
        if index == self.partial.content.len() {
            self.partial
                .content
                .push(ContentBlock::Text(TextContent::default()));
        }
        Ok(())
    }

    fn start_text(&mut self, index: usize) {
        if index < self.partial.content.len()
            && matches!(self.partial.content[index], ContentBlock::Text(_))
        {
            return;
        }
        if self.ensure_index(index).is_err() {
            return;
        }
        self.partial.content[index] = ContentBlock::Text(TextContent::default());
        self.send(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: self.partial.clone(),
        });
    }

    fn start_thinking(&mut self, index: usize) {
        if index < self.partial.content.len()
            && matches!(self.partial.content[index], ContentBlock::Thinking(_))
        {
            return;
        }
        if self.ensure_index(index).is_err() {
            return;
        }
        self.partial.content[index] = ContentBlock::Thinking(ThinkingContent::default());
        self.send(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: self.partial.clone(),
        });
    }

    fn fail(&mut self, error: &AiError, cancelled: bool) {
        if self
            .sender
            .as_ref()
            .is_some_and(AssistantEventSender::is_terminal)
        {
            return;
        }
        let reason = if cancelled || matches!(error, AiError::Aborted) {
            StopReason::Aborted
        } else {
            StopReason::Error
        };
        self.partial.stop_reason = reason;
        self.partial.error_message = Some(error.to_string());
        self.finalize_cost();
        self.send(AssistantMessageEvent::Error {
            reason,
            error: self.partial.clone(),
        });
    }

    fn close(&mut self) {
        if let Some(sender) = &mut self.sender {
            sender.close();
        }
    }

    fn finalize_cost(&mut self) {
        if self.partial.usage.cost == crate::message::UsageCost::default() {
            calculate_cost(&self.model, &mut self.partial.usage);
        }
    }
}

/// Applies defaults, model headers, and caller headers case-insensitively.
pub fn merge_wire_headers(
    defaults: impl IntoIterator<Item = (String, String)>,
    model: &Model,
    overrides: &ProviderHeaders,
) -> HttpHeaders {
    let mut headers = HttpHeaders::new();
    for (name, value) in defaults {
        insert_header(&mut headers, &name, Some(&value));
    }
    for (name, value) in &model.headers {
        insert_header(&mut headers, name, Some(value));
    }
    for (name, value) in overrides {
        insert_header(&mut headers, name, value.as_ref());
    }
    headers
}

/// Resolves cache retention, including the legacy environment opt-in.
pub fn resolve_cache_retention(options: &WireRequestOptions) -> CacheRetention {
    options.cache_retention.unwrap_or_else(|| {
        if cache_retention_env(options).as_deref() == Some("long") {
            CacheRetention::Long
        } else {
            CacheRetention::Short
        }
    })
}

/// Resolves cache retention when the remote protocol owns its default.
pub fn resolve_optional_cache_retention(options: &WireRequestOptions) -> Option<CacheRetention> {
    options.cache_retention.or_else(|| {
        (cache_retention_env(options).as_deref() == Some("long")).then_some(CacheRetention::Long)
    })
}

fn cache_retention_env(options: &WireRequestOptions) -> Option<String> {
    options
        .env
        .get("PI_CACHE_RETENTION")
        .cloned()
        .or_else(|| std::env::var("PI_CACHE_RETENTION").ok())
}

fn insert_header(headers: &mut HttpHeaders, name: &str, value: Option<&String>) {
    let lower = name.to_ascii_lowercase();
    headers.retain(|existing, _| existing.to_ascii_lowercase() != lower);
    if let Some(value) = value {
        headers.insert(name.to_owned(), value.clone());
    }
}

/// Serializes a request body and applies common transport options.
///
/// # Errors
///
/// Returns a validation error when the JSON body cannot be serialized.
pub fn json_request(
    url: url::Url,
    body: &Value,
    headers: HttpHeaders,
    options: &WireRequestOptions,
) -> Result<HttpRequest, AiError> {
    Ok(HttpRequest {
        method: http::Method::POST,
        url,
        headers,
        body: bytes::Bytes::from(
            serde_json::to_vec(body).map_err(|error| AiError::Validation(error.to_string()))?,
        ),
        timeout: options.timeout,
        cancellation: options.cancellation.clone(),
    })
}

/// Converts provider headers to concrete HTTP headers, dropping suppressions.
pub fn concrete_headers(headers: &ProviderHeaders) -> HttpHeaders {
    headers
        .iter()
        .filter_map(|(name, value)| value.as_ref().map(|value| (name.clone(), value.clone())))
        .collect()
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream;
    use serde_json::json;

    use super::*;
    use crate::{
        message::UserMessage,
        transport::{HttpResponse, HttpTransport, SseResponse},
        wire::{anthropic::AnthropicMessagesAdapter, pi_messages::PiMessagesAdapter},
    };

    #[derive(Clone, Debug)]
    struct MemoryTransport {
        frames: Vec<SseFrame>,
    }

    #[async_trait]
    impl HttpTransport for MemoryTransport {
        async fn execute(&self, _request: HttpRequest) -> Result<HttpResponse, AiError> {
            Ok(HttpResponse {
                status: 200,
                headers: HttpHeaders::new(),
                body: Bytes::new(),
            })
        }

        async fn execute_sse(&self, _request: HttpRequest) -> Result<SseResponse, AiError> {
            Ok(SseResponse {
                status: 200,
                headers: HttpHeaders::new(),
                events: Box::pin(stream::iter(self.frames.clone().into_iter().map(Ok))),
            })
        }
    }

    #[test]
    fn header_overrides_are_case_insensitive_and_can_suppress() {
        let mut model = Model::new("p", "m", "a", "https://example.test");
        model.headers.insert("X-Model".into(), "model".into());
        let headers = merge_wire_headers(
            [("Authorization".into(), "default".into())],
            &model,
            &BTreeMap::from([
                ("authorization".into(), None),
                ("x-model".into(), Some("override".into())),
            ]),
        );
        assert!(
            !headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case("authorization"))
        );
        assert_eq!(headers.get("x-model").map(String::as_str), Some("override"));
    }

    #[test]
    fn cache_retention_honors_explicit_value_before_legacy_env() {
        let mut options = WireRequestOptions {
            env: ProviderEnv::from([("PI_CACHE_RETENTION".into(), "long".into())]),
            ..WireRequestOptions::default()
        };
        assert_eq!(resolve_cache_retention(&options), CacheRetention::Long);
        assert_eq!(
            resolve_optional_cache_retention(&options),
            Some(CacheRetention::Long)
        );

        options.cache_retention = Some(CacheRetention::Short);
        assert_eq!(resolve_cache_retention(&options), CacheRetention::Short);
        assert_eq!(
            resolve_optional_cache_retention(&options),
            Some(CacheRetention::Short)
        );
    }

    #[test]
    fn request_reasoning_is_clamped_to_model_capabilities() {
        let mut model = Model::new("provider", "model", "api", "https://example.test");
        model.reasoning = true;
        model
            .thinking_level_map
            .insert(ThinkingLevel::Xhigh, Some("xhigh".into()));
        model.thinking_level_map.insert(ThinkingLevel::High, None);
        let mut options = WireRequestOptions {
            reasoning: Some(ThinkingLevel::High),
            ..WireRequestOptions::default()
        };
        options.normalize_for_model(&model);
        assert_eq!(options.reasoning, Some(ThinkingLevel::Xhigh));
    }

    #[tokio::test]
    async fn error_stop_reasons_use_error_terminal_events() {
        let model = Model::new("provider", "model", "api", "https://example.test");
        let (sender, stream) = create_assistant_message_event_stream();
        let mut assembler = StreamAssembler::new(model, sender);
        assembler.start();
        assert!(
            assembler
                .apply(WireDelta::Done(StopReason::Error))
                .expect("apply")
        );
        assembler.close();
        let result = stream.result().await.expect("terminal result");
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn decode_state_allocates_stable_dense_slots() {
        let mut state = WireDecodeState::default();
        assert_eq!(state.slot("a"), 0);
        assert_eq!(state.slot("b"), 1);
        assert_eq!(state.slot("a"), 0);
    }

    #[tokio::test]
    async fn in_memory_transport_assembles_a_real_adapter_stream() {
        let frames = [
            serde_json::json!({
                "type": "message_start",
                "message": {"id": "msg", "model": "routed", "usage": {"input_tokens": 2}}
            }),
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"}
            }),
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1}
            }),
            serde_json::json!({"type": "message_stop"}),
        ]
        .into_iter()
        .map(|value| SseFrame {
            data: value.to_string(),
            ..SseFrame::default()
        })
        .collect();
        let transport: DynHttpTransport = Arc::new(MemoryTransport { frames });
        let model = Model::new(
            "anthropic",
            "claude-test",
            "anthropic-messages",
            "https://api.anthropic.com",
        );
        let context = Context {
            system_prompt: None,
            messages: vec![crate::message::Message::User(UserMessage::new("hi"))],
            tools: Vec::new(),
        };
        let stream = execute_text_stream(
            transport,
            Arc::new(AnthropicMessagesAdapter),
            model,
            context,
            WireRequestOptions {
                api_key: Some("key".into()),
                ..WireRequestOptions::default()
            },
        );
        let result = stream.result().await.expect("result");
        assert_eq!(result.response_id.as_deref(), Some("msg"));
        assert_eq!(result.response_model.as_deref(), Some("routed"));
        assert_eq!(
            result.content,
            vec![ContentBlock::Text(TextContent::new("hello"))]
        );
        assert_eq!(result.usage.input, 2);
        assert_eq!(result.usage.output, 1);
    }

    #[tokio::test]
    async fn pi_end_events_are_authoritative_and_keep_abort_reason() {
        let frames = [
            json!({"type": "start"}),
            json!({"type": "text_start", "contentIndex": 0}),
            json!({"type": "text_delta", "contentIndex": 0, "delta": "partial"}),
            json!({"type": "text_end", "contentIndex": 0, "content": "final"}),
            json!({
                "type": "done",
                "reason": "stop",
                "usage": {
                    "cost": {
                        "input": 1.0,
                        "output": 2.0,
                        "cacheRead": 0.0,
                        "cacheWrite": 0.0,
                        "total": 3.0
                    }
                }
            }),
        ]
        .into_iter()
        .map(|value| SseFrame {
            data: value.to_string(),
            ..SseFrame::default()
        })
        .collect();
        let model = Model::new("radius", "model", "pi-messages", "https://example.test");
        let stream = execute_text_stream(
            Arc::new(MemoryTransport { frames }),
            Arc::new(PiMessagesAdapter),
            model.clone(),
            Context::default(),
            WireRequestOptions {
                api_key: Some("key".into()),
                ..WireRequestOptions::default()
            },
        );
        let result = stream.result().await.expect("result");
        assert!(matches!(
            result.content.as_slice(),
            [ContentBlock::Text(text)] if text.text == "final"
        ));
        assert!((result.usage.cost.total - 3.0).abs() <= f64::EPSILON);

        let frames = vec![SseFrame {
            data: json!({"type": "error", "reason": "aborted"}).to_string(),
            ..SseFrame::default()
        }];
        let stream = execute_text_stream(
            Arc::new(MemoryTransport { frames }),
            Arc::new(PiMessagesAdapter),
            model,
            Context::default(),
            WireRequestOptions {
                api_key: Some("key".into()),
                ..WireRequestOptions::default()
            },
        );
        assert_eq!(
            stream.result().await.expect("aborted result").stop_reason,
            StopReason::Aborted
        );
    }
}
