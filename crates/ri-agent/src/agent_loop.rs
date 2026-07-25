//! Low-level prompt and continuation loops.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context as TaskContext, Poll},
};

use async_trait::async_trait;
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture, stream::FuturesUnordered};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::{
    AgentError, AgentEvent, AgentMessage, StreamFn, StreamOptions, Tool, ToolCallContext,
    ToolExecutionMode, ToolResult, ToolUpdateSink,
};

/// Transcript and tool snapshot consumed by a low-level run.
#[derive(Clone, Debug)]
pub struct AgentContext<M> {
    /// System instruction included in provider requests.
    pub system_prompt: String,
    /// Application transcript.
    pub messages: Vec<M>,
    /// Tools available for this run.
    pub tools: Vec<Arc<dyn Tool>>,
}

impl<M> AgentContext<M> {
    /// Creates a context without tools.
    pub fn new(system_prompt: impl Into<String>, messages: Vec<M>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            messages,
            tools: Vec::new(),
        }
    }
}

/// Result returned by a pre-execution tool hook.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeforeToolCallResult {
    /// Prevent the tool from executing.
    pub block: bool,
    /// Text used for a blocked error result.
    pub reason: Option<String>,
    /// Replacement for the already-validated arguments.
    ///
    /// A replacement is deliberately not revalidated, matching the reference
    /// hook's mutable-argument behavior.
    pub arguments: Option<Value>,
}

/// Partial replacement returned by a post-execution tool hook.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterToolCallResult {
    /// Full replacement content.
    pub content: Option<Vec<ri_ai::message::InputContent>>,
    /// Full replacement details value.
    pub details: Option<Value>,
    /// Replacement tool-local usage.
    pub usage: Option<ri_ai::Usage>,
    /// Replacement error flag.
    pub is_error: Option<bool>,
    /// Replacement batch-termination hint.
    pub terminate: Option<bool>,
}

/// Owned snapshot passed to a pre-execution tool hook.
#[derive(Clone, Debug)]
pub struct BeforeToolCallContext<M> {
    /// Assistant message requesting the invocation.
    pub assistant_message: ri_ai::AssistantMessage,
    /// Raw tool call from the assistant message.
    pub tool_call: ri_ai::ToolCall,
    /// Validated arguments.
    pub arguments: Value,
    /// Current application context.
    pub context: AgentContext<M>,
}

/// Owned snapshot passed to a post-execution tool hook.
#[derive(Clone, Debug)]
pub struct AfterToolCallContext<M> {
    /// Assistant message requesting the invocation.
    pub assistant_message: ri_ai::AssistantMessage,
    /// Raw tool call from the assistant message.
    pub tool_call: ri_ai::ToolCall,
    /// Validated, possibly pre-hook-replaced arguments.
    pub arguments: Value,
    /// Result before post-hook overrides.
    pub result: ToolResult,
    /// Error flag before post-hook overrides.
    pub is_error: bool,
    /// Current application context.
    pub context: AgentContext<M>,
}

/// Completed-turn snapshot supplied to stop and next-turn callbacks.
#[derive(Clone, Debug)]
pub struct CompletedTurn<M> {
    /// Assistant message that completed the turn.
    pub message: ri_ai::AssistantMessage,
    /// Tool result messages in source order.
    pub tool_results: Vec<ri_ai::ToolResultMessage>,
    /// Current context after appending the turn artifacts.
    pub context: AgentContext<M>,
    /// Messages produced by this loop invocation.
    pub new_messages: Vec<M>,
}

/// Replacement runtime values for the next provider turn.
#[derive(Clone, Debug)]
pub struct AgentLoopTurnUpdate<M> {
    /// Replacement context.
    pub context: Option<AgentContext<M>>,
    /// Replacement model.
    pub model: Option<ri_ai::Model>,
    /// Replacement reasoning level.
    pub thinking_level: Option<ri_ai::ThinkingLevel>,
}

impl<M> Default for AgentLoopTurnUpdate<M> {
    fn default() -> Self {
        Self {
            context: None,
            model: None,
            thinking_level: None,
        }
    }
}

type TransformContext<M> = dyn Fn(Vec<M>, CancellationToken) -> BoxFuture<'static, Result<Vec<M>, AgentError>>
    + Send
    + Sync;
type ConvertToLlm<M> =
    dyn Fn(Vec<M>) -> BoxFuture<'static, Result<Vec<ri_ai::Message>, AgentError>> + Send + Sync;
type GetApiKey =
    dyn Fn(String) -> BoxFuture<'static, Result<Option<String>, AgentError>> + Send + Sync;
type BeforeToolCall<M> = dyn Fn(
        BeforeToolCallContext<M>,
        CancellationToken,
    ) -> BoxFuture<'static, Result<BeforeToolCallResult, AgentError>>
    + Send
    + Sync;
type AfterToolCall<M> = dyn Fn(
        AfterToolCallContext<M>,
        CancellationToken,
    ) -> BoxFuture<'static, Result<AfterToolCallResult, AgentError>>
    + Send
    + Sync;
type PrepareNextTurn<M> = dyn Fn(
        CompletedTurn<M>,
        CancellationToken,
    ) -> BoxFuture<'static, Result<Option<AgentLoopTurnUpdate<M>>, AgentError>>
    + Send
    + Sync;
type ShouldStopAfterTurn<M> = dyn Fn(CompletedTurn<M>, CancellationToken) -> BoxFuture<'static, Result<bool, AgentError>>
    + Send
    + Sync;
type GetQueuedMessages<M> =
    dyn Fn() -> BoxFuture<'static, Result<Vec<M>, AgentError>> + Send + Sync;

/// Configuration for a low-level agent loop.
#[must_use]
#[derive(Clone)]
pub struct AgentLoopConfig<M> {
    /// Model used for the next provider request.
    pub model: ri_ai::Model,
    /// Requested reasoning effort.
    pub thinking_level: ri_ai::ThinkingLevel,
    /// Default tool scheduling policy.
    pub tool_execution: ToolExecutionMode,
    /// Optional provider cache-affinity identifier.
    pub session_id: Option<String>,
    /// Static API key used when no dynamic resolver returns one.
    pub api_key: Option<String>,
    /// Provider-specific options forwarded without interpretation.
    pub stream_extensions: indexmap::IndexMap<String, Value>,
    transform_context: Option<Arc<TransformContext<M>>>,
    convert_to_llm: Arc<ConvertToLlm<M>>,
    get_api_key: Option<Arc<GetApiKey>>,
    before_tool_call: Option<Arc<BeforeToolCall<M>>>,
    after_tool_call: Option<Arc<AfterToolCall<M>>>,
    prepare_next_turn: Option<Arc<PrepareNextTurn<M>>>,
    should_stop_after_turn: Option<Arc<ShouldStopAfterTurn<M>>>,
    get_steering_messages: Option<Arc<GetQueuedMessages<M>>>,
    get_follow_up_messages: Option<Arc<GetQueuedMessages<M>>>,
}

impl<M> std::fmt::Debug for AgentLoopConfig<M> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AgentLoopConfig")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tool_execution", &self.tool_execution)
            .field("session_id", &self.session_id)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("stream_extensions", &self.stream_extensions)
            .finish_non_exhaustive()
    }
}

impl<M: AgentMessage> AgentLoopConfig<M> {
    /// Creates configuration with the default application-message projection.
    pub fn new(model: ri_ai::Model) -> Self {
        Self {
            model,
            thinking_level: ri_ai::ThinkingLevel::Off,
            tool_execution: ToolExecutionMode::Parallel,
            session_id: None,
            api_key: None,
            stream_extensions: indexmap::IndexMap::new(),
            transform_context: None,
            convert_to_llm: Arc::new(|messages: Vec<M>| {
                async move { Ok(messages.iter().filter_map(AgentMessage::project).collect()) }
                    .boxed()
            }),
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            prepare_next_turn: None,
            should_stop_after_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
        }
    }

    /// Replaces the application-level context transform.
    pub fn with_transform_context<F, Fut>(mut self, transform: F) -> Self
    where
        F: Fn(Vec<M>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<M>, AgentError>> + Send + 'static,
    {
        self.transform_context = Some(Arc::new(move |messages, cancellation| {
            transform(messages, cancellation).boxed()
        }));
        self
    }

    /// Replaces the provider-message projection.
    pub fn with_convert_to_llm<F, Fut>(mut self, convert: F) -> Self
    where
        F: Fn(Vec<M>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<ri_ai::Message>, AgentError>> + Send + 'static,
    {
        self.convert_to_llm = Arc::new(move |messages| convert(messages).boxed());
        self
    }

    /// Installs a dynamic API-key resolver.
    pub fn with_api_key_resolver<F, Fut>(mut self, resolver: F) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>, AgentError>> + Send + 'static,
    {
        self.get_api_key = Some(Arc::new(move |provider| resolver(provider).boxed()));
        self
    }

    /// Installs a pre-execution tool hook.
    pub fn with_before_tool_call<F, Fut>(mut self, hook: F) -> Self
    where
        F: Fn(BeforeToolCallContext<M>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BeforeToolCallResult, AgentError>> + Send + 'static,
    {
        self.before_tool_call = Some(Arc::new(move |context, cancellation| {
            hook(context, cancellation).boxed()
        }));
        self
    }

    /// Installs a post-execution tool hook.
    pub fn with_after_tool_call<F, Fut>(mut self, hook: F) -> Self
    where
        F: Fn(AfterToolCallContext<M>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<AfterToolCallResult, AgentError>> + Send + 'static,
    {
        self.after_tool_call = Some(Arc::new(move |context, cancellation| {
            hook(context, cancellation).boxed()
        }));
        self
    }

    /// Installs the completed-turn runtime snapshot updater.
    pub fn with_prepare_next_turn<F, Fut>(mut self, prepare: F) -> Self
    where
        F: Fn(CompletedTurn<M>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<AgentLoopTurnUpdate<M>>, AgentError>> + Send + 'static,
    {
        self.prepare_next_turn = Some(Arc::new(move |turn, cancellation| {
            prepare(turn, cancellation).boxed()
        }));
        self
    }

    /// Installs a graceful completed-turn stop predicate.
    pub fn with_should_stop_after_turn<F, Fut>(mut self, predicate: F) -> Self
    where
        F: Fn(CompletedTurn<M>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<bool, AgentError>> + Send + 'static,
    {
        self.should_stop_after_turn = Some(Arc::new(move |turn, cancellation| {
            predicate(turn, cancellation).boxed()
        }));
        self
    }

    /// Installs the steering queue drain callback.
    pub fn with_steering_messages<F, Fut>(mut self, drain: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<M>, AgentError>> + Send + 'static,
    {
        self.get_steering_messages = Some(Arc::new(move || drain().boxed()));
        self
    }

    /// Installs the follow-up queue drain callback.
    pub fn with_follow_up_messages<F, Fut>(mut self, drain: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<M>, AgentError>> + Send + 'static,
    {
        self.get_follow_up_messages = Some(Arc::new(move || drain().boxed()));
        self
    }
}

/// Awaited destination for low-level loop events.
#[async_trait]
pub trait AgentEventSink<M>: Send + Sync + 'static {
    /// Delivers one event. The loop does not proceed until this future settles.
    ///
    /// # Errors
    ///
    /// Returns an event-consumer failure when delivery cannot complete.
    async fn emit(&self, event: AgentEvent<M>) -> Result<(), AgentError>;
}

#[async_trait]
impl<M, F, Fut> AgentEventSink<M> for F
where
    M: Send + 'static,
    F: Fn(AgentEvent<M>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), AgentError>> + Send,
{
    async fn emit(&self, event: AgentEvent<M>) -> Result<(), AgentError> {
        self(event).await
    }
}

/// Shared event sink used by loop and tool tasks.
pub type SharedEventSink<M> = Arc<dyn AgentEventSink<M>>;

/// Runs a new prompt and returns only messages produced by this invocation.
///
/// # Errors
///
/// Returns an error when event delivery fails. Provider and callback failures
/// are represented by a terminal assistant message and complete lifecycle.
pub async fn run_agent_loop<M>(
    prompts: Vec<M>,
    context: AgentContext<M>,
    config: AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: SharedEventSink<M>,
    stream_fn: Arc<dyn StreamFn>,
) -> Result<Vec<M>, AgentError>
where
    M: AgentMessage,
{
    let mut new_messages = prompts.clone();
    let mut current_context = context;
    current_context.messages.extend(prompts.iter().cloned());
    let failure_model = config.model.clone();

    emit.emit(AgentEvent::AgentStart).await?;
    emit.emit(AgentEvent::TurnStart).await?;
    for prompt in prompts {
        emit_message(&emit, prompt).await?;
    }

    if let Err(error) = run_loop(
        &mut current_context,
        &mut new_messages,
        config,
        cancellation.clone(),
        &emit,
        &stream_fn,
    )
    .await
    {
        complete_failure_lifecycle(
            &mut current_context,
            &mut new_messages,
            &failure_model,
            cancellation,
            error,
            &emit,
        )
        .await?;
    }
    Ok(new_messages)
}

/// Continues from an existing user, tool-result, or custom transcript tail.
///
/// # Errors
///
/// Returns an invalid-continuation error before starting, or an event-delivery
/// error during the run. Operational failures complete as assistant messages.
pub async fn run_agent_loop_continue<M>(
    mut context: AgentContext<M>,
    config: AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: SharedEventSink<M>,
    stream_fn: Arc<dyn StreamFn>,
) -> Result<Vec<M>, AgentError>
where
    M: AgentMessage,
{
    validate_continuation(&context)?;
    let mut new_messages = Vec::new();
    let failure_model = config.model.clone();
    emit.emit(AgentEvent::AgentStart).await?;
    emit.emit(AgentEvent::TurnStart).await?;
    if let Err(error) = run_loop(
        &mut context,
        &mut new_messages,
        config,
        cancellation.clone(),
        &emit,
        &stream_fn,
    )
    .await
    {
        complete_failure_lifecycle(
            &mut context,
            &mut new_messages,
            &failure_model,
            cancellation,
            error,
            &emit,
        )
        .await?;
    }
    Ok(new_messages)
}

fn validate_continuation<M: AgentMessage>(context: &AgentContext<M>) -> Result<(), AgentError> {
    let Some(last) = context.messages.last() else {
        return Err(AgentError::EmptyContinuation);
    };
    if matches!(last.as_llm_message(), Some(ri_ai::Message::Assistant(_))) {
        return Err(AgentError::AssistantContinuation);
    }
    Ok(())
}

async fn run_loop<M: AgentMessage>(
    initial_context: &mut AgentContext<M>,
    new_messages: &mut Vec<M>,
    mut config: AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
    stream_fn: &Arc<dyn StreamFn>,
) -> Result<(), AgentError> {
    let mut current_context = initial_context.clone();
    let mut first_turn = true;
    let mut pending_messages = drain_messages(config.get_steering_messages.as_ref()).await?;

    loop {
        let mut has_more_tool_calls = true;
        while has_more_tool_calls || !pending_messages.is_empty() {
            if first_turn {
                first_turn = false;
            } else {
                emit.emit(AgentEvent::TurnStart).await?;
            }

            for message in std::mem::take(&mut pending_messages) {
                emit_message(emit, message.clone()).await?;
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            let assistant = stream_assistant_response(
                &mut current_context,
                &config,
                cancellation.clone(),
                emit,
                stream_fn,
            )
            .await?;
            let assistant_message =
                M::from_llm_message(ri_ai::Message::Assistant(assistant.clone()));
            new_messages.push(assistant_message.clone());

            if assistant.stop_reason.is_error() {
                emit.emit(AgentEvent::TurnEnd {
                    message: assistant_message,
                    tool_results: Vec::new(),
                })
                .await?;
                emit.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                *initial_context = current_context;
                return Ok(());
            }

            let tool_calls = assistant
                .content
                .iter()
                .filter_map(|block| match block {
                    ri_ai::ContentBlock::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let batch = if tool_calls.is_empty() {
                ExecutedToolBatch::default()
            } else if assistant.stop_reason == ri_ai::StopReason::Length {
                fail_truncated_tool_calls(tool_calls, emit).await?
            } else {
                execute_tool_calls(
                    &current_context,
                    &assistant,
                    tool_calls,
                    &config,
                    cancellation.clone(),
                    emit,
                )
                .await?
            };
            has_more_tool_calls = !tool_calls_were_absent_or_terminated(&assistant, &batch);

            for result in &batch.messages {
                let wrapped = M::from_llm_message(ri_ai::Message::ToolResult(result.clone()));
                current_context.messages.push(wrapped.clone());
                new_messages.push(wrapped);
            }

            emit.emit(AgentEvent::TurnEnd {
                message: assistant_message,
                tool_results: batch.messages.clone(),
            })
            .await?;

            let completed = CompletedTurn {
                message: assistant.clone(),
                tool_results: batch.messages.clone(),
                context: current_context.clone(),
                new_messages: new_messages.clone(),
            };
            if let Some(prepare) = &config.prepare_next_turn
                && let Some(update) = prepare(completed.clone(), cancellation.clone()).await?
            {
                if let Some(context) = update.context {
                    current_context = context;
                }
                if let Some(model) = update.model {
                    config.model = model;
                }
                if let Some(thinking_level) = update.thinking_level {
                    config.thinking_level = thinking_level;
                }
            }

            let completed = CompletedTurn {
                message: assistant,
                tool_results: batch.messages,
                context: current_context.clone(),
                new_messages: new_messages.clone(),
            };
            if let Some(should_stop) = &config.should_stop_after_turn
                && should_stop(completed, cancellation.clone()).await?
            {
                emit.emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await?;
                *initial_context = current_context;
                return Ok(());
            }

            pending_messages = drain_messages(config.get_steering_messages.as_ref()).await?;
        }

        let follow_ups = drain_messages(config.get_follow_up_messages.as_ref()).await?;
        if follow_ups.is_empty() {
            break;
        }
        pending_messages = follow_ups;
    }

    emit.emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await?;
    *initial_context = current_context;
    Ok(())
}

fn tool_calls_were_absent_or_terminated(
    assistant: &ri_ai::AssistantMessage,
    batch: &ExecutedToolBatch,
) -> bool {
    let had_tool_calls = assistant
        .content
        .iter()
        .any(|block| matches!(block, ri_ai::ContentBlock::ToolCall(_)));
    !had_tool_calls || batch.terminate
}

async fn drain_messages<M>(
    drain: Option<&Arc<GetQueuedMessages<M>>>,
) -> Result<Vec<M>, AgentError> {
    match drain {
        Some(drain) => drain().await,
        None => Ok(Vec::new()),
    }
}

async fn stream_assistant_response<M: AgentMessage>(
    context: &mut AgentContext<M>,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
    stream_fn: &Arc<dyn StreamFn>,
) -> Result<ri_ai::AssistantMessage, AgentError> {
    let mut application_messages = context.messages.clone();
    if let Some(transform) = &config.transform_context {
        application_messages = transform(application_messages, cancellation.clone()).await?;
    }
    let llm_messages = (config.convert_to_llm)(application_messages).await?;
    let llm_context = ri_ai::Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: context
            .tools
            .iter()
            .map(|tool| tool.definition().clone())
            .collect(),
    };
    let api_key = if let Some(resolve) = &config.get_api_key {
        resolve(config.model.provider.clone())
            .await?
            .or_else(|| config.api_key.clone())
    } else {
        config.api_key.clone()
    };
    let options = StreamOptions {
        cancellation: cancellation.clone(),
        thinking_level: config.thinking_level,
        session_id: config.session_id.clone(),
        api_key,
        extensions: config.stream_extensions.clone(),
    };

    let stream_result = tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            return finish_synthetic_assistant(
                context,
                config,
                None,
                false,
                ri_ai::StopReason::Aborted,
                "Operation aborted".to_owned(),
                emit,
            ).await;
        }
        result = stream_fn.stream(config.model.clone(), llm_context, options) => result,
    };
    let mut stream = match stream_result {
        Ok(stream) => stream,
        Err(error) => {
            let reason = if matches!(error, ri_ai::AiError::Aborted) {
                ri_ai::StopReason::Aborted
            } else {
                ri_ai::StopReason::Error
            };
            return finish_synthetic_assistant(
                context,
                config,
                None,
                false,
                reason,
                error.to_string(),
                emit,
            )
            .await;
        }
    };

    let mut partial: Option<ri_ai::AssistantMessage> = None;
    let mut added_partial = false;
    loop {
        let next = tokio::select! {
            biased;
            () = cancellation.cancelled() => {
                return finish_synthetic_assistant(
                    context,
                    config,
                    partial,
                    added_partial,
                    ri_ai::StopReason::Aborted,
                    "Operation aborted".to_owned(),
                    emit,
                ).await;
            }
            next = stream.next() => next,
        };
        let Some(event) = next else {
            let result = stream.result().await;
            return match result {
                Ok(message) => finish_final_assistant(context, message, added_partial, emit).await,
                Err(error) => {
                    finish_synthetic_assistant(
                        context,
                        config,
                        partial,
                        added_partial,
                        ri_ai::StopReason::Error,
                        error.to_string(),
                        emit,
                    )
                    .await
                }
            };
        };

        match &event {
            ri_ai::AssistantMessageEvent::Start { partial: started } => {
                partial = Some(started.clone());
                if added_partial {
                    replace_last_assistant(context, started.clone());
                } else {
                    context
                        .messages
                        .push(M::from_llm_message(ri_ai::Message::Assistant(
                            started.clone(),
                        )));
                    added_partial = true;
                    emit.emit(AgentEvent::MessageStart {
                        message: M::from_llm_message(ri_ai::Message::Assistant(started.clone())),
                    })
                    .await?;
                }
            }
            ri_ai::AssistantMessageEvent::Done { message, .. } => {
                return finish_final_assistant(context, message.clone(), added_partial, emit).await;
            }
            ri_ai::AssistantMessageEvent::Error { reason, error } => {
                let mut error = error.clone();
                error.stop_reason = *reason;
                return finish_final_assistant(context, error, added_partial, emit).await;
            }
            _ => {
                if let Some(snapshot) = event.partial().cloned() {
                    partial = Some(snapshot.clone());
                    if !added_partial {
                        context
                            .messages
                            .push(M::from_llm_message(ri_ai::Message::Assistant(
                                snapshot.clone(),
                            )));
                        added_partial = true;
                        emit.emit(AgentEvent::MessageStart {
                            message: M::from_llm_message(ri_ai::Message::Assistant(
                                snapshot.clone(),
                            )),
                        })
                        .await?;
                    }
                    replace_last_assistant(context, snapshot.clone());
                    emit.emit(AgentEvent::MessageUpdate {
                        message: M::from_llm_message(ri_ai::Message::Assistant(snapshot)),
                        assistant_event: event.clone(),
                    })
                    .await?;
                }
            }
        }
    }
}

fn replace_last_assistant<M: AgentMessage>(
    context: &mut AgentContext<M>,
    assistant: ri_ai::AssistantMessage,
) {
    if let Some(last) = context.messages.last_mut() {
        *last = M::from_llm_message(ri_ai::Message::Assistant(assistant));
    }
}

async fn finish_final_assistant<M: AgentMessage>(
    context: &mut AgentContext<M>,
    message: ri_ai::AssistantMessage,
    added_partial: bool,
    emit: &SharedEventSink<M>,
) -> Result<ri_ai::AssistantMessage, AgentError> {
    let wrapped = M::from_llm_message(ri_ai::Message::Assistant(message.clone()));
    if added_partial {
        if let Some(last) = context.messages.last_mut() {
            *last = wrapped.clone();
        }
    } else {
        context.messages.push(wrapped.clone());
        emit.emit(AgentEvent::MessageStart {
            message: wrapped.clone(),
        })
        .await?;
    }
    emit.emit(AgentEvent::MessageEnd { message: wrapped })
        .await?;
    Ok(message)
}

async fn finish_synthetic_assistant<M: AgentMessage>(
    context: &mut AgentContext<M>,
    config: &AgentLoopConfig<M>,
    partial: Option<ri_ai::AssistantMessage>,
    added_partial: bool,
    reason: ri_ai::StopReason,
    error_message: String,
    emit: &SharedEventSink<M>,
) -> Result<ri_ai::AssistantMessage, AgentError> {
    let mut message =
        partial.unwrap_or_else(|| failure_assistant(&config.model, reason, error_message.clone()));
    message.stop_reason = reason;
    message.error_message = Some(error_message);
    finish_final_assistant(context, message, added_partial, emit).await
}

#[derive(Debug)]
enum PreparedToolCall {
    Immediate(FinalizedToolCall),
    Prepared {
        source_index: usize,
        tool_call: ri_ai::ToolCall,
        tool: Arc<dyn Tool>,
        arguments: Value,
    },
}

#[derive(Clone, Debug)]
struct FinalizedToolCall {
    source_index: usize,
    tool_call: ri_ai::ToolCall,
    result: ToolResult,
    is_error: bool,
}

#[derive(Debug, Default)]
struct ExecutedToolBatch {
    messages: Vec<ri_ai::ToolResultMessage>,
    terminate: bool,
}

async fn fail_truncated_tool_calls<M: AgentMessage>(
    tool_calls: Vec<ri_ai::ToolCall>,
    emit: &SharedEventSink<M>,
) -> Result<ExecutedToolBatch, AgentError> {
    let mut messages = Vec::with_capacity(tool_calls.len());
    for (source_index, tool_call) in tool_calls.into_iter().enumerate() {
        emit_tool_start(&tool_call, emit).await?;
        let finalized = FinalizedToolCall {
            source_index,
            result: ToolResult::error(format!(
                "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                tool_call.name
            )),
            tool_call,
            is_error: true,
        };
        emit_tool_end(&finalized, emit).await?;
        let message = create_tool_result_message(&finalized);
        emit_message(
            emit,
            M::from_llm_message(ri_ai::Message::ToolResult(message.clone())),
        )
        .await?;
        messages.push(message);
    }
    Ok(ExecutedToolBatch {
        messages,
        terminate: false,
    })
}

async fn execute_tool_calls<M: AgentMessage>(
    context: &AgentContext<M>,
    assistant: &ri_ai::AssistantMessage,
    tool_calls: Vec<ri_ai::ToolCall>,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
) -> Result<ExecutedToolBatch, AgentError> {
    let has_sequential_tool = tool_calls.iter().any(|call| {
        context
            .tools
            .iter()
            .find(|tool| tool.definition().name == call.name)
            .and_then(|tool| tool.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });
    if config.tool_execution == ToolExecutionMode::Sequential || has_sequential_tool {
        execute_tool_calls_sequential(context, assistant, tool_calls, config, cancellation, emit)
            .await
    } else {
        execute_tool_calls_parallel(context, assistant, tool_calls, config, cancellation, emit)
            .await
    }
}

async fn execute_tool_calls_sequential<M: AgentMessage>(
    context: &AgentContext<M>,
    assistant: &ri_ai::AssistantMessage,
    tool_calls: Vec<ri_ai::ToolCall>,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
) -> Result<ExecutedToolBatch, AgentError> {
    let mut finalized_calls = Vec::new();
    let mut messages = Vec::new();
    for (source_index, tool_call) in tool_calls.into_iter().enumerate() {
        emit_tool_start(&tool_call, emit).await?;
        let prepared = prepare_tool_call(
            source_index,
            context,
            assistant,
            tool_call,
            config,
            cancellation.clone(),
        )
        .await;
        let finalized = match prepared {
            PreparedToolCall::Immediate(finalized) => finalized,
            prepared @ PreparedToolCall::Prepared { .. } => {
                execute_prepared_tool_call(
                    context,
                    assistant,
                    prepared,
                    config,
                    cancellation.clone(),
                    emit,
                )
                .await?
            }
        };
        emit_tool_end(&finalized, emit).await?;
        let message = create_tool_result_message(&finalized);
        emit_message(
            emit,
            M::from_llm_message(ri_ai::Message::ToolResult(message.clone())),
        )
        .await?;
        finalized_calls.push(finalized);
        messages.push(message);
        if cancellation.is_cancelled() {
            break;
        }
    }
    Ok(ExecutedToolBatch {
        terminate: !cancellation.is_cancelled() && should_terminate_tool_batch(&finalized_calls),
        messages,
    })
}

async fn execute_tool_calls_parallel<M: AgentMessage>(
    context: &AgentContext<M>,
    assistant: &ri_ai::AssistantMessage,
    tool_calls: Vec<ri_ai::ToolCall>,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
) -> Result<ExecutedToolBatch, AgentError> {
    let mut finalized_calls = Vec::new();
    let mut prepared_calls = Vec::new();
    for (source_index, tool_call) in tool_calls.into_iter().enumerate() {
        emit_tool_start(&tool_call, emit).await?;
        match prepare_tool_call(
            source_index,
            context,
            assistant,
            tool_call,
            config,
            cancellation.clone(),
        )
        .await
        {
            PreparedToolCall::Immediate(finalized) => {
                emit_tool_end(&finalized, emit).await?;
                finalized_calls.push(finalized);
            }
            prepared @ PreparedToolCall::Prepared { .. } => prepared_calls.push(prepared),
        }
        if cancellation.is_cancelled() {
            break;
        }
    }

    let mut executions = FuturesUnordered::new();
    for prepared in prepared_calls {
        executions.push(execute_prepared_tool_call(
            context,
            assistant,
            prepared,
            config,
            cancellation.clone(),
            emit,
        ));
    }
    while let Some(finalized) = executions.next().await {
        let finalized = finalized?;
        emit_tool_end(&finalized, emit).await?;
        finalized_calls.push(finalized);
    }

    finalized_calls.sort_by_key(|finalized| finalized.source_index);
    let mut messages = Vec::with_capacity(finalized_calls.len());
    for finalized in &finalized_calls {
        let message = create_tool_result_message(finalized);
        emit_message(
            emit,
            M::from_llm_message(ri_ai::Message::ToolResult(message.clone())),
        )
        .await?;
        messages.push(message);
    }
    Ok(ExecutedToolBatch {
        terminate: !cancellation.is_cancelled() && should_terminate_tool_batch(&finalized_calls),
        messages,
    })
}

async fn prepare_tool_call<M: AgentMessage>(
    source_index: usize,
    context: &AgentContext<M>,
    assistant: &ri_ai::AssistantMessage,
    tool_call: ri_ai::ToolCall,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
) -> PreparedToolCall {
    let Some(tool) = context
        .tools
        .iter()
        .find(|tool| tool.definition().name == tool_call.name)
        .cloned()
    else {
        return immediate_tool_error(
            source_index,
            tool_call.clone(),
            format!("Tool {} not found", tool_call.name),
        );
    };

    if cancellation.is_cancelled() {
        return immediate_tool_error(source_index, tool_call, "Operation aborted");
    }

    let prepared_arguments = match tool.prepare_arguments(tool_call.arguments.clone()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return immediate_tool_error(source_index, tool_call, error.to_string());
        }
    };
    let prepared_call = ri_ai::ToolCall {
        arguments: prepared_arguments,
        ..tool_call.clone()
    };
    let mut arguments =
        match ri_ai::tool::validate_tool_arguments(tool.definition(), &prepared_call) {
            Ok(arguments) => arguments,
            Err(error) => {
                return immediate_tool_error(source_index, tool_call, error.to_string());
            }
        };

    if let Some(before) = &config.before_tool_call {
        let before_result = before(
            BeforeToolCallContext {
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                arguments: arguments.clone(),
                context: context.clone(),
            },
            cancellation.clone(),
        )
        .await;
        match before_result {
            Ok(before_result) => {
                if let Some(replacement) = before_result.arguments {
                    arguments = replacement;
                }
                if cancellation.is_cancelled() {
                    return immediate_tool_error(source_index, tool_call, "Operation aborted");
                }
                if before_result.block {
                    return immediate_tool_error(
                        source_index,
                        tool_call,
                        before_result
                            .reason
                            .unwrap_or_else(|| "Tool execution was blocked".to_owned()),
                    );
                }
            }
            Err(error) => {
                return immediate_tool_error(source_index, tool_call, error.to_string());
            }
        }
    }
    if cancellation.is_cancelled() {
        return immediate_tool_error(source_index, tool_call, "Operation aborted");
    }
    PreparedToolCall::Prepared {
        source_index,
        tool_call,
        tool,
        arguments,
    }
}

fn immediate_tool_error(
    source_index: usize,
    tool_call: ri_ai::ToolCall,
    message: impl Into<String>,
) -> PreparedToolCall {
    PreparedToolCall::Immediate(FinalizedToolCall {
        source_index,
        tool_call,
        result: ToolResult::error(message),
        is_error: true,
    })
}

async fn execute_prepared_tool_call<M: AgentMessage>(
    context: &AgentContext<M>,
    assistant: &ri_ai::AssistantMessage,
    prepared: PreparedToolCall,
    config: &AgentLoopConfig<M>,
    cancellation: CancellationToken,
    emit: &SharedEventSink<M>,
) -> Result<FinalizedToolCall, AgentError> {
    let PreparedToolCall::Prepared {
        source_index,
        tool_call,
        tool,
        arguments,
    } = prepared
    else {
        return Err(AgentError::Callback(
            "internal tool scheduler received an immediate result as executable".to_owned(),
        ));
    };

    let update_emit = Arc::clone(emit);
    let update_call = tool_call.clone();
    let updates = ToolUpdateSink::new(move |partial_result| {
        let emit = Arc::clone(&update_emit);
        let call = update_call.clone();
        async move {
            emit.emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: call.id,
                tool_name: call.name,
                arguments: call.arguments,
                partial_result,
            })
            .await
        }
    });
    let tool_context = ToolCallContext {
        tool_call_id: tool_call.id.clone(),
        cancellation: cancellation.clone(),
        updates: updates.clone(),
    };
    let execution = tool.execute(tool_context, arguments.clone());
    let executed = tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            Err(crate::ToolError::message("Operation aborted"))
        }
        result = execution => result,
    };
    updates.close_and_wait().await?;

    let (mut result, mut is_error) = match executed {
        Ok(result) => (result, false),
        Err(error) => (ToolResult::error(error.to_string()), true),
    };
    if let Some(after) = &config.after_tool_call {
        match after(
            AfterToolCallContext {
                assistant_message: assistant.clone(),
                tool_call: tool_call.clone(),
                arguments: arguments.clone(),
                result: result.clone(),
                is_error,
                context: context.clone(),
            },
            cancellation,
        )
        .await
        {
            Ok(overrides) => {
                if let Some(content) = overrides.content {
                    result.content = content;
                }
                if let Some(details) = overrides.details {
                    result.details = Some(details);
                }
                if let Some(usage) = overrides.usage {
                    result.usage = Some(usage);
                }
                if let Some(terminate) = overrides.terminate {
                    result.terminate = terminate;
                }
                if let Some(error) = overrides.is_error {
                    is_error = error;
                }
            }
            Err(error) => {
                result = ToolResult::error(error.to_string());
                is_error = true;
            }
        }
    }
    Ok(FinalizedToolCall {
        source_index,
        tool_call,
        result,
        is_error,
    })
}

fn should_terminate_tool_batch(finalized_calls: &[FinalizedToolCall]) -> bool {
    !finalized_calls.is_empty()
        && finalized_calls
            .iter()
            .all(|finalized| finalized.result.terminate)
}

async fn emit_tool_start<M: 'static>(
    tool_call: &ri_ai::ToolCall,
    emit: &SharedEventSink<M>,
) -> Result<(), AgentError> {
    emit.emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        arguments: tool_call.arguments.clone(),
    })
    .await
}

async fn emit_tool_end<M: 'static>(
    finalized: &FinalizedToolCall,
    emit: &SharedEventSink<M>,
) -> Result<(), AgentError> {
    emit.emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
    })
    .await
}

fn create_tool_result_message(finalized: &FinalizedToolCall) -> ri_ai::ToolResultMessage {
    ri_ai::ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: finalized.result.content.clone(),
        details: finalized.result.details.clone(),
        usage: finalized.result.usage.clone(),
        added_tool_names: finalized.result.added_tool_names.clone(),
        is_error: finalized.is_error,
        timestamp: ri_ai::message::now_millis(),
    }
}

async fn emit_message<M: Clone + 'static>(
    emit: &SharedEventSink<M>,
    message: M,
) -> Result<(), AgentError> {
    emit.emit(AgentEvent::MessageStart {
        message: message.clone(),
    })
    .await?;
    emit.emit(AgentEvent::MessageEnd { message }).await
}

fn failure_assistant(
    model: &ri_ai::Model,
    reason: ri_ai::StopReason,
    error_message: String,
) -> ri_ai::AssistantMessage {
    let mut message =
        ri_ai::AssistantMessage::empty(model.api.clone(), model.provider.clone(), model.id.clone());
    message
        .content
        .push(ri_ai::ContentBlock::Text(ri_ai::TextContent::new("")));
    message.stop_reason = reason;
    message.error_message = Some(error_message);
    message
}

async fn complete_failure_lifecycle<M: AgentMessage>(
    context: &mut AgentContext<M>,
    new_messages: &mut Vec<M>,
    model: &ri_ai::Model,
    cancellation: CancellationToken,
    error: AgentError,
    emit: &SharedEventSink<M>,
) -> Result<(), AgentError> {
    let reason = if cancellation.is_cancelled()
        || matches!(&error, AgentError::Ai(ri_ai::AiError::Aborted))
    {
        ri_ai::StopReason::Aborted
    } else {
        ri_ai::StopReason::Error
    };
    let assistant = failure_assistant(model, reason, error.to_string());
    let message = M::from_llm_message(ri_ai::Message::Assistant(assistant.clone()));
    context.messages.push(message.clone());
    new_messages.push(message.clone());
    emit_message(emit, message.clone()).await?;
    emit.emit(AgentEvent::TurnEnd {
        message,
        tool_results: Vec::new(),
    })
    .await?;
    emit.emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await
}

/// Event stream returned by [`agent_loop`] and [`agent_loop_continue`].
#[derive(Debug)]
pub struct AgentEventStream<M> {
    events: mpsc::UnboundedReceiver<AgentEvent<M>>,
    result: watch::Receiver<Option<Result<Vec<M>, AgentError>>>,
}

impl<M: Clone> AgentEventStream<M> {
    /// Waits for the run's produced messages independently of event iteration.
    ///
    /// # Errors
    ///
    /// Returns the loop failure, or [`AgentError::EventStreamClosed`] if the
    /// producer disappears without publishing a result.
    pub async fn result(&self) -> Result<Vec<M>, AgentError> {
        let mut result = self.result.clone();
        loop {
            if let Some(result) = result.borrow().clone() {
                return result;
            }
            if result.changed().await.is_err() {
                return Err(AgentError::EventStreamClosed);
            }
        }
    }
}

impl<M> Stream for AgentEventStream<M> {
    type Item = AgentEvent<M>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(context)
    }
}

#[derive(Debug)]
struct ChannelEventSink<M> {
    events: mpsc::UnboundedSender<AgentEvent<M>>,
}

#[async_trait]
impl<M: Send + 'static> AgentEventSink<M> for ChannelEventSink<M> {
    async fn emit(&self, event: AgentEvent<M>) -> Result<(), AgentError> {
        self.events
            .send(event)
            .map_err(|_| AgentError::EventStreamClosed)
    }
}

/// Starts a prompt loop in a Tokio task and returns its event stream.
pub fn agent_loop<M>(
    prompts: Vec<M>,
    context: AgentContext<M>,
    config: AgentLoopConfig<M>,
    cancellation: CancellationToken,
    stream_fn: Arc<dyn StreamFn>,
) -> AgentEventStream<M>
where
    M: AgentMessage,
{
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = watch::channel(None);
    let emit: SharedEventSink<M> = Arc::new(ChannelEventSink { events: events_tx });
    tokio::spawn(async move {
        let result = run_agent_loop(prompts, context, config, cancellation, emit, stream_fn).await;
        result_tx.send_replace(Some(result));
    });
    AgentEventStream {
        events: events_rx,
        result: result_rx,
    }
}

/// Starts a continuation loop in a Tokio task.
///
/// # Errors
///
/// Returns an invalid-continuation error without spawning a task.
pub fn agent_loop_continue<M>(
    context: AgentContext<M>,
    config: AgentLoopConfig<M>,
    cancellation: CancellationToken,
    stream_fn: Arc<dyn StreamFn>,
) -> Result<AgentEventStream<M>, AgentError>
where
    M: AgentMessage,
{
    validate_continuation(&context)?;
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = watch::channel(None);
    let emit: SharedEventSink<M> = Arc::new(ChannelEventSink { events: events_tx });
    tokio::spawn(async move {
        let result = run_agent_loop_continue(context, config, cancellation, emit, stream_fn).await;
        result_tx.send_replace(Some(result));
    });
    Ok(AgentEventStream {
        events: events_rx,
        result: result_rx,
    })
}
