//! Concrete narrow adapter from the harness to `ri-agent`.

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ri_agent::{
    AfterToolCallContext, AfterToolCallResult, AgentContext, AgentError, AgentEvent,
    AgentLoopConfig, BeforeToolCallContext, BeforeToolCallResult, SharedEventSink, StreamFn, Tool,
    run_agent_loop_continue,
};
use ri_ai::{ContentBlock, Message, Model, StopReason, UserMessage};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::backend::HarnessBackend;
use crate::error::{BackendError, BackendErrorKind, Error, Result};
use crate::projection::assistant_text;
use crate::types::{SummaryRequest, SummaryResponse, TurnOutput, TurnRequest};

/// Model/auth access required by [`AgentBackend`].
#[async_trait]
pub trait ModelAccess: Send + Sync + fmt::Debug {
    /// Validate model availability and authentication.
    async fn preflight(&self, model: &Model) -> std::result::Result<(), BackendError>;

    /// Resolve a fresh request credential. Ambient providers may return `None`.
    async fn api_key(&self, model: &Model) -> std::result::Result<Option<String>, BackendError>;
}

/// Optional low-level extension/hook adapter.
#[async_trait]
pub trait AgentBackendHooks: Send + Sync + fmt::Debug {
    /// Observe an awaited low-level event.
    async fn event(
        &self,
        _event: &AgentEvent<Message>,
        _cancellation: CancellationToken,
    ) -> std::result::Result<(), AgentError> {
        Ok(())
    }

    /// Intercept validated tool arguments.
    async fn before_tool_call(
        &self,
        _context: BeforeToolCallContext<Message>,
        _cancellation: CancellationToken,
    ) -> std::result::Result<BeforeToolCallResult, AgentError> {
        Ok(BeforeToolCallResult::default())
    }

    /// Patch a finalized tool result.
    async fn after_tool_call(
        &self,
        _context: AfterToolCallContext<Message>,
        _cancellation: CancellationToken,
    ) -> std::result::Result<AfterToolCallResult, AgentError> {
        Ok(AfterToolCallResult::default())
    }

    /// Replaces a finalized message before the harness persists it.
    async fn message_end(
        &self,
        _message: &Message,
    ) -> std::result::Result<Option<Message>, AgentError> {
        Ok(None)
    }
}

/// `ri-agent`-backed implementation of the harness runtime boundary.
///
/// The low-level loop is stopped after one assistant/tool batch. This lets the
/// harness persist a save point and create a fresh immutable snapshot before the
/// next provider request without reimplementing the tool scheduler.
pub struct AgentBackend {
    stream: Arc<dyn StreamFn>,
    access: Arc<dyn ModelAccess>,
    tools: Arc<[Arc<dyn Tool>]>,
    hooks: Option<Arc<dyn AgentBackendHooks>>,
}

impl fmt::Debug for AgentBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentBackend")
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.definition().name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("has_hooks", &self.hooks.is_some())
            .finish_non_exhaustive()
    }
}

impl AgentBackend {
    /// Creates an adapter from explicit runtime dependencies.
    ///
    /// # Errors
    /// Returns an error when a tool name is blank or registered more than once.
    pub fn new(
        stream: Arc<dyn StreamFn>,
        access: Arc<dyn ModelAccess>,
        tools: Vec<Arc<dyn Tool>>,
    ) -> Result<Self> {
        let mut names = HashSet::new();
        for tool in &tools {
            let name = tool.definition().name.as_str();
            if name.trim().is_empty() {
                return Err(Error::InvalidArgument(
                    "agent tool names cannot be blank".to_owned(),
                ));
            }
            if !names.insert(name.to_owned()) {
                return Err(Error::InvalidArgument(format!(
                    "duplicate agent tool name {name:?}"
                )));
            }
        }
        Ok(Self {
            stream,
            access,
            tools: tools.into(),
            hooks: None,
        })
    }

    /// Installs the concrete extension adapter.
    #[must_use]
    pub fn with_hooks(mut self, hooks: Arc<dyn AgentBackendHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Provider-facing schemas in stable registration order.
    pub fn tool_definitions(&self) -> Vec<ri_ai::Tool> {
        self.tools
            .iter()
            .map(|tool| tool.definition().clone())
            .collect()
    }
}

#[derive(Debug, Default)]
struct TurnTracker {
    terminations: Vec<bool>,
    continue_after_tools: bool,
}

#[async_trait]
impl HarnessBackend for AgentBackend {
    async fn preflight(&self, model: &Model) -> std::result::Result<(), BackendError> {
        self.access.preflight(model).await
    }

    async fn execute_turn(
        &self,
        request: TurnRequest,
        cancellation: CancellationToken,
    ) -> std::result::Result<TurnOutput, BackendError> {
        let active: HashSet<_> = request
            .snapshot
            .active_tool_names
            .iter()
            .map(String::as_str)
            .collect();
        let tools = self
            .tools
            .iter()
            .filter(|tool| active.contains(tool.definition().name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let context = AgentContext {
            system_prompt: request.snapshot.system_prompt.to_string(),
            messages: request.snapshot.context.messages.clone(),
            tools,
        };
        let mut config = AgentLoopConfig::new((*request.snapshot.model).clone());
        config.thinking_level = request.snapshot.thinking_level;
        config.session_id = Some(request.snapshot.session_id.to_string());
        config.api_key = self.access.api_key(&request.snapshot.model).await?;
        config.stream_extensions.insert(
            "ri.request_options".to_owned(),
            serde_json::to_value(&request.snapshot.request_options)
                .map_err(|error| BackendError::new(BackendErrorKind::Fatal, error.to_string()))?,
        );
        if let Some(hooks) = self.hooks.clone() {
            let before = hooks.clone();
            config = config.with_before_tool_call(move |context, cancellation| {
                let hooks = before.clone();
                async move { hooks.before_tool_call(context, cancellation).await }
            });
            let after = hooks;
            config = config.with_after_tool_call(move |context, cancellation| {
                let hooks = after.clone();
                async move { hooks.after_tool_call(context, cancellation).await }
            });
        }
        // A single low-level provider/tool turn is the harness save-point unit.
        config = config.with_should_stop_after_turn(|_, _| async { Ok(true) });

        let tracker = Arc::new(Mutex::new(TurnTracker::default()));
        let event_tracker = tracker.clone();
        let hooks = self.hooks.clone();
        let event_cancellation = cancellation.clone();
        let event_sink: SharedEventSink<Message> = Arc::new(move |event: AgentEvent<Message>| {
            let tracker = event_tracker.clone();
            let hooks = hooks.clone();
            let cancellation = event_cancellation.clone();
            async move {
                {
                    let mut state = tracker
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    match &event {
                        AgentEvent::TurnStart => state.terminations.clear(),
                        AgentEvent::ToolExecutionEnd { result, .. } => {
                            state.terminations.push(result.terminate);
                        }
                        AgentEvent::TurnEnd {
                            message,
                            tool_results,
                        } => {
                            let has_calls = matches!(
                                message,
                                Message::Assistant(message)
                                    if message.content.iter().any(|block| {
                                        matches!(block, ContentBlock::ToolCall(_))
                                    })
                            );
                            let terminated = !state.terminations.is_empty()
                                && state.terminations.iter().all(|value| *value);
                            state.continue_after_tools =
                                has_calls && !tool_results.is_empty() && !terminated;
                        }
                        _ => {}
                    }
                }
                if let Some(hooks) = hooks {
                    hooks.event(&event, cancellation).await?;
                }
                Ok(())
            }
        });
        let mut messages = run_agent_loop_continue(
            context,
            config,
            cancellation,
            event_sink,
            self.stream.clone(),
        )
        .await
        .map_err(map_agent_error)?;
        if let Some(hooks) = &self.hooks {
            for message in &mut messages {
                if let Some(replacement) =
                    hooks.message_end(message).await.map_err(map_agent_error)?
                {
                    if std::mem::discriminant(&replacement) != std::mem::discriminant(message) {
                        return Err(BackendError::new(
                            BackendErrorKind::Fatal,
                            "message_end hooks must preserve the message role",
                        ));
                    }
                    *message = replacement;
                }
            }
        }
        let continue_after_tools = tracker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .continue_after_tools;
        Ok(TurnOutput {
            messages,
            continue_after_tools,
        })
    }

    async fn summarize(
        &self,
        request: SummaryRequest,
        cancellation: CancellationToken,
    ) -> std::result::Result<SummaryResponse, BackendError> {
        self.access.preflight(&request.model).await?;
        let api_key = self.access.api_key(&request.model).await?;
        let context = ri_ai::Context {
            system_prompt: Some(request.system_prompt),
            messages: vec![Message::User(UserMessage::new(request.prompt))],
            tools: Vec::new(),
        };
        let mut extensions = indexmap::IndexMap::new();
        extensions.insert(
            "ri.request_options".to_owned(),
            serde_json::to_value(&request.request_options)
                .map_err(|error| BackendError::new(BackendErrorKind::Fatal, error.to_string()))?,
        );
        extensions.insert("ri.max_tokens".to_owned(), json!(request.max_tokens));
        let options = ri_agent::StreamOptions {
            cancellation: cancellation.clone(),
            thinking_level: request.thinking_level,
            session_id: Some(request.request_id),
            api_key,
            extensions,
        };
        let stream = self
            .stream
            .stream((*request.model).clone(), context, options)
            .await
            .map_err(|error| map_ai_error(&error))?;
        let message = stream
            .result()
            .await
            .map_err(|error| map_ai_error(&error))?;
        match message.stop_reason {
            StopReason::Aborted => {
                return Err(BackendError::new(
                    BackendErrorKind::Aborted,
                    message
                        .error_message
                        .unwrap_or_else(|| "summarization aborted".to_owned()),
                ));
            }
            StopReason::Error => {
                let text = message
                    .error_message
                    .unwrap_or_else(|| "summarization failed".to_owned());
                return Err(classify_message_error(text));
            }
            StopReason::Stop | StopReason::Length | StopReason::ToolUse => {}
        }
        Ok(SummaryResponse {
            text: assistant_text(&message),
            usage: message.usage,
        })
    }
}

fn map_agent_error(error: AgentError) -> BackendError {
    match error {
        AgentError::Ai(error) => map_ai_error(&error),
        AgentError::Busy
        | AgentError::EmptyContinuation
        | AgentError::AssistantContinuation
        | AgentError::MissingStreamFunction
        | AgentError::Projection(_)
        | AgentError::EventStreamClosed
        | AgentError::Callback(_) => BackendError::new(BackendErrorKind::Fatal, error.to_string()),
    }
}

fn map_ai_error(error: &ri_ai::AiError) -> BackendError {
    let kind = match error {
        ri_ai::AiError::Aborted => BackendErrorKind::Aborted,
        error if error.is_retryable() => BackendErrorKind::Transient,
        _ => BackendErrorKind::Fatal,
    };
    BackendError::new(kind, error.to_string())
}

fn classify_message_error(message: String) -> BackendError {
    let lower = message.to_ascii_lowercase();
    let kind = if [
        "context window",
        "prompt is too long",
        "too many tokens",
        "token limit exceeded",
    ]
    .iter()
    .any(|pattern| lower.contains(pattern))
    {
        BackendErrorKind::ContextOverflow
    } else if [
        "429",
        "502",
        "503",
        "504",
        "529",
        "rate limit",
        "overloaded",
        "timeout",
        "temporar",
        "connection",
        "terminated",
    ]
    .iter()
    .any(|pattern| lower.contains(pattern))
    {
        BackendErrorKind::Transient
    } else {
        BackendErrorKind::Fatal
    };
    BackendError::new(kind, message)
}
