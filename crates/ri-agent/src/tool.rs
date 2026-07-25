//! Object-safe tool contracts and closure adapters.

use std::{fmt, future::Future, sync::Arc};

use async_trait::async_trait;
use futures::{FutureExt, future::BoxFuture};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::{AgentError, ToolError};

/// How calls in one assistant tool batch are scheduled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    /// Execute and finalize one call before preparing the next call.
    Sequential,
    /// Prepare calls in source order, then execute allowed calls concurrently.
    #[default]
    Parallel,
}

/// Final or partial output produced by a tool.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    /// Text and image blocks returned to the model.
    #[serde(default)]
    pub content: Vec<ri_ai::message::InputContent>,
    /// Structured application details not sent to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Usage incurred inside the tool itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ri_ai::Usage>,
    /// Names of deferred tools introduced at this transcript point.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added_tool_names: Vec<String>,
    /// Hint to stop after this batch when every result sets the same hint.
    #[serde(default)]
    pub terminate: bool,
}

impl ToolResult {
    /// Creates a plain-text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ri_ai::message::InputContent::Text(ri_ai::TextContent::new(
                text,
            ))],
            ..Self::default()
        }
    }

    pub(crate) fn error(text: impl Into<String>) -> Self {
        Self {
            details: Some(Value::Object(serde_json::Map::new())),
            ..Self::text(text)
        }
    }
}

type UpdateEmitter = dyn Fn(ToolResult) -> BoxFuture<'static, Result<(), AgentError>> + Send + Sync;

#[derive(Debug, Default)]
struct UpdateGate {
    accepting: bool,
    pending: usize,
    error: Option<AgentError>,
}

struct ToolUpdateInner {
    gate: Mutex<UpdateGate>,
    settled: Notify,
    emit: Arc<UpdateEmitter>,
}

impl fmt::Debug for ToolUpdateInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gate = self.gate.lock();
        formatter
            .debug_struct("ToolUpdateInner")
            .field("accepting", &gate.accepting)
            .field("pending", &gate.pending)
            .finish_non_exhaustive()
    }
}

/// Execution-scoped sink for partial tool updates.
///
/// Clones may safely outlive `Tool::execute`; updates begun after execution
/// settles return `Ok(false)` and are not emitted.
#[derive(Clone, Debug)]
pub struct ToolUpdateSink {
    inner: Arc<ToolUpdateInner>,
}

impl ToolUpdateSink {
    pub(crate) fn new<F, Fut>(emit: F) -> Self
    where
        F: Fn(ToolResult) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AgentError>> + Send + 'static,
    {
        Self {
            inner: Arc::new(ToolUpdateInner {
                gate: Mutex::new(UpdateGate {
                    accepting: true,
                    ..UpdateGate::default()
                }),
                settled: Notify::new(),
                emit: Arc::new(move |result| emit(result).boxed()),
            }),
        }
    }

    /// Emits a partial result if this execution is still active.
    ///
    /// The boolean is `false` for a late update.
    ///
    /// # Errors
    ///
    /// Returns an event-sink error when an accepted update cannot be delivered.
    pub async fn send(&self, result: ToolResult) -> Result<bool, AgentError> {
        {
            let mut gate = self.inner.gate.lock();
            if !gate.accepting {
                return Ok(false);
            }
            gate.pending += 1;
        }

        let emitted = (self.inner.emit)(result).await;
        let mut gate = self.inner.gate.lock();
        gate.pending = gate.pending.saturating_sub(1);
        if let Err(error) = &emitted
            && gate.error.is_none()
        {
            gate.error = Some(error.clone());
        }
        if !gate.accepting && gate.pending == 0 {
            self.inner.settled.notify_waiters();
        }
        emitted.map(|()| true)
    }

    pub(crate) async fn close_and_wait(&self) -> Result<(), AgentError> {
        loop {
            let notified = self.inner.settled.notified();
            {
                let mut gate = self.inner.gate.lock();
                gate.accepting = false;
                if gate.pending == 0 {
                    return gate.error.clone().map_or(Ok(()), Err);
                }
            }
            notified.await;
        }
    }
}

/// Metadata and runtime services supplied to a tool invocation.
#[derive(Clone, Debug)]
pub struct ToolCallContext {
    /// Provider-assigned tool call identifier.
    pub tool_call_id: String,
    /// Cancellation token for the active agent run.
    pub cancellation: CancellationToken,
    /// Sink for partial execution updates.
    pub updates: ToolUpdateSink,
}

/// Object-safe application tool executed by the agent loop.
///
/// This JSON-value boundary is intentionally suitable for generated
/// `#[ri::tool]` implementations and adapters around built-in tools.
#[async_trait]
pub trait Tool: fmt::Debug + Send + Sync + 'static {
    /// Provider-facing name, description, and JSON Schema.
    fn definition(&self) -> &ri_ai::Tool;

    /// Human-readable label for interfaces.
    fn label(&self) -> &str;

    /// Optional per-tool scheduling override.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }

    /// Compatibility rewrite applied before schema validation.
    ///
    /// # Errors
    ///
    /// Returns a tool-defined compatibility error when arguments cannot be
    /// normalized safely.
    fn prepare_arguments(&self, arguments: Value) -> Result<Value, ToolError> {
        Ok(arguments)
    }

    /// Executes a validated invocation.
    ///
    /// # Errors
    ///
    /// Returns a tool-defined execution failure. The scheduler converts it into
    /// an error tool-result message.
    async fn execute(
        &self,
        context: ToolCallContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError>;
}

type PrepareArguments = dyn Fn(Value) -> Result<Value, ToolError> + Send + Sync;
type ToolHandler = dyn Fn(ToolCallContext, Value) -> BoxFuture<'static, Result<ToolResult, ToolError>>
    + Send
    + Sync;

/// A [`Tool`] backed by async Rust closures.
#[must_use]
pub struct FnTool {
    definition: ri_ai::Tool,
    label: String,
    execution_mode: Option<ToolExecutionMode>,
    prepare_arguments: Option<Arc<PrepareArguments>>,
    handler: Arc<ToolHandler>,
}

impl fmt::Debug for FnTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FnTool")
            .field("definition", &self.definition)
            .field("label", &self.label)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

impl FnTool {
    /// Creates a JSON-value closure tool.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(ToolCallContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolResult, ToolError>> + Send + 'static,
    {
        Self {
            definition: ri_ai::Tool::new(name, description, parameters),
            label: label.into(),
            execution_mode: None,
            prepare_arguments: None,
            handler: Arc::new(move |context, arguments| handler(context, arguments).boxed()),
        }
    }

    /// Creates a typed closure tool and derives its JSON Schema.
    ///
    /// # Errors
    ///
    /// Returns an error if the generated schema cannot be serialized.
    pub fn typed<P, F, Fut>(
        name: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        handler: F,
    ) -> Result<Self, ToolError>
    where
        P: DeserializeOwned + JsonSchema + Send + 'static,
        F: Fn(ToolCallContext, P) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolResult, ToolError>> + Send + 'static,
    {
        let schema = schemars::schema_for!(P);
        let parameters =
            serde_json::to_value(schema).map_err(|error| ToolError::Schema(error.to_string()))?;
        Ok(Self::new(
            name,
            label,
            description,
            parameters,
            move |context, arguments| {
                let parsed = serde_json::from_value::<P>(arguments)
                    .map_err(|error| ToolError::Arguments(error.to_string()));
                let future = parsed.map(|parameters| handler(context, parameters));
                async move {
                    match future {
                        Ok(future) => future.await,
                        Err(error) => Err(error),
                    }
                }
            },
        ))
    }

    /// Sets a pre-validation argument compatibility rewrite.
    pub fn with_prepare_arguments<F>(mut self, prepare: F) -> Self
    where
        F: Fn(Value) -> Result<Value, ToolError> + Send + Sync + 'static,
    {
        self.prepare_arguments = Some(Arc::new(prepare));
        self
    }

    /// Sets this tool's scheduling override.
    pub fn with_execution_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    /// Mutably accesses the provider-facing definition.
    pub fn definition_mut(&mut self) -> &mut ri_ai::Tool {
        &mut self.definition
    }
}

#[async_trait]
impl Tool for FnTool {
    fn definition(&self) -> &ri_ai::Tool {
        &self.definition
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.execution_mode
    }

    fn prepare_arguments(&self, arguments: Value) -> Result<Value, ToolError> {
        match &self.prepare_arguments {
            Some(prepare) => prepare(arguments),
            None => Ok(arguments),
        }
    }

    async fn execute(
        &self,
        context: ToolCallContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError> {
        (self.handler)(context, arguments).await
    }
}
