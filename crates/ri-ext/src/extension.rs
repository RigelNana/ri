//! Native extensions, reducer hooks, registries, stale contexts, and event bus.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use async_trait::async_trait;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};
use thiserror::Error;
use tokio_util::sync::CancellationToken;
use url::Url;

pub use ri_protocol_core::CompactionReason;

use crate::source::{Diagnostic, ResourceKind, SourceInfo};

fn mutex_lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ---------------------------------------------------------------------------
// Shared event bus
// ---------------------------------------------------------------------------

/// Error returned by a shared event-bus subscriber.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{message}")]
pub struct BusHandlerError {
    message: String,
}

impl BusHandlerError {
    /// Construct a subscriber error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// An extension-to-extension event. JSON is intentionally confined to this
/// open payload boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BusEvent {
    /// Channel on which the event was emitted.
    pub channel: String,
    /// Extensible channel-specific payload.
    pub payload: Value,
}

/// Failure isolated while delivering a bus event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BusDeliveryError {
    /// Channel being delivered.
    pub channel: String,
    /// Stable process-local subscriber identifier.
    pub subscriber_id: u64,
    /// Subscriber error message.
    pub message: String,
}

/// Subscriber for an [`EventBus`] channel.
#[async_trait]
pub trait BusHandler: Send + Sync {
    /// Handle one event. Delivery continues after an error.
    ///
    /// # Errors
    ///
    /// Returns [`BusHandlerError`] when this subscriber cannot process the
    /// event; other subscribers still run.
    async fn handle(&self, event: &BusEvent) -> Result<(), BusHandlerError>;
}

struct EventBusInner {
    next_id: AtomicU64,
    subscribers: RwLock<IndexMap<String, IndexMap<u64, Arc<dyn BusHandler>>>>,
}

/// Process-local, ordered, error-isolating extension event bus.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<EventBusInner>,
}

impl fmt::Debug for EventBus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventBus")
            .field("channels", &read_lock(&self.inner.subscribers).len())
            .finish()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self {
            inner: Arc::new(EventBusInner {
                next_id: AtomicU64::new(1),
                subscribers: RwLock::new(IndexMap::new()),
            }),
        }
    }
}

impl EventBus {
    /// Subscribe to a channel. Dropping the returned token unsubscribes.
    pub fn subscribe(
        &self,
        channel: impl Into<String>,
        handler: Arc<dyn BusHandler>,
    ) -> BusSubscription {
        let channel = channel.into();
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        write_lock(&self.inner.subscribers)
            .entry(channel.clone())
            .or_default()
            .insert(id, handler);
        BusSubscription {
            inner: Arc::downgrade(&self.inner),
            channel,
            id,
        }
    }

    /// Emit to subscribers in registration order. Every subscriber is awaited;
    /// failures are returned after delivery rather than aborting the sequence.
    pub async fn emit(&self, channel: impl Into<String>, payload: Value) -> Vec<BusDeliveryError> {
        let channel = channel.into();
        let handlers = read_lock(&self.inner.subscribers)
            .get(&channel)
            .map(|handlers| {
                handlers
                    .iter()
                    .map(|(id, handler)| (*id, Arc::clone(handler)))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let event = BusEvent {
            channel: channel.clone(),
            payload,
        };
        let mut errors = Vec::new();
        for (subscriber_id, handler) in handlers {
            if let Err(error) = handler.handle(&event).await {
                errors.push(BusDeliveryError {
                    channel: channel.clone(),
                    subscriber_id,
                    message: error.to_string(),
                });
            }
        }
        errors
    }

    /// Remove every subscription.
    pub fn clear(&self) {
        write_lock(&self.inner.subscribers).clear();
    }
}

/// RAII event-bus subscription.
#[derive(Debug)]
pub struct BusSubscription {
    inner: Weak<EventBusInner>,
    channel: String,
    id: u64,
}

impl Drop for BusSubscription {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut subscribers = write_lock(&inner.subscribers);
        if let Some(channel) = subscribers.get_mut(&self.channel) {
            channel.shift_remove(&self.id);
            if channel.is_empty() {
                subscribers.shift_remove(&self.channel);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generation-checked context
// ---------------------------------------------------------------------------

/// Monotonic runtime generation shared by extension, resource, and package
/// reload paths.
#[derive(Clone, Debug)]
pub struct GenerationClock {
    value: Arc<AtomicU64>,
}

impl Default for GenerationClock {
    fn default() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(1)),
        }
    }
}

impl GenerationClock {
    /// Current generation.
    pub fn current(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Invalidate all previously captured contexts and return the new value.
    pub fn advance(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel) + 1
    }
}

/// User-facing mode in which an extension executes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionMode {
    /// Interactive terminal user interface.
    Tui,
    /// Structured RPC host.
    Rpc,
    /// JSON-stream host.
    Json,
    /// Non-interactive print mode.
    #[default]
    Print,
}

/// Mutable runtime facts resolved whenever a new context is created.
#[derive(Clone, Debug)]
pub struct ContextBinding {
    /// Host execution mode.
    pub mode: ExtensionMode,
    /// Whether interactive UI capabilities are available.
    pub has_ui: bool,
    /// Current project working directory.
    pub cwd: PathBuf,
    /// Whether project-local capabilities are trusted.
    pub project_trusted: bool,
    /// Cancellation token for the active operation, when any.
    pub cancellation: Option<CancellationToken>,
    /// Current base system prompt.
    pub system_prompt: String,
}

impl Default for ContextBinding {
    fn default() -> Self {
        Self {
            mode: ExtensionMode::Print,
            has_ui: false,
            cwd: PathBuf::from("."),
            project_trusted: false,
            cancellation: None,
            system_prompt: String::new(),
        }
    }
}

/// Custom message accepted at the native extension boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomMessage {
    /// Extension-defined message type.
    pub custom_type: String,
    /// Human- or model-readable message text.
    pub content: String,
    /// Whether the host should display the message.
    #[serde(default)]
    pub display: bool,
    /// Optional extension-defined structured details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// Error produced by a host action.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{message}")]
pub struct ActionError {
    message: String,
}

impl ActionError {
    /// Construct an action error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Host capabilities available through an [`ExtensionContext`].
///
/// The trait intentionally avoids depending on the still-evolving harness,
/// agent, and session crates. The harness can implement this narrow facade.
#[async_trait]
pub trait ContextActions: Send + Sync {
    /// Add a custom extension message to the active conversation.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if the host cannot accept the message.
    async fn send_message(&self, message: CustomMessage) -> Result<(), ActionError>;
    /// Submit user text to the active conversation.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if the host cannot accept the input.
    async fn send_user_message(&self, text: String) -> Result<(), ActionError>;
    /// Append an extension-defined session entry.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if the host cannot persist the entry.
    async fn append_entry(&self, custom_type: String, data: Value) -> Result<(), ActionError>;
    /// Return names of currently active tools.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if active tool state is unavailable.
    async fn active_tools(&self) -> Result<Vec<String>, ActionError>;
    /// Replace the active tool set.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if the host rejects the tool set.
    async fn set_active_tools(&self, names: Vec<String>) -> Result<(), ActionError>;
    /// Request orderly host shutdown.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if shutdown cannot be requested.
    async fn shutdown(&self) -> Result<(), ActionError>;
}

/// Safe default facade for hosts that only need hooks and registries.
#[derive(Debug, Default)]
pub struct NoopContextActions;

#[async_trait]
impl ContextActions for NoopContextActions {
    async fn send_message(&self, _message: CustomMessage) -> Result<(), ActionError> {
        Err(ActionError::new("send_message is not bound"))
    }

    async fn send_user_message(&self, _text: String) -> Result<(), ActionError> {
        Err(ActionError::new("send_user_message is not bound"))
    }

    async fn append_entry(&self, _custom_type: String, _data: Value) -> Result<(), ActionError> {
        Err(ActionError::new("append_entry is not bound"))
    }

    async fn active_tools(&self) -> Result<Vec<String>, ActionError> {
        Ok(Vec::new())
    }

    async fn set_active_tools(&self, _names: Vec<String>) -> Result<(), ActionError> {
        Err(ActionError::new("set_active_tools is not bound"))
    }

    async fn shutdown(&self) -> Result<(), ActionError> {
        Err(ActionError::new("shutdown is not bound"))
    }
}

/// Captured context was used after session replacement or reload.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("stale extension context: captured generation {captured}, current generation {current}")]
pub struct StaleContextError {
    /// Generation captured when the context was created.
    pub captured: u64,
    /// Current runtime generation.
    pub current: u64,
}

/// Error from a generation-checked context action.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ContextError {
    /// Context belongs to an invalidated generation.
    #[error(transparent)]
    Stale(#[from] StaleContextError),
    /// Host capability failed.
    #[error(transparent)]
    Action(#[from] ActionError),
}

/// Native extension context. Every observation and action checks generation,
/// so a captured context cannot mutate a replacement session.
#[derive(Clone)]
pub struct ExtensionContext {
    clock: GenerationClock,
    captured_generation: u64,
    binding: Arc<RwLock<ContextBinding>>,
    actions: Arc<dyn ContextActions>,
    event_bus: EventBus,
    system_prompt_override: Option<Arc<Mutex<String>>>,
}

impl fmt::Debug for ExtensionContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionContext")
            .field("captured_generation", &self.captured_generation)
            .finish_non_exhaustive()
    }
}

impl ExtensionContext {
    /// Assert that this context belongs to the active runtime generation.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] after any shared generation advance.
    pub fn ensure_active(&self) -> Result<(), StaleContextError> {
        let current = self.clock.current();
        if current == self.captured_generation {
            Ok(())
        } else {
            Err(StaleContextError {
                captured: self.captured_generation,
                current,
            })
        }
    }

    /// Generation captured by this context.
    pub fn generation(&self) -> u64 {
        self.captured_generation
    }

    /// Return the current host execution mode.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn mode(&self) -> Result<ExtensionMode, StaleContextError> {
        self.ensure_active()?;
        Ok(read_lock(&self.binding).mode)
    }

    /// Return whether interactive UI capabilities are available.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn has_ui(&self) -> Result<bool, StaleContextError> {
        self.ensure_active()?;
        Ok(read_lock(&self.binding).has_ui)
    }

    /// Return the current project working directory.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn cwd(&self) -> Result<PathBuf, StaleContextError> {
        self.ensure_active()?;
        Ok(read_lock(&self.binding).cwd.clone())
    }

    /// Return whether project-local capabilities are trusted.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn is_project_trusted(&self) -> Result<bool, StaleContextError> {
        self.ensure_active()?;
        Ok(read_lock(&self.binding).project_trusted)
    }

    /// Return the current operation's cancellation token, when present.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn cancellation(&self) -> Result<Option<CancellationToken>, StaleContextError> {
        self.ensure_active()?;
        Ok(read_lock(&self.binding).cancellation.clone())
    }

    /// Current system prompt. During `before_agent_start`, this follows the
    /// reducer's chained prompt rather than the base binding.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn system_prompt(&self) -> Result<String, StaleContextError> {
        self.ensure_active()?;
        if let Some(prompt) = &self.system_prompt_override {
            return Ok(mutex_lock(prompt).clone());
        }
        Ok(read_lock(&self.binding).system_prompt.clone())
    }

    /// Return the shared extension event bus.
    ///
    /// # Errors
    ///
    /// Returns [`StaleContextError`] if this context was invalidated.
    pub fn event_bus(&self) -> Result<EventBus, StaleContextError> {
        self.ensure_active()?;
        Ok(self.event_bus.clone())
    }

    /// Add a custom extension message to the conversation.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if the host rejects the message.
    pub async fn send_message(&self, message: CustomMessage) -> Result<(), ContextError> {
        self.ensure_active()?;
        self.actions.send_message(message).await?;
        Ok(())
    }

    /// Submit user text through the host.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if the host rejects the input.
    pub async fn send_user_message(&self, text: impl Into<String>) -> Result<(), ContextError> {
        self.ensure_active()?;
        self.actions.send_user_message(text.into()).await?;
        Ok(())
    }

    /// Append an extension-defined entry to the active session.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if the host cannot append the entry.
    pub async fn append_entry(
        &self,
        custom_type: impl Into<String>,
        data: Value,
    ) -> Result<(), ContextError> {
        self.ensure_active()?;
        self.actions.append_entry(custom_type.into(), data).await?;
        Ok(())
    }

    /// Return currently active tool names.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if tool state is unavailable.
    pub async fn active_tools(&self) -> Result<Vec<String>, ContextError> {
        self.ensure_active()?;
        Ok(self.actions.active_tools().await?)
    }

    /// Replace the active tool set.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if the host rejects the tool set.
    pub async fn set_active_tools(&self, names: Vec<String>) -> Result<(), ContextError> {
        self.ensure_active()?;
        self.actions.set_active_tools(names).await?;
        Ok(())
    }

    /// Request orderly host shutdown.
    ///
    /// # Errors
    ///
    /// Returns [`ContextError::Stale`] if invalidated or
    /// [`ContextError::Action`] if shutdown cannot be requested.
    pub async fn shutdown(&self) -> Result<(), ContextError> {
        self.ensure_active()?;
        self.actions.shutdown().await?;
        Ok(())
    }
}

/// Factory used by the runner to create fresh generation-bound contexts.
#[derive(Clone)]
pub struct ContextFactory {
    clock: GenerationClock,
    binding: Arc<RwLock<ContextBinding>>,
    actions: Arc<dyn ContextActions>,
    event_bus: EventBus,
}

impl fmt::Debug for ContextFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ContextFactory")
            .field("generation", &self.clock.current())
            .field("binding", &read_lock(&self.binding))
            .finish_non_exhaustive()
    }
}

impl ContextFactory {
    /// Create a factory around host bindings.
    pub fn new(
        clock: GenerationClock,
        binding: ContextBinding,
        actions: Arc<dyn ContextActions>,
        event_bus: EventBus,
    ) -> Self {
        Self {
            clock,
            binding: Arc::new(RwLock::new(binding)),
            actions,
            event_bus,
        }
    }

    /// Replace dynamic binding facts without invalidating the generation.
    pub fn set_binding(&self, binding: ContextBinding) {
        *write_lock(&self.binding) = binding;
    }

    /// Capture a normal event context.
    pub fn create(&self) -> ExtensionContext {
        self.create_inner(None)
    }

    fn create_with_system_prompt(&self, prompt: Arc<Mutex<String>>) -> ExtensionContext {
        self.create_inner(Some(prompt))
    }

    fn create_inner(&self, system_prompt_override: Option<Arc<Mutex<String>>>) -> ExtensionContext {
        ExtensionContext {
            clock: self.clock.clone(),
            captured_generation: self.clock.current(),
            binding: Arc::clone(&self.binding),
            actions: Arc::clone(&self.actions),
            event_bus: self.event_bus.clone(),
            system_prompt_override,
        }
    }
}

// ---------------------------------------------------------------------------
// Event values and reducer hook trait
// ---------------------------------------------------------------------------

/// Role of a context message.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// User-authored input.
    User,
    /// Assistant-authored output.
    Assistant,
    /// Result associated with a tool call.
    ToolResult,
    /// Extension-defined message.
    Custom,
}

/// Typed text/image content used by hook reducers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// UTF-8 text content.
    Text {
        /// Text value.
        text: String,
    },
    /// Image content represented at the provider boundary.
    Image {
        /// MIME media type.
        media_type: String,
        /// Provider- or host-specific image reference.
        source: Value,
    },
}

/// Message representation local to the extension boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextMessage {
    /// Stable message role.
    pub role: MessageRole,
    /// Ordered text and image content.
    pub content: Vec<ContentPart>,
    /// Optional extension- or host-defined metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Provider headers. `None` means delete the named header.
pub type ProviderHeaders = IndexMap<String, Option<String>>;

/// Usage optionally reported by a tool.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolUsage {
    /// Input tokens attributed to the tool operation.
    pub input_tokens: u64,
    /// Output tokens attributed to the tool operation.
    pub output_tokens: u64,
}

/// Event sent through the context reducer.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextEvent {
    /// Current message list after preceding reducer handlers.
    pub messages: Vec<ContextMessage>,
}

/// Provider request event. JSON is the provider payload boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderRequestEvent {
    /// Current provider-specific request payload.
    pub payload: Value,
}

/// Header event passed mutably to each handler.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderHeadersEvent {
    /// Mutable header map; `None` removes a header.
    pub headers: ProviderHeaders,
}

/// Pre-agent event. Later handlers observe the preceding prompt override.
#[derive(Clone, Debug, PartialEq)]
pub struct BeforeAgentStartEvent {
    /// User prompt starting the run.
    pub prompt: String,
    /// Images attached to the prompt.
    pub images: Vec<ContentPart>,
    /// Current system prompt after preceding handlers.
    pub system_prompt: String,
}

/// Optional result from one pre-agent handler.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeforeAgentStartResult {
    /// Optional custom message injected before the run.
    pub message: Option<CustomMessage>,
    /// Optional replacement system prompt.
    pub system_prompt: Option<String>,
}

/// Combined pre-agent result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeforeAgentStartReduction {
    /// Messages accumulated from every handler.
    pub messages: Vec<CustomMessage>,
    /// Final replacement system prompt, if any handler changed it.
    pub system_prompt: Option<String>,
}

/// Tool call event. Handlers may mutate `input`; no post-hook validation is
/// performed, matching the Pi extension contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallEvent {
    /// Provider-assigned tool-call identifier.
    pub tool_call_id: String,
    /// Registered tool name.
    pub tool_name: String,
    /// Mutable tool argument object.
    pub input: JsonMap<String, Value>,
}

/// Tool-call control result.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ToolCallResult {
    /// Whether execution must be blocked.
    pub block: bool,
    /// Optional human-readable block reason.
    pub reason: Option<String>,
}

/// Tool-call reducer output, including in-place argument mutations.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallReduction {
    /// Tool call after all argument mutations.
    pub event: ToolCallEvent,
    /// Last control result, or the first blocking result.
    pub result: Option<ToolCallResult>,
}

/// Tool result event.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResultEvent {
    /// Provider-assigned tool-call identifier.
    pub tool_call_id: String,
    /// Registered tool name.
    pub tool_name: String,
    /// Arguments used for the call.
    pub input: JsonMap<String, Value>,
    /// Current result content.
    pub content: Vec<ContentPart>,
    /// Optional tool-specific details.
    pub details: Option<Value>,
    /// Whether the tool execution failed.
    pub is_error: bool,
    /// Optional usage attributed to the tool.
    pub usage: Option<ToolUsage>,
}

/// Partial patch returned by a tool-result handler.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolResultPatch {
    /// Replacement content, or `None` to preserve current content.
    pub content: Option<Vec<ContentPart>>,
    /// Replacement details, or `None` to preserve current details.
    pub details: Option<Value>,
    /// Replacement error state, or `None` to preserve it.
    pub is_error: Option<bool>,
    /// Replacement usage, or `None` to preserve it.
    pub usage: Option<ToolUsage>,
}

/// Source of submitted input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSource {
    /// Interactive user input.
    Interactive,
    /// Input submitted through RPC.
    Rpc,
    /// Input submitted by an extension.
    Extension,
}

/// Streaming delivery behavior attached to input.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingBehavior {
    /// Redirect the currently streaming run.
    Steer,
    /// Queue input for a subsequent turn.
    FollowUp,
}

/// Input reducer event.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEvent {
    /// Current input text after preceding transforms.
    pub text: String,
    /// Current attached images after preceding transforms.
    pub images: Vec<ContentPart>,
    /// Origin of the submitted input.
    pub source: InputSource,
    /// Requested streaming delivery behavior.
    pub streaming_behavior: Option<StreamingBehavior>,
}

/// Input reducer action.
#[derive(Clone, Debug, PartialEq)]
pub enum InputResult {
    /// Leave input unchanged and continue reducer execution.
    Continue,
    /// Replace input fields and continue reducer execution.
    Transform {
        /// Replacement text.
        text: String,
        /// `None` preserves images from the previous reducer step.
        images: Option<Vec<ContentPart>>,
    },
    /// Mark input handled and stop reducer execution.
    Handled,
}

/// Session operation that can be cancelled or overridden.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionBeforeEvent {
    /// Session switch requested.
    Switch {
        /// Host-provided switch reason.
        reason: String,
        /// Optional target session path.
        target_session: Option<PathBuf>,
    },
    /// Session fork requested.
    Fork {
        /// Entry around which to fork.
        entry_id: String,
        /// Fork position relative to the entry.
        position: ForkPosition,
    },
    /// Session compaction requested.
    Compact {
        /// Why compaction is running.
        reason: CompactionReason,
        /// Host-specific prepared compaction data.
        preparation: Value,
    },
    /// Session tree navigation requested.
    Tree {
        /// Target tree entry.
        target_id: String,
        /// Host-specific prepared tree data.
        preparation: Value,
    },
}

/// Position used when forking a session.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForkPosition {
    /// Fork before the selected entry.
    Before,
    /// Fork at the selected entry.
    At,
}

/// Type-specific session override.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionOverride {
    /// Override whether existing conversation messages are restored.
    SkipConversationRestore(bool),
    /// Replace prepared compaction data.
    Compaction(Value),
    /// Replace a generated tree summary.
    TreeSummary {
        /// Replacement summary text.
        summary: String,
        /// Optional structured summary details.
        details: Option<Value>,
        /// Optional instructions used to produce the summary.
        custom_instructions: Option<String>,
        /// Whether custom instructions replace rather than augment defaults.
        replace_instructions: Option<bool>,
        /// Optional display label.
        label: Option<String>,
    },
}

/// Result from a session-before handler. The first cancellation wins;
/// otherwise the last returned value wins as a whole.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionBeforeResult {
    /// Whether the session operation is cancelled.
    pub cancel: bool,
    /// Optional operation-specific override.
    pub override_value: Option<SessionOverride>,
}

/// Project trust decision contributed by an extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectTrustDecision {
    /// Trust the project.
    Yes,
    /// Deny project trust.
    No,
    /// Defer to subsequent trust sources.
    Undecided,
}

/// Project trust hook result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectTrustResult {
    /// Extension-contributed decision.
    pub decision: ProjectTrustDecision,
    /// Whether the resolver should persist the decision.
    pub remember: bool,
}

/// Finalized-message hook event.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageEndEvent {
    /// Current finalized message after preceding replacements.
    pub message: ContextMessage,
}

/// Broadcast lifecycle event.
#[derive(Clone, Debug, PartialEq)]
pub enum NotificationEvent {
    /// A session became active.
    SessionStart {
        /// Host-provided lifecycle reason.
        reason: String,
    },
    /// The active session is shutting down.
    SessionShutdown {
        /// Host-provided lifecycle reason.
        reason: String,
    },
    /// Agent execution started.
    AgentStart,
    /// Agent execution ended.
    AgentEnd {
        /// Final conversation messages.
        messages: Vec<ContextMessage>,
    },
    /// Agent execution reached an idle state.
    AgentSettled,
    /// A model turn started.
    TurnStart {
        /// Zero-based turn index.
        index: usize,
        /// Start timestamp in Unix milliseconds.
        timestamp_ms: u64,
    },
    /// A model turn ended.
    TurnEnd {
        /// Zero-based turn index.
        index: usize,
    },
    /// Streaming of a message started.
    MessageStart {
        /// Initial message state.
        message: ContextMessage,
    },
    /// A streaming message changed.
    MessageUpdate {
        /// Current complete message state.
        message: ContextMessage,
        /// Provider- or host-specific incremental delta.
        delta: Value,
    },
    /// Tool execution started.
    ToolExecutionStart {
        /// Provider-assigned tool-call identifier.
        tool_call_id: String,
        /// Registered tool name.
        tool_name: String,
    },
    /// Tool execution produced a partial result.
    ToolExecutionUpdate {
        /// Provider-assigned tool-call identifier.
        tool_call_id: String,
        /// Registered tool name.
        tool_name: String,
        /// Tool-specific partial result payload.
        partial_result: Value,
    },
    /// Tool execution ended.
    ToolExecutionEnd {
        /// Provider-assigned tool-call identifier.
        tool_call_id: String,
        /// Registered tool name.
        tool_name: String,
        /// Whether execution failed.
        is_error: bool,
    },
    /// Active model selection changed.
    ModelSelect {
        /// Selected provider identifier.
        provider: String,
        /// Selected model identifier.
        model: String,
    },
    /// Active reasoning level changed.
    ThinkingLevelSelect {
        /// Selected level.
        level: String,
    },
    /// Host-defined lifecycle notification.
    Custom {
        /// Notification name.
        name: String,
        /// Extensible notification payload.
        payload: Value,
    },
}

/// Normalized hook failure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{message}")]
pub struct HookError {
    message: String,
}

impl HookError {
    /// Construct a hook failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Event hook implemented by a native extension. Default methods opt out.
#[async_trait]
pub trait EventHook: Send + Sync {
    /// Observe a broadcast lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] when this handler fails. The runner records the
    /// failure and continues with subsequent handlers.
    async fn on_notification(
        &self,
        _event: &NotificationEvent,
        _context: &ExtensionContext,
    ) -> Result<(), HookError> {
        Ok(())
    }

    /// Optionally replace the message list for the next context handler.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to leave the current message list unchanged and
    /// continue reducer execution.
    async fn on_context(
        &self,
        _event: &ContextEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<Vec<ContextMessage>>, HookError> {
        Ok(None)
    }

    /// Optionally replace the provider request for the next handler.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to leave the current payload unchanged and
    /// continue reducer execution.
    async fn on_provider_request(
        &self,
        _event: &ProviderRequestEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<Value>, HookError> {
        Ok(None)
    }

    /// Mutate provider headers in place.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to record failure and continue with the same
    /// header map.
    async fn on_provider_headers(
        &self,
        _event: &mut ProviderHeadersEvent,
        _context: &ExtensionContext,
    ) -> Result<(), HookError> {
        Ok(())
    }

    /// Inject a message or replace the system prompt before agent execution.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to contribute no result and continue.
    async fn on_before_agent_start(
        &self,
        _event: &BeforeAgentStartEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<BeforeAgentStartResult>, HookError> {
        Ok(None)
    }

    /// Mutate tool arguments or block execution.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to preserve mutations already made by this
    /// handler and continue reducer execution.
    async fn on_tool_call(
        &self,
        _event: &mut ToolCallEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<ToolCallResult>, HookError> {
        Ok(None)
    }

    /// Return a partial patch for a completed tool result.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to leave the current result unchanged and
    /// continue reducer execution.
    async fn on_tool_result(
        &self,
        _event: &ToolResultEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<ToolResultPatch>, HookError> {
        Ok(None)
    }

    /// Transform or fully handle submitted input.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to leave current input unchanged and continue.
    async fn on_input(
        &self,
        _event: &InputEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<InputResult>, HookError> {
        Ok(None)
    }

    /// Cancel or override a pending session operation.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to contribute no result and continue.
    async fn on_session_before(
        &self,
        _event: &SessionBeforeEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<SessionBeforeResult>, HookError> {
        Ok(None)
    }

    /// Contribute a project trust decision.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to defer to subsequent trust handlers.
    async fn on_project_trust(
        &self,
        _cwd: &std::path::Path,
        _context: &ExtensionContext,
    ) -> Result<Option<ProjectTrustResult>, HookError> {
        Ok(None)
    }

    /// Optionally replace a finalized message while preserving its role.
    ///
    /// # Errors
    ///
    /// Returns [`HookError`] to leave the current message unchanged and
    /// continue reducer execution.
    async fn on_message_end(
        &self,
        _event: &MessageEndEvent,
        _context: &ExtensionContext,
    ) -> Result<Option<ContextMessage>, HookError> {
        Ok(None)
    }
}

/// One hook plus its provenance.
#[derive(Clone)]
pub struct HookRegistration {
    /// Identifier of the extension owning the hook.
    pub extension_id: String,
    /// Extension provenance.
    pub source: SourceInfo,
    /// Hook implementation.
    pub hook: Arc<dyn EventHook>,
}

impl fmt::Debug for HookRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HookRegistration")
            .field("extension_id", &self.extension_id)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Error emitted for one handler while the reducer continues.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandlerFailure {
    /// Identifier of the extension whose handler failed.
    pub extension_id: String,
    /// Extension provenance.
    pub source: SourceInfo,
    /// Normalized event name.
    pub event: String,
    /// Hook or contract-validation failure.
    pub message: String,
}

/// Ordered reducer runner. Every event method isolates handler errors.
pub struct ExtensionRunner {
    hooks: Vec<HookRegistration>,
    contexts: ContextFactory,
    errors: Mutex<Vec<HandlerFailure>>,
}

impl fmt::Debug for ExtensionRunner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionRunner")
            .field("hook_count", &self.hooks.len())
            .field("contexts", &self.contexts)
            .finish_non_exhaustive()
    }
}

impl ExtensionRunner {
    /// Construct a runner in load order.
    pub fn new(hooks: Vec<HookRegistration>, contexts: ContextFactory) -> Self {
        Self {
            hooks,
            contexts,
            errors: Mutex::new(Vec::new()),
        }
    }

    /// Whether any extension registered a hook.
    pub fn has_hooks(&self) -> bool {
        !self.hooks.is_empty()
    }

    /// Create a fresh context for tool execution or a command.
    pub fn context(&self) -> ExtensionContext {
        self.contexts.create()
    }

    /// Drain normalized handler failures.
    pub fn drain_errors(&self) -> Vec<HandlerFailure> {
        std::mem::take(&mut *mutex_lock(&self.errors))
    }

    fn record_error(&self, registration: &HookRegistration, event: &str, error: &HookError) {
        mutex_lock(&self.errors).push(HandlerFailure {
            extension_id: registration.extension_id.clone(),
            source: registration.source.clone(),
            event: event.to_owned(),
            message: error.to_string(),
        });
    }

    fn record_message_error(
        &self,
        registration: &HookRegistration,
        event: &str,
        message: impl Into<String>,
    ) {
        mutex_lock(&self.errors).push(HandlerFailure {
            extension_id: registration.extension_id.clone(),
            source: registration.source.clone(),
            event: event.to_owned(),
            message: message.into(),
        });
    }

    /// Broadcast a notification in load order.
    pub async fn emit_notification(&self, event: &NotificationEvent) {
        let context = self.contexts.create();
        for registration in &self.hooks {
            if let Err(error) = registration.hook.on_notification(event, &context).await {
                self.record_error(registration, notification_name(event), &error);
            }
        }
    }

    /// Clone the initial context and chain message replacements.
    pub async fn emit_context(&self, messages: &[ContextMessage]) -> Vec<ContextMessage> {
        let context = self.contexts.create();
        let mut current = messages.to_vec();
        for registration in &self.hooks {
            let event = ContextEvent {
                messages: current.clone(),
            };
            match registration.hook.on_context(&event, &context).await {
                Ok(Some(messages)) => current = messages,
                Ok(None) => {}
                Err(error) => self.record_error(registration, "context", &error),
            }
        }
        current
    }

    /// Chain provider payload replacements; `None` means no replacement.
    pub async fn emit_provider_request(&self, payload: Value) -> Value {
        let context = self.contexts.create();
        let mut current = payload;
        for registration in &self.hooks {
            let event = ProviderRequestEvent {
                payload: current.clone(),
            };
            match registration
                .hook
                .on_provider_request(&event, &context)
                .await
            {
                Ok(Some(payload)) => current = payload,
                Ok(None) => {}
                Err(error) => self.record_error(registration, "before_provider_request", &error),
            }
        }
        current
    }

    /// Pass one mutable header map through all handlers.
    pub async fn emit_provider_headers(&self, headers: ProviderHeaders) -> ProviderHeaders {
        let context = self.contexts.create();
        let mut event = ProviderHeadersEvent { headers };
        for registration in &self.hooks {
            if let Err(error) = registration
                .hook
                .on_provider_headers(&mut event, &context)
                .await
            {
                self.record_error(registration, "before_provider_headers", &error);
            }
        }
        event.headers
    }

    /// Accumulate injected messages and chain system-prompt replacements.
    pub async fn emit_before_agent_start(
        &self,
        prompt: impl Into<String>,
        images: Vec<ContentPart>,
        system_prompt: impl Into<String>,
    ) -> BeforeAgentStartReduction {
        let prompt = prompt.into();
        let original_system_prompt = system_prompt.into();
        let shared_prompt = Arc::new(Mutex::new(original_system_prompt.clone()));
        let context = self
            .contexts
            .create_with_system_prompt(Arc::clone(&shared_prompt));
        let mut current_system_prompt = original_system_prompt;
        let mut reduction = BeforeAgentStartReduction::default();

        for registration in &self.hooks {
            let event = BeforeAgentStartEvent {
                prompt: prompt.clone(),
                images: images.clone(),
                system_prompt: current_system_prompt.clone(),
            };
            match registration
                .hook
                .on_before_agent_start(&event, &context)
                .await
            {
                Ok(Some(result)) => {
                    if let Some(message) = result.message {
                        reduction.messages.push(message);
                    }
                    if let Some(system_prompt) = result.system_prompt {
                        current_system_prompt = system_prompt;
                        mutex_lock(&shared_prompt).clone_from(&current_system_prompt);
                        reduction.system_prompt = Some(current_system_prompt.clone());
                    }
                }
                Ok(None) => {}
                Err(error) => self.record_error(registration, "before_agent_start", &error),
            }
        }
        reduction
    }

    /// Chain in-place tool argument mutations. The first block short-circuits.
    pub async fn emit_tool_call(&self, mut event: ToolCallEvent) -> ToolCallReduction {
        let context = self.contexts.create();
        let mut latest = None;
        for registration in &self.hooks {
            match registration.hook.on_tool_call(&mut event, &context).await {
                Ok(Some(result)) => {
                    let blocked = result.block;
                    latest = Some(result);
                    if blocked {
                        break;
                    }
                }
                Ok(None) => {}
                Err(error) => self.record_error(registration, "tool_call", &error),
            }
        }
        ToolCallReduction {
            event,
            result: latest,
        }
    }

    /// Chain partial tool-result patches. Later handlers observe all previous
    /// patches and omitted fields preserve earlier changes.
    pub async fn emit_tool_result(&self, event: ToolResultEvent) -> Option<ToolResultEvent> {
        let context = self.contexts.create();
        let mut current = event;
        let mut modified = false;
        for registration in &self.hooks {
            match registration.hook.on_tool_result(&current, &context).await {
                Ok(Some(patch)) => {
                    if let Some(content) = patch.content {
                        current.content = content;
                        modified = true;
                    }
                    if let Some(details) = patch.details {
                        current.details = Some(details);
                        modified = true;
                    }
                    if let Some(is_error) = patch.is_error {
                        current.is_error = is_error;
                        modified = true;
                    }
                    if let Some(usage) = patch.usage {
                        current.usage = Some(usage);
                        modified = true;
                    }
                }
                Ok(None) => {}
                Err(error) => self.record_error(registration, "tool_result", &error),
            }
        }
        modified.then_some(current)
    }

    /// Chain input transforms. `Handled` is terminal.
    pub async fn emit_input(&self, mut event: InputEvent) -> InputResult {
        let context = self.contexts.create();
        let original = event.clone();
        for registration in &self.hooks {
            match registration.hook.on_input(&event, &context).await {
                Ok(Some(InputResult::Handled)) => return InputResult::Handled,
                Ok(Some(InputResult::Transform { text, images })) => {
                    event.text = text;
                    if let Some(images) = images {
                        event.images = images;
                    }
                }
                Ok(Some(InputResult::Continue) | None) => {}
                Err(error) => self.record_error(registration, "input", &error),
            }
        }
        if event.text != original.text || event.images != original.images {
            InputResult::Transform {
                text: event.text,
                images: Some(event.images),
            }
        } else {
            InputResult::Continue
        }
    }

    /// Return the first cancellation, otherwise the last returned result.
    pub async fn emit_session_before(
        &self,
        event: &SessionBeforeEvent,
    ) -> Option<SessionBeforeResult> {
        let context = self.contexts.create();
        let mut latest = None;
        for registration in &self.hooks {
            match registration.hook.on_session_before(event, &context).await {
                Ok(Some(result)) => {
                    let cancelled = result.cancel;
                    latest = Some(result);
                    if cancelled {
                        return latest;
                    }
                }
                Ok(None) => {}
                Err(error) => self.record_error(registration, "session_before", &error),
            }
        }
        latest
    }

    /// Continue past undecided handlers and failures; first yes/no wins.
    pub async fn emit_project_trust(&self, cwd: &std::path::Path) -> Option<ProjectTrustResult> {
        let context = self.contexts.create();
        for registration in &self.hooks {
            match registration.hook.on_project_trust(cwd, &context).await {
                Ok(Some(result)) if result.decision != ProjectTrustDecision::Undecided => {
                    return Some(result);
                }
                Ok(_) => {}
                Err(error) => self.record_error(registration, "project_trust", &error),
            }
        }
        None
    }

    /// Chain finalized-message replacements while enforcing role stability.
    pub async fn emit_message_end(&self, message: ContextMessage) -> Option<ContextMessage> {
        let context = self.contexts.create();
        let mut current = message;
        let mut modified = false;
        for registration in &self.hooks {
            let event = MessageEndEvent {
                message: current.clone(),
            };
            match registration.hook.on_message_end(&event, &context).await {
                Ok(Some(replacement)) if replacement.role == current.role => {
                    current = replacement;
                    modified = true;
                }
                Ok(Some(_)) => self.record_message_error(
                    registration,
                    "message_end",
                    "message_end handlers must preserve the message role",
                ),
                Ok(None) => {}
                Err(error) => self.record_error(registration, "message_end", &error),
            }
        }
        modified.then_some(current)
    }
}

fn notification_name(event: &NotificationEvent) -> &'static str {
    match event {
        NotificationEvent::SessionStart { .. } => "session_start",
        NotificationEvent::SessionShutdown { .. } => "session_shutdown",
        NotificationEvent::AgentStart => "agent_start",
        NotificationEvent::AgentEnd { .. } => "agent_end",
        NotificationEvent::AgentSettled => "agent_settled",
        NotificationEvent::TurnStart { .. } => "turn_start",
        NotificationEvent::TurnEnd { .. } => "turn_end",
        NotificationEvent::MessageStart { .. } => "message_start",
        NotificationEvent::MessageUpdate { .. } => "message_update",
        NotificationEvent::ToolExecutionStart { .. } => "tool_execution_start",
        NotificationEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        NotificationEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        NotificationEvent::ModelSelect { .. } => "model_select",
        NotificationEvent::ThinkingLevelSelect { .. } => "thinking_level_select",
        NotificationEvent::Custom { .. } => "custom",
    }
}

// ---------------------------------------------------------------------------
// Registries
// ---------------------------------------------------------------------------

/// Native tool metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Stable tool invocation name.
    pub name: String,
    /// Human-readable tool label.
    pub label: String,
    /// Model-visible tool description.
    pub description: String,
    /// JSON Schema is an extension/provider boundary.
    pub parameter_schema: Value,
    /// Optional short system-prompt listing entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_snippet: Option<String>,
    /// Optional model usage guidance.
    #[serde(default)]
    pub prompt_guidelines: Vec<String>,
}

/// Native tool result.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutput {
    /// Text or image result content.
    pub content: Vec<ContentPart>,
    /// Optional tool-defined structured details.
    pub details: Option<Value>,
    /// Whether execution failed.
    pub is_error: bool,
    /// Optional token usage attributed to the operation.
    pub usage: Option<ToolUsage>,
}

/// LLM-callable native tool.
#[async_trait]
pub trait NativeTool: Send + Sync {
    /// Return immutable metadata for registration and provider binding.
    fn descriptor(&self) -> &ToolDescriptor;
    /// Execute one validated tool call.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] when execution cannot complete.
    async fn execute(
        &self,
        tool_call_id: &str,
        arguments: JsonMap<String, Value>,
        context: &ExtensionContext,
    ) -> Result<ToolOutput, ActionError>;
}

/// Slash-command handler.
#[async_trait]
pub trait CommandHandler: Send + Sync {
    /// Execute a slash command with its raw argument suffix.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] when command execution fails.
    async fn execute(&self, arguments: &str, context: &ExtensionContext)
    -> Result<(), ActionError>;
}

/// Registered command before invocation-name resolution.
#[derive(Clone)]
pub struct CommandRegistration {
    /// Requested slash-command name.
    pub name: String,
    /// Optional command description.
    pub description: Option<String>,
    /// Optional command argument hint.
    pub argument_hint: Option<String>,
    /// Command callback.
    pub handler: Arc<dyn CommandHandler>,
}

impl fmt::Debug for CommandRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CommandRegistration")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("argument_hint", &self.argument_hint)
            .finish_non_exhaustive()
    }
}

/// Command with a collision-safe name used for invocation.
#[derive(Clone)]
pub struct ResolvedCommand {
    /// Original requested command name.
    pub name: String,
    /// Collision-safe name exposed for invocation.
    pub invocation_name: String,
    /// Optional command description.
    pub description: Option<String>,
    /// Optional command argument hint.
    pub argument_hint: Option<String>,
    /// Command callback.
    pub handler: Arc<dyn CommandHandler>,
    /// Registration provenance.
    pub source: SourceInfo,
}

impl fmt::Debug for ResolvedCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedCommand")
            .field("name", &self.name)
            .field("invocation_name", &self.invocation_name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// CLI flag type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlagKind {
    /// Boolean switch.
    Boolean,
    /// String-valued option.
    String,
}

/// CLI flag value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FlagValue {
    /// Boolean flag value.
    Boolean(bool),
    /// String flag value.
    String(String),
}

/// CLI flag registration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlagRegistration {
    /// Long flag name, with or without leading dashes.
    pub name: String,
    /// Optional help text.
    pub description: Option<String>,
    /// Accepted value type.
    pub kind: FlagKind,
    /// Optional initial value.
    pub default: Option<FlagValue>,
}

/// Keyboard shortcut callback.
#[async_trait]
pub trait ShortcutHandler: Send + Sync {
    /// Execute the shortcut action.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] when the action fails.
    async fn execute(&self, context: &ExtensionContext) -> Result<(), ActionError>;
}

/// Extension shortcut registration.
#[derive(Clone)]
pub struct ShortcutRegistration {
    /// Key chord, normalized case-insensitively during resolution.
    pub key: String,
    /// Optional shortcut description.
    pub description: Option<String>,
    /// Shortcut callback.
    pub handler: Arc<dyn ShortcutHandler>,
}

impl fmt::Debug for ShortcutRegistration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShortcutRegistration")
            .field("key", &self.key)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Built-in binding used when resolving extension shortcuts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuiltinShortcut {
    /// Built-in action identifier.
    pub action: String,
    /// Reserved bindings cannot be replaced by extensions.
    pub restricted: bool,
}

/// Provider registration kept independent from `ri-ai`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderRegistration {
    /// Stable provider identifier.
    pub id: String,
    /// Human-readable provider name.
    pub display_name: String,
    /// Optional API base URL override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<Url>,
    /// Models exposed by the provider.
    #[serde(default)]
    pub models: Vec<ProviderModel>,
    /// Provider-specific configuration boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
}

/// Provider model metadata needed before the provider is bound.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProviderModel {
    /// Provider-specific model identifier.
    pub id: String,
    /// Human-readable model name.
    pub display_name: String,
    /// Maximum input plus output context size.
    pub context_window: u64,
    /// Maximum generated output tokens.
    pub max_output_tokens: u64,
    /// Whether image input is supported.
    #[serde(default)]
    pub supports_images: bool,
    /// Whether explicit reasoning controls are supported.
    #[serde(default)]
    pub supports_reasoning: bool,
}

/// Renderer output independent from the terminal crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedContent {
    /// Rendered plain-text lines.
    pub lines: Vec<String>,
}

/// Renderer for custom session messages or entries.
pub trait Renderer: Send + Sync {
    /// Render an extension-defined value.
    ///
    /// `Ok(None)` asks the host to use its fallback representation.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] when rendering fails.
    fn render(&self, value: &Value, expanded: bool)
    -> Result<Option<RenderedContent>, ActionError>;
}

#[derive(Clone)]
struct Sourced<T> {
    value: T,
    source: SourceInfo,
}

#[derive(Clone)]
struct RegisteredTool {
    tool: Arc<dyn NativeTool>,
    source: SourceInfo,
}

#[derive(Clone)]
struct RegisteredCommand {
    command: CommandRegistration,
    source: SourceInfo,
}

#[derive(Clone)]
struct RegisteredShortcut {
    shortcut: ShortcutRegistration,
    source: SourceInfo,
}

/// Fully resolved shortcut.
#[derive(Clone)]
pub struct ResolvedShortcut {
    /// Normalized key chord.
    pub key: String,
    /// Optional shortcut description.
    pub description: Option<String>,
    /// Shortcut callback.
    pub handler: Arc<dyn ShortcutHandler>,
    /// Winning registration provenance.
    pub source: SourceInfo,
}

impl fmt::Debug for ResolvedShortcut {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedShortcut")
            .field("key", &self.key)
            .field("description", &self.description)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Registry validation failure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RegistryError {
    /// A registration supplied an empty name or key.
    #[error("registry name must not be empty")]
    EmptyName,
    /// A flag default or assigned value has the wrong type.
    #[error("default value for flag {name:?} does not match its declared type")]
    FlagDefaultType {
        /// Flag name.
        name: String,
    },
}

/// Native extension registries and their explicit collision policies.
///
/// Tools, flags, and renderers are first-wins. Commands all remain available
/// with suffixed invocation names. Extension shortcuts are last-wins unless a
/// restricted built-in owns the key. Providers are last-wins, enabling an
/// extension to override a provider descriptor deliberately.
#[derive(Default)]
pub struct Registries {
    tools: IndexMap<String, RegisteredTool>,
    commands: Vec<RegisteredCommand>,
    providers: IndexMap<String, Sourced<ProviderRegistration>>,
    flags: IndexMap<String, Sourced<FlagRegistration>>,
    flag_values: IndexMap<String, FlagValue>,
    shortcuts: Vec<RegisteredShortcut>,
    message_renderers: IndexMap<String, Sourced<Arc<dyn Renderer>>>,
    entry_renderers: IndexMap<String, Sourced<Arc<dyn Renderer>>>,
    diagnostics: Vec<Diagnostic>,
}

impl fmt::Debug for Registries {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Registries")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("commands", &self.commands.len())
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .field("flags", &self.flags.keys().collect::<Vec<_>>())
            .field("flag_values", &self.flag_values)
            .field("shortcuts", &self.shortcuts.len())
            .field(
                "message_renderers",
                &self.message_renderers.keys().collect::<Vec<_>>(),
            )
            .field(
                "entry_renderers",
                &self.entry_renderers.keys().collect::<Vec<_>>(),
            )
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}

impl Registries {
    /// Register a tool using first-wins semantics.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when the tool descriptor has no
    /// usable name.
    pub fn register_tool(
        &mut self,
        source: SourceInfo,
        tool: Arc<dyn NativeTool>,
    ) -> Result<(), RegistryError> {
        let name = tool.descriptor().name.trim().to_owned();
        if name.is_empty() {
            return Err(RegistryError::EmptyName);
        }
        if let Some(existing) = self.tools.get(&name) {
            self.diagnostics.push(Diagnostic::collision(
                ResourceKind::Tool,
                name,
                existing.source.clone(),
                source,
            ));
            return Ok(());
        }
        self.tools.insert(name, RegisteredTool { tool, source });
        Ok(())
    }

    /// Register a command. Duplicate names are retained and resolved later.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when the command has no usable
    /// name.
    pub fn register_command(
        &mut self,
        source: SourceInfo,
        command: CommandRegistration,
    ) -> Result<(), RegistryError> {
        if command.name.trim().is_empty() {
            return Err(RegistryError::EmptyName);
        }
        self.commands.push(RegisteredCommand { command, source });
        Ok(())
    }

    /// Register or override a provider using last-wins semantics.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when the provider has no usable
    /// identifier.
    pub fn register_provider(
        &mut self,
        source: SourceInfo,
        provider: ProviderRegistration,
    ) -> Result<(), RegistryError> {
        let name = provider.id.trim().to_owned();
        if name.is_empty() {
            return Err(RegistryError::EmptyName);
        }
        if let Some(previous) = self.providers.shift_remove(&name) {
            self.diagnostics.push(Diagnostic::collision(
                ResourceKind::Provider,
                name.clone(),
                source.clone(),
                previous.source,
            ));
        }
        self.providers.insert(
            name,
            Sourced {
                value: provider,
                source,
            },
        );
        Ok(())
    }

    /// Register a flag using first-wins semantics.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the name is empty or its default value
    /// does not match the declared kind.
    pub fn register_flag(
        &mut self,
        source: SourceInfo,
        flag: FlagRegistration,
    ) -> Result<(), RegistryError> {
        let name = flag.name.trim().trim_start_matches('-').to_owned();
        if name.is_empty() {
            return Err(RegistryError::EmptyName);
        }
        if !flag_default_matches(&flag) {
            return Err(RegistryError::FlagDefaultType { name });
        }
        if let Some(previous) = self.flags.get(&name) {
            self.diagnostics.push(Diagnostic::collision(
                ResourceKind::Flag,
                name,
                previous.source.clone(),
                source,
            ));
            return Ok(());
        }
        if let Some(default) = flag.default.clone() {
            self.flag_values.insert(name.clone(), default);
        }
        self.flags.insert(
            name,
            Sourced {
                value: flag,
                source,
            },
        );
        Ok(())
    }

    /// Set a parsed CLI value for a registered flag.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the flag is unknown or the value has the
    /// wrong type.
    pub fn set_flag_value(&mut self, name: &str, value: FlagValue) -> Result<(), RegistryError> {
        let normalized = name.trim_start_matches('-');
        let Some(flag) = self.flags.get(normalized) else {
            return Err(RegistryError::EmptyName);
        };
        let matches = matches!(
            (&flag.value.kind, &value),
            (FlagKind::Boolean, FlagValue::Boolean(_)) | (FlagKind::String, FlagValue::String(_))
        );
        if !matches {
            return Err(RegistryError::FlagDefaultType {
                name: normalized.to_owned(),
            });
        }
        self.flag_values.insert(normalized.to_owned(), value);
        Ok(())
    }

    /// Register a shortcut. Conflict policy is applied by
    /// [`Self::resolve_shortcuts`] once built-ins are known.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when the key chord is empty.
    pub fn register_shortcut(
        &mut self,
        source: SourceInfo,
        shortcut: ShortcutRegistration,
    ) -> Result<(), RegistryError> {
        if shortcut.key.trim().is_empty() {
            return Err(RegistryError::EmptyName);
        }
        self.shortcuts.push(RegisteredShortcut { shortcut, source });
        Ok(())
    }

    /// Register a custom-message renderer using first-wins semantics.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when `custom_type` is empty.
    pub fn register_message_renderer(
        &mut self,
        source: SourceInfo,
        custom_type: impl Into<String>,
        renderer: Arc<dyn Renderer>,
    ) -> Result<(), RegistryError> {
        let custom_type = custom_type.into();
        register_renderer(
            &mut self.message_renderers,
            &mut self.diagnostics,
            ResourceKind::MessageRenderer,
            custom_type,
            source,
            renderer,
        )
    }

    /// Register a custom-entry renderer using first-wins semantics.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::EmptyName`] when `custom_type` is empty.
    pub fn register_entry_renderer(
        &mut self,
        source: SourceInfo,
        custom_type: impl Into<String>,
        renderer: Arc<dyn Renderer>,
    ) -> Result<(), RegistryError> {
        let custom_type = custom_type.into();
        register_renderer(
            &mut self.entry_renderers,
            &mut self.diagnostics,
            ResourceKind::EntryRenderer,
            custom_type,
            source,
            renderer,
        )
    }

    /// Tool names in deterministic registration order.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Get one native tool and its provenance.
    pub fn tool(&self, name: &str) -> Option<(Arc<dyn NativeTool>, &SourceInfo)> {
        self.tools
            .get(name)
            .map(|tool| (Arc::clone(&tool.tool), &tool.source))
    }

    /// Get one provider descriptor.
    pub fn provider(&self, id: &str) -> Option<(&ProviderRegistration, &SourceInfo)> {
        self.providers
            .get(id)
            .map(|provider| (&provider.value, &provider.source))
    }

    /// Current value of a registered flag.
    pub fn flag_value(&self, name: &str) -> Option<&FlagValue> {
        self.flag_values.get(name.trim_start_matches('-'))
    }

    /// Resolve duplicate command names while keeping every command invokable.
    pub fn commands(&self) -> Vec<ResolvedCommand> {
        let mut counts = BTreeMap::<&str, usize>::new();
        for command in &self.commands {
            *counts.entry(command.command.name.as_str()).or_default() += 1;
        }

        let mut seen = BTreeMap::<&str, usize>::new();
        let mut taken = std::collections::BTreeSet::<String>::new();
        let mut resolved = Vec::with_capacity(self.commands.len());
        for registered in &self.commands {
            let name = registered.command.name.as_str();
            let occurrence = seen.entry(name).or_default();
            *occurrence += 1;
            let mut invocation_name = if counts.get(name).copied().unwrap_or_default() > 1 {
                format!("{name}:{occurrence}")
            } else {
                name.to_owned()
            };
            if taken.contains(&invocation_name) {
                let mut suffix = *occurrence;
                loop {
                    suffix += 1;
                    invocation_name = format!("{name}:{suffix}");
                    if !taken.contains(&invocation_name) {
                        break;
                    }
                }
            }
            taken.insert(invocation_name.clone());
            resolved.push(ResolvedCommand {
                name: registered.command.name.clone(),
                invocation_name,
                description: registered.command.description.clone(),
                argument_hint: registered.command.argument_hint.clone(),
                handler: Arc::clone(&registered.command.handler),
                source: registered.source.clone(),
            });
        }
        resolved
    }

    /// Resolve extension shortcuts against built-ins. Restricted built-ins
    /// always win; non-restricted built-ins are replaced with a warning;
    /// extension-to-extension conflicts are last-wins.
    pub fn resolve_shortcuts(
        &self,
        builtins: &IndexMap<String, BuiltinShortcut>,
    ) -> (IndexMap<String, ResolvedShortcut>, Vec<Diagnostic>) {
        let normalized_builtins = builtins
            .iter()
            .map(|(key, binding)| (key.to_lowercase(), binding))
            .collect::<BTreeMap<_, _>>();
        let mut resolved = IndexMap::<String, ResolvedShortcut>::new();
        let mut diagnostics = Vec::new();

        for registered in &self.shortcuts {
            let key = registered.shortcut.key.to_lowercase();
            if let Some(binding) = normalized_builtins.get(&key) {
                if binding.restricted {
                    diagnostics.push(Diagnostic::warning(
                        format!(
                            "extension shortcut {:?} conflicts with reserved built-in action {}",
                            registered.shortcut.key, binding.action
                        ),
                        registered.source.clone(),
                    ));
                    continue;
                }
                diagnostics.push(Diagnostic::warning(
                    format!(
                        "extension shortcut {:?} replaces built-in action {}",
                        registered.shortcut.key, binding.action
                    ),
                    registered.source.clone(),
                ));
            }

            if let Some(previous) = resolved.shift_remove(&key) {
                diagnostics.push(Diagnostic::collision(
                    ResourceKind::Shortcut,
                    key.clone(),
                    registered.source.clone(),
                    previous.source,
                ));
            }
            resolved.insert(
                key.clone(),
                ResolvedShortcut {
                    key,
                    description: registered.shortcut.description.clone(),
                    handler: Arc::clone(&registered.shortcut.handler),
                    source: registered.source.clone(),
                },
            );
        }
        (resolved, diagnostics)
    }

    /// Get a message renderer.
    pub fn message_renderer(&self, custom_type: &str) -> Option<Arc<dyn Renderer>> {
        self.message_renderers
            .get(custom_type)
            .map(|renderer| Arc::clone(&renderer.value))
    }

    /// Get an entry renderer.
    pub fn entry_renderer(&self, custom_type: &str) -> Option<Arc<dyn Renderer>> {
        self.entry_renderers
            .get(custom_type)
            .map(|renderer| Arc::clone(&renderer.value))
    }

    /// Registry diagnostics accumulated while loading.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    fn merge(&mut self, other: Registries) -> Result<(), RegistryError> {
        for (_, tool) in other.tools {
            self.register_tool(tool.source, tool.tool)?;
        }
        for command in other.commands {
            self.register_command(command.source, command.command)?;
        }
        for (_, provider) in other.providers {
            self.register_provider(provider.source, provider.value)?;
        }
        for (_, flag) in other.flags {
            self.register_flag(flag.source, flag.value)?;
        }
        for shortcut in other.shortcuts {
            self.register_shortcut(shortcut.source, shortcut.shortcut)?;
        }
        for (custom_type, renderer) in other.message_renderers {
            self.register_message_renderer(renderer.source, custom_type, renderer.value)?;
        }
        for (custom_type, renderer) in other.entry_renderers {
            self.register_entry_renderer(renderer.source, custom_type, renderer.value)?;
        }
        self.diagnostics.extend(other.diagnostics);
        Ok(())
    }
}

fn flag_default_matches(flag: &FlagRegistration) -> bool {
    matches!(
        (&flag.kind, &flag.default),
        (_, None)
            | (FlagKind::Boolean, Some(FlagValue::Boolean(_)))
            | (FlagKind::String, Some(FlagValue::String(_)))
    )
}

fn register_renderer(
    renderers: &mut IndexMap<String, Sourced<Arc<dyn Renderer>>>,
    diagnostics: &mut Vec<Diagnostic>,
    kind: ResourceKind,
    custom_type: String,
    source: SourceInfo,
    renderer: Arc<dyn Renderer>,
) -> Result<(), RegistryError> {
    if custom_type.trim().is_empty() {
        return Err(RegistryError::EmptyName);
    }
    if let Some(previous) = renderers.get(&custom_type) {
        diagnostics.push(Diagnostic::collision(
            kind,
            custom_type,
            previous.source.clone(),
            source,
        ));
        return Ok(());
    }
    renderers.insert(
        custom_type,
        Sourced {
            value: renderer,
            source,
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Native extension loading and lifecycle
// ---------------------------------------------------------------------------

/// Native extension identity and provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensionDescriptor {
    /// Stable extension identifier.
    pub id: String,
    /// Human-readable extension name.
    pub name: String,
    /// Optional extension version.
    pub version: Option<String>,
    /// Extension provenance.
    pub source: SourceInfo,
}

/// Extension initialization failure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{message}")]
pub struct ExtensionInitError {
    message: String,
}

impl ExtensionInitError {
    /// Construct an initialization error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Native, statically linked extension.
#[async_trait]
pub trait Extension: Send + Sync {
    /// Return extension identity and provenance.
    fn descriptor(&self) -> ExtensionDescriptor;
    /// Register hooks and runtime contributions transactionally.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionInitError`] to discard every contribution from this
    /// extension while allowing later extensions to load.
    async fn register(&self, registrar: &mut ExtensionRegistrar) -> Result<(), ExtensionInitError>;
}

/// Transactional registration surface passed to one extension.
///
/// Contributions are merged only after `Extension::register` succeeds.
pub struct ExtensionRegistrar {
    descriptor: ExtensionDescriptor,
    hooks: Vec<Arc<dyn EventHook>>,
    registries: Registries,
}

impl fmt::Debug for ExtensionRegistrar {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionRegistrar")
            .field("descriptor", &self.descriptor)
            .field("hook_count", &self.hooks.len())
            .field("registries", &self.registries)
            .finish()
    }
}

impl ExtensionRegistrar {
    fn new(descriptor: ExtensionDescriptor) -> Self {
        Self {
            descriptor,
            hooks: Vec::new(),
            registries: Registries::default(),
        }
    }

    /// Subscribe one hook object. A single extension may call this repeatedly;
    /// callbacks retain registration order.
    pub fn add_hook(&mut self, hook: Arc<dyn EventHook>) {
        self.hooks.push(hook);
    }

    /// Register a native tool owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registration is invalid.
    pub fn register_tool(&mut self, tool: Arc<dyn NativeTool>) -> Result<(), RegistryError> {
        self.registries
            .register_tool(self.descriptor.source.clone(), tool)
    }

    /// Register a slash command owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registration is invalid.
    pub fn register_command(&mut self, command: CommandRegistration) -> Result<(), RegistryError> {
        self.registries
            .register_command(self.descriptor.source.clone(), command)
    }

    /// Register a provider descriptor owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registration is invalid.
    pub fn register_provider(
        &mut self,
        provider: ProviderRegistration,
    ) -> Result<(), RegistryError> {
        self.registries
            .register_provider(self.descriptor.source.clone(), provider)
    }

    /// Register a command-line flag owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registration is invalid.
    pub fn register_flag(&mut self, flag: FlagRegistration) -> Result<(), RegistryError> {
        self.registries
            .register_flag(self.descriptor.source.clone(), flag)
    }

    /// Register a keyboard shortcut owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the registration is invalid.
    pub fn register_shortcut(
        &mut self,
        shortcut: ShortcutRegistration,
    ) -> Result<(), RegistryError> {
        self.registries
            .register_shortcut(self.descriptor.source.clone(), shortcut)
    }

    /// Register a custom-message renderer owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the custom type is empty.
    pub fn register_message_renderer(
        &mut self,
        custom_type: impl Into<String>,
        renderer: Arc<dyn Renderer>,
    ) -> Result<(), RegistryError> {
        self.registries.register_message_renderer(
            self.descriptor.source.clone(),
            custom_type,
            renderer,
        )
    }

    /// Register a custom-entry renderer owned by this extension.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError`] when the custom type is empty.
    pub fn register_entry_renderer(
        &mut self,
        custom_type: impl Into<String>,
        renderer: Arc<dyn Renderer>,
    ) -> Result<(), RegistryError> {
        self.registries.register_entry_renderer(
            self.descriptor.source.clone(),
            custom_type,
            renderer,
        )
    }
}

/// One extension that failed during transactional loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExtensionLoadError {
    /// Extension that failed to initialize.
    pub descriptor: ExtensionDescriptor,
    /// Initialization or registry-merge failure.
    pub message: String,
}

/// Loaded native extension runtime.
pub struct ExtensionHost {
    clock: GenerationClock,
    binding: ContextBinding,
    actions: Arc<dyn ContextActions>,
    event_bus: EventBus,
    runner: ExtensionRunner,
    registries: Registries,
    load_errors: Vec<ExtensionLoadError>,
}

impl fmt::Debug for ExtensionHost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionHost")
            .field("generation", &self.clock.current())
            .field("runner", &self.runner)
            .field("registries", &self.registries)
            .field("load_errors", &self.load_errors)
            .finish_non_exhaustive()
    }
}

impl ExtensionHost {
    /// Load native extensions transactionally in the provided order.
    pub async fn load(
        clock: GenerationClock,
        binding: ContextBinding,
        actions: Arc<dyn ContextActions>,
        event_bus: EventBus,
        extensions: &[Arc<dyn Extension>],
    ) -> Self {
        let (runner, registries, load_errors) = load_contributions(
            &clock,
            &binding,
            Arc::clone(&actions),
            event_bus.clone(),
            extensions,
        )
        .await;
        Self {
            clock,
            binding,
            actions,
            event_bus,
            runner,
            registries,
            load_errors,
        }
    }

    /// Emit `session_shutdown`, invalidate captured contexts, and load the
    /// replacement extension set. This ordering prevents old contexts from
    /// becoming stale before shutdown handlers finish.
    pub async fn reload(&mut self, reason: impl Into<String>, extensions: &[Arc<dyn Extension>]) {
        self.runner
            .emit_notification(&NotificationEvent::SessionShutdown {
                reason: reason.into(),
            })
            .await;
        self.clock.advance();
        let (runner, registries, load_errors) = load_contributions(
            &self.clock,
            &self.binding,
            Arc::clone(&self.actions),
            self.event_bus.clone(),
            extensions,
        )
        .await;
        self.runner = runner;
        self.registries = registries;
        self.load_errors = load_errors;
    }

    /// Update mode/cwd/trust facts for newly created contexts.
    pub fn set_binding(&mut self, binding: ContextBinding) {
        self.binding = binding.clone();
        self.runner.contexts.set_binding(binding);
    }

    /// Return the active hook runner.
    pub fn runner(&self) -> &ExtensionRunner {
        &self.runner
    }

    /// Return active extension registries.
    pub fn registries(&self) -> &Registries {
        &self.registries
    }

    /// Mutably access active extension registries.
    pub fn registries_mut(&mut self) -> &mut Registries {
        &mut self.registries
    }

    /// Return isolated errors from the most recent load.
    pub fn load_errors(&self) -> &[ExtensionLoadError] {
        &self.load_errors
    }

    /// Return the shared extension event bus.
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    /// Return the active runtime generation.
    pub fn generation(&self) -> u64 {
        self.clock.current()
    }
}

async fn load_contributions(
    clock: &GenerationClock,
    binding: &ContextBinding,
    actions: Arc<dyn ContextActions>,
    event_bus: EventBus,
    extensions: &[Arc<dyn Extension>],
) -> (ExtensionRunner, Registries, Vec<ExtensionLoadError>) {
    let mut hooks = Vec::new();
    let mut registries = Registries::default();
    let mut load_errors = Vec::new();

    for extension in extensions {
        let descriptor = extension.descriptor();
        let mut registrar = ExtensionRegistrar::new(descriptor.clone());
        match extension.register(&mut registrar).await {
            Ok(()) => {
                for hook in registrar.hooks {
                    hooks.push(HookRegistration {
                        extension_id: descriptor.id.clone(),
                        source: descriptor.source.clone(),
                        hook,
                    });
                }
                if let Err(error) = registries.merge(registrar.registries) {
                    load_errors.push(ExtensionLoadError {
                        descriptor,
                        message: error.to_string(),
                    });
                }
            }
            Err(error) => load_errors.push(ExtensionLoadError {
                descriptor,
                message: error.to_string(),
            }),
        }
    }

    let contexts = ContextFactory::new(clock.clone(), binding.clone(), actions, event_bus);
    (
        ExtensionRunner::new(hooks, contexts),
        registries,
        load_errors,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use serde_json::json;

    use super::*;

    fn message(text: &str) -> ContextMessage {
        ContextMessage {
            role: MessageRole::User,
            content: vec![ContentPart::Text {
                text: text.to_owned(),
            }],
            metadata: None,
        }
    }

    fn runner(hooks: Vec<Arc<dyn EventHook>>) -> ExtensionRunner {
        let clock = GenerationClock::default();
        let contexts = ContextFactory::new(
            clock,
            ContextBinding::default(),
            Arc::new(NoopContextActions),
            EventBus::default(),
        );
        let hooks = hooks
            .into_iter()
            .enumerate()
            .map(|(index, hook)| HookRegistration {
                extension_id: format!("extension-{index}"),
                source: SourceInfo::inline(format!("extension-{index}")),
                hook,
            })
            .collect();
        ExtensionRunner::new(hooks, contexts)
    }

    struct ContextHook {
        suffix: &'static str,
        fail: bool,
    }

    #[async_trait]
    impl EventHook for ContextHook {
        async fn on_context(
            &self,
            event: &ContextEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<Vec<ContextMessage>>, HookError> {
            if self.fail {
                return Err(HookError::new("boom"));
            }
            let mut messages = event.messages.clone();
            messages.push(message(self.suffix));
            Ok(Some(messages))
        }
    }

    #[tokio::test]
    async fn context_reducer_chains_and_isolates_errors() {
        let runner = runner(vec![
            Arc::new(ContextHook {
                suffix: "one",
                fail: false,
            }),
            Arc::new(ContextHook {
                suffix: "ignored",
                fail: true,
            }),
            Arc::new(ContextHook {
                suffix: "two",
                fail: false,
            }),
        ]);
        let result = runner.emit_context(&[message("base")]).await;
        assert_eq!(
            result,
            vec![message("base"), message("one"), message("two")]
        );
        assert_eq!(runner.drain_errors().len(), 1);
    }

    struct InputHook {
        value: InputResult,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EventHook for InputHook {
        async fn on_input(
            &self,
            _event: &InputEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<InputResult>, HookError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Some(self.value.clone()))
        }
    }

    #[tokio::test]
    async fn input_transform_chains_and_handled_short_circuits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = runner(vec![
            Arc::new(InputHook {
                value: InputResult::Transform {
                    text: "changed".to_owned(),
                    images: None,
                },
                calls: Arc::clone(&calls),
            }),
            Arc::new(InputHook {
                value: InputResult::Handled,
                calls: Arc::clone(&calls),
            }),
            Arc::new(InputHook {
                value: InputResult::Continue,
                calls: Arc::clone(&calls),
            }),
        ]);
        let result = runner
            .emit_input(InputEvent {
                text: "original".to_owned(),
                images: vec![],
                source: InputSource::Interactive,
                streaming_behavior: None,
            })
            .await;
        assert_eq!(result, InputResult::Handled);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    struct HeaderHook {
        name: &'static str,
        fail: bool,
    }

    #[async_trait]
    impl EventHook for HeaderHook {
        async fn on_provider_headers(
            &self,
            event: &mut ProviderHeadersEvent,
            _context: &ExtensionContext,
        ) -> Result<(), HookError> {
            if self.fail {
                return Err(HookError::new("header failure"));
            }
            event
                .headers
                .insert(self.name.to_owned(), Some("yes".to_owned()));
            Ok(())
        }
    }

    #[tokio::test]
    async fn header_handlers_share_one_map_after_failure() {
        let runner = runner(vec![
            Arc::new(HeaderHook {
                name: "X-One",
                fail: false,
            }),
            Arc::new(HeaderHook {
                name: "X-Bad",
                fail: true,
            }),
            Arc::new(HeaderHook {
                name: "X-Two",
                fail: false,
            }),
        ]);
        let headers = runner.emit_provider_headers(IndexMap::new()).await;
        assert_eq!(headers["X-One"], Some("yes".to_owned()));
        assert_eq!(headers["X-Two"], Some("yes".to_owned()));
        assert_eq!(runner.drain_errors().len(), 1);
    }

    struct RequestHook {
        key: &'static str,
        value: i64,
        fail: bool,
    }

    #[async_trait]
    impl EventHook for RequestHook {
        async fn on_provider_request(
            &self,
            event: &ProviderRequestEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<Value>, HookError> {
            if self.fail {
                return Err(HookError::new("request failure"));
            }
            let mut payload = event
                .payload
                .as_object()
                .cloned()
                .ok_or_else(|| HookError::new("expected object"))?;
            payload.insert(self.key.to_owned(), json!(self.value));
            Ok(Some(Value::Object(payload)))
        }
    }

    #[tokio::test]
    async fn provider_request_replacements_chain_across_failures() {
        let runner = runner(vec![
            Arc::new(RequestHook {
                key: "one",
                value: 1,
                fail: false,
            }),
            Arc::new(RequestHook {
                key: "bad",
                value: 0,
                fail: true,
            }),
            Arc::new(RequestHook {
                key: "two",
                value: 2,
                fail: false,
            }),
        ]);
        let payload = runner.emit_provider_request(json!({"base": true})).await;
        assert_eq!(payload, json!({"base": true, "one": 1, "two": 2}));
        let errors = runner.drain_errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].event, "before_provider_request");
    }

    struct PromptHook {
        suffix: &'static str,
    }

    #[async_trait]
    impl EventHook for PromptHook {
        async fn on_before_agent_start(
            &self,
            event: &BeforeAgentStartEvent,
            context: &ExtensionContext,
        ) -> Result<Option<BeforeAgentStartResult>, HookError> {
            assert_eq!(
                context.system_prompt().expect("active"),
                event.system_prompt
            );
            Ok(Some(BeforeAgentStartResult {
                message: None,
                system_prompt: Some(format!("{}\n{}", event.system_prompt, self.suffix)),
            }))
        }
    }

    #[tokio::test]
    async fn system_prompt_and_context_chain_together() {
        let runner = runner(vec![
            Arc::new(PromptHook { suffix: "one" }),
            Arc::new(PromptHook { suffix: "two" }),
        ]);
        let result = runner
            .emit_before_agent_start("hello", vec![], "base")
            .await;
        assert_eq!(result.system_prompt.as_deref(), Some("base\none\ntwo"));
    }

    struct ToolHook {
        value: i64,
        block: bool,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EventHook for ToolHook {
        async fn on_tool_call(
            &self,
            event: &mut ToolCallEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<ToolCallResult>, HookError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            event.input.insert("value".to_owned(), json!(self.value));
            Ok(Some(ToolCallResult {
                block: self.block,
                reason: self.block.then(|| "blocked".to_owned()),
            }))
        }
    }

    #[tokio::test]
    async fn first_tool_block_stops_reducer_after_mutation() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = runner(vec![
            Arc::new(ToolHook {
                value: 1,
                block: false,
                calls: Arc::clone(&calls),
            }),
            Arc::new(ToolHook {
                value: 2,
                block: true,
                calls: Arc::clone(&calls),
            }),
            Arc::new(ToolHook {
                value: 3,
                block: false,
                calls: Arc::clone(&calls),
            }),
        ]);
        let reduced = runner
            .emit_tool_call(ToolCallEvent {
                tool_call_id: "call".to_owned(),
                tool_name: "tool".to_owned(),
                input: JsonMap::new(),
            })
            .await;
        assert_eq!(reduced.event.input["value"], json!(2));
        assert!(reduced.result.expect("result").block);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    struct ToolResultHook {
        stage: usize,
        fail: bool,
    }

    #[async_trait]
    impl EventHook for ToolResultHook {
        async fn on_tool_result(
            &self,
            event: &ToolResultEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<ToolResultPatch>, HookError> {
            if self.fail {
                return Err(HookError::new("result failure"));
            }
            match self.stage {
                1 => Ok(Some(ToolResultPatch {
                    content: Some(vec![ContentPart::Text {
                        text: "changed".to_owned(),
                    }]),
                    details: Some(json!({"stage": 1})),
                    ..ToolResultPatch::default()
                })),
                2 => {
                    assert_eq!(event.content, message("changed").content);
                    assert_eq!(event.details, Some(json!({"stage": 1})));
                    Ok(Some(ToolResultPatch {
                        is_error: Some(true),
                        usage: Some(ToolUsage {
                            input_tokens: 3,
                            output_tokens: 5,
                        }),
                        ..ToolResultPatch::default()
                    }))
                }
                _ => Ok(None),
            }
        }
    }

    #[tokio::test]
    async fn tool_result_patches_chain_and_preserve_omitted_fields() {
        let runner = runner(vec![
            Arc::new(ToolResultHook {
                stage: 1,
                fail: false,
            }),
            Arc::new(ToolResultHook {
                stage: 0,
                fail: true,
            }),
            Arc::new(ToolResultHook {
                stage: 2,
                fail: false,
            }),
        ]);
        let result = runner
            .emit_tool_result(ToolResultEvent {
                tool_call_id: "call".to_owned(),
                tool_name: "tool".to_owned(),
                input: JsonMap::new(),
                content: message("original").content,
                details: None,
                is_error: false,
                usage: None,
            })
            .await
            .expect("modified result");
        assert_eq!(result.content, message("changed").content);
        assert_eq!(result.details, Some(json!({"stage": 1})));
        assert!(result.is_error);
        assert_eq!(result.usage.expect("usage").output_tokens, 5);
        assert_eq!(runner.drain_errors().len(), 1);
    }

    struct SessionHook {
        result: Option<SessionBeforeResult>,
        fail: bool,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EventHook for SessionHook {
        async fn on_session_before(
            &self,
            _event: &SessionBeforeEvent,
            _context: &ExtensionContext,
        ) -> Result<Option<SessionBeforeResult>, HookError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                Err(HookError::new("session failure"))
            } else {
                Ok(self.result.clone())
            }
        }
    }

    #[tokio::test]
    async fn session_before_uses_last_result_but_first_cancellation() {
        let calls = Arc::new(AtomicUsize::new(0));
        let event = SessionBeforeEvent::Switch {
            reason: "test".to_owned(),
            target_session: None,
        };
        let last_result_runner = runner(vec![
            Arc::new(SessionHook {
                result: Some(SessionBeforeResult {
                    cancel: false,
                    override_value: Some(SessionOverride::SkipConversationRestore(true)),
                }),
                fail: false,
                calls: Arc::clone(&calls),
            }),
            Arc::new(SessionHook {
                result: None,
                fail: true,
                calls: Arc::clone(&calls),
            }),
            Arc::new(SessionHook {
                result: Some(SessionBeforeResult {
                    cancel: false,
                    override_value: Some(SessionOverride::SkipConversationRestore(false)),
                }),
                fail: false,
                calls: Arc::clone(&calls),
            }),
        ]);
        let result = last_result_runner
            .emit_session_before(&event)
            .await
            .expect("last result");
        assert_eq!(
            result.override_value,
            Some(SessionOverride::SkipConversationRestore(false))
        );
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(last_result_runner.drain_errors().len(), 1);

        calls.store(0, Ordering::Relaxed);
        let cancellation_runner = runner(vec![
            Arc::new(SessionHook {
                result: Some(SessionBeforeResult {
                    cancel: true,
                    override_value: None,
                }),
                fail: false,
                calls: Arc::clone(&calls),
            }),
            Arc::new(SessionHook {
                result: None,
                fail: false,
                calls: Arc::clone(&calls),
            }),
        ]);
        assert!(
            cancellation_runner
                .emit_session_before(&event)
                .await
                .expect("cancellation")
                .cancel
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn captured_context_becomes_stale() {
        let clock = GenerationClock::default();
        let factory = ContextFactory::new(
            clock.clone(),
            ContextBinding::default(),
            Arc::new(NoopContextActions),
            EventBus::default(),
        );
        let context = factory.create();
        assert!(context.cwd().is_ok());
        clock.advance();
        assert!(matches!(context.cwd(), Err(StaleContextError { .. })));
    }

    struct RecordingBusHandler {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait]
    impl BusHandler for RecordingBusHandler {
        async fn handle(&self, _event: &BusEvent) -> Result<(), BusHandlerError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                Err(BusHandlerError::new("bad subscriber"))
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn event_bus_isolates_errors_and_unsubscribes_on_drop() {
        let bus = EventBus::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let failed = bus.subscribe(
            "channel",
            Arc::new(RecordingBusHandler {
                calls: Arc::clone(&calls),
                fail: true,
            }),
        );
        let successful = bus.subscribe(
            "channel",
            Arc::new(RecordingBusHandler {
                calls: Arc::clone(&calls),
                fail: false,
            }),
        );
        let errors = bus.emit("channel", json!({"ok": true})).await;
        assert_eq!(errors.len(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        drop(failed);
        drop(successful);
        bus.emit("channel", Value::Null).await;
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    struct EmptyCommand;

    #[async_trait]
    impl CommandHandler for EmptyCommand {
        async fn execute(
            &self,
            _arguments: &str,
            _context: &ExtensionContext,
        ) -> Result<(), ActionError> {
            Ok(())
        }
    }

    struct EmptyShortcut;

    #[async_trait]
    impl ShortcutHandler for EmptyShortcut {
        async fn execute(&self, _context: &ExtensionContext) -> Result<(), ActionError> {
            Ok(())
        }
    }

    #[test]
    fn registry_collision_policies_are_explicit() {
        let mut registries = Registries::default();
        for id in ["one", "two"] {
            registries
                .register_command(
                    SourceInfo::inline(id),
                    CommandRegistration {
                        name: "same".to_owned(),
                        description: Some(id.to_owned()),
                        argument_hint: None,
                        handler: Arc::new(EmptyCommand),
                    },
                )
                .expect("command");
            registries
                .register_shortcut(
                    SourceInfo::inline(id),
                    ShortcutRegistration {
                        key: "CTRL+X".to_owned(),
                        description: Some(id.to_owned()),
                        handler: Arc::new(EmptyShortcut),
                    },
                )
                .expect("shortcut");
        }
        let commands = registries.commands();
        assert_eq!(commands[0].invocation_name, "same:1");
        assert_eq!(commands[1].invocation_name, "same:2");

        let (shortcuts, diagnostics) = registries.resolve_shortcuts(&IndexMap::new());
        assert_eq!(shortcuts["ctrl+x"].description.as_deref(), Some("two"));
        assert_eq!(diagnostics.len(), 1);
    }

    struct ShutdownHook {
        called: Arc<AtomicBool>,
        context_was_active: Arc<AtomicBool>,
    }

    #[async_trait]
    impl EventHook for ShutdownHook {
        async fn on_notification(
            &self,
            event: &NotificationEvent,
            context: &ExtensionContext,
        ) -> Result<(), HookError> {
            if matches!(event, NotificationEvent::SessionShutdown { .. }) {
                self.called.store(true, Ordering::Relaxed);
                self.context_was_active
                    .store(context.ensure_active().is_ok(), Ordering::Relaxed);
            }
            Ok(())
        }
    }

    struct HookExtension {
        hook: Arc<dyn EventHook>,
    }

    #[async_trait]
    impl Extension for HookExtension {
        fn descriptor(&self) -> ExtensionDescriptor {
            ExtensionDescriptor {
                id: "hook".to_owned(),
                name: "Hook".to_owned(),
                version: None,
                source: SourceInfo::inline("hook"),
            }
        }

        async fn register(
            &self,
            registrar: &mut ExtensionRegistrar,
        ) -> Result<(), ExtensionInitError> {
            registrar.add_hook(Arc::clone(&self.hook));
            Ok(())
        }
    }

    #[tokio::test]
    async fn reload_emits_shutdown_before_invalidating_contexts() {
        let called = Arc::new(AtomicBool::new(false));
        let context_was_active = Arc::new(AtomicBool::new(false));
        let extension: Arc<dyn Extension> = Arc::new(HookExtension {
            hook: Arc::new(ShutdownHook {
                called: Arc::clone(&called),
                context_was_active: Arc::clone(&context_was_active),
            }),
        });
        let mut host = ExtensionHost::load(
            GenerationClock::default(),
            ContextBinding::default(),
            Arc::new(NoopContextActions),
            EventBus::default(),
            &[Arc::clone(&extension)],
        )
        .await;
        let captured = host.runner().context();
        host.reload("reload", &[extension]).await;
        assert!(called.load(Ordering::Relaxed));
        assert!(context_was_active.load(Ordering::Relaxed));
        assert!(captured.ensure_active().is_err());
    }
}
