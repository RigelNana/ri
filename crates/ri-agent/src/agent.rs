//! Stateful agent wrapper, ordered listeners, and message queues.

use std::{
    collections::{BTreeSet, VecDeque},
    future::Future,
    panic::AssertUnwindSafe,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use futures::FutureExt;
use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
pub use ri_protocol_core::QueueMode;
use tokio::sync::{Mutex as AsyncMutex, watch};
use tokio_util::sync::CancellationToken;

use crate::{
    AgentContext, AgentError, AgentEvent, AgentEventSink, AgentLoopConfig, AgentMessage, Prompt,
    StreamFn, Tool, default_stream_fn, run_agent_loop, run_agent_loop_continue,
};

#[derive(Clone, Debug)]
struct PendingMessageQueue<M> {
    messages: VecDeque<M>,
    mode: QueueMode,
}

impl<M> PendingMessageQueue<M> {
    fn new(mode: QueueMode) -> Self {
        Self {
            messages: VecDeque::new(),
            mode,
        }
    }

    fn drain(&mut self) -> Vec<M> {
        match self.mode {
            QueueMode::All => self.messages.drain(..).collect(),
            QueueMode::OneAtATime => self.messages.pop_front().into_iter().collect(),
        }
    }
}

/// Public snapshot of state owned by an [`Agent`].
#[derive(Clone, Debug)]
pub struct AgentState<M> {
    /// System instruction used by future provider turns.
    pub system_prompt: String,
    /// Model used by future provider turns.
    pub model: ri_ai::Model,
    /// Requested reasoning effort.
    pub thinking_level: ri_ai::ThinkingLevel,
    /// Active tools.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Complete application transcript.
    pub messages: Vec<M>,
    /// Whether a prompt or continuation is active.
    ///
    /// This remains true through the final awaited listener barrier.
    pub is_streaming: bool,
    /// Current partial message, if one is being emitted.
    pub streaming_message: Option<M>,
    /// Tool call ids that have started but not ended.
    pub pending_tool_calls: BTreeSet<String>,
    /// Error text from the most recent failed or aborted assistant turn.
    pub error_message: Option<String>,
}

impl<M> AgentState<M> {
    /// Creates empty state for a model.
    pub fn new(model: ri_ai::Model) -> Self {
        Self {
            system_prompt: String::new(),
            model,
            thinking_level: ri_ai::ThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: BTreeSet::new(),
            error_message: None,
        }
    }
}

/// Construction options for a stateful agent.
#[must_use]
#[derive(Clone)]
pub struct AgentOptions<M> {
    /// Initial public state.
    pub initial_state: AgentState<M>,
    /// Loop callbacks and provider-request options.
    pub loop_config: AgentLoopConfig<M>,
    /// Explicit provider stream boundary.
    pub stream_fn: Option<Arc<dyn StreamFn>>,
    /// Steering queue policy.
    pub steering_mode: QueueMode,
    /// Follow-up queue policy.
    pub follow_up_mode: QueueMode,
}

impl<M> std::fmt::Debug for AgentOptions<M>
where
    M: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AgentOptions")
            .field("initial_state", &self.initial_state)
            .field("loop_config", &self.loop_config)
            .field("has_stream_fn", &self.stream_fn.is_some())
            .field("steering_mode", &self.steering_mode)
            .field("follow_up_mode", &self.follow_up_mode)
            .finish()
    }
}

impl<M: AgentMessage> AgentOptions<M> {
    /// Creates options with empty state and default loop behavior.
    pub fn new(model: ri_ai::Model) -> Self {
        Self {
            initial_state: AgentState::new(model.clone()),
            loop_config: AgentLoopConfig::new(model),
            stream_fn: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
        }
    }

    /// Sets an explicit stream function.
    pub fn with_stream_fn(mut self, stream_fn: Arc<dyn StreamFn>) -> Self {
        self.stream_fn = Some(stream_fn);
        self
    }
}

/// Opaque id returned when an event listener is registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListenerId(u64);

/// Ordered asynchronous agent event listener.
#[async_trait]
pub trait AgentEventListener<M>: Send + Sync + 'static {
    /// Handles one event and the active run's cancellation token.
    async fn on_event(&self, event: AgentEvent<M>, cancellation: CancellationToken);
}

#[async_trait]
impl<M, F, Fut> AgentEventListener<M> for F
where
    M: Send + 'static,
    F: Fn(AgentEvent<M>, CancellationToken) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send,
{
    async fn on_event(&self, event: AgentEvent<M>, cancellation: CancellationToken) {
        self(event, cancellation).await;
    }
}

struct AgentSettings<M> {
    loop_config: AgentLoopConfig<M>,
    stream_fn: Option<Arc<dyn StreamFn>>,
}

struct AgentControl<M> {
    active: Option<CancellationToken>,
    steering: PendingMessageQueue<M>,
    follow_up: PendingMessageQueue<M>,
    listeners: IndexMap<ListenerId, Arc<dyn AgentEventListener<M>>>,
    next_listener_id: u64,
}

struct AgentInner<M> {
    state: RwLock<AgentState<M>>,
    settings: RwLock<AgentSettings<M>>,
    control: Mutex<AgentControl<M>>,
    event_barrier: AsyncMutex<()>,
    busy: watch::Sender<bool>,
}

/// Stateful, cloneable owner of a transcript and its agent runtime.
pub struct Agent<M = ri_ai::Message> {
    inner: Arc<AgentInner<M>>,
}

impl<M> Clone for Agent<M> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<M> std::fmt::Debug for Agent<M>
where
    M: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("state", &*self.inner.state.read())
            .finish_non_exhaustive()
    }
}

impl<M: AgentMessage> Agent<M> {
    /// Creates an agent with an explicit stream function.
    pub fn new<S>(model: ri_ai::Model, stream_fn: S) -> Self
    where
        S: StreamFn,
    {
        Self::from_options(AgentOptions::new(model).with_stream_fn(Arc::new(stream_fn)))
    }

    /// Creates an agent from an already type-erased stream function.
    pub fn with_stream_fn(model: ri_ai::Model, stream_fn: Arc<dyn StreamFn>) -> Self {
        Self::from_options(AgentOptions::new(model).with_stream_fn(stream_fn))
    }

    /// Creates an agent from complete options.
    pub fn from_options(options: AgentOptions<M>) -> Self {
        let (busy, _) = watch::channel(false);
        Self {
            inner: Arc::new(AgentInner {
                state: RwLock::new(options.initial_state),
                settings: RwLock::new(AgentSettings {
                    loop_config: options.loop_config,
                    stream_fn: options.stream_fn,
                }),
                control: Mutex::new(AgentControl {
                    active: None,
                    steering: PendingMessageQueue::new(options.steering_mode),
                    follow_up: PendingMessageQueue::new(options.follow_up_mode),
                    listeners: IndexMap::new(),
                    next_listener_id: 0,
                }),
                event_barrier: AsyncMutex::new(()),
                busy,
            }),
        }
    }

    /// Returns a cloned public-state snapshot.
    pub fn state(&self) -> AgentState<M> {
        self.inner.state.read().clone()
    }

    /// Mutates future-run state under a short exclusive lock.
    pub fn update_state(&self, update: impl FnOnce(&mut AgentState<M>)) {
        update(&mut self.inner.state.write());
    }

    /// Replaces the top-level transcript vector.
    pub fn set_messages(&self, messages: Vec<M>) {
        self.inner.state.write().messages = messages;
    }

    /// Replaces the top-level tool vector.
    pub fn set_tools(&self, tools: Vec<Arc<dyn Tool>>) {
        self.inner.state.write().tools = tools;
    }

    /// Returns a clone of the loop configuration template.
    pub fn loop_config(&self) -> AgentLoopConfig<M> {
        self.inner.settings.read().loop_config.clone()
    }

    /// Mutates configuration used by future runs.
    pub fn update_loop_config(&self, update: impl FnOnce(&mut AgentLoopConfig<M>)) {
        update(&mut self.inner.settings.write().loop_config);
    }

    /// Replaces the stream function used by future runs.
    pub fn set_stream_fn(&self, stream_fn: Option<Arc<dyn StreamFn>>) {
        self.inner.settings.write().stream_fn = stream_fn;
    }

    /// Registers an awaited event listener.
    ///
    /// Listener snapshots are invoked in registration order for each event.
    pub fn subscribe<L>(&self, listener: L) -> ListenerId
    where
        L: AgentEventListener<M>,
    {
        let mut control = self.inner.control.lock();
        let id = ListenerId(control.next_listener_id);
        control.next_listener_id = control.next_listener_id.wrapping_add(1);
        control.listeners.insert(id, Arc::new(listener));
        id
    }

    /// Removes a registered listener.
    pub fn unsubscribe(&self, id: ListenerId) -> bool {
        self.inner
            .control
            .lock()
            .listeners
            .shift_remove(&id)
            .is_some()
    }

    /// Queues a message for the next post-turn steering safe point.
    pub fn steer(&self, message: M) {
        self.inner
            .control
            .lock()
            .steering
            .messages
            .push_back(message);
    }

    /// Queues a message for when the run would otherwise stop.
    pub fn follow_up(&self, message: M) {
        self.inner
            .control
            .lock()
            .follow_up
            .messages
            .push_back(message);
    }

    /// Alias for [`Self::follow_up`].
    pub fn follow(&self, message: M) {
        self.follow_up(message);
    }

    /// Removes queued steering messages.
    pub fn clear_steering_queue(&self) {
        self.inner.control.lock().steering.messages.clear();
    }

    /// Removes queued follow-up messages.
    pub fn clear_follow_up_queue(&self) {
        self.inner.control.lock().follow_up.messages.clear();
    }

    /// Removes all queued messages.
    pub fn clear_all_queues(&self) {
        let mut control = self.inner.control.lock();
        control.steering.messages.clear();
        control.follow_up.messages.clear();
    }

    /// Whether either queue contains a message.
    pub fn has_queued_messages(&self) -> bool {
        let control = self.inner.control.lock();
        !control.steering.messages.is_empty() || !control.follow_up.messages.is_empty()
    }

    /// Changes the steering delivery policy.
    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.inner.control.lock().steering.mode = mode;
    }

    /// Returns the steering delivery policy.
    pub fn steering_mode(&self) -> QueueMode {
        self.inner.control.lock().steering.mode
    }

    /// Changes the follow-up delivery policy.
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.inner.control.lock().follow_up.mode = mode;
    }

    /// Alias for [`Self::set_follow_up_mode`].
    pub fn set_follow_mode(&self, mode: QueueMode) {
        self.set_follow_up_mode(mode);
    }

    /// Returns the follow-up delivery policy.
    pub fn follow_up_mode(&self) -> QueueMode {
        self.inner.control.lock().follow_up.mode
    }

    /// Alias for [`Self::follow_up_mode`].
    pub fn follow_mode(&self) -> QueueMode {
        self.follow_up_mode()
    }

    /// Returns the active cancellation token, if any.
    pub fn cancellation_token(&self) -> Option<CancellationToken> {
        self.inner.control.lock().active.clone()
    }

    /// Cancels the active run. Calling this while idle is a no-op.
    pub fn abort(&self) {
        if let Some(cancellation) = self.cancellation_token() {
            cancellation.cancel();
        }
    }

    /// Resolves when the current run and its final listeners have settled.
    pub async fn wait_for_idle(&self) {
        let mut busy = self.inner.busy.subscribe();
        loop {
            if !*busy.borrow() {
                return;
            }
            if busy.changed().await.is_err() {
                return;
            }
        }
    }

    /// Clears transcript, runtime state, and both queues while idle.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Busy`] while a run is active.
    pub fn reset(&self) -> Result<(), AgentError> {
        if self.inner.control.lock().active.is_some() {
            return Err(AgentError::Busy);
        }
        {
            let mut state = self.inner.state.write();
            state.messages.clear();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message = None;
        }
        self.clear_all_queues();
        Ok(())
    }

    /// Starts a prompt from one message, a batch, or a supported convenience
    /// conversion such as `&str` for `Agent<ri_ai::Message>`.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Busy`] if another run is active, a stream
    /// configuration error before startup, or an unexpected runtime failure.
    pub async fn prompt<P>(&self, prompt: P) -> Result<(), AgentError>
    where
        P: Into<Prompt<M>>,
    {
        let messages = prompt.into().into_messages();
        let cancellation = self.begin_run()?;
        let stream_fn = match self.resolve_stream_fn() {
            Ok(stream_fn) => stream_fn,
            Err(error) => {
                self.finish_run();
                return Err(error);
            }
        };
        let agent = self.clone();
        let task = tokio::spawn(async move {
            let run = AssertUnwindSafe(agent.run_prompt_messages(
                messages,
                false,
                cancellation,
                stream_fn,
            ))
            .catch_unwind()
            .await;
            agent.finish_run();
            run.map_err(|panic| {
                AgentError::Callback(format!(
                    "agent run panicked: {}",
                    panic_payload(panic.as_ref())
                ))
            })?
        });
        task.await
            .map_err(|error| AgentError::Callback(format!("agent run task failed: {error}")))?
    }

    /// Continues from the current transcript.
    ///
    /// When the transcript ends in an assistant message, one queued steering
    /// message is preferred over one queued follow-up message.
    ///
    /// # Errors
    ///
    /// Returns an invalid-continuation error, [`AgentError::Busy`], a stream
    /// configuration error, or an unexpected runtime failure.
    pub async fn continue_run(&self) -> Result<(), AgentError> {
        let cancellation = self.begin_run()?;
        let stream_fn = match self.resolve_stream_fn() {
            Ok(stream_fn) => stream_fn,
            Err(error) => {
                self.finish_run();
                return Err(error);
            }
        };
        let agent = self.clone();
        let task = tokio::spawn(async move {
            let run = AssertUnwindSafe(agent.run_continuation(cancellation, stream_fn))
                .catch_unwind()
                .await;
            agent.finish_run();
            run.map_err(|panic| {
                AgentError::Callback(format!(
                    "agent run panicked: {}",
                    panic_payload(panic.as_ref())
                ))
            })?
        });
        task.await
            .map_err(|error| AgentError::Callback(format!("agent run task failed: {error}")))?
    }

    /// Alias for [`Self::continue_run`] using Rust's raw-identifier syntax.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::continue_run`].
    pub async fn r#continue(&self) -> Result<(), AgentError> {
        self.continue_run().await
    }

    fn begin_run(&self) -> Result<CancellationToken, AgentError> {
        let cancellation = CancellationToken::new();
        {
            let mut control = self.inner.control.lock();
            if control.active.is_some() {
                return Err(AgentError::Busy);
            }
            control.active = Some(cancellation.clone());
        }
        {
            let mut state = self.inner.state.write();
            state.is_streaming = true;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message = None;
        }
        self.inner.busy.send_replace(true);
        Ok(cancellation)
    }

    fn finish_run(&self) {
        {
            let mut state = self.inner.state.write();
            state.is_streaming = false;
            state.streaming_message = None;
            state.pending_tool_calls.clear();
        }
        self.inner.control.lock().active = None;
        self.inner.busy.send_replace(false);
    }

    fn resolve_stream_fn(&self) -> Result<Arc<dyn StreamFn>, AgentError> {
        self.inner
            .settings
            .read()
            .stream_fn
            .clone()
            .map_or_else(default_stream_fn, Ok)
    }

    fn context_snapshot(&self) -> AgentContext<M> {
        let state = self.inner.state.read();
        AgentContext {
            system_prompt: state.system_prompt.clone(),
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }
    }

    fn configured_loop(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig<M> {
        let mut config = self.inner.settings.read().loop_config.clone();
        {
            let state = self.inner.state.read();
            config.model = state.model.clone();
            config.thinking_level = state.thinking_level;
        }
        let skip = Arc::new(AtomicBool::new(skip_initial_steering_poll));
        let steering_agent = self.clone();
        config = config.with_steering_messages(move || {
            let agent = steering_agent.clone();
            let skip = Arc::clone(&skip);
            async move {
                if skip.swap(false, Ordering::AcqRel) {
                    Ok(Vec::new())
                } else {
                    Ok(agent.inner.control.lock().steering.drain())
                }
            }
        });
        let follow_agent = self.clone();
        config.with_follow_up_messages(move || {
            let agent = follow_agent.clone();
            async move { Ok(agent.inner.control.lock().follow_up.drain()) }
        })
    }

    async fn run_prompt_messages(
        &self,
        messages: Vec<M>,
        skip_initial_steering_poll: bool,
        cancellation: CancellationToken,
        stream_fn: Arc<dyn StreamFn>,
    ) -> Result<(), AgentError> {
        run_agent_loop(
            messages,
            self.context_snapshot(),
            self.configured_loop(skip_initial_steering_poll),
            cancellation.clone(),
            Arc::new(StatefulEventSink {
                agent: self.clone(),
                cancellation,
            }),
            stream_fn,
        )
        .await
        .map(|_| ())
    }

    async fn run_continuation(
        &self,
        cancellation: CancellationToken,
        stream_fn: Arc<dyn StreamFn>,
    ) -> Result<(), AgentError> {
        let context = self.context_snapshot();
        let Some(last) = context.messages.last() else {
            return Err(AgentError::EmptyContinuation);
        };
        if matches!(last.as_llm_message(), Some(ri_ai::Message::Assistant(_))) {
            let steering = self.inner.control.lock().steering.drain();
            if !steering.is_empty() {
                return self
                    .run_prompt_messages(steering, true, cancellation, stream_fn)
                    .await;
            }
            let follow_ups = self.inner.control.lock().follow_up.drain();
            if !follow_ups.is_empty() {
                return self
                    .run_prompt_messages(follow_ups, false, cancellation, stream_fn)
                    .await;
            }
            return Err(AgentError::AssistantContinuation);
        }

        run_agent_loop_continue(
            context,
            self.configured_loop(false),
            cancellation.clone(),
            Arc::new(StatefulEventSink {
                agent: self.clone(),
                cancellation,
            }),
            stream_fn,
        )
        .await
        .map(|_| ())
    }

    async fn process_event(&self, event: AgentEvent<M>, cancellation: CancellationToken) {
        let _barrier = self.inner.event_barrier.lock().await;
        {
            let mut state = self.inner.state.write();
            match &event {
                AgentEvent::MessageStart { message }
                | AgentEvent::MessageUpdate { message, .. } => {
                    state.streaming_message = Some(message.clone());
                }
                AgentEvent::MessageEnd { message } => {
                    state.streaming_message = None;
                    state.messages.push(message.clone());
                }
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                    state.pending_tool_calls.insert(tool_call_id.clone());
                }
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                    state.pending_tool_calls.remove(tool_call_id);
                }
                AgentEvent::TurnEnd { message, .. } => {
                    state.error_message =
                        message.as_llm_message().and_then(|message| match message {
                            ri_ai::Message::Assistant(message) => message.error_message.clone(),
                            _ => None,
                        });
                }
                AgentEvent::AgentEnd { .. } => {
                    state.streaming_message = None;
                }
                AgentEvent::AgentStart
                | AgentEvent::TurnStart
                | AgentEvent::ToolExecutionUpdate { .. } => {}
            }
        }
        let listeners = self
            .inner
            .control
            .lock()
            .listeners
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for listener in listeners {
            listener.on_event(event.clone(), cancellation.clone()).await;
        }
    }
}

fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

#[derive(Clone)]
struct StatefulEventSink<M> {
    agent: Agent<M>,
    cancellation: CancellationToken,
}

impl<M> std::fmt::Debug for StatefulEventSink<M>
where
    M: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StatefulEventSink")
            .field("agent", &self.agent)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl<M: AgentMessage> AgentEventSink<M> for StatefulEventSink<M> {
    async fn emit(&self, event: AgentEvent<M>) -> Result<(), AgentError> {
        self.agent
            .process_event(event, self.cancellation.clone())
            .await;
        Ok(())
    }
}
