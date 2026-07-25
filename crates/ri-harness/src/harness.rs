//! Unified high-level session, prompt, retry, queue, and compaction lifecycle.

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::FutureExt;
use ri_ai::{
    AssistantMessage, CacheRetention, Context, Message, StopReason, ThinkingLevel, UserContent,
    UserMessage, classify_context_overflow, message::InputContent, now_millis,
};
use ri_session::{Session, SessionEntry};
use serde_json::Value;
use tokio::sync::{Mutex, RwLock, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::backend::{
    BeforeAgentStart, BeforeCompactionResult, BeforeNavigation, BeforeNavigationResult,
    HarnessBackend, HarnessHooks, HarnessObserver, HookContext, InputAction, InputEvent,
};
use crate::compaction::{
    CompactionPreparation, CompactionResult, SUMMARIZATION_SYSTEM_PROMPT, TURN_PREFIX_PROMPT,
    append_file_lists, branch_request_text, collect_abandoned_branch, combine_usage,
    compaction_request_text, context_tokens, estimate_context_tokens, estimate_message_tokens,
    prepare_branch, prepare_compaction, session_usage, should_compact,
};
use crate::error::{BackendError, BackendErrorKind, Error, Result};
use crate::projection::{message_value, project_session, user_text};
use crate::prompt::expand_resources;
use crate::types::{
    CompactionReason, HarnessConfig, HarnessEvent, HarnessStatus, NavigateOptions, NavigateResult,
    Phase, PromptOptions, PromptOutcome, QueueLengths, QueueMode, RequestOptions, Resources,
    RetryOperation, SessionWrite, StreamingBehavior, SummaryKind, SummaryRequest, SummaryResponse,
    TurnOutput, TurnRequest, TurnSnapshot,
};

/// Shared unified high-level harness.
#[derive(Clone)]
pub struct Harness {
    inner: Arc<Inner>,
}

struct CompactionExecution<'a> {
    operation: u64,
    reason: CompactionReason,
    will_retry: bool,
    custom_instructions: Option<&'a str>,
    cancellation: CancellationToken,
    session: Session,
    config: HarnessConfig,
    preparation: CompactionPreparation,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Harness").finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct Inner {
    backend: Arc<dyn HarnessBackend>,
    hooks: Option<Arc<dyn HarnessHooks>>,
    control: Mutex<Control>,
    observers: RwLock<Vec<(u64, Arc<dyn HarnessObserver>)>>,
    observer_ids: AtomicU64,
    lifecycle_tx: watch::Sender<LifecycleMarker>,
}

#[derive(Clone, Debug)]
struct SessionBinding {
    session: Session,
    id: Arc<str>,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LifecycleMarker {
    phase: Phase,
    active: Option<u64>,
    settled: u64,
    generation: u64,
}

#[derive(Debug)]
struct Lifecycle {
    phase: Phase,
    root_phase: Phase,
    next_operation: u64,
    active: Option<u64>,
    settled: u64,
}

#[derive(Clone, Debug)]
struct QueueItem {
    id: u64,
    message: Message,
}

#[derive(Debug, Default)]
struct Queues {
    next_id: u64,
    steer: VecDeque<QueueItem>,
    follow_up: VecDeque<QueueItem>,
    next_turn: VecDeque<QueueItem>,
}

impl Queues {
    fn lengths(&self) -> QueueLengths {
        QueueLengths {
            steer: self.steer.len(),
            follow_up: self.follow_up.len(),
            next_turn: self.next_turn.len(),
        }
    }

    fn push(&mut self, queue: Queue, message: Message) -> u64 {
        self.next_id = self.next_id.saturating_add(1);
        let id = self.next_id;
        let item = QueueItem { id, message };
        self.get_mut(queue).push_back(item);
        id
    }

    fn get_mut(&mut self, queue: Queue) -> &mut VecDeque<QueueItem> {
        match queue {
            Queue::Steer => &mut self.steer,
            Queue::FollowUp => &mut self.follow_up,
            Queue::NextTurn => &mut self.next_turn,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Queue {
    Steer,
    FollowUp,
    NextTurn,
}

#[derive(Debug)]
enum AgentRetryPreparation {
    Retry,
    Stop {
        attempt: Option<u32>,
        final_error: Option<String>,
    },
}

#[derive(Debug)]
struct Control {
    lifecycle: Lifecycle,
    binding: SessionBinding,
    config: HarnessConfig,
    queues: Queues,
    pending_writes: VecDeque<SessionWrite>,
    cancellation: Option<CancellationToken>,
    retry_cancellation: Option<CancellationToken>,
}

impl Harness {
    /// Creates a harness around an explicit session and runtime backend.
    ///
    /// No provider or storage fallback is constructed. Existing branch-scoped
    /// model, reasoning, and active-tool state is restored and validated.
    ///
    /// # Errors
    /// Returns an error when configuration, model preflight, session state, or hook binding fails.
    pub async fn new(
        session: Session,
        mut config: HarnessConfig,
        backend: Arc<dyn HarnessBackend>,
        hooks: Option<Arc<dyn HarnessHooks>>,
    ) -> Result<Self> {
        validate_config(&config)?;
        backend
            .preflight(&config.model)
            .await
            .map_err(model_backend_error)?;
        reconcile_session_config(&session, &mut config, true).await?;
        let metadata = session.metadata().await?;
        let binding = SessionBinding {
            session,
            id: metadata.id.into(),
            generation: 1,
        };
        backend
            .bind_session(&binding.id, binding.generation)
            .await
            .map_err(runtime_backend_error)?;
        let hook_context = HookContext {
            session_id: binding.id.clone(),
            generation: binding.generation,
        };
        if let Some(hooks) = &hooks {
            hooks.bind_session(&hook_context).await?;
        }
        let marker = LifecycleMarker {
            phase: Phase::Idle,
            active: None,
            settled: 0,
            generation: binding.generation,
        };
        let (lifecycle_tx, _) = watch::channel(marker);
        Ok(Self {
            inner: Arc::new(Inner {
                backend,
                hooks,
                control: Mutex::new(Control {
                    lifecycle: Lifecycle {
                        phase: Phase::Idle,
                        root_phase: Phase::Idle,
                        next_operation: 1,
                        active: None,
                        settled: 0,
                    },
                    binding,
                    config,
                    queues: Queues::default(),
                    pending_writes: VecDeque::new(),
                    cancellation: None,
                    retry_cancellation: None,
                }),
                observers: RwLock::new(Vec::new()),
                observer_ids: AtomicU64::new(1),
                lifecycle_tx,
            }),
        })
    }

    /// Runs the canonical prompt pipeline.
    ///
    /// # Errors
    /// Returns an error when hooks, resources, persistence, compaction, or agent execution fails.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        options: PromptOptions,
    ) -> Result<PromptOutcome> {
        let mut text = text.into();
        let mut images = options.images.clone();
        let hook_context = self.hook_context().await;
        if options.expand_resources
            && text.starts_with('/')
            && let Some(hooks) = &self.inner.hooks
            && hooks.command(&hook_context, &text).await?
        {
            return Ok(PromptOutcome::Handled);
        }
        if let Some(hooks) = &self.inner.hooks {
            match hooks
                .input(
                    &hook_context,
                    InputEvent {
                        text: text.clone(),
                        images: images.clone(),
                        source: options.source,
                        streaming_behavior: options.streaming_behavior,
                    },
                )
                .await?
            {
                InputAction::Continue => {}
                InputAction::Transform {
                    text: replacement,
                    images: replacement_images,
                } => {
                    text = replacement;
                    if let Some(replacement) = replacement_images {
                        images = replacement;
                    }
                }
                InputAction::Handled => return Ok(PromptOutcome::Handled),
            }
        }
        if options.expand_resources {
            let resources = {
                let control = self.inner.control.lock().await;
                control.config.resources.clone()
            };
            text = self.expand_resources(&text, &resources).await?;
        }

        match self.begin_operation(Phase::Turn).await {
            Ok(operation) => {
                let harness = self.clone();
                self.run_owned_operation(operation, async move {
                    harness.run_prompt(operation, text, images).await
                })
                .await
                .map(|message| PromptOutcome::Completed(Box::new(message)))
            }
            Err(Error::Busy { .. }) => {
                let behavior = options.streaming_behavior.ok_or_else(|| Error::Busy {
                    phase: self.inner.lifecycle_tx.borrow().phase.as_str(),
                })?;
                match behavior {
                    StreamingBehavior::Steer => {
                        self.enqueue(Queue::Steer, create_user_message(text, images))
                            .await?;
                    }
                    StreamingBehavior::FollowUp => {
                        self.enqueue(Queue::FollowUp, create_user_message(text, images))
                            .await?;
                    }
                }
                Ok(PromptOutcome::Queued(behavior))
            }
            Err(error) => Err(error),
        }
    }

    /// Convenience prompt with ordinary resource expansion.
    ///
    /// # Errors
    /// Returns an error when the canonical prompt pipeline fails.
    pub async fn prompt_text(&self, text: impl Into<String>) -> Result<PromptOutcome> {
        self.prompt(text, PromptOptions::interactive()).await
    }

    /// Runs plain text directly and returns the terminal assistant message.
    ///
    /// This typed entry point skips command and input hooks, expands registered
    /// skills and prompt templates, and never queues a busy prompt. It is
    /// intended for SDK-owned agents whose input is controlled by the caller.
    ///
    /// # Errors
    /// Returns an error when the harness is busy or model, tool, persistence,
    /// compaction, or agent execution fails.
    pub async fn prompt_message(&self, text: impl Into<String>) -> Result<AssistantMessage> {
        let resources = self.inner.control.lock().await.config.resources.clone();
        let text = self.expand_resources(&text.into(), &resources).await?;
        let operation = self.begin_operation(Phase::Turn).await?;
        let harness = self.clone();
        self.run_owned_operation(operation, async move {
            harness.run_prompt(operation, text, Vec::new()).await
        })
        .await
    }

    /// Queues a steering message for the next post-turn safe point.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn steer(&self, text: impl Into<String>) -> Result<()> {
        self.steer_with_images(text, Vec::new()).await
    }

    /// Queues steering text and images for the next post-turn safe point.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn steer_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<ri_ai::ImageContent>,
    ) -> Result<()> {
        self.enqueue(Queue::Steer, create_user_message(text.into(), images))
            .await
    }

    /// Queues a follow-up message for after the current run would otherwise stop.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn follow_up(&self, text: impl Into<String>) -> Result<()> {
        self.follow_up_with_images(text, Vec::new()).await
    }

    /// Queues follow-up text and images for after the current run would otherwise stop.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn follow_up_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<ri_ai::ImageContent>,
    ) -> Result<()> {
        self.enqueue(Queue::FollowUp, create_user_message(text.into(), images))
            .await
    }

    /// Queues context for the next user-initiated prompt. This queue survives
    /// abort and is accepted while idle.
    ///
    /// # Errors
    /// Returns an error when queue event delivery fails.
    pub async fn next_turn(&self, message: Message) -> Result<()> {
        self.enqueue(Queue::NextTurn, message).await
    }

    /// Requests cancellation, clears active-run queues, and waits through the
    /// precise settlement barrier. Next-turn messages are preserved.
    ///
    /// # Errors
    /// Returns an error when queue update observers fail.
    pub async fn abort(&self) -> Result<QueueLengths> {
        let (target, cleared) = {
            let mut control = self.inner.control.lock().await;
            let target = control.lifecycle.active;
            if let Some(cancellation) = &control.cancellation {
                cancellation.cancel();
            }
            control.queues.steer.clear();
            control.queues.follow_up.clear();
            (target, control.queues.lengths())
        };
        let event_result = self.emit_event(HarnessEvent::QueueUpdated(cleared)).await;
        if let Some(target) = target {
            self.wait_for_operation(target).await;
        }
        event_result?;
        Ok(cleared)
    }

    /// Cancels only the active agent retry backoff, preserving the enclosing
    /// prompt operation and its queues.
    ///
    /// Returns whether a retry delay was active.
    pub async fn abort_retry(&self) -> bool {
        let cancellation = self.inner.control.lock().await.retry_cancellation.take();
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
            true
        } else {
            false
        }
    }

    /// Waits until the operation active at call time has fully settled,
    /// including queued continuations, awaited callbacks, and writes accepted by
    /// those callbacks.
    pub async fn wait_settled(&self) {
        let target = self.inner.lifecycle_tx.borrow().active;
        if let Some(target) = target {
            self.wait_for_operation(target).await;
        }
    }

    /// Manually compacts the active branch.
    ///
    /// # Errors
    /// Returns an error when compaction cannot start, summarize, persist, or settle.
    pub async fn compact(&self, custom_instructions: Option<String>) -> Result<CompactionResult> {
        let operation = self.begin_operation(Phase::Compaction).await?;
        let harness = self.clone();
        self.run_owned_operation(operation, async move {
            let cancellation = harness.operation_cancellation(operation).await?;
            let model = {
                let control = harness.inner.control.lock().await;
                control.config.model.clone()
            };
            harness
                .inner
                .backend
                .preflight(&model)
                .await
                .map_err(model_backend_error)?;
            harness
                .run_compaction(
                    operation,
                    CompactionReason::Manual,
                    false,
                    custom_instructions.as_deref(),
                    cancellation,
                    Phase::Compaction,
                )
                .await?
                .ok_or_else(|| Error::Compaction("nothing to compact".to_owned()))
        })
        .await
    }

    /// Navigates the session tree and optionally summarizes the abandoned branch.
    ///
    /// # Errors
    /// Returns an error when navigation, branch summarization, persistence, or settlement fails.
    pub async fn navigate(
        &self,
        target_id: impl Into<String>,
        options: NavigateOptions,
    ) -> Result<NavigateResult> {
        let operation = self.begin_operation(Phase::BranchSummary).await?;
        let target_id = target_id.into();
        let harness = self.clone();
        self.run_owned_operation(operation, async move {
            let cancellation = harness.operation_cancellation(operation).await?;
            harness
                .run_navigation(operation, target_id, options, cancellation)
                .await
        })
        .await
    }

    /// Replaces the bound session using shutdown → invalidate → bind ordering.
    ///
    /// The caller supplies a fully resolved configuration for the destination;
    /// model mismatches are rejected rather than silently falling back.
    ///
    /// # Errors
    /// Returns an error when validation, lifecycle hooks, backend rebinding, or persistence fails.
    pub async fn replace_session(&self, session: Session, mut config: HarnessConfig) -> Result<()> {
        validate_config(&config)?;
        self.inner
            .backend
            .preflight(&config.model)
            .await
            .map_err(model_backend_error)?;
        reconcile_session_config(&session, &mut config, true).await?;
        let metadata = session.metadata().await?;
        let operation = self.begin_operation(Phase::ReplacingSession).await?;
        let harness = self.clone();
        self.run_owned_operation(operation, async move {
            let old = {
                let control = harness.inner.control.lock().await;
                control.binding.clone()
            };
            harness
                .emit_event(HarnessEvent::SessionReplacing {
                    old_session_id: old.id.to_string(),
                })
                .await?;
            let old_context = HookContext {
                session_id: old.id.clone(),
                generation: old.generation,
            };
            if let Some(hooks) = &harness.inner.hooks {
                hooks.unbind_session(&old_context).await?;
            }
            harness
                .inner
                .backend
                .unbind_session(&old.id)
                .await
                .map_err(runtime_backend_error)?;

            let generation = old.generation.saturating_add(1);
            let new_binding = SessionBinding {
                session,
                id: metadata.id.into(),
                generation,
            };
            {
                let mut control = harness.inner.control.lock().await;
                control.binding = new_binding.clone();
                control.config = config;
                control.queues = Queues::default();
                control.pending_writes.clear();
            }
            harness.publish_lifecycle().await;
            harness
                .inner
                .backend
                .bind_session(&new_binding.id, generation)
                .await
                .map_err(runtime_backend_error)?;
            let context = HookContext {
                session_id: new_binding.id.clone(),
                generation,
            };
            if let Some(hooks) = &harness.inner.hooks {
                hooks.bind_session(&context).await?;
            }
            harness
                .emit_event(HarnessEvent::SessionReplaced {
                    session_id: new_binding.id.to_string(),
                    generation,
                })
                .await
        })
        .await
    }

    /// Returns a clone of the currently bound session handle.
    pub async fn session(&self) -> Session {
        self.inner.control.lock().await.binding.session.clone()
    }

    /// Returns the latest mutable configuration, not an in-flight turn snapshot.
    pub async fn config(&self) -> HarnessConfig {
        self.inner.control.lock().await.config.clone()
    }

    /// Returns a consistent lifecycle diagnostic snapshot.
    pub async fn status(&self) -> HarnessStatus {
        let control = self.inner.control.lock().await;
        HarnessStatus {
            phase: control.lifecycle.phase,
            operation: control.lifecycle.active,
            settled_operation: control.lifecycle.settled,
            generation: control.binding.generation,
            queues: control.queues.lengths(),
        }
    }

    /// Validates that a captured hook context still belongs to this session.
    ///
    /// # Errors
    /// Returns an error when the context belongs to a replaced session generation.
    pub async fn validate_hook_context(&self, context: &HookContext) -> Result<()> {
        let control = self.inner.control.lock().await;
        if context.generation != control.binding.generation
            || context.session_id != control.binding.id
        {
            return Err(Error::InvalidState(
                "captured context is stale after session replacement".to_owned(),
            ));
        }
        Ok(())
    }

    /// Adds an awaited observational subscriber and returns its id.
    pub async fn add_observer(&self, observer: Arc<dyn HarnessObserver>) -> u64 {
        let id = self.inner.observer_ids.fetch_add(1, Ordering::Relaxed);
        self.inner.observers.write().await.push((id, observer));
        id
    }

    /// Removes an observer.
    pub async fn remove_observer(&self, id: u64) -> bool {
        let mut observers = self.inner.observers.write().await;
        let before = observers.len();
        observers.retain(|(candidate, _)| *candidate != id);
        observers.len() != before
    }

    /// Appends or queues a session write with deterministic save-point ordering.
    ///
    /// # Errors
    /// Returns an error when an idle write cannot be persisted.
    pub async fn write_session(&self, write: SessionWrite) -> Result<()> {
        let mut control = self.inner.control.lock().await;
        if control.lifecycle.phase == Phase::Idle {
            persist_write(&control.binding.session, &write).await?;
        } else {
            control.pending_writes.push_back(write);
        }
        Ok(())
    }

    /// Changes the model for future snapshots and persists the branch-scoped
    /// selection before committing it while idle.
    ///
    /// # Errors
    /// Returns an error when model preflight or idle persistence fails.
    pub async fn set_model(&self, model: Arc<ri_ai::Model>) -> Result<()> {
        self.inner
            .backend
            .preflight(&model)
            .await
            .map_err(model_backend_error)?;
        let write = SessionWrite::Model {
            provider: model.provider.clone(),
            model_id: model.id.clone(),
        };
        let mut control = self.inner.control.lock().await;
        if control.lifecycle.phase == Phase::Idle {
            persist_write(&control.binding.session, &write).await?;
        } else {
            control.pending_writes.push_back(write);
        }
        control.config.model = model;
        Ok(())
    }

    /// Changes reasoning level for future snapshots.
    ///
    /// # Errors
    /// Returns an error when the branch-scoped update cannot be persisted.
    pub async fn set_thinking_level(&self, level: ThinkingLevel) -> Result<()> {
        let write = SessionWrite::Thinking(level);
        let mut control = self.inner.control.lock().await;
        if control.lifecycle.phase == Phase::Idle {
            persist_write(&control.binding.session, &write).await?;
        } else {
            control.pending_writes.push_back(write);
        }
        control.config.thinking_level = level;
        Ok(())
    }

    /// Replaces active tools for future snapshots after validating names.
    ///
    /// # Errors
    /// Returns an error for unknown tool names or when persistence fails.
    pub async fn set_active_tools(&self, names: Vec<String>) -> Result<()> {
        let mut control = self.inner.control.lock().await;
        validate_active_tools(&control.config.tools, &names)?;
        let write = SessionWrite::ActiveTools(names.clone());
        if control.lifecycle.phase == Phase::Idle {
            persist_write(&control.binding.session, &write).await?;
        } else {
            control.pending_writes.push_back(write);
        }
        control.config.active_tool_names = names.into();
        Ok(())
    }

    /// Replaces resources for future snapshots.
    pub async fn set_resources(&self, resources: Resources) {
        self.inner.control.lock().await.config.resources = resources;
    }

    /// Replaces the base system prompt for future snapshots.
    pub async fn set_system_prompt(&self, prompt: impl Into<String>) {
        self.inner.control.lock().await.config.system_prompt = prompt.into();
    }

    /// Replaces provider options for future snapshots.
    pub async fn set_request_options(&self, options: RequestOptions) {
        self.inner.control.lock().await.config.request_options = options;
    }

    /// Changes live queue drain modes.
    pub async fn set_queue_modes(&self, steering: QueueMode, follow_up: QueueMode) {
        let mut control = self.inner.control.lock().await;
        control.config.steering_mode = steering;
        control.config.follow_up_mode = follow_up;
    }

    /// Enables or disables automatic context compaction for future checks.
    pub async fn set_auto_compaction_enabled(&self, enabled: bool) {
        self.inner.control.lock().await.config.compaction.enabled = enabled;
    }

    /// Enables or disables automatic retry for future retry decisions.
    pub async fn set_auto_retry_enabled(&self, enabled: bool) {
        self.inner.control.lock().await.config.retry.enabled = enabled;
    }

    async fn run_prompt(
        &self,
        operation: u64,
        text: String,
        images: Vec<ri_ai::ImageContent>,
    ) -> Result<AssistantMessage> {
        let cancellation = self.operation_cancellation(operation).await?;
        let model = {
            let control = self.inner.control.lock().await;
            control.config.model.clone()
        };
        self.inner
            .backend
            .preflight(&model)
            .await
            .map_err(model_backend_error)?;

        if let Some((reason, will_retry)) = self.compaction_check(false).await? {
            let _ = self
                .run_compaction(
                    operation,
                    reason,
                    will_retry,
                    None,
                    cancellation.child_token(),
                    Phase::Turn,
                )
                .await?;
        }

        self.emit_event(HarnessEvent::PromptAccepted { operation })
            .await?;
        let mut snapshot = self.create_turn_snapshot(None).await?;
        let mut initial = self.drain_queue(Queue::NextTurn, QueueMode::All).await?;
        let user = create_user_message(text.clone(), images.clone());
        initial.push(user);
        let hook_context = self.hook_context().await;
        if let Some(hooks) = &self.inner.hooks {
            let result = hooks
                .before_agent_start(
                    &hook_context,
                    BeforeAgentStart {
                        prompt: text,
                        images,
                        system_prompt: snapshot.system_prompt.to_string(),
                        resources: snapshot.resources.clone(),
                        model: snapshot.model.clone(),
                        thinking_level: snapshot.thinking_level,
                        active_tool_names: snapshot.active_tool_names.clone(),
                    },
                )
                .await?;
            initial.extend(result.messages);
            if let Some(system_prompt) = result.system_prompt {
                snapshot.system_prompt = system_prompt.clone().into();
                snapshot.context.system_prompt = Some(system_prompt);
            }
        }
        self.persist_messages(&initial).await?;
        snapshot.context.messages.extend(initial);

        let mut continuation = false;
        let mut retry_attempt = 0_u32;
        let mut overflow_recovery_attempted = false;
        let mut excluded_error_timestamp = None;
        let mut final_assistant = None;

        loop {
            if cancellation.is_cancelled() {
                break;
            }
            let execution = self
                .inner
                .backend
                .execute_turn(
                    TurnRequest {
                        snapshot: snapshot.clone(),
                        continuation,
                    },
                    cancellation.child_token(),
                )
                .await;
            let (output, backend_failure) = match execution {
                Ok(output) => (output, None),
                Err(error) => {
                    let message = backend_failure_message(&snapshot.model, &error);
                    (
                        TurnOutput {
                            messages: vec![Message::Assistant(message)],
                            continue_after_tools: false,
                        },
                        Some(error),
                    )
                }
            };
            if output.messages.is_empty() {
                return Err(Error::Agent(
                    "low-level turn completed without messages".to_owned(),
                ));
            }
            self.persist_messages(&output.messages).await?;
            let had_pending = self.flush_pending_writes().await?;
            self.emit_event(HarnessEvent::SavePoint {
                operation,
                had_pending_writes: had_pending,
            })
            .await?;
            let assistant = output
                .assistant()
                .cloned()
                .ok_or_else(|| Error::Agent("turn produced no assistant message".to_owned()))?;
            final_assistant = Some(assistant.clone());

            let overflow = backend_failure
                .as_ref()
                .is_some_and(|error| error.kind == BackendErrorKind::ContextOverflow)
                || same_model(&assistant, &snapshot.model)
                    && classify_context_overflow(&assistant, Some(snapshot.model.context_window))
                        .is_some();
            let retryable = backend_failure
                .as_ref()
                .is_some_and(BackendError::is_retryable)
                || assistant_retryable(&assistant);

            if overflow {
                let retry_overflow = assistant.stop_reason != StopReason::Stop;
                if retry_overflow && overflow_recovery_attempted {
                    break;
                }
                if retry_overflow {
                    overflow_recovery_attempted = true;
                    excluded_error_timestamp = Some(assistant.timestamp);
                }
                let compacted = self
                    .run_compaction(
                        operation,
                        CompactionReason::Overflow,
                        retry_overflow,
                        None,
                        cancellation.child_token(),
                        Phase::Turn,
                    )
                    .await?;
                if retry_overflow && compacted.is_some() {
                    snapshot = self.create_turn_snapshot(excluded_error_timestamp).await?;
                    continuation = true;
                    continue;
                }
            } else if retryable {
                let next_attempt = retry_attempt.saturating_add(1);
                match self
                    .prepare_agent_retry(
                        operation,
                        next_attempt,
                        backend_failure.as_ref(),
                        &assistant,
                        cancellation.child_token(),
                    )
                    .await?
                {
                    AgentRetryPreparation::Retry => {
                        retry_attempt = next_attempt;
                        excluded_error_timestamp = Some(assistant.timestamp);
                        snapshot = self.create_turn_snapshot(excluded_error_timestamp).await?;
                        continuation = true;
                        continue;
                    }
                    AgentRetryPreparation::Stop {
                        attempt,
                        final_error,
                    } => {
                        let attempt = attempt.unwrap_or(retry_attempt);
                        if attempt > 0 {
                            self.emit_event(HarnessEvent::RetryFinished {
                                operation: RetryOperation::Agent,
                                success: false,
                                attempt,
                                final_error: final_error
                                    .or_else(|| assistant.error_message.clone()),
                            })
                            .await?;
                        }
                        retry_attempt = 0;
                    }
                }
            } else if retry_attempt > 0 {
                let success = assistant.stop_reason != StopReason::Error;
                self.emit_event(HarnessEvent::RetryFinished {
                    operation: RetryOperation::Agent,
                    success,
                    attempt: retry_attempt,
                    final_error: (!success)
                        .then(|| assistant.error_message.clone())
                        .flatten(),
                })
                .await?;
                retry_attempt = 0;
            }

            if let Some((CompactionReason::Threshold, _)) = self.compaction_check(true).await? {
                let _ = self
                    .run_compaction(
                        operation,
                        CompactionReason::Threshold,
                        false,
                        None,
                        cancellation.child_token(),
                        Phase::Turn,
                    )
                    .await?;
            }

            let mut next_messages = self.drain_queue_live(Queue::Steer, true).await?;
            let mut should_continue = output.continue_after_tools || !next_messages.is_empty();
            if !should_continue {
                next_messages = self.drain_queue_live(Queue::FollowUp, false).await?;
                should_continue = !next_messages.is_empty();
            }
            if !should_continue {
                // Awaited end observers may enqueue a continuation. Re-check both
                // queues before exposing settlement.
                self.emit_event(HarnessEvent::SavePoint {
                    operation,
                    had_pending_writes: false,
                })
                .await?;
                next_messages = self.drain_queue_live(Queue::Steer, true).await?;
                if next_messages.is_empty() {
                    next_messages = self.drain_queue_live(Queue::FollowUp, false).await?;
                }
                should_continue = !next_messages.is_empty();
            }
            if !should_continue || cancellation.is_cancelled() {
                break;
            }
            if !next_messages.is_empty() {
                self.persist_messages(&next_messages).await?;
            }
            snapshot = self.create_turn_snapshot(None).await?;
            continuation = true;
        }

        final_assistant.ok_or_else(|| {
            if cancellation.is_cancelled() {
                Error::Aborted
            } else {
                Error::Agent("prompt settled without an assistant response".to_owned())
            }
        })
    }

    async fn run_compaction(
        &self,
        operation: u64,
        reason: CompactionReason,
        will_retry: bool,
        custom_instructions: Option<&str>,
        cancellation: CancellationToken,
        return_phase: Phase,
    ) -> Result<Option<CompactionResult>> {
        self.set_phase(operation, Phase::Compaction).await?;
        let (session, config) = {
            let control = self.inner.control.lock().await;
            (control.binding.session.clone(), control.config.clone())
        };
        let snapshot = session.snapshot().await?;
        let Some(preparation) = prepare_compaction(&snapshot, config.compaction)? else {
            self.set_phase(operation, return_phase).await?;
            return Ok(None);
        };
        self.emit_event(HarnessEvent::CompactionStarted { reason })
            .await?;
        let result = self
            .perform_compaction(CompactionExecution {
                operation,
                reason,
                will_retry,
                custom_instructions,
                cancellation,
                session,
                config,
                preparation,
            })
            .await;
        let (event_result, aborted, event_will_retry, error_message) = match &result {
            Ok(Some(result)) => (Some(Box::new(result.clone())), false, will_retry, None),
            Ok(None) | Err(Error::Aborted) => (None, true, false, None),
            Err(error) => (None, false, false, Some(error.to_string())),
        };
        let event_delivery = self
            .emit_event(HarnessEvent::CompactionFinished {
                reason,
                result: event_result,
                aborted,
                will_retry: event_will_retry,
                error_message,
            })
            .await;
        match result {
            Err(error) => Err(error),
            Ok(result) => {
                event_delivery?;
                self.set_phase(operation, return_phase).await?;
                Ok(result)
            }
        }
    }

    async fn perform_compaction(
        &self,
        execution: CompactionExecution<'_>,
    ) -> Result<Option<CompactionResult>> {
        let CompactionExecution {
            operation,
            reason,
            will_retry,
            custom_instructions,
            cancellation,
            session,
            config,
            preparation,
        } = execution;
        let hook_context = self.hook_context().await;
        let hook_result = if let Some(hooks) = &self.inner.hooks {
            hooks
                .before_compaction(
                    &hook_context,
                    &preparation,
                    reason,
                    will_retry,
                    custom_instructions,
                    cancellation.child_token(),
                )
                .await?
        } else {
            BeforeCompactionResult::default()
        };
        if hook_result.cancel {
            return Ok(None);
        }
        if cancellation.is_cancelled() {
            return Err(Error::Aborted);
        }

        let from_hook = hook_result.replacement.is_some();
        let (summary, first_kept_entry_id, tokens_before, retained_tail, details, usage) =
            if let Some(replacement) = hook_result.replacement {
                (
                    replacement.summary,
                    replacement.first_kept_entry_id,
                    replacement.tokens_before,
                    replacement
                        .retained_tail
                        .unwrap_or_else(|| preparation.retained_tail.clone()),
                    replacement.details,
                    replacement.usage,
                )
            } else {
                let generated = self
                    .generate_compaction(
                        operation,
                        &preparation,
                        &config,
                        reason,
                        custom_instructions,
                        cancellation.child_token(),
                    )
                    .await?;
                let files = preparation.file_operations.lists();
                (
                    generated.0,
                    preparation.first_kept_entry_id.clone(),
                    preparation.tokens_before,
                    preparation.retained_tail.clone(),
                    Some(serde_json::to_value(&files)?),
                    Some(generated.1),
                )
            };
        session
            .append_compaction_with(
                summary.clone(),
                Some(first_kept_entry_id.clone()),
                tokens_before,
                Some(retained_tail.clone()),
                details.clone(),
                usage.as_ref().map(session_usage),
                Some(from_hook),
            )
            .await?;
        let estimated_tokens_after = project_session(&session)
            .await?
            .iter()
            .map(estimate_message_tokens)
            .sum();
        let result = CompactionResult {
            summary,
            first_kept_entry_id,
            tokens_before,
            estimated_tokens_after,
            usage,
            retained_tail,
            details,
            from_hook,
        };
        Ok(Some(result))
    }

    async fn generate_compaction(
        &self,
        operation: u64,
        preparation: &CompactionPreparation,
        config: &HarnessConfig,
        reason: CompactionReason,
        custom_instructions: Option<&str>,
        cancellation: CancellationToken,
    ) -> Result<(String, ri_ai::Usage)> {
        let max_tokens = config.compaction.reserve_tokens.saturating_mul(4) / 5;
        let max_tokens = max_tokens.min(config.model.max_tokens).max(1);
        if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
            let (history, history_usage) = if preparation.messages_to_summarize.is_empty() {
                ("No prior history.".to_owned(), None)
            } else {
                let response = self
                    .summarize_with_retry(
                        operation,
                        summary_request(
                            SummaryKind::Compaction,
                            config,
                            compaction_request_text(
                                &preparation.messages_to_summarize,
                                preparation.previous_summary.as_deref(),
                                custom_instructions,
                            ),
                            max_tokens,
                        ),
                        RetryOperation::Compaction,
                        Some(reason),
                        Phase::Compaction,
                        cancellation.child_token(),
                    )
                    .await?;
                (response.text, Some(response.usage))
            };
            let prefix = self
                .summarize_with_retry(
                    operation,
                    summary_request(
                        SummaryKind::TurnPrefix,
                        config,
                        format!(
                            "<conversation>\n{}\n</conversation>\n\n{}",
                            crate::compaction::serialize_conversation(
                                &preparation.turn_prefix_messages
                            ),
                            TURN_PREFIX_PROMPT
                        ),
                        (config.compaction.reserve_tokens / 2)
                            .min(config.model.max_tokens)
                            .max(1),
                    ),
                    RetryOperation::TurnPrefix,
                    Some(reason),
                    Phase::Compaction,
                    cancellation,
                )
                .await?;
            let mut summary = format!(
                "{history}\n\n---\n\n**Turn Context (split turn):**\n\n{}",
                prefix.text
            );
            let files = preparation.file_operations.lists();
            append_file_lists(&mut summary, &files);
            let usage = history_usage
                .as_ref()
                .map_or(prefix.usage.clone(), |history| {
                    combine_usage(history, &prefix.usage)
                });
            Ok((summary, usage))
        } else {
            let response = self
                .summarize_with_retry(
                    operation,
                    summary_request(
                        SummaryKind::Compaction,
                        config,
                        compaction_request_text(
                            &preparation.messages_to_summarize,
                            preparation.previous_summary.as_deref(),
                            custom_instructions,
                        ),
                        max_tokens,
                    ),
                    RetryOperation::Compaction,
                    Some(reason),
                    Phase::Compaction,
                    cancellation,
                )
                .await?;
            let mut summary = response.text;
            append_file_lists(&mut summary, &preparation.file_operations.lists());
            Ok((summary, response.usage))
        }
    }

    async fn run_navigation(
        &self,
        operation: u64,
        target_id: String,
        mut options: NavigateOptions,
        cancellation: CancellationToken,
    ) -> Result<NavigateResult> {
        let (session, config) = {
            let control = self.inner.control.lock().await;
            (control.binding.session.clone(), control.config.clone())
        };
        let snapshot = session.snapshot().await?;
        let old_leaf = snapshot.leaf_id().map(str::to_owned);
        if old_leaf.as_deref() == Some(target_id.as_str()) {
            return Ok(NavigateResult::default());
        }
        let target = snapshot
            .entry(&target_id)
            .cloned()
            .ok_or_else(|| Error::InvalidArgument(format!("entry {target_id} was not found")))?;
        let (entries, common_ancestor) =
            collect_abandoned_branch(&snapshot, old_leaf.as_deref(), Some(&target_id))?;
        let hook_context = self.hook_context().await;
        let hook_result = if let Some(hooks) = &self.inner.hooks {
            hooks
                .before_navigation(
                    &hook_context,
                    BeforeNavigation {
                        target_id: target_id.clone(),
                        old_leaf_id: old_leaf.clone(),
                        common_ancestor_id: common_ancestor,
                        entries: entries.clone(),
                        options: options.clone(),
                    },
                    cancellation.child_token(),
                )
                .await?
        } else {
            BeforeNavigationResult::default()
        };
        if hook_result.cancel {
            return Ok(NavigateResult {
                cancelled: true,
                ..NavigateResult::default()
            });
        }
        if let Some(custom) = hook_result.custom_instructions {
            options.custom_instructions = Some(custom);
        }
        if let Some(replace) = hook_result.replace_instructions {
            options.replace_instructions = replace;
        }
        if let Some(label) = hook_result.label {
            options.label = Some(label);
        }
        if cancellation.is_cancelled() {
            return Err(Error::Aborted);
        }

        let mut generated_summary = None;
        let mut summary_details = None;
        let mut summary_usage = None;
        let from_hook = hook_result.summary.is_some();
        if let Some(summary) = hook_result.summary {
            generated_summary = Some(summary.summary);
            summary_details = summary.details;
            summary_usage = summary.usage;
        } else if options.summarize && !entries.is_empty() {
            self.inner
                .backend
                .preflight(&config.model)
                .await
                .map_err(model_backend_error)?;
            let budget = config
                .model
                .context_window
                .saturating_sub(config.compaction.reserve_tokens);
            let preparation = prepare_branch(&entries, budget)?;
            if !preparation.messages.is_empty() {
                let response = self
                    .summarize_with_retry(
                        operation,
                        summary_request(
                            SummaryKind::Branch,
                            &config,
                            branch_request_text(
                                &preparation.messages,
                                options.custom_instructions.as_deref(),
                                options.replace_instructions,
                            ),
                            2_048_u64.min(config.model.max_tokens).max(1),
                        ),
                        RetryOperation::BranchSummary,
                        None,
                        Phase::BranchSummary,
                        cancellation.child_token(),
                    )
                    .await?;
                let files = preparation.file_operations.lists();
                let mut summary = format!(
                    "The user explored a different conversation branch before returning here.\n\
                     Summary of that exploration:\n\n{}",
                    response.text
                );
                append_file_lists(&mut summary, &files);
                generated_summary = Some(summary);
                summary_details = Some(serde_json::to_value(files)?);
                summary_usage = Some(response.usage);
            }
        }

        let (new_leaf, editor_text) = navigation_position(&target.entry);
        if let Some(label) = options.label {
            session.append_label(target_id.clone(), Some(label)).await?;
        }
        session.move_to(new_leaf.clone()).await?;
        let summary_entry_id = if let Some(summary) = generated_summary {
            Some(
                session
                    .append_branch_summary_with(
                        old_leaf.clone().unwrap_or_else(|| "root".to_owned()),
                        summary,
                        summary_details,
                        summary_usage.as_ref().map(session_usage),
                        Some(from_hook),
                    )
                    .await?,
            )
        } else {
            None
        };
        self.emit_event(HarnessEvent::BranchNavigated {
            old_leaf,
            new_leaf: session.leaf_id().await?,
            summary_entry: summary_entry_id.clone(),
        })
        .await?;
        Ok(NavigateResult {
            cancelled: false,
            editor_text,
            summary_entry_id,
        })
    }

    async fn summarize_with_retry(
        &self,
        operation: u64,
        request: SummaryRequest,
        retry_operation: RetryOperation,
        compaction_reason: Option<CompactionReason>,
        resume_phase: Phase,
        cancellation: CancellationToken,
    ) -> Result<SummaryResponse> {
        let policy = {
            let control = self.inner.control.lock().await;
            control.config.retry
        };
        let mut attempt = 0_u32;
        loop {
            if cancellation.is_cancelled() {
                if attempt > 0 {
                    self.emit_event(HarnessEvent::RetryFinished {
                        operation: retry_operation,
                        success: false,
                        attempt,
                        final_error: Some("Retry cancelled".to_owned()),
                    })
                    .await?;
                }
                return Err(Error::Aborted);
            }
            match self
                .inner
                .backend
                .summarize(request.clone(), cancellation.child_token())
                .await
            {
                Ok(response) => {
                    if attempt > 0 {
                        self.emit_event(HarnessEvent::RetryFinished {
                            operation: retry_operation,
                            success: true,
                            attempt,
                            final_error: None,
                        })
                        .await?;
                    }
                    self.set_phase(operation, resume_phase).await?;
                    return Ok(response);
                }
                Err(error)
                    if error.is_retryable() && policy.enabled && attempt < policy.max_retries =>
                {
                    attempt = attempt.saturating_add(1);
                    let delay = policy.delay(attempt, error.retry_after);
                    self.set_phase(operation, Phase::Retry).await?;
                    self.emit_event(HarnessEvent::RetryScheduled {
                        operation: retry_operation,
                        attempt,
                        max_attempts: policy.max_retries,
                        delay,
                        error: error.message,
                    })
                    .await?;
                    if let Err(error) = abortable_sleep(delay, &cancellation).await {
                        self.emit_event(HarnessEvent::RetryFinished {
                            operation: retry_operation,
                            success: false,
                            attempt,
                            final_error: Some(error.to_string()),
                        })
                        .await?;
                        return Err(error);
                    }
                    self.emit_event(HarnessEvent::RetryAttemptStarted {
                        kind: request.kind,
                        reason: compaction_reason,
                    })
                    .await?;
                    self.set_phase(operation, resume_phase).await?;
                }
                Err(error) => {
                    if attempt > 0 {
                        self.emit_event(HarnessEvent::RetryFinished {
                            operation: retry_operation,
                            success: false,
                            attempt,
                            final_error: Some(error.message.clone()),
                        })
                        .await?;
                    }
                    return Err(summary_backend_error(error, retry_operation));
                }
            }
        }
    }

    async fn prepare_agent_retry(
        &self,
        operation: u64,
        attempt: u32,
        backend_error: Option<&BackendError>,
        assistant: &AssistantMessage,
        cancellation: CancellationToken,
    ) -> Result<AgentRetryPreparation> {
        let policy = {
            let control = self.inner.control.lock().await;
            control.config.retry
        };
        if !policy.enabled || attempt > policy.max_retries {
            return Ok(AgentRetryPreparation::Stop {
                attempt: None,
                final_error: None,
            });
        }
        let requested = backend_error.and_then(|error| error.retry_after);
        let delay = policy.delay(attempt, requested);
        let retry_cancellation = CancellationToken::new();
        self.inner.control.lock().await.retry_cancellation = Some(retry_cancellation.clone());
        self.set_phase(operation, Phase::Retry).await?;
        let event_result = self
            .emit_event(HarnessEvent::RetryScheduled {
                operation: RetryOperation::Agent,
                attempt,
                max_attempts: policy.max_retries,
                delay,
                error: assistant
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "transient agent failure".to_owned()),
            })
            .await;
        if let Err(error) = event_result {
            self.inner.control.lock().await.retry_cancellation = None;
            return Err(error);
        }
        let retry = tokio::select! {
            () = tokio::time::sleep(delay) => true,
            () = retry_cancellation.cancelled() => false,
            () = cancellation.cancelled() => {
                self.inner.control.lock().await.retry_cancellation = None;
                return Err(Error::Aborted);
            }
        };
        self.inner.control.lock().await.retry_cancellation = None;
        self.set_phase(operation, Phase::Turn).await?;
        Ok(if retry {
            AgentRetryPreparation::Retry
        } else {
            AgentRetryPreparation::Stop {
                attempt: Some(attempt),
                final_error: Some("Retry cancelled".to_owned()),
            }
        })
    }

    async fn compaction_check(&self, post_turn: bool) -> Result<Option<(CompactionReason, bool)>> {
        let (session, config) = {
            let control = self.inner.control.lock().await;
            (control.binding.session.clone(), control.config.clone())
        };
        if !config.compaction.enabled {
            return Ok(None);
        }
        let snapshot = session.snapshot().await?;
        let path = snapshot.active_path()?;
        let latest_compaction_time = path.iter().rev().find_map(|stored| {
            let SessionEntry::Compaction(entry) = &stored.entry else {
                return None;
            };
            Some(entry.base.timestamp.timestamp_millis())
        });
        let assistant = path.iter().rev().find_map(|stored| {
            let SessionEntry::Message(entry) = &stored.entry else {
                return None;
            };
            if entry.message.get("role").and_then(Value::as_str) != Some("assistant") {
                return None;
            }
            serde_json::from_value::<AssistantMessage>(entry.message.clone()).ok()
        });
        let Some(assistant) = assistant else {
            return Ok(None);
        };
        if latest_compaction_time.is_some_and(|time| assistant.timestamp <= time) {
            return Ok(None);
        }
        if post_turn && assistant.stop_reason == StopReason::Aborted {
            return Ok(None);
        }
        if same_model(&assistant, &config.model)
            && classify_context_overflow(&assistant, Some(config.model.context_window)).is_some()
        {
            return Ok(Some((
                CompactionReason::Overflow,
                assistant.stop_reason != StopReason::Stop,
            )));
        }
        let direct = context_tokens(&assistant.usage);
        let tokens = if assistant.stop_reason == StopReason::Error || direct == 0 {
            let messages = project_session(&session).await?;
            let estimate = estimate_context_tokens(&messages);
            let Some(index) = estimate.last_usage_index else {
                return Ok(None);
            };
            let usage_time = messages.get(index).and_then(|message| match message {
                Message::Assistant(message) => Some(message.timestamp),
                Message::User(_) | Message::ToolResult(_) => None,
            });
            if latest_compaction_time
                .zip(usage_time)
                .is_some_and(|(boundary, usage)| usage <= boundary)
            {
                return Ok(None);
            }
            estimate.tokens
        } else {
            direct
        };
        Ok(
            should_compact(tokens, config.model.context_window, config.compaction)
                .then_some((CompactionReason::Threshold, false)),
        )
    }

    async fn create_turn_snapshot(
        &self,
        excluded_error_timestamp: Option<i64>,
    ) -> Result<TurnSnapshot> {
        let (binding, config) = {
            let control = self.inner.control.lock().await;
            (control.binding.clone(), control.config.clone())
        };
        let mut messages = project_session(&binding.session).await?;
        if let Some(timestamp) = excluded_error_timestamp
            && let Some(index) = messages.iter().rposition(|message| {
                matches!(
                    message,
                    Message::Assistant(message)
                        if message.timestamp == timestamp
                            && matches!(message.stop_reason, StopReason::Error | StopReason::Aborted | StopReason::Length)
                )
            })
        {
            messages.remove(index);
        }
        let hook_context = HookContext {
            session_id: binding.id.clone(),
            generation: binding.generation,
        };
        if let Some(hooks) = &self.inner.hooks {
            messages = hooks.context(&hook_context, messages).await?;
        }
        let active_names: HashSet<_> = config
            .active_tool_names
            .iter()
            .map(String::as_str)
            .collect();
        let active_tools = config
            .tools
            .iter()
            .filter(|tool| active_names.contains(tool.name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let mut base_prompt = config.system_prompt.clone();
        if !config.resources.context.is_empty() {
            if !base_prompt.is_empty() {
                base_prompt.push_str("\n\n");
            }
            base_prompt.push_str(&config.resources.context.join("\n\n"));
        }
        let system_prompt = if let Some(hooks) = &self.inner.hooks {
            hooks
                .system_prompt(
                    &hook_context,
                    &base_prompt,
                    &config.resources,
                    &config.model,
                    config.thinking_level,
                    &config.active_tool_names,
                )
                .await?
                .unwrap_or(base_prompt)
        } else {
            base_prompt
        };
        Ok(TurnSnapshot {
            generation: binding.generation,
            session_id: binding.id,
            context: Context {
                system_prompt: Some(system_prompt.clone()),
                messages,
                tools: active_tools,
            },
            model: config.model,
            thinking_level: config.thinking_level,
            tools: config.tools,
            active_tool_names: config.active_tool_names,
            resources: config.resources,
            system_prompt: system_prompt.into(),
            request_options: config.request_options,
        })
    }

    async fn persist_messages(&self, messages: &[Message]) -> Result<Vec<String>> {
        let session = self.inner.control.lock().await.binding.session.clone();
        let mut ids = Vec::with_capacity(messages.len());
        for message in messages {
            let id = session.append_message(message_value(message)?).await?;
            self.emit_event(HarnessEvent::MessagePersisted {
                entry_id: id.clone(),
                role: message_role(message),
            })
            .await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn flush_pending_writes(&self) -> Result<bool> {
        let mut control = self.inner.control.lock().await;
        let had_pending = !control.pending_writes.is_empty();
        while let Some(write) = control.pending_writes.front().cloned() {
            persist_write(&control.binding.session, &write).await?;
            control.pending_writes.pop_front();
        }
        Ok(had_pending)
    }

    async fn begin_operation(&self, phase: Phase) -> Result<u64> {
        let (operation, marker) = {
            let mut control = self.inner.control.lock().await;
            if control.lifecycle.phase != Phase::Idle {
                return Err(Error::Busy {
                    phase: control.lifecycle.phase.as_str(),
                });
            }
            let operation = control.lifecycle.next_operation;
            control.lifecycle.next_operation = operation.saturating_add(1);
            control.lifecycle.phase = phase;
            control.lifecycle.root_phase = phase;
            control.lifecycle.active = Some(operation);
            control.cancellation = Some(CancellationToken::new());
            control.retry_cancellation = None;
            (operation, marker(&control))
        };
        self.inner.lifecycle_tx.send_replace(marker);
        Ok(operation)
    }

    async fn set_phase(&self, operation: u64, phase: Phase) -> Result<()> {
        let marker = {
            let mut control = self.inner.control.lock().await;
            if control.lifecycle.active != Some(operation) {
                return Err(Error::InvalidState(format!(
                    "operation {operation} is no longer active"
                )));
            }
            control.lifecycle.phase = phase;
            marker(&control)
        };
        self.inner.lifecycle_tx.send_replace(marker);
        Ok(())
    }

    async fn run_owned_operation<T, F>(&self, operation: u64, future: F) -> Result<T>
    where
        T: Send + 'static,
        F: Future<Output = Result<T>> + Send + 'static,
    {
        let harness = self.clone();
        let task = tokio::spawn(async move {
            let completed = AssertUnwindSafe(async {
                let result = future.await;
                harness.settle_result(operation, result).await
            })
            .catch_unwind()
            .await;
            match completed {
                Ok(result) => result,
                Err(payload) => {
                    let cleanup = harness.force_finish_operation(operation).await;
                    let panic = panic_message(&payload);
                    match cleanup {
                        Ok(()) => Err(Error::InvalidState(format!(
                            "operation {operation} panicked: {panic}"
                        ))),
                        Err(error) => Err(Error::InvalidState(format!(
                            "operation {operation} panicked: {panic}; forced settlement failed: {error}"
                        ))),
                    }
                }
            }
        });
        match task.await {
            Ok(result) => result,
            Err(task_error) => {
                let cleanup = self.force_finish_operation(operation).await;
                match cleanup {
                    Ok(()) => Err(Error::InvalidState(format!(
                        "operation {operation} task failed: {task_error}"
                    ))),
                    Err(error) => Err(Error::InvalidState(format!(
                        "operation {operation} task failed: {task_error}; forced settlement failed: {error}"
                    ))),
                }
            }
        }
    }

    async fn settle_result<T>(&self, operation: u64, result: Result<T>) -> Result<T> {
        let settlement = self.finish_operation(operation).await;
        match (result, settlement) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(operation_error), Err(settlement_error)) => Err(Error::InvalidState(format!(
                "{operation_error}; settlement also failed: {settlement_error}"
            ))),
        }
    }

    async fn finish_operation(&self, operation: u64) -> Result<()> {
        // Settlement is a barrier, not merely an event: accepted writes and
        // awaited callbacks finish before idle becomes observable. Cleanup must
        // still publish idle if either persistence or a callback fails.
        let mut failure = self.set_phase(operation, Phase::Settling).await.err();
        if failure.is_none()
            && let Err(error) = self.flush_pending_writes().await
        {
            failure = Some(error);
        }
        if failure.is_none() {
            let next_turn = {
                let control = self.inner.control.lock().await;
                control.queues.next_turn.len()
            };
            if let Err(error) = self
                .emit_event(HarnessEvent::Settled {
                    operation,
                    next_turn,
                })
                .await
            {
                failure = Some(error);
            }
        }

        let marker = {
            let mut control = self.inner.control.lock().await;
            while let Some(write) = control.pending_writes.front().cloned() {
                if let Err(error) = persist_write(&control.binding.session, &write).await {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                    break;
                }
                control.pending_writes.pop_front();
            }
            if control.lifecycle.active == Some(operation) {
                control.lifecycle.phase = Phase::Idle;
                control.lifecycle.root_phase = Phase::Idle;
                control.lifecycle.active = None;
                control.lifecycle.settled = operation;
                control.cancellation = None;
                control.retry_cancellation = None;
                marker(&control)
            } else {
                if failure.is_none() {
                    failure = Some(Error::InvalidState(format!(
                        "operation {operation} lost ownership during settlement"
                    )));
                }
                marker(&control)
            }
        };
        self.inner.lifecycle_tx.send_replace(marker);
        failure.map_or(Ok(()), Err)
    }

    async fn force_finish_operation(&self, operation: u64) -> Result<()> {
        let mut failure = None;
        let marker = {
            let mut control = self.inner.control.lock().await;
            while let Some(write) = control.pending_writes.front().cloned() {
                if let Err(error) = persist_write(&control.binding.session, &write).await {
                    failure = Some(error);
                    break;
                }
                control.pending_writes.pop_front();
            }
            if control.lifecycle.active == Some(operation) {
                control.lifecycle.phase = Phase::Idle;
                control.lifecycle.root_phase = Phase::Idle;
                control.lifecycle.active = None;
                control.lifecycle.settled = operation;
                control.cancellation = None;
                control.retry_cancellation = None;
            } else if failure.is_none() {
                failure = Some(Error::InvalidState(format!(
                    "operation {operation} lost ownership during forced settlement"
                )));
            }
            marker(&control)
        };
        self.inner.lifecycle_tx.send_replace(marker);
        failure.map_or(Ok(()), Err)
    }

    async fn operation_cancellation(&self, operation: u64) -> Result<CancellationToken> {
        let control = self.inner.control.lock().await;
        if control.lifecycle.active != Some(operation) {
            return Err(Error::InvalidState(format!(
                "operation {operation} is not active"
            )));
        }
        control
            .cancellation
            .clone()
            .ok_or_else(|| Error::InvalidState("active operation has no cancellation token".into()))
    }

    async fn enqueue(&self, queue: Queue, message: Message) -> Result<()> {
        let (id, lengths) = {
            let mut control = self.inner.control.lock().await;
            if !matches!(queue, Queue::NextTurn)
                && (control.lifecycle.root_phase != Phase::Turn
                    || control.lifecycle.phase == Phase::Settling)
            {
                return Err(Error::InvalidState(
                    "steering and follow-up messages require an active turn".to_owned(),
                ));
            }
            let id = control.queues.push(queue, message);
            (id, control.queues.lengths())
        };
        if let Err(error) = self.emit_event(HarnessEvent::QueueUpdated(lengths)).await {
            let mut control = self.inner.control.lock().await;
            control.queues.get_mut(queue).retain(|item| item.id != id);
            return Err(error);
        }
        Ok(())
    }

    async fn drain_queue_live(&self, queue: Queue, steering: bool) -> Result<Vec<Message>> {
        let mode = {
            let control = self.inner.control.lock().await;
            if steering {
                control.config.steering_mode
            } else {
                control.config.follow_up_mode
            }
        };
        self.drain_queue(queue, mode).await
    }

    async fn drain_queue(&self, queue: Queue, mode: QueueMode) -> Result<Vec<Message>> {
        let (removed, lengths) = {
            let mut control = self.inner.control.lock().await;
            let target = control.queues.get_mut(queue);
            let count = match mode {
                QueueMode::OneAtATime => usize::from(!target.is_empty()),
                QueueMode::All => target.len(),
            };
            let removed = target.drain(..count).collect::<Vec<_>>();
            (removed, control.queues.lengths())
        };
        if removed.is_empty() {
            return Ok(Vec::new());
        }
        if let Err(error) = self.emit_event(HarnessEvent::QueueUpdated(lengths)).await {
            let mut control = self.inner.control.lock().await;
            let target = control.queues.get_mut(queue);
            for item in removed.into_iter().rev() {
                target.push_front(item);
            }
            return Err(error);
        }
        Ok(removed.into_iter().map(|item| item.message).collect())
    }

    async fn hook_context(&self) -> HookContext {
        let control = self.inner.control.lock().await;
        HookContext {
            session_id: control.binding.id.clone(),
            generation: control.binding.generation,
        }
    }

    async fn expand_resources(
        &self,
        input: &str,
        resources: &crate::types::Resources,
    ) -> Result<String> {
        let expansion = expand_resources(input, resources);
        if let Some(resource) = expansion.resource {
            self.emit_event(HarnessEvent::ResourceExpanded {
                resource,
                text: expansion.text.clone(),
            })
            .await?;
        }
        Ok(expansion.text)
    }

    async fn emit_event(&self, event: HarnessEvent) -> Result<()> {
        let context = self.hook_context().await;
        if let Some(hooks) = &self.inner.hooks {
            hooks.event(&context, &event).await?;
        }
        let observers = self
            .inner
            .observers
            .read()
            .await
            .iter()
            .map(|(_, observer)| observer.clone())
            .collect::<Vec<_>>();
        for observer in observers {
            observer.on_event(&event).await?;
        }
        Ok(())
    }

    async fn wait_for_operation(&self, target: u64) {
        let mut receiver = self.inner.lifecycle_tx.subscribe();
        loop {
            if receiver.borrow().settled >= target {
                return;
            }
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }

    async fn publish_lifecycle(&self) {
        let marker = {
            let control = self.inner.control.lock().await;
            marker(&control)
        };
        self.inner.lifecycle_tx.send_replace(marker);
    }
}

fn marker(control: &Control) -> LifecycleMarker {
    LifecycleMarker {
        phase: control.lifecycle.phase,
        active: control.lifecycle.active,
        settled: control.lifecycle.settled,
        generation: control.binding.generation,
    }
}

fn validate_config(config: &HarnessConfig) -> Result<()> {
    let mut names = HashSet::new();
    for tool in config.tools.iter() {
        if tool.name.trim().is_empty() {
            return Err(Error::InvalidArgument(
                "tool names cannot be blank".to_owned(),
            ));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "duplicate tool name {:?}",
                tool.name
            )));
        }
    }
    validate_active_tools(&config.tools, &config.active_tool_names)
}

fn validate_active_tools(tools: &[ri_ai::Tool], active: &[String]) -> Result<()> {
    let tools: HashSet<_> = tools.iter().map(|tool| tool.name.as_str()).collect();
    let mut seen = HashSet::new();
    for name in active {
        if !seen.insert(name.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "duplicate active tool name {name:?}"
            )));
        }
        if !tools.contains(name.as_str()) {
            return Err(Error::InvalidArgument(format!(
                "unknown active tool {name:?}"
            )));
        }
    }
    Ok(())
}

async fn reconcile_session_config(
    session: &Session,
    config: &mut HarnessConfig,
    persist_missing: bool,
) -> Result<()> {
    let context = session.context().await?;
    if let Some(selection) = context.model {
        if selection.provider != config.model.provider || selection.model_id != config.model.id {
            return Err(Error::Model(format!(
                "session requires {}/{}, but builder supplied {}/{}",
                selection.provider, selection.model_id, config.model.provider, config.model.id
            )));
        }
    } else if persist_missing {
        session
            .append_model_change(config.model.provider.clone(), config.model.id.clone())
            .await?;
    }
    if context.thinking_level != "off" || session_has_thinking_entry(session).await? {
        config.thinking_level = parse_thinking_level(&context.thinking_level)?;
    } else if persist_missing {
        session
            .append_thinking_level_change(config.thinking_level.as_str())
            .await?;
    }
    if let Some(active) = context.active_tool_names {
        validate_active_tools(&config.tools, &active)?;
        config.active_tool_names = active.into();
    } else if persist_missing {
        session
            .append_active_tools_change(config.active_tool_names.to_vec())
            .await?;
    }
    Ok(())
}

async fn session_has_thinking_entry(session: &Session) -> Result<bool> {
    Ok(session
        .branch()
        .await?
        .iter()
        .any(|stored| matches!(stored.entry, SessionEntry::ThinkingLevelChange(_))))
}

fn parse_thinking_level(value: &str) -> Result<ThinkingLevel> {
    match value {
        "off" => Ok(ThinkingLevel::Off),
        "minimal" => Ok(ThinkingLevel::Minimal),
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "xhigh" => Ok(ThinkingLevel::Xhigh),
        "max" => Ok(ThinkingLevel::Max),
        _ => Err(Error::Session(format!(
            "session contains unknown thinking level {value:?}"
        ))),
    }
}

async fn persist_write(session: &Session, write: &SessionWrite) -> Result<()> {
    match write {
        SessionWrite::Message(message) => {
            session.append_message(message_value(message)?).await?;
        }
        SessionWrite::RawMessage(message) => {
            session.append_message(message.clone()).await?;
        }
        SessionWrite::Model { provider, model_id } => {
            session
                .append_model_change(provider.clone(), model_id.clone())
                .await?;
        }
        SessionWrite::Thinking(level) => {
            session.append_thinking_level_change(level.as_str()).await?;
        }
        SessionWrite::ActiveTools(names) => {
            session.append_active_tools_change(names.clone()).await?;
        }
        SessionWrite::Custom { kind, data } => {
            session.append_custom(kind.clone(), data.clone()).await?;
        }
        SessionWrite::CustomMessage {
            kind,
            content,
            display,
            details,
        } => {
            session
                .append_custom_message(kind.clone(), content.clone(), *display, details.clone())
                .await?;
        }
    }
    Ok(())
}

fn create_user_message(text: impl Into<String>, images: Vec<ri_ai::ImageContent>) -> Message {
    let text = text.into();
    if images.is_empty() {
        return Message::User(UserMessage {
            content: UserContent::Text(text),
            timestamp: now_millis(),
        });
    }
    let mut blocks = vec![InputContent::Text(ri_ai::TextContent::new(text))];
    for image in images {
        blocks.push(InputContent::Image(image));
    }
    Message::User(UserMessage {
        content: UserContent::Blocks(blocks),
        timestamp: now_millis(),
    })
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
    }
}

fn same_model(message: &AssistantMessage, model: &ri_ai::Model) -> bool {
    message.provider == model.provider && message.model == model.id
}

fn assistant_retryable(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let error = message
        .error_message
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if [
        "insufficient_quota",
        "invalid api key",
        "authentication",
        "permission",
        "context",
        "prompt is too long",
    ]
    .iter()
    .any(|pattern| error.contains(pattern))
    {
        return false;
    }
    [
        "429",
        "502",
        "503",
        "504",
        "529",
        "rate limit",
        "overloaded",
        "temporar",
        "timeout",
        "timed out",
        "terminated",
        "connection",
        "server error",
        "service unavailable",
    ]
    .iter()
    .any(|pattern| error.contains(pattern))
}

fn backend_failure_message(model: &ri_ai::Model, error: &BackendError) -> AssistantMessage {
    let mut message =
        AssistantMessage::empty(model.api.clone(), model.provider.clone(), model.id.clone());
    message.stop_reason = if error.kind == BackendErrorKind::Aborted {
        StopReason::Aborted
    } else {
        StopReason::Error
    };
    message.error_message = Some(error.message.clone());
    message
}

fn model_backend_error(error: BackendError) -> Error {
    match error.kind {
        BackendErrorKind::Aborted => Error::Aborted,
        _ => Error::Model(error.message),
    }
}

fn runtime_backend_error(error: BackendError) -> Error {
    match error.kind {
        BackendErrorKind::Aborted => Error::Aborted,
        _ => Error::Agent(error.message),
    }
}

fn summary_backend_error(error: BackendError, operation: RetryOperation) -> Error {
    match error.kind {
        BackendErrorKind::Aborted => Error::Aborted,
        _ if operation == RetryOperation::BranchSummary => Error::BranchSummary(error.message),
        _ => Error::Compaction(error.message),
    }
}

fn summary_request(
    kind: SummaryKind,
    config: &HarnessConfig,
    prompt: String,
    max_tokens: u64,
) -> SummaryRequest {
    let mut request_options = config.request_options.clone();
    request_options.cache_retention = Some(CacheRetention::None);
    SummaryRequest {
        kind,
        model: config.model.clone(),
        system_prompt: SUMMARIZATION_SYSTEM_PROMPT.to_owned(),
        prompt,
        max_tokens,
        thinking_level: config.thinking_level,
        request_id: Uuid::now_v7().hyphenated().to_string(),
        request_options,
    }
}

fn navigation_position(entry: &SessionEntry) -> (Option<String>, Option<String>) {
    match entry {
        SessionEntry::Message(entry)
            if entry.message.get("role").and_then(Value::as_str) == Some("user") =>
        {
            let editor_text = serde_json::from_value::<UserMessage>(entry.message.clone())
                .ok()
                .map(|message| user_text(&message));
            (entry.base.parent_id.clone(), editor_text)
        }
        SessionEntry::CustomMessage(entry) => {
            let text = entry.content.as_str().map(str::to_owned).or_else(|| {
                entry.content.as_array().map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(|block| block.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
            });
            (entry.base.parent_id.clone(), text)
        }
        entry => (Some(entry.id().to_owned()), None),
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_owned())
}

async fn abortable_sleep(delay: Duration, cancellation: &CancellationToken) -> Result<()> {
    if delay.is_zero() {
        return if cancellation.is_cancelled() {
            Err(Error::Aborted)
        } else {
            Ok(())
        };
    }
    tokio::select! {
        () = cancellation.cancelled() => Err(Error::Aborted),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}
