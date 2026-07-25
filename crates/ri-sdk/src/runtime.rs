//! Shared runtime handles for interactive, print, JSON, and RPC frontends.

use std::sync::Arc;

use ri_ai::{Message, Model, ThinkingLevel, UserMessage};
use ri_harness::{
    CompactionResult, Harness, HarnessConfig, HarnessObserver, HarnessStatus, NavigateOptions,
    NavigateResult, PromptOptions, PromptOutcome, PromptSource, QueueLengths, QueueMode,
    RequestOptions, SessionWrite,
};
use ri_session::Session;

use crate::{ModelRuntime, ResourceRuntime};

#[derive(Debug)]
struct RuntimeInner {
    harness: Harness,
    models: Arc<ModelRuntime>,
    resources: ResourceRuntime,
}

/// The sole mutable session runtime shared by every presentation mode.
#[derive(Clone, Debug)]
pub struct SessionRuntime {
    inner: Arc<RuntimeInner>,
}

impl SessionRuntime {
    pub(crate) fn new(
        harness: Harness,
        models: Arc<ModelRuntime>,
        resources: ResourceRuntime,
    ) -> Self {
        Self {
            inner: Arc::new(RuntimeInner {
                harness,
                models,
                resources,
            }),
        }
    }

    /// Accesses the unified harness.
    pub fn harness(&self) -> &Harness {
        &self.inner.harness
    }

    /// Accesses the shared provider/auth runtime.
    pub fn models(&self) -> &Arc<ModelRuntime> {
        &self.inner.models
    }

    /// Accesses the originally resolved resource snapshot.
    pub fn resources(&self) -> &ResourceRuntime {
        &self.inner.resources
    }

    /// Creates a mode-specific view that delegates to this exact instance.
    pub fn frontend(&self, mode: FrontendMode) -> SessionFrontend {
        SessionFrontend {
            runtime: self.clone(),
            mode,
        }
    }

    /// Runs input as an in-process SDK caller.
    ///
    /// # Errors
    /// Returns an error when the harness prompt pipeline fails.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        mut options: PromptOptions,
    ) -> ri_harness::Result<PromptOutcome> {
        options.source = PromptSource::Sdk;
        self.inner.harness.prompt(text, options).await
    }

    /// Runs plain text through the normal resource-expanding SDK pipeline.
    ///
    /// # Errors
    /// Returns an error when the harness prompt pipeline fails.
    pub async fn prompt_text(&self, text: impl Into<String>) -> ri_harness::Result<PromptOutcome> {
        self.prompt(text, PromptOptions::interactive()).await
    }

    /// Queues steering input.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn steer(&self, text: impl Into<String>) -> ri_harness::Result<()> {
        self.inner.harness.steer(text).await
    }

    /// Queues steering text and images.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn steer_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<ri_ai::ImageContent>,
    ) -> ri_harness::Result<()> {
        self.inner.harness.steer_with_images(text, images).await
    }

    /// Queues follow-up input.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn follow_up(&self, text: impl Into<String>) -> ri_harness::Result<()> {
        self.inner.harness.follow_up(text).await
    }

    /// Queues follow-up text and images.
    ///
    /// # Errors
    /// Returns an error when no run is active or queue event delivery fails.
    pub async fn follow_up_with_images(
        &self,
        text: impl Into<String>,
        images: Vec<ri_ai::ImageContent>,
    ) -> ri_harness::Result<()> {
        self.inner.harness.follow_up_with_images(text, images).await
    }

    /// Queues input for the next independently submitted turn.
    ///
    /// # Errors
    /// Returns an error when queue event delivery fails.
    pub async fn next_turn(&self, message: Message) -> ri_harness::Result<()> {
        self.inner.harness.next_turn(message).await
    }

    /// Queues plain text for the next independently submitted turn.
    ///
    /// # Errors
    /// Returns an error when queue event delivery fails.
    pub async fn next_turn_text(&self, text: impl Into<String>) -> ri_harness::Result<()> {
        self.next_turn(Message::User(UserMessage::new(text.into())))
            .await
    }

    /// Cancels the active operation, clears live queues, and waits for settlement.
    ///
    /// # Errors
    /// Returns an error when queue update observers fail.
    pub async fn abort(&self) -> ri_harness::Result<QueueLengths> {
        self.inner.harness.abort().await
    }

    /// Cancels only an active automatic-retry delay.
    ///
    /// Returns whether a retry delay was active.
    pub async fn abort_retry(&self) -> bool {
        self.inner.harness.abort_retry().await
    }

    /// Waits through all callbacks and accepted persistence writes.
    pub async fn wait_settled(&self) {
        self.inner.harness.wait_settled().await;
    }

    /// Runs manual compaction.
    ///
    /// # Errors
    /// Returns an error when compaction or settlement fails.
    pub async fn compact(
        &self,
        instructions: Option<String>,
    ) -> ri_harness::Result<CompactionResult> {
        self.inner.harness.compact(instructions).await
    }

    /// Navigates the session tree with abandoned-branch summarization.
    ///
    /// # Errors
    /// Returns an error when navigation, summarization, persistence, or settlement fails.
    pub async fn navigate(
        &self,
        target_id: impl Into<String>,
        options: NavigateOptions,
    ) -> ri_harness::Result<NavigateResult> {
        self.inner.harness.navigate(target_id, options).await
    }

    /// Rebinds this same runtime to a replacement durable session.
    ///
    /// # Errors
    /// Returns an error when validation, lifecycle hooks, or backend rebinding fails.
    pub async fn replace_session(&self, session: Session) -> ri_harness::Result<()> {
        let config = self.inner.harness.config().await;
        self.inner.harness.replace_session(session, config).await
    }

    /// Rebinds this runtime with an explicitly resolved destination config.
    ///
    /// # Errors
    /// Returns an error when validation, lifecycle hooks, or backend rebinding fails.
    pub async fn replace_session_with_config(
        &self,
        session: Session,
        config: HarnessConfig,
    ) -> ri_harness::Result<()> {
        self.inner.harness.replace_session(session, config).await
    }

    /// Current session handle.
    pub async fn session(&self) -> Session {
        self.inner.harness.session().await
    }

    /// Current lifecycle and queue state.
    pub async fn status(&self) -> HarnessStatus {
        self.inner.harness.status().await
    }

    /// Appends or save-point-queues an application write.
    ///
    /// # Errors
    /// Returns an error when an idle write cannot be persisted.
    pub async fn write_session(&self, write: SessionWrite) -> ri_harness::Result<()> {
        self.inner.harness.write_session(write).await
    }

    /// Selects a model for future immutable turn snapshots.
    ///
    /// # Errors
    /// Returns an error when model preflight or persistence fails.
    pub async fn set_model(&self, model: Model) -> ri_harness::Result<()> {
        self.inner.harness.set_model(Arc::new(model)).await
    }

    /// Selects a reasoning level for future turn snapshots.
    ///
    /// # Errors
    /// Returns an error when the branch-scoped update cannot be persisted.
    pub async fn set_thinking_level(&self, level: ThinkingLevel) -> ri_harness::Result<()> {
        self.inner.harness.set_thinking_level(level).await
    }

    /// Replaces the active tool subset after validating registered names.
    ///
    /// # Errors
    /// Returns an error for unknown tool names or when persistence fails.
    pub async fn set_active_tools(&self, names: Vec<String>) -> ri_harness::Result<()> {
        self.inner.harness.set_active_tools(names).await
    }

    /// Replaces the base model-visible prompt for future turns.
    pub async fn set_system_prompt(&self, prompt: impl Into<String>) {
        self.inner.harness.set_system_prompt(prompt).await;
    }

    /// Replaces provider request options for future turns.
    pub async fn set_request_options(&self, options: RequestOptions) {
        self.inner.harness.set_request_options(options).await;
    }

    /// Changes live steering and follow-up queue policies.
    pub async fn set_queue_modes(&self, steering: QueueMode, follow_up: QueueMode) {
        self.inner
            .harness
            .set_queue_modes(steering, follow_up)
            .await;
    }

    /// Enables or disables automatic context compaction for future checks.
    pub async fn set_auto_compaction_enabled(&self, enabled: bool) {
        self.inner
            .harness
            .set_auto_compaction_enabled(enabled)
            .await;
    }

    /// Enables or disables automatic retry for future retry decisions.
    pub async fn set_auto_retry_enabled(&self, enabled: bool) {
        self.inner.harness.set_auto_retry_enabled(enabled).await;
    }

    /// Adds an awaited event observer.
    pub async fn add_observer(&self, observer: Arc<dyn HarnessObserver>) -> u64 {
        self.inner.harness.add_observer(observer).await
    }

    /// Removes an event observer.
    pub async fn remove_observer(&self, id: u64) -> bool {
        self.inner.harness.remove_observer(id).await
    }
}

/// Presentation protocol using the shared runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrontendMode {
    /// Interactive terminal application.
    #[default]
    Interactive,
    /// One-shot human-readable output.
    Print,
    /// Structured event/output stream.
    Json,
    /// Bidirectional RPC host.
    Rpc,
}

impl FrontendMode {
    const fn prompt_source(self) -> PromptSource {
        match self {
            Self::Interactive => PromptSource::Interactive,
            Self::Print => PromptSource::Print,
            Self::Json => PromptSource::Json,
            Self::Rpc => PromptSource::Rpc,
        }
    }
}

/// Lightweight mode-specific handle over one [`SessionRuntime`].
#[derive(Clone, Debug)]
pub struct SessionFrontend {
    runtime: SessionRuntime,
    mode: FrontendMode,
}

impl SessionFrontend {
    /// This frontend's presentation mode.
    pub const fn mode(&self) -> FrontendMode {
        self.mode
    }

    /// Underlying shared runtime.
    pub fn runtime(&self) -> &SessionRuntime {
        &self.runtime
    }

    /// Submits input through the canonical prompt pipeline with mode metadata.
    ///
    /// # Errors
    /// Returns an error when the harness prompt pipeline fails.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        mut options: PromptOptions,
    ) -> ri_harness::Result<PromptOutcome> {
        options.source = self.mode.prompt_source();
        self.runtime.harness().prompt(text, options).await
    }
}
