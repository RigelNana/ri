//! Narrow runtime and extension boundaries used by the unified harness.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use ri_ai::{Message, Model, ThinkingLevel, Usage};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::compaction::CompactionPreparation;
use crate::error::{BackendError, Result};
use crate::types::{
    HarnessEvent, NavigateOptions, Resources, SummaryRequest, SummaryResponse, TurnOutput,
    TurnRequest,
};

/// Low-level agent/model runtime consumed by the high-level harness.
///
/// Implementations adapt the concrete `ri-agent` loop and `ri-ai` model/auth
/// registry. The harness deliberately depends on this small boundary so CLI,
/// JSON, RPC, and in-process callers share one lifecycle implementation.
#[async_trait]
pub trait HarnessBackend: Send + Sync + fmt::Debug {
    /// Resolve and validate model authentication before accepting a new run.
    async fn preflight(&self, model: &Model) -> std::result::Result<(), BackendError>;

    /// Execute exactly one assistant response and its tool-result batch.
    ///
    /// Returning `continue_after_tools` asks the harness to create a fresh turn
    /// snapshot at the save point before invoking this method again.
    async fn execute_turn(
        &self,
        request: TurnRequest,
        cancellation: CancellationToken,
    ) -> std::result::Result<TurnOutput, BackendError>;

    /// Execute a standalone compaction or branch-summary request.
    async fn summarize(
        &self,
        request: SummaryRequest,
        cancellation: CancellationToken,
    ) -> std::result::Result<SummaryResponse, BackendError>;

    /// Shut down runtime state tied to a session before replacement.
    async fn unbind_session(&self, _session_id: &str) -> std::result::Result<(), BackendError> {
        Ok(())
    }

    /// Bind runtime state to a newly selected session.
    async fn bind_session(
        &self,
        _session_id: &str,
        _generation: u64,
    ) -> std::result::Result<(), BackendError> {
        Ok(())
    }
}

/// Session-scoped identity supplied to hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HookContext {
    /// Session id.
    pub session_id: Arc<str>,
    /// Binding generation. Captured contexts become stale after replacement.
    pub generation: u64,
}

/// Result of the input interception stage.
#[derive(Clone, Debug, PartialEq)]
pub enum InputAction {
    /// Continue with unchanged input.
    Continue,
    /// Continue with transformed text and optional replacement images.
    Transform {
        /// Replacement text.
        text: String,
        /// Replacement image blocks. `None` preserves the original images.
        images: Option<Vec<ri_ai::ImageContent>>,
    },
    /// The hook handled the input without starting an agent run.
    Handled,
}

/// Input supplied to the canonical input hook.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEvent {
    /// Raw prompt text, before skill/template expansion.
    pub text: String,
    /// Image blocks.
    pub images: Vec<ri_ai::ImageContent>,
    /// Input origin.
    pub source: crate::types::PromptSource,
    /// Delivery mode when a turn is active.
    pub streaming_behavior: Option<crate::types::StreamingBehavior>,
}

/// Hook input immediately before the low-level run begins.
#[derive(Clone, Debug)]
pub struct BeforeAgentStart {
    /// Expanded user prompt.
    pub prompt: String,
    /// Images accompanying the user prompt.
    pub images: Vec<ri_ai::ImageContent>,
    /// System prompt resolved for the immutable turn snapshot.
    pub system_prompt: String,
    /// Resource snapshot.
    pub resources: Resources,
    /// Selected model.
    pub model: Arc<Model>,
    /// Requested reasoning level.
    pub thinking_level: ThinkingLevel,
    /// Active tool names.
    pub active_tool_names: Arc<[String]>,
}

/// Hook patch applied before an agent starts.
#[derive(Clone, Debug, Default)]
pub struct BeforeAgentStartResult {
    /// Additional messages appended after the user message.
    pub messages: Vec<Message>,
    /// Per-run system prompt override.
    pub system_prompt: Option<String>,
}

/// Extension-provided compaction result.
#[derive(Clone, Debug)]
pub struct CompactionOverride {
    /// Summary text.
    pub summary: String,
    /// First retained entry.
    pub first_kept_entry_id: String,
    /// Estimated context tokens before compaction.
    pub tokens_before: u64,
    /// Optional implementation metadata.
    pub details: Option<Value>,
    /// Summary request usage.
    pub usage: Option<Usage>,
    /// Self-contained retained tail.
    pub retained_tail: Option<Vec<Value>>,
}

/// Result of the pre-compaction hook.
#[derive(Clone, Debug, Default)]
pub struct BeforeCompactionResult {
    /// Cancel the operation.
    pub cancel: bool,
    /// Replace generated compaction content.
    pub replacement: Option<CompactionOverride>,
}

/// Extension-provided branch summary.
#[derive(Clone, Debug)]
pub struct BranchSummaryOverride {
    /// Summary text.
    pub summary: String,
    /// Optional implementation metadata.
    pub details: Option<Value>,
    /// Summary request usage.
    pub usage: Option<Usage>,
}

/// Result of the pre-navigation hook.
#[derive(Clone, Debug, Default)]
pub struct BeforeNavigationResult {
    /// Cancel navigation.
    pub cancel: bool,
    /// Replace generated branch summary content.
    pub summary: Option<BranchSummaryOverride>,
    /// Override custom summary instructions.
    pub custom_instructions: Option<String>,
    /// Override whether custom instructions replace defaults.
    pub replace_instructions: Option<bool>,
    /// Override target label.
    pub label: Option<String>,
}

/// Hook input for branch navigation.
#[derive(Clone, Debug)]
pub struct BeforeNavigation {
    /// Requested target entry.
    pub target_id: String,
    /// Current leaf.
    pub old_leaf_id: Option<String>,
    /// Deepest common ancestor.
    pub common_ancestor_id: Option<String>,
    /// Entries on the branch being abandoned.
    pub entries: Vec<ri_session::SequencedEntry>,
    /// Caller options.
    pub options: NavigateOptions,
}

/// Extension boundary. A `ri-ext` adapter owns registration and result
/// reduction; the harness only invokes already-reduced lifecycle methods.
#[async_trait]
pub trait HarnessHooks: Send + Sync + fmt::Debug {
    /// Execute an extension command before ordinary input interception.
    async fn command(&self, _context: &HookContext, _input: &str) -> Result<bool> {
        Ok(false)
    }

    /// Intercept or transform raw input.
    async fn input(&self, _context: &HookContext, _event: InputEvent) -> Result<InputAction> {
        Ok(InputAction::Continue)
    }

    /// Resolve a dynamic system prompt once for a turn snapshot.
    async fn system_prompt(
        &self,
        _context: &HookContext,
        _base: &str,
        _resources: &Resources,
        _model: &Model,
        _thinking_level: ThinkingLevel,
        _active_tool_names: &[String],
    ) -> Result<Option<String>> {
        Ok(None)
    }

    /// Inject messages or override the system prompt before an agent run.
    async fn before_agent_start(
        &self,
        _context: &HookContext,
        _event: BeforeAgentStart,
    ) -> Result<BeforeAgentStartResult> {
        Ok(BeforeAgentStartResult::default())
    }

    /// Transform projected messages immediately before a provider request.
    async fn context(
        &self,
        _context: &HookContext,
        messages: Vec<Message>,
    ) -> Result<Vec<Message>> {
        Ok(messages)
    }

    /// Intercept compaction after deterministic preparation.
    async fn before_compaction(
        &self,
        _context: &HookContext,
        _preparation: &CompactionPreparation,
        _reason: crate::types::CompactionReason,
        _will_retry: bool,
        _custom_instructions: Option<&str>,
        _cancellation: CancellationToken,
    ) -> Result<BeforeCompactionResult> {
        Ok(BeforeCompactionResult::default())
    }

    /// Intercept branch navigation and summary generation.
    async fn before_navigation(
        &self,
        _context: &HookContext,
        _event: BeforeNavigation,
        _cancellation: CancellationToken,
    ) -> Result<BeforeNavigationResult> {
        Ok(BeforeNavigationResult::default())
    }

    /// Observe an awaited lifecycle event.
    async fn event(&self, _context: &HookContext, _event: &HarnessEvent) -> Result<()> {
        Ok(())
    }

    /// Invalidate hook-owned state before a session is replaced.
    async fn unbind_session(&self, _context: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Bind hook-owned state to a new session generation.
    async fn bind_session(&self, _context: &HookContext) -> Result<()> {
        Ok(())
    }
}

/// Additional observational subscriber.
#[async_trait]
pub trait HarnessObserver: Send + Sync + fmt::Debug {
    /// Observe an event. Events are delivered sequentially in registration order.
    async fn on_event(&self, event: &HarnessEvent) -> Result<()>;
}
