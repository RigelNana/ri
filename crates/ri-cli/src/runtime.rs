//! Narrow boundary between frontend bindings and `ri-sdk`.

use async_trait::async_trait;
use ri_sdk::FrontendMode;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::cli::Command;
use crate::error::Result;
use crate::input::ImageAttachment;

/// Delivery requested while another run is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptDelivery {
    /// Deliver after the current turn's tool calls.
    Steer,
    /// Deliver after the active run settles.
    FollowUp,
}

/// One prompt submitted through the shared runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptRequest {
    /// User-visible text.
    pub text: String,
    /// Inline images, used only by the first CLI message.
    pub images: Vec<ImageAttachment>,
    /// Frontend origin.
    pub source: FrontendMode,
    /// Queue behavior during an active run.
    pub delivery: Option<PromptDelivery>,
}

/// Settled result used by text and interactive bindings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptCompletion {
    /// Concatenated assistant text from the terminal message.
    pub text: String,
    /// Complete assistant message for structured event output.
    pub message: Option<Value>,
}

/// Short runtime identity displayed by the terminal UI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeStatus {
    /// Stable session id.
    pub session_id: String,
    /// `provider/model` selection.
    pub model: Option<String>,
    /// Active reasoning level.
    pub thinking: String,
}

/// Result of an administrative command.
#[derive(Clone, Debug, PartialEq)]
pub enum CommandOutput {
    /// Operation succeeded without user-facing output.
    Silent,
    /// Human-readable or raw exported text.
    Text(String),
    /// Structured metadata.
    Json(Value),
}

/// Frontend contract implemented only by the localized SDK adapter.
///
/// Production implementations must resolve a real configured provider. The
/// CLI never manufactures a provider or storage fallback.
#[async_trait]
pub trait CliRuntime: Send + Sync {
    /// Subscribe to exact serialized SDK events.
    fn subscribe(&self) -> broadcast::Receiver<Value>;

    /// Native session header emitted first in JSON mode.
    async fn session_header(&self) -> Result<Option<Value>>;

    /// Submit input and wait for the complete session-level run to settle.
    async fn prompt(&self, request: PromptRequest) -> Result<PromptCompletion>;

    /// Cancel the active run, if any.
    async fn abort(&self) -> Result<()>;

    /// Read identity for interactive status.
    async fn status(&self) -> Result<RuntimeStatus>;

    /// Execute a metadata/auth/session/resource/package command.
    async fn command(&self, command: &Command) -> Result<CommandOutput>;

    /// Execute one typed RPC request against this same runtime.
    #[cfg(feature = "rpc")]
    async fn rpc(
        &self,
        request: ri_rpc::Request,
        context: ri_rpc::DispatchContext,
    ) -> Result<ri_rpc::ResponsePayload>;

    /// Drain accepted writes and release runtime resources.
    async fn shutdown(&self) -> Result<()>;
}
