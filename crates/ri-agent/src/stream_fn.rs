//! Narrow provider integration boundary.
//!
//! `ri-agent` intentionally depends only on `ri-ai`'s provider-neutral model,
//! context, event-stream, and error types. A provider catalog can adapt its
//! streaming method to [`StreamFn`] without the agent crate depending on
//! provider registration or transport ownership.

use std::{
    future::Future,
    sync::{Arc, OnceLock},
};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::AgentError;

/// Request options owned by the agent layer.
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    /// Structured cancellation for the active run.
    pub cancellation: CancellationToken,
    /// Requested reasoning effort.
    pub thinking_level: ri_ai::ThinkingLevel,
    /// Optional stable session/cache-affinity identifier.
    pub session_id: Option<String>,
    /// Dynamically resolved provider credential.
    pub api_key: Option<String>,
    /// Application-owned provider options not interpreted by this crate.
    pub extensions: indexmap::IndexMap<String, Value>,
}

/// Provider function consumed by the agent loop.
///
/// Implementations should encode ordinary provider failures as terminal
/// assistant error events. Returning `Err` is still supported for failures that
/// occur before a provider-neutral assistant message can be created.
#[async_trait]
pub trait StreamFn: Send + Sync + 'static {
    /// Starts one provider request.
    ///
    /// # Errors
    ///
    /// Returns a provider-neutral failure when no assistant error message can
    /// be constructed.
    async fn stream(
        &self,
        model: ri_ai::Model,
        context: ri_ai::Context,
        options: StreamOptions,
    ) -> Result<ri_ai::AssistantEventStream, ri_ai::AiError>;
}

#[async_trait]
impl<F, Fut> StreamFn for F
where
    F: Fn(ri_ai::Model, ri_ai::Context, StreamOptions) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ri_ai::AssistantEventStream, ri_ai::AiError>> + Send,
{
    async fn stream(
        &self,
        model: ri_ai::Model,
        context: ri_ai::Context,
        options: StreamOptions,
    ) -> Result<ri_ai::AssistantEventStream, ri_ai::AiError> {
        self(model, context, options).await
    }
}

#[async_trait]
impl StreamFn for ri_ai::Models {
    async fn stream(
        &self,
        model: ri_ai::Model,
        context: ri_ai::Context,
        options: StreamOptions,
    ) -> Result<ri_ai::AssistantEventStream, ri_ai::AiError> {
        let reasoning =
            (options.thinking_level != ri_ai::ThinkingLevel::Off).then_some(options.thinking_level);
        let provider_options = ri_ai::StreamOptions {
            api_key: options.api_key,
            reasoning,
            session_id: options.session_id,
            cancellation: Some(options.cancellation),
            extra: options.extensions.into_iter().collect(),
            ..ri_ai::StreamOptions::default()
        };
        Ok(ri_ai::Models::stream(
            self,
            model,
            context,
            provider_options,
        ))
    }
}

fn default_slot() -> &'static RwLock<Option<Arc<dyn StreamFn>>> {
    static DEFAULT: OnceLock<RwLock<Option<Arc<dyn StreamFn>>>> = OnceLock::new();
    DEFAULT.get_or_init(|| RwLock::new(None))
}

/// Replaces the process-wide compatibility fallback stream function.
///
/// New code should generally pass an explicit stream function to preserve
/// dependency visibility.
pub fn set_default_stream_fn(stream: Option<Arc<dyn StreamFn>>) {
    *default_slot().write() = stream;
}

/// Returns the configured process-wide stream fallback.
///
/// # Errors
///
/// Returns [`AgentError::MissingStreamFunction`] when no fallback is installed.
pub fn default_stream_fn() -> Result<Arc<dyn StreamFn>, AgentError> {
    default_slot()
        .read()
        .clone()
        .ok_or(AgentError::MissingStreamFunction)
}
