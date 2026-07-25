//! Shared model, provider, and authentication runtime.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use ri_agent::{StreamFn, StreamOptions as AgentStreamOptions};
use ri_ai::{
    AiError, AssistantEventStream, AuthResolutionOverrides, Context, InMemoryCredentialStore,
    Model, Models, StreamOptions as ModelStreamOptions, SystemAuthContext, builtin_providers,
};
use ri_harness::{BackendError, BackendErrorKind, ModelAccess, RequestOptions};

/// Cloneable provider catalog and authentication runtime used by all sessions.
#[derive(Clone, Debug)]
pub struct ModelRuntime {
    models: Models,
}

impl ModelRuntime {
    /// Wraps an explicitly configured `ri-ai` runtime.
    pub fn new(models: Models) -> Self {
        Self { models }
    }

    /// Creates the built-in provider catalog using ambient authentication.
    ///
    /// API keys are resolved from each provider's documented environment
    /// variables (for example, `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). The
    /// in-memory credential store intentionally does not read or create a
    /// private SDK-specific credential file.
    pub fn builtin_from_environment() -> Self {
        Self::new(Models::with_providers(
            Arc::new(InMemoryCredentialStore::default()),
            Arc::new(SystemAuthContext),
            builtin_providers(),
        ))
    }

    /// Accesses the underlying provider registry.
    pub fn models(&self) -> &Models {
        &self.models
    }

    /// Resolves a registered model.
    pub fn model(&self, provider: &str, model: &str) -> Option<Model> {
        self.models.model(provider, model)
    }

    /// Lists models whose providers currently have resolvable authentication.
    ///
    /// # Errors
    /// Returns an error when provider authentication cannot be resolved.
    pub async fn available(&self, provider: Option<&str>) -> Result<Vec<Model>, AiError> {
        self.models.available(provider).await
    }
}

#[async_trait]
impl StreamFn for ModelRuntime {
    async fn stream(
        &self,
        model: Model,
        context: Context,
        mut options: AgentStreamOptions,
    ) -> Result<AssistantEventStream, AiError> {
        let request = options
            .extensions
            .shift_remove("ri.request_options")
            .map(serde_json::from_value::<RequestOptions>)
            .transpose()
            .map_err(|error| AiError::Validation(error.to_string()))?
            .unwrap_or_default();
        let max_tokens = options
            .extensions
            .shift_remove("ri.max_tokens")
            .and_then(|value| value.as_u64());
        let mut extra = request.metadata;
        for (name, value) in options.extensions {
            extra.insert(name, value);
        }
        let headers = request
            .headers
            .into_iter()
            .map(|(name, value)| (name, Some(value)))
            .collect();
        Ok(self.models.stream(
            model,
            context,
            ModelStreamOptions {
                api_key: options.api_key,
                headers,
                env: BTreeMap::new(),
                temperature: None,
                max_tokens,
                reasoning: Some(options.thinking_level),
                cache_retention: request.cache_retention,
                session_id: options.session_id,
                tool_choice: None,
                timeout: request.timeout,
                cancellation: Some(options.cancellation),
                extra,
            },
        ))
    }
}

#[async_trait]
impl ModelAccess for ModelRuntime {
    async fn preflight(&self, model: &Model) -> Result<(), BackendError> {
        if self.models.provider(&model.provider).is_none() {
            return Err(BackendError::new(
                BackendErrorKind::Model,
                format!("unknown provider {}", model.provider),
            ));
        }
        let check = self
            .models
            .check_auth(&model.provider)
            .await
            .map_err(|error| map_ai_error(&error))?
            .ok_or_else(|| {
                BackendError::new(
                    BackendErrorKind::Model,
                    format!("unknown provider {}", model.provider),
                )
            })?;
        if !check.configured {
            return Err(BackendError::new(
                BackendErrorKind::Model,
                format!(
                    "no configured authentication for provider {}",
                    model.provider
                ),
            ));
        }
        Ok(())
    }

    async fn api_key(&self, _model: &Model) -> Result<Option<String>, BackendError> {
        // `Models::stream` owns the complete API-key/OAuth resolution and
        // refresh transaction. Passing no override preserves that authority.
        Ok(None)
    }
}

fn map_ai_error(error: &AiError) -> BackendError {
    let kind = if error.is_retryable() {
        BackendErrorKind::Transient
    } else {
        match error {
            AiError::Aborted => BackendErrorKind::Aborted,
            AiError::Auth(_)
            | AiError::Oauth(_)
            | AiError::Provider(_)
            | AiError::ModelSource(_)
            | AiError::ModelValidation(_) => BackendErrorKind::Model,
            AiError::Stream(_)
            | AiError::ProviderResponse { .. }
            | AiError::Validation(_)
            | AiError::Http(_) => BackendErrorKind::Fatal,
        }
    };
    BackendError::new(kind, error.to_string())
}

/// Resolves model authentication eagerly without starting a request.
///
/// # Errors
/// Returns an error when credentials are missing, invalid, or cannot be refreshed.
pub async fn resolve_model_auth(
    runtime: &ModelRuntime,
    model: &Model,
) -> Result<Option<ri_ai::AuthResult>, AiError> {
    runtime
        .models
        .get_model_auth(model, &AuthResolutionOverrides::default())
        .await
}
