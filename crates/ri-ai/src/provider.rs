//! Provider runtime and provider collection.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::{StreamExt, future::join_all};
use indexmap::IndexMap;
use parking_lot::RwLock;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{
    auth::{
        AuthContext, AuthResolutionOverrides, AuthResult, Credential, CredentialKind,
        CredentialStore, InMemoryCredentialStore, ProviderAuth, ProviderEnv, ProviderHeaders,
        SystemAuthContext, resolve_provider_auth,
    },
    error::AiError,
    message::{AssistantMessage, AssistantMessageEvent, Context, StopReason},
    model::{CacheRetention, Model, ThinkingLevel},
    stream::{AssistantEventStream, create_assistant_message_event_stream},
    transport::{DynHttpTransport, ReqwestTransport},
    wire::{WireAdapter, WireRequestOptions, execute_text_stream},
};

/// Request options accepted by [`Models::stream`].
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    /// Explicit API key; takes precedence over stored and ambient credentials.
    pub api_key: Option<String>,
    /// Caller/auth header overrides.
    pub headers: ProviderHeaders,
    /// Provider-scoped environment overlay.
    pub env: ProviderEnv,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Output token cap.
    pub max_tokens: Option<u64>,
    /// Requested reasoning effort.
    pub reasoning: Option<ThinkingLevel>,
    /// Prompt-cache retention.
    pub cache_retention: Option<CacheRetention>,
    /// Stable cache/session affinity key.
    pub session_id: Option<String>,
    /// Provider-neutral tool choice.
    pub tool_choice: Option<String>,
    /// Whole-request timeout.
    pub timeout: Option<Duration>,
    /// Cooperative cancellation.
    pub cancellation: Option<CancellationToken>,
    /// Explicit adapter-specific options.
    pub extra: BTreeMap<String, Value>,
}

impl StreamOptions {
    fn auth_overrides(&self) -> AuthResolutionOverrides {
        AuthResolutionOverrides {
            api_key: self.api_key.clone(),
            env: self.env.clone(),
        }
    }

    fn into_wire(self, auth: AuthResult) -> WireRequestOptions {
        let mut headers = auth.auth.headers;
        headers.extend(self.headers);
        let mut env = auth.env;
        env.extend(self.env);
        WireRequestOptions {
            api_key: auth.auth.api_key,
            headers,
            env,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            reasoning: self.reasoning,
            cache_retention: self.cache_retention,
            session_id: self.session_id,
            tool_choice: self.tool_choice,
            timeout: self.timeout,
            cancellation: self.cancellation,
            extra: self.extra,
        }
    }
}

/// Result of a non-refreshing provider auth check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthCheck {
    /// Whether enough configuration exists to attempt requests.
    pub configured: bool,
    /// Credential kind when a credential is stored.
    pub credential_kind: Option<CredentialKind>,
    /// Resolution source for ambient/API-key auth.
    pub source: Option<String>,
}

/// Options controlling provider catalog refresh.
#[derive(Clone, Debug, Default)]
pub struct RefreshOptions {
    /// Permit provider network requests.
    pub allow_network: bool,
    /// Ignore provider freshness checks.
    pub force: bool,
    /// Cooperative cancellation.
    pub cancellation: Option<CancellationToken>,
}

/// Best-effort result of refreshing all dynamic providers.
#[derive(Debug, Default)]
pub struct RefreshResult {
    /// Whether cancellation was observed.
    pub aborted: bool,
    /// Provider-scoped failures.
    pub errors: BTreeMap<String, AiError>,
}

/// Concrete provider runtime contract.
#[async_trait]
pub trait Provider: Send + Sync + std::fmt::Debug {
    /// Stable provider id.
    fn id(&self) -> &str;
    /// Display name.
    fn name(&self) -> &str;
    /// Provider-level headers.
    fn headers(&self) -> ProviderHeaders {
        ProviderHeaders::new()
    }
    /// Provider-owned auth methods.
    fn auth(&self) -> &ProviderAuth;
    /// Current synchronous model catalog.
    fn models(&self) -> Vec<Model>;
    /// Whether this provider has a refreshable dynamic catalog.
    fn is_dynamic(&self) -> bool {
        false
    }
    /// Refresh a dynamic model overlay, retaining old state on failure.
    async fn refresh_models(
        &self,
        _auth: &AuthResult,
        _options: &RefreshOptions,
    ) -> Result<(), AiError> {
        Ok(())
    }
    /// Optional credential-specific model availability policy.
    fn filter_models(&self, models: Vec<Model>, _credential: Option<&Credential>) -> Vec<Model> {
        models
    }
    /// Dispatch a fully authenticated text request.
    fn stream(
        &self,
        model: Model,
        context: Context,
        options: WireRequestOptions,
    ) -> AssistantEventStream;
}

/// Static provider implementation backed by real wire adapters.
#[derive(Clone)]
pub struct ProviderDescriptor {
    /// Stable id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Provider defaults.
    pub headers: ProviderHeaders,
    /// Provider auth methods.
    pub auth: ProviderAuth,
    /// Static model catalog.
    pub models: Vec<Model>,
    /// API-id to wire adapter dispatch table.
    pub adapters: BTreeMap<String, Arc<dyn WireAdapter>>,
    /// HTTP transport used by adapters.
    pub transport: DynHttpTransport,
}

impl std::fmt::Debug for ProviderDescriptor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderDescriptor")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("headers", &self.headers)
            .field("auth", &self.auth)
            .field("models", &self.models)
            .field("adapters", &self.adapters.keys().collect::<Vec<_>>())
            .field("transport", &self.transport)
            .finish()
    }
}

impl ProviderDescriptor {
    /// Creates a production descriptor using [`ReqwestTransport`].
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        auth: ProviderAuth,
        models: Vec<Model>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            headers: ProviderHeaders::new(),
            auth,
            models,
            adapters: BTreeMap::new(),
            transport: Arc::new(ReqwestTransport::default()),
        }
    }

    /// Uses an injectable transport.
    #[must_use]
    pub fn with_transport(mut self, transport: DynHttpTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Adds one concrete wire adapter.
    #[must_use]
    pub fn with_adapter(mut self, adapter: Arc<dyn WireAdapter>) -> Self {
        self.adapters.insert(adapter.api().to_owned(), adapter);
        self
    }

    /// Adds provider-level request headers.
    #[must_use]
    pub fn with_headers(mut self, headers: ProviderHeaders) -> Self {
        self.headers = headers;
        self
    }
}

#[async_trait]
impl Provider for ProviderDescriptor {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn headers(&self) -> ProviderHeaders {
        self.headers.clone()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: WireRequestOptions,
    ) -> AssistantEventStream {
        match self.adapters.get(&model.api) {
            Some(adapter) => execute_text_stream(
                self.transport.clone(),
                adapter.clone(),
                model,
                context,
                options,
            ),
            None => error_stream(
                &model,
                &AiError::Provider(format!(
                    "provider {} has no adapter for API {}",
                    self.id, model.api
                )),
            ),
        }
    }
}

/// Runtime provider registry and auth resolver.
#[derive(Clone)]
pub struct Models {
    providers: Arc<RwLock<IndexMap<String, Arc<dyn Provider>>>>,
    credentials: Arc<dyn CredentialStore>,
    auth_context: Arc<dyn AuthContext>,
}

impl std::fmt::Debug for Models {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Models")
            .field(
                "providers",
                &self.providers.read().keys().collect::<Vec<_>>(),
            )
            .field("credentials", &self.credentials)
            .field("auth_context", &self.auth_context)
            .finish()
    }
}

impl Default for Models {
    fn default() -> Self {
        Self::new(
            Arc::new(InMemoryCredentialStore::default()),
            Arc::new(SystemAuthContext),
        )
    }
}

impl Models {
    /// Creates an empty runtime with app-owned credential and auth contexts.
    pub fn new(credentials: Arc<dyn CredentialStore>, auth_context: Arc<dyn AuthContext>) -> Self {
        Self {
            providers: Arc::new(RwLock::new(IndexMap::new())),
            credentials,
            auth_context,
        }
    }

    /// Creates a runtime preloaded with providers.
    pub fn with_providers(
        credentials: Arc<dyn CredentialStore>,
        auth_context: Arc<dyn AuthContext>,
        providers: impl IntoIterator<Item = Arc<dyn Provider>>,
    ) -> Self {
        let runtime = Self::new(credentials, auth_context);
        for provider in providers {
            runtime.set_provider(provider);
        }
        runtime
    }

    /// Upserts a provider by id.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        self.providers
            .write()
            .insert(provider.id().to_owned(), provider);
    }

    /// Removes a provider.
    pub fn delete_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.write().shift_remove(id)
    }

    /// Removes all providers.
    pub fn clear_providers(&self) {
        self.providers.write().clear();
    }

    /// Returns provider snapshots in registration order.
    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.providers.read().values().cloned().collect()
    }

    /// Looks up one provider.
    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.read().get(id).cloned()
    }

    /// Returns the latest known model catalog.
    pub fn models(&self, provider_id: Option<&str>) -> Vec<Model> {
        match provider_id {
            Some(provider_id) => self
                .provider(provider_id)
                .map_or_else(Vec::new, |provider| provider.models()),
            None => self
                .providers()
                .into_iter()
                .flat_map(|provider| provider.models())
                .collect(),
        }
    }

    /// Looks up a model by provider and model id.
    pub fn model(&self, provider_id: &str, model_id: &str) -> Option<Model> {
        self.provider(provider_id)?
            .models()
            .into_iter()
            .find(|model| model.id == model_id)
    }

    /// Resolves provider auth with explicit > stored > ambient precedence.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when credential access or provider
    /// authentication resolution fails.
    pub async fn get_auth(
        &self,
        provider_id: &str,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, AiError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        resolve_provider_auth(
            provider_id,
            provider.auth(),
            self.credentials.as_ref(),
            self.auth_context.as_ref(),
            overrides,
        )
        .await
    }

    /// Resolves auth and incorporates provider/model static headers.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when provider authentication cannot be
    /// resolved.
    pub async fn get_model_auth(
        &self,
        model: &Model,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, AiError> {
        let Some(provider) = self.provider(&model.provider) else {
            return Ok(None);
        };
        let Some(mut result) = self.get_auth(&model.provider, overrides).await? else {
            return Ok(None);
        };
        let mut headers = provider.headers();
        headers.extend(
            model
                .headers
                .iter()
                .map(|(name, value)| (name.clone(), Some(value.clone()))),
        );
        headers.extend(result.auth.headers);
        result.auth.headers = headers;
        Ok(Some(result))
    }

    /// Checks provider configuration without refreshing an OAuth token.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when stored or ambient credentials
    /// cannot be inspected.
    pub async fn check_auth(&self, provider_id: &str) -> Result<Option<AuthCheck>, AiError> {
        let Some(provider) = self.provider(provider_id) else {
            return Ok(None);
        };
        let stored = self.credentials.read(provider_id).await?;
        match stored.as_ref() {
            Some(Credential::Oauth(_)) => Ok(Some(AuthCheck {
                configured: provider.auth().oauth.is_some(),
                credential_kind: Some(CredentialKind::Oauth),
                source: Some("stored OAuth".into()),
            })),
            Some(Credential::ApiKey(credential)) => {
                let configured = if let Some(auth) = &provider.auth().api_key {
                    auth.resolve(self.auth_context.as_ref(), Some(credential))
                        .await?
                        .is_some()
                } else {
                    false
                };
                Ok(Some(AuthCheck {
                    configured,
                    credential_kind: Some(CredentialKind::ApiKey),
                    source: configured.then(|| "stored credential".into()),
                }))
            }
            None => {
                let resolved = match &provider.auth().api_key {
                    Some(auth) => auth.resolve(self.auth_context.as_ref(), None).await?,
                    None => None,
                };
                Ok(Some(AuthCheck {
                    configured: resolved.is_some(),
                    credential_kind: None,
                    source: resolved.and_then(|result| result.source),
                }))
            }
        }
    }

    /// Returns models belonging to configured providers.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when any selected provider's
    /// configuration cannot be inspected.
    pub async fn available(&self, provider_id: Option<&str>) -> Result<Vec<Model>, AiError> {
        let providers = match provider_id {
            Some(id) => self.provider(id).into_iter().collect::<Vec<_>>(),
            None => self.providers(),
        };
        let mut output = Vec::new();
        for provider in providers {
            if !self
                .check_auth(provider.id())
                .await?
                .is_some_and(|check| check.configured)
            {
                continue;
            }
            let credential = self.credentials.read(provider.id()).await?;
            output.extend(provider.filter_models(provider.models(), credential.as_ref()));
        }
        Ok(output)
    }

    /// Stores a successful provider-owned login credential.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is unknown or the credential-store
    /// mutation fails.
    pub async fn store_credential(
        &self,
        provider_id: &str,
        credential: Credential,
    ) -> Result<(), AiError> {
        if self.provider(provider_id).is_none() {
            return Err(AiError::Provider(format!("unknown provider {provider_id}")));
        }
        self.credentials
            .modify(
                provider_id,
                Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
            )
            .await?;
        Ok(())
    }

    /// Deletes a stored provider credential.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when the credential store cannot delete
    /// the provider entry.
    pub async fn logout(&self, provider_id: &str) -> Result<(), AiError> {
        self.credentials.delete(provider_id).await
    }

    /// Refreshes all configured dynamic providers concurrently.
    pub async fn refresh(&self, options: RefreshOptions) -> RefreshResult {
        let futures = self
            .providers()
            .into_iter()
            .filter(|provider| provider.is_dynamic())
            .map(|provider| {
                let runtime = self.clone();
                let options = options.clone();
                async move {
                    if options
                        .cancellation
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
                    {
                        return (provider.id().to_owned(), Err(AiError::Aborted));
                    }
                    let auth = runtime
                        .get_auth(provider.id(), &AuthResolutionOverrides::default())
                        .await;
                    match auth {
                        Ok(Some(auth)) => (
                            provider.id().to_owned(),
                            provider.refresh_models(&auth, &options).await,
                        ),
                        Ok(None) => (provider.id().to_owned(), Ok(())),
                        Err(error) => (provider.id().to_owned(), Err(error)),
                    }
                }
            });
        let mut result = RefreshResult::default();
        for (provider, outcome) in join_all(futures).await {
            if let Err(error) = outcome {
                if matches!(error, AiError::Aborted) {
                    result.aborted = true;
                } else {
                    result.errors.insert(provider, error);
                }
            }
        }
        result
    }

    /// Resolves auth lazily and streams through the owning provider.
    pub fn stream(
        &self,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantEventStream {
        let (mut sender, stream) = create_assistant_message_event_stream();
        let runtime = self.clone();
        tokio::spawn(async move {
            let result = async {
                let provider = runtime.provider(&model.provider).ok_or_else(|| {
                    AiError::Provider(format!("unknown provider {}", model.provider))
                })?;
                let mut request_model = model.clone();
                let auth = runtime
                    .get_model_auth(&model, &options.auth_overrides())
                    .await?
                    .ok_or_else(|| {
                        AiError::Auth(format!(
                            "no configured authentication for provider {}",
                            model.provider
                        ))
                    })?;
                if let Some(base_url) = &auth.auth.base_url {
                    request_model.base_url = base_url.clone();
                }
                let mut provider_headers = provider.headers();
                provider_headers.extend(
                    request_model
                        .headers
                        .iter()
                        .map(|(name, value)| (name.clone(), Some(value.clone()))),
                );
                request_model.headers = provider_headers
                    .iter()
                    .filter_map(|(name, value)| {
                        value.as_ref().map(|value| (name.clone(), value.clone()))
                    })
                    .collect();
                let mut wire_options = options.clone().into_wire(auth);
                wire_options.normalize_for_model(&request_model);
                let mut provider_stream =
                    provider.stream(request_model, context.clone(), wire_options);
                while let Some(event) = provider_stream.next().await {
                    let terminal = event.final_message().is_some();
                    if !sender.send(event) {
                        return Ok(());
                    }
                    if terminal {
                        sender.close();
                        return Ok(());
                    }
                }
                Err(AiError::Stream(
                    "provider stream ended without a terminal event".into(),
                ))
            }
            .await;
            if let Err(error) = result {
                let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
                message.stop_reason = if matches!(error, AiError::Aborted) {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                message.error_message = Some(error.to_string());
                sender.send(AssistantMessageEvent::Error {
                    reason: message.stop_reason,
                    error: message,
                });
                sender.close();
            }
        });
        stream
    }

    /// Waits for the final assistant result.
    ///
    /// # Errors
    ///
    /// Returns a stream or authentication error if no terminal assistant
    /// message can be produced.
    pub async fn complete(
        &self,
        model: Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessage, AiError> {
        self.stream(model, context, options).result().await
    }
}

fn error_stream(model: &Model, error: &AiError) -> AssistantEventStream {
    let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
    message.stop_reason = StopReason::Error;
    message.error_message = Some(error.to_string());
    AssistantEventStream::completed(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::{auth::EnvApiKeyAuth, message::UserMessage};

    use super::*;

    #[derive(Debug, Default)]
    struct TestContext {
        env: BTreeMap<String, String>,
    }

    #[async_trait]
    impl AuthContext for TestContext {
        async fn env(&self, name: &str) -> Option<String> {
            self.env.get(name).cloned()
        }

        async fn file_exists(&self, _path: &str) -> bool {
            false
        }
    }

    #[derive(Debug)]
    struct TestProvider {
        auth: ProviderAuth,
        models: Vec<Model>,
        refreshes: AtomicUsize,
    }

    #[async_trait]
    impl Provider for TestProvider {
        fn id(&self) -> &'static str {
            "test"
        }

        fn name(&self) -> &'static str {
            "Test"
        }

        fn auth(&self) -> &ProviderAuth {
            &self.auth
        }

        fn models(&self) -> Vec<Model> {
            self.models.clone()
        }

        fn is_dynamic(&self) -> bool {
            true
        }

        async fn refresh_models(
            &self,
            _auth: &AuthResult,
            _options: &RefreshOptions,
        ) -> Result<(), AiError> {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn stream(
            &self,
            model: Model,
            _context: Context,
            _options: WireRequestOptions,
        ) -> AssistantEventStream {
            let message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
            AssistantEventStream::completed(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message,
            })
        }
    }

    fn runtime() -> (Models, Arc<TestProvider>) {
        let provider = Arc::new(TestProvider {
            auth: ProviderAuth::api_key(Arc::new(EnvApiKeyAuth::new("test", ["TEST_KEY"]))),
            models: vec![Model::new(
                "test",
                "model",
                "test-api",
                "https://example.test",
            )],
            refreshes: AtomicUsize::new(0),
        });
        let runtime = Models::with_providers(
            Arc::new(InMemoryCredentialStore::default()),
            Arc::new(TestContext {
                env: BTreeMap::from([("TEST_KEY".into(), "ambient".into())]),
            }),
            [provider.clone() as Arc<dyn Provider>],
        );
        (runtime, provider)
    }

    #[tokio::test]
    async fn models_store_arc_trait_objects_and_lookup_catalog() {
        let (runtime, _) = runtime();
        assert_eq!(runtime.providers().len(), 1);
        assert_eq!(runtime.models(None).len(), 1);
        assert_eq!(
            runtime.model("test", "model").map(|model| model.id),
            Some("model".into())
        );
        assert!(runtime.model("missing", "model").is_none());
    }

    #[tokio::test]
    async fn availability_and_refresh_use_provider_auth() {
        let (runtime, provider) = runtime();
        assert_eq!(runtime.available(None).await.expect("available").len(), 1);
        let result = runtime
            .refresh(RefreshOptions {
                allow_network: true,
                ..RefreshOptions::default()
            })
            .await;
        assert!(result.errors.is_empty());
        assert_eq!(provider.refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn refresh_reports_cancellation_separately_from_failures() {
        let (runtime, provider) = runtime();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let result = runtime
            .refresh(RefreshOptions {
                allow_network: true,
                cancellation: Some(cancellation),
                ..RefreshOptions::default()
            })
            .await;
        assert!(result.aborted);
        assert!(result.errors.is_empty());
        assert_eq!(provider.refreshes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn stream_resolves_auth_before_dispatch() {
        let (runtime, _) = runtime();
        let model = runtime.model("test", "model").expect("model");
        let context = Context {
            system_prompt: None,
            messages: vec![crate::message::Message::User(UserMessage::new("hello"))],
            tools: Vec::new(),
        };
        let result = runtime
            .complete(model, context, StreamOptions::default())
            .await
            .expect("complete");
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn unknown_or_unconfigured_provider_becomes_terminal_error() {
        let runtime = Models::default();
        let model = Model::new("missing", "model", "api", "https://example.test");
        let result = runtime
            .complete(model, Context::default(), StreamOptions::default())
            .await
            .expect("error message");
        assert_eq!(result.stop_reason, StopReason::Error);
        assert!(result.error_message.is_some());
    }
}
