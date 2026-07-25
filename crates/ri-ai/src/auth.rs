//! Provider-scoped credential storage and authentication resolution.

use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use async_trait::async_trait;
use indexmap::IndexMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex as AsyncMutex, RwLock};

use crate::error::AiError;

/// Case-preserving provider headers. `None` suppresses a default header.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;
/// Provider-scoped environment/configuration.
pub type ProviderEnv = BTreeMap<String, String>;

/// Authentication material applied to one request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelAuth {
    /// Bearer/API key consumed by the adapter.
    pub api_key: Option<String>,
    /// Additional or overriding headers.
    pub headers: ProviderHeaders,
    /// Credential-specific endpoint override.
    pub base_url: Option<String>,
}

/// Stored API-key credential.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeyCredential {
    /// Secret key. Ambient-only providers may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Provider-specific configuration such as project/account ids.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: ProviderEnv,
}

/// Stored OAuth token set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthCredential {
    /// Refresh token.
    pub refresh: String,
    /// Access token.
    pub access: String,
    /// Expiration in Unix milliseconds.
    pub expires: i64,
    /// Provider-owned non-secret token metadata.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, Value>,
}

/// One credential per provider.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    /// API-key or ambient configuration.
    ApiKey(ApiKeyCredential),
    /// OAuth token set.
    Oauth(OAuthCredential),
}

impl Credential {
    /// Stable credential type label.
    pub const fn kind(&self) -> CredentialKind {
        match self {
            Self::ApiKey(_) => CredentialKind::ApiKey,
            Self::Oauth(_) => CredentialKind::Oauth,
        }
    }
}

/// Non-secret credential type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialKind {
    /// API key.
    ApiKey,
    /// OAuth.
    Oauth,
}

/// Non-secret stored credential metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialInfo {
    /// Provider id.
    pub provider_id: String,
    /// Stored credential type.
    pub kind: CredentialKind,
}

/// Async mutation executed while a provider credential lock is held.
pub type CredentialModifier = Box<
    dyn FnOnce(
            Option<Credential>,
        ) -> Pin<Box<dyn Future<Output = Result<Option<Credential>, AiError>> + Send>>
        + Send,
>;

/// App-owned credential persistence.
#[async_trait]
pub trait CredentialStore: Send + Sync + std::fmt::Debug {
    /// Reads a credential without refreshing it.
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, AiError>;
    /// Lists non-secret metadata.
    async fn list(&self) -> Result<Vec<CredentialInfo>, AiError>;
    /// Performs a serialized read-modify-write for one provider.
    ///
    /// Returning `None` leaves the current entry unchanged. The returned value
    /// is always the post-transaction credential.
    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
    ) -> Result<Option<Credential>, AiError>;
    /// Deletes a credential, serialized against [`Self::modify`].
    async fn delete(&self, provider_id: &str) -> Result<(), AiError>;
}

/// In-memory store with per-provider asynchronous locks.
#[derive(Debug, Default)]
pub struct InMemoryCredentialStore {
    credentials: RwLock<IndexMap<String, Credential>>,
    locks: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

impl InMemoryCredentialStore {
    fn provider_lock(&self, provider_id: &str) -> Arc<AsyncMutex<()>> {
        self.locks
            .lock()
            .entry(provider_id.to_owned())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    /// Inserts a credential through the same locked mutation path used by
    /// login and refresh.
    ///
    /// # Errors
    ///
    /// Returns an authentication error when the credential-store mutation
    /// cannot be completed.
    pub async fn set(
        &self,
        provider_id: impl Into<String>,
        credential: Credential,
    ) -> Result<(), AiError> {
        let provider_id = provider_id.into();
        self.modify(
            &provider_id,
            Box::new(move |_| Box::pin(async move { Ok(Some(credential)) })),
        )
        .await?;
        Ok(())
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, AiError> {
        Ok(self.credentials.read().await.get(provider_id).cloned())
    }

    async fn list(&self) -> Result<Vec<CredentialInfo>, AiError> {
        Ok(self
            .credentials
            .read()
            .await
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                kind: credential.kind(),
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
    ) -> Result<Option<Credential>, AiError> {
        let lock = self.provider_lock(provider_id);
        let _guard = lock.lock().await;
        let current = self.credentials.read().await.get(provider_id).cloned();
        let replacement = modifier(current.clone()).await?;
        if let Some(replacement) = replacement {
            self.credentials
                .write()
                .await
                .insert(provider_id.to_owned(), replacement.clone());
            Ok(Some(replacement))
        } else {
            Ok(current)
        }
    }

    async fn delete(&self, provider_id: &str) -> Result<(), AiError> {
        let lock = self.provider_lock(provider_id);
        let _guard = lock.lock().await;
        self.credentials.write().await.shift_remove(provider_id);
        Ok(())
    }
}

/// Environment and filesystem access used during auth resolution.
#[async_trait]
pub trait AuthContext: Send + Sync + std::fmt::Debug {
    /// Reads an environment value.
    async fn env(&self, name: &str) -> Option<String>;
    /// Checks a path, supporting a leading `~`.
    async fn file_exists(&self, path: &str) -> bool;
}

/// Host process auth context.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemAuthContext;

#[async_trait]
impl AuthContext for SystemAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|value| !value.is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        tokio::fs::try_exists(expand_home(path))
            .await
            .unwrap_or(false)
    }
}

fn expand_home(path: &str) -> PathBuf {
    let path = Path::new(path);
    if !path.starts_with("~") {
        return path.to_path_buf();
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    match home {
        Some(home) => path
            .strip_prefix("~")
            .map_or_else(|_| path.to_path_buf(), |suffix| home.join(suffix)),
        None => path.to_path_buf(),
    }
}

/// Result of provider authentication resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthResult {
    /// Request auth.
    pub auth: ModelAuth,
    /// Resolved provider environment/configuration.
    pub env: ProviderEnv,
    /// Human-readable source label.
    pub source: Option<String>,
}

/// API-key resolver owned by a provider.
#[async_trait]
pub trait ApiKeyAuth: Send + Sync + std::fmt::Debug {
    /// Display name.
    fn name(&self) -> &str;
    /// Resolves stored and/or ambient configuration.
    async fn resolve(
        &self,
        context: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, AiError>;
}

/// OAuth refresh and request-auth adapter owned by a provider.
#[async_trait]
pub trait OAuthAuth: Send + Sync + std::fmt::Debug {
    /// Display name.
    fn name(&self) -> &str;
    /// Exchanges an expired credential.
    async fn refresh(&self, credential: OAuthCredential) -> Result<OAuthCredential, AiError>;
    /// Derives request auth from a valid token.
    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, AiError>;
}

/// Provider authentication methods.
#[derive(Clone, Default)]
pub struct ProviderAuth {
    /// API-key/ambient method.
    pub api_key: Option<Arc<dyn ApiKeyAuth>>,
    /// OAuth method.
    pub oauth: Option<Arc<dyn OAuthAuth>>,
}

impl std::fmt::Debug for ProviderAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|auth| auth.name()))
            .field("oauth", &self.oauth.as_ref().map(|auth| auth.name()))
            .finish()
    }
}

impl ProviderAuth {
    /// Creates API-key-only auth.
    pub fn api_key(auth: Arc<dyn ApiKeyAuth>) -> Self {
        Self {
            api_key: Some(auth),
            oauth: None,
        }
    }
}

/// Per-request auth overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthResolutionOverrides {
    /// Explicit API key. Takes precedence when the provider has API-key auth.
    pub api_key: Option<String>,
    /// Environment/configuration overlay.
    pub env: ProviderEnv,
}

#[derive(Debug)]
struct OverlayAuthContext<'a> {
    base: &'a dyn AuthContext,
    env: &'a ProviderEnv,
}

#[async_trait]
impl AuthContext for OverlayAuthContext<'_> {
    async fn env(&self, name: &str) -> Option<String> {
        if let Some(value) = self.env.get(name).filter(|value| !value.is_empty()) {
            return Some(value.clone());
        }
        self.base.env(name).await
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}

/// Resolves provider auth with explicit > stored > ambient precedence.
///
/// A stored credential owns the provider. If its type has no matching handler,
/// or OAuth refresh fails, ambient credentials are not consulted.
///
/// # Errors
///
/// Returns an authentication error when credential-store access, API-key
/// resolution, OAuth refresh, or OAuth request-auth conversion fails.
pub async fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &dyn CredentialStore,
    context: &dyn AuthContext,
    overrides: &AuthResolutionOverrides,
) -> Result<Option<AuthResult>, AiError> {
    let overlay = OverlayAuthContext {
        base: context,
        env: &overrides.env,
    };
    let request_context: &dyn AuthContext = if overrides.env.is_empty() {
        context
    } else {
        &overlay
    };

    if let (Some(explicit), Some(api_key_auth)) = (&overrides.api_key, &auth.api_key) {
        let credential = ApiKeyCredential {
            key: Some(explicit.clone()),
            env: overrides.env.clone(),
        };
        return api_key_auth
            .resolve(request_context, Some(&credential))
            .await
            .map_err(|error| wrap_auth_error(provider_id, error));
    }

    let stored = credentials.read(provider_id).await.map_err(|error| {
        AiError::Auth(format!(
            "credential store read failed for {provider_id}: {error}"
        ))
    })?;
    if let Some(stored) = stored {
        return match stored {
            Credential::Oauth(credential) => {
                let Some(oauth) = &auth.oauth else {
                    return Ok(None);
                };
                resolve_stored_oauth(provider_id, oauth.clone(), credentials, credential).await
            }
            Credential::ApiKey(mut credential) => {
                let Some(api_key_auth) = &auth.api_key else {
                    return Ok(None);
                };
                credential.env.extend(overrides.env.clone());
                api_key_auth
                    .resolve(request_context, Some(&credential))
                    .await
                    .map_err(|error| wrap_auth_error(provider_id, error))
            }
        };
    }

    let Some(api_key_auth) = &auth.api_key else {
        return Ok(None);
    };
    api_key_auth
        .resolve(request_context, None)
        .await
        .map_err(|error| wrap_auth_error(provider_id, error))
}

fn wrap_auth_error(provider_id: &str, error: AiError) -> AiError {
    match error {
        AiError::Auth(_) => error,
        _ => AiError::Auth(format!(
            "API key auth failed for provider {provider_id}: {error}"
        )),
    }
}

async fn resolve_stored_oauth(
    provider_id: &str,
    oauth: Arc<dyn OAuthAuth>,
    credentials: &dyn CredentialStore,
    mut credential: OAuthCredential,
) -> Result<Option<AuthResult>, AiError> {
    if chrono::Utc::now().timestamp_millis() >= credential.expires {
        let id = provider_id.to_owned();
        let refresh = oauth.clone();
        let post = credentials
            .modify(
                provider_id,
                Box::new(move |current| {
                    Box::pin(async move {
                        let Some(Credential::Oauth(current)) = current else {
                            return Ok(None);
                        };
                        if chrono::Utc::now().timestamp_millis() < current.expires {
                            return Ok(None);
                        }
                        refresh
                            .refresh(current)
                            .await
                            .map(Credential::Oauth)
                            .map(Some)
                            .map_err(|error| {
                                AiError::Oauth(format!("OAuth refresh failed for {id}: {error}"))
                            })
                    })
                }),
            )
            .await
            .map_err(|error| match error {
                AiError::Oauth(_) => error,
                _ => AiError::Auth(format!(
                    "credential store modify failed for {provider_id}: {error}"
                )),
            })?;
        let Some(Credential::Oauth(refreshed)) = post else {
            return Ok(None);
        };
        credential = refreshed;
    }
    let auth = oauth.to_auth(&credential).await.map_err(|error| {
        AiError::Oauth(format!(
            "OAuth auth derivation failed for {provider_id}: {error}"
        ))
    })?;
    Ok(Some(AuthResult {
        auth,
        env: ProviderEnv::new(),
        source: Some("OAuth".into()),
    }))
}

/// Simple environment-variable API-key resolver used by built-in providers.
#[derive(Clone, Debug)]
pub struct EnvApiKeyAuth {
    name: String,
    variables: Vec<String>,
}

impl EnvApiKeyAuth {
    /// Creates an environment API-key resolver.
    pub fn new(
        name: impl Into<String>,
        variables: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            variables: variables.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn resolve(
        &self,
        context: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, AiError> {
        if let Some(key) = credential.and_then(|credential| credential.key.as_ref()) {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key.clone()),
                    ..ModelAuth::default()
                },
                env: credential.map_or_else(ProviderEnv::new, |credential| credential.env.clone()),
                source: Some("stored credential".into()),
            }));
        }
        for variable in &self.variables {
            if let Some(key) = context.env(variable).await {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(key),
                        ..ModelAuth::default()
                    },
                    env: ProviderEnv::new(),
                    source: Some(variable.clone()),
                }));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

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
    struct TestOAuth {
        refreshes: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait]
    impl OAuthAuth for TestOAuth {
        fn name(&self) -> &'static str {
            "test oauth"
        }

        async fn refresh(
            &self,
            mut credential: OAuthCredential,
        ) -> Result<OAuthCredential, AiError> {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            if self.fail {
                return Err(AiError::Http("invalid_grant".into()));
            }
            credential.access = "fresh".into();
            credential.expires = chrono::Utc::now().timestamp_millis() + 60_000;
            Ok(credential)
        }

        async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, AiError> {
            Ok(ModelAuth {
                api_key: Some(credential.access.clone()),
                ..ModelAuth::default()
            })
        }
    }

    fn expired() -> Credential {
        Credential::Oauth(OAuthCredential {
            access: "expired".into(),
            refresh: "refresh".into(),
            expires: 0,
            extra: BTreeMap::new(),
        })
    }

    #[tokio::test]
    async fn explicit_then_stored_then_ambient_precedence() {
        let store = InMemoryCredentialStore::default();
        let context = TestContext {
            env: BTreeMap::from([("TEST_KEY".into(), "ambient".into())]),
        };
        let auth = ProviderAuth::api_key(Arc::new(EnvApiKeyAuth::new("key", ["TEST_KEY"])));

        let ambient = resolve_provider_auth(
            "p",
            &auth,
            &store,
            &context,
            &AuthResolutionOverrides::default(),
        )
        .await
        .expect("resolve")
        .expect("ambient");
        assert_eq!(ambient.auth.api_key.as_deref(), Some("ambient"));

        store
            .set(
                "p",
                Credential::ApiKey(ApiKeyCredential {
                    key: Some("stored".into()),
                    env: ProviderEnv::new(),
                }),
            )
            .await
            .expect("store");
        let stored = resolve_provider_auth(
            "p",
            &auth,
            &store,
            &context,
            &AuthResolutionOverrides::default(),
        )
        .await
        .expect("resolve")
        .expect("stored");
        assert_eq!(stored.auth.api_key.as_deref(), Some("stored"));

        let explicit = resolve_provider_auth(
            "p",
            &auth,
            &store,
            &context,
            &AuthResolutionOverrides {
                api_key: Some("explicit".into()),
                env: ProviderEnv::new(),
            },
        )
        .await
        .expect("resolve")
        .expect("explicit");
        assert_eq!(explicit.auth.api_key.as_deref(), Some("explicit"));
    }

    #[tokio::test]
    async fn concurrent_expired_tokens_refresh_once() {
        let store = Arc::new(InMemoryCredentialStore::default());
        store.set("p", expired()).await.expect("store");
        let refreshes = Arc::new(AtomicUsize::new(0));
        let auth = ProviderAuth {
            api_key: None,
            oauth: Some(Arc::new(TestOAuth {
                refreshes: refreshes.clone(),
                fail: false,
            })),
        };
        let context = TestContext::default();
        let first_overrides = AuthResolutionOverrides::default();
        let second_overrides = AuthResolutionOverrides::default();
        let first = resolve_provider_auth("p", &auth, store.as_ref(), &context, &first_overrides);
        let second = resolve_provider_auth("p", &auth, store.as_ref(), &context, &second_overrides);
        let (first, second) = tokio::join!(first, second);
        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert_eq!(
            first
                .expect("first")
                .expect("first auth")
                .auth
                .api_key
                .as_deref(),
            Some("fresh")
        );
        assert_eq!(
            second
                .expect("second")
                .expect("second auth")
                .auth
                .api_key
                .as_deref(),
            Some("fresh")
        );
    }

    #[tokio::test]
    async fn failed_refresh_preserves_credential_and_never_falls_back() {
        let store = InMemoryCredentialStore::default();
        store.set("p", expired()).await.expect("store");
        let auth = ProviderAuth {
            api_key: Some(Arc::new(EnvApiKeyAuth::new("key", ["TEST_KEY"]))),
            oauth: Some(Arc::new(TestOAuth {
                refreshes: Arc::new(AtomicUsize::new(0)),
                fail: true,
            })),
        };
        let context = TestContext {
            env: BTreeMap::from([("TEST_KEY".into(), "ambient".into())]),
        };
        assert!(matches!(
            resolve_provider_auth(
                "p",
                &auth,
                &store,
                &context,
                &AuthResolutionOverrides::default()
            )
            .await,
            Err(AiError::Oauth(_))
        ));
        assert_eq!(store.read("p").await.expect("read"), Some(expired()));
    }

    #[tokio::test]
    async fn mismatched_stored_type_blocks_ambient() {
        let store = InMemoryCredentialStore::default();
        store.set("p", expired()).await.expect("store");
        let auth = ProviderAuth::api_key(Arc::new(EnvApiKeyAuth::new("key", ["TEST_KEY"])));
        let context = TestContext {
            env: BTreeMap::from([("TEST_KEY".into(), "ambient".into())]),
        };
        assert!(
            resolve_provider_auth(
                "p",
                &auth,
                &store,
                &context,
                &AuthResolutionOverrides::default()
            )
            .await
            .expect("resolve")
            .is_none()
        );
    }
}
