//! Durable provider credentials owned by the CLI application.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ri_ai::auth::CredentialModifier;
use ri_ai::{AiError, Credential, CredentialInfo, CredentialStore};
use ri_ext::atomic::{AtomicWriteOptions, atomic_write};
use tokio::sync::Mutex;

/// JSON credential store used by the standalone binary.
///
/// The SDK deliberately accepts an application-owned store. Keeping this
/// implementation here avoids teaching the SDK where a CLI should put secrets.
#[derive(Debug)]
pub(crate) struct FileCredentialStore {
    path: PathBuf,
    transaction: Mutex<()>,
}

impl FileCredentialStore {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            path,
            transaction: Mutex::new(()),
        }
    }

    async fn load(&self) -> Result<BTreeMap<String, Credential>, AiError> {
        match tokio::fs::read(&self.path).await {
            Ok(bytes) if bytes.is_empty() => Ok(BTreeMap::new()),
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| {
                AiError::Auth(format!(
                    "failed to decode credential file `{}`: {error}",
                    self.path.display()
                ))
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(error) => Err(auth_io("read", &self.path, &error)),
        }
    }

    async fn save(&self, credentials: &BTreeMap<String, Credential>) -> Result<(), AiError> {
        self.path.parent().ok_or_else(|| {
            AiError::Auth(format!(
                "credential path `{}` has no parent directory",
                self.path.display()
            ))
        })?;
        let mut bytes = serde_json::to_vec_pretty(credentials)
            .map_err(|error| AiError::Auth(format!("failed to encode credentials: {error}")))?;
        bytes.push(b'\n');

        atomic_write(
            &self.path,
            &bytes,
            AtomicWriteOptions {
                unix_mode: Some(0o600),
            },
        )
        .await
        .map_err(|error| auth_io("replace", &self.path, &error))
    }
}

#[async_trait]
impl CredentialStore for FileCredentialStore {
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, AiError> {
        let _transaction = self.transaction.lock().await;
        Ok(self.load().await?.get(provider_id).cloned())
    }

    async fn list(&self) -> Result<Vec<CredentialInfo>, AiError> {
        let _transaction = self.transaction.lock().await;
        Ok(self
            .load()
            .await?
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
                kind: credential.kind(),
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
    ) -> Result<Option<Credential>, AiError> {
        let _transaction = self.transaction.lock().await;
        let mut credentials = self.load().await?;
        let current = credentials.get(provider_id).cloned();
        let replacement = modifier(current.clone()).await?;
        let Some(replacement) = replacement else {
            return Ok(current);
        };
        credentials.insert(provider_id.to_owned(), replacement.clone());
        self.save(&credentials).await?;
        Ok(Some(replacement))
    }

    async fn delete(&self, provider_id: &str) -> Result<(), AiError> {
        let _transaction = self.transaction.lock().await;
        let mut credentials = self.load().await?;
        if credentials.remove(provider_id).is_some() {
            self.save(&credentials).await?;
        }
        Ok(())
    }
}

/// Non-persistent one-run credentials layered over a durable store.
#[derive(Debug)]
pub(crate) struct OverlayCredentialStore {
    base: Arc<dyn CredentialStore>,
    overrides: BTreeMap<String, Credential>,
}

impl OverlayCredentialStore {
    pub(crate) fn new(
        base: Arc<dyn CredentialStore>,
        overrides: BTreeMap<String, Credential>,
    ) -> Self {
        Self { base, overrides }
    }
}

#[async_trait]
impl CredentialStore for OverlayCredentialStore {
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, AiError> {
        match self.overrides.get(provider_id) {
            Some(credential) => Ok(Some(credential.clone())),
            None => self.base.read(provider_id).await,
        }
    }

    async fn list(&self) -> Result<Vec<CredentialInfo>, AiError> {
        let mut listed = self.base.list().await?;
        for (provider_id, credential) in &self.overrides {
            if let Some(existing) = listed
                .iter_mut()
                .find(|candidate| candidate.provider_id == *provider_id)
            {
                existing.kind = credential.kind();
            } else {
                listed.push(CredentialInfo {
                    provider_id: provider_id.clone(),
                    kind: credential.kind(),
                });
            }
        }
        listed.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
        Ok(listed)
    }

    async fn modify(
        &self,
        provider_id: &str,
        modifier: CredentialModifier,
    ) -> Result<Option<Credential>, AiError> {
        self.base.modify(provider_id, modifier).await
    }

    async fn delete(&self, provider_id: &str) -> Result<(), AiError> {
        self.base.delete(provider_id).await
    }
}

fn auth_io(operation: &str, path: &Path, error: &std::io::Error) -> AiError {
    AiError::Auth(format!("{operation} `{}`: {error}", path.display()))
}
