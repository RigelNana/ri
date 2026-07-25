//! Layered, typed settings with serialized asynchronous persistence.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{Mutex as AsyncMutex, RwLock, mpsc, oneshot};

pub use ri_protocol_core::QueueMode;
use ri_protocol_core::ThinkingLevel;

use crate::atomic::{AtomicWriteOptions, atomic_write};
use crate::package::PackageSpec;

fn mutex_lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Default behavior when a project has trust-requiring resources.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultProjectTrust {
    /// Ask interactively when a UI is available; otherwise deny.
    #[default]
    Ask,
    /// Trust projects without prompting.
    Always,
    /// Deny trust without prompting.
    Never,
}

/// Turn compaction configuration. `None` means inherit from the lower layer.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct CompactionSettings {
    /// Whether automatic conversation compaction is enabled.
    pub enabled: Option<bool>,
    /// Tokens reserved for the next model response.
    pub reserve_tokens: Option<u64>,
    /// Recent tokens preserved verbatim during compaction.
    pub keep_recent_tokens: Option<u64>,
}

impl CompactionSettings {
    fn overlay(&self, higher: &Self) -> Self {
        Self {
            enabled: higher.enabled.or(self.enabled),
            reserve_tokens: higher.reserve_tokens.or(self.reserve_tokens),
            keep_recent_tokens: higher.keep_recent_tokens.or(self.keep_recent_tokens),
        }
    }
}

/// Provider-level retry settings.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct ProviderRetrySettings {
    /// Per-request provider timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Provider-specific retry limit.
    pub max_retries: Option<u32>,
    /// Maximum provider retry delay in milliseconds.
    pub max_retry_delay_ms: Option<u64>,
}

impl ProviderRetrySettings {
    fn overlay(&self, higher: &Self) -> Self {
        Self {
            timeout_ms: higher.timeout_ms.or(self.timeout_ms),
            max_retries: higher.max_retries.or(self.max_retries),
            max_retry_delay_ms: higher.max_retry_delay_ms.or(self.max_retry_delay_ms),
        }
    }
}

/// Agent retry settings.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct RetrySettings {
    /// Whether retries are enabled.
    pub enabled: Option<bool>,
    /// Agent-level retry limit.
    pub max_retries: Option<u32>,
    /// Initial retry delay in milliseconds.
    pub base_delay_ms: Option<u64>,
    /// Optional provider-specific overrides.
    pub provider: Option<ProviderRetrySettings>,
}

impl RetrySettings {
    fn overlay(&self, higher: &Self) -> Self {
        Self {
            enabled: higher.enabled.or(self.enabled),
            max_retries: higher.max_retries.or(self.max_retries),
            base_delay_ms: higher.base_delay_ms.or(self.base_delay_ms),
            provider: merge_optional(
                self.provider.as_ref(),
                higher.provider.as_ref(),
                ProviderRetrySettings::overlay,
            ),
        }
    }
}

/// Resource path settings. Arrays replace rather than concatenate across
/// layers, matching Pi settings semantics.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct ResourceSettings {
    /// Extension paths replacing the lower-precedence list.
    pub extensions: Option<Vec<PathBuf>>,
    /// Skill paths replacing the lower-precedence list.
    pub skills: Option<Vec<PathBuf>>,
    /// Prompt-template paths replacing the lower-precedence list.
    pub prompts: Option<Vec<PathBuf>>,
    /// Theme paths replacing the lower-precedence list.
    pub themes: Option<Vec<PathBuf>>,
}

impl ResourceSettings {
    fn overlay(&self, higher: &Self) -> Self {
        Self {
            extensions: higher
                .extensions
                .clone()
                .or_else(|| self.extensions.clone()),
            skills: higher.skills.clone().or_else(|| self.skills.clone()),
            prompts: higher.prompts.clone().or_else(|| self.prompts.clone()),
            themes: higher.themes.clone().or_else(|| self.themes.clone()),
        }
    }
}

/// Typed settings stored in global and project JSON files.
///
/// Fields not yet owned by this runtime are preserved verbatim so loading and
/// updating a complete Pi settings document does not erase them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Default provider identifier.
    pub default_provider: Option<String>,
    /// Default model identifier.
    pub default_model: Option<String>,
    /// Default model thinking/reasoning level.
    pub default_thinking_level: Option<ThinkingLevel>,
    /// Default policy for projects requiring trust.
    pub default_project_trust: Option<DefaultProjectTrust>,
    /// Delivery mode for steering input.
    pub steering_mode: Option<QueueMode>,
    /// Delivery mode for follow-up input.
    pub follow_up_mode: Option<QueueMode>,
    /// Conversation-compaction settings.
    pub compaction: Option<CompactionSettings>,
    /// Agent and provider retry settings.
    pub retry: Option<RetrySettings>,
    /// Configured extension, skill, prompt, and theme paths. Pi stores these
    /// arrays at the top level of `settings.json`.
    #[serde(flatten)]
    pub resources: ResourceSettings,
    /// Configured package specifications.
    pub packages: Option<Vec<PackageSpec>>,
    /// Whether discovered skills are exposed as slash commands.
    pub enable_skill_commands: Option<bool>,
    /// Optional directory used for session persistence.
    pub session_dir: Option<PathBuf>,
    /// Pi settings not yet interpreted by this runtime.
    #[serde(flatten)]
    pub extension: BTreeMap<String, Value>,
}

impl Settings {
    /// Overlay a higher-precedence settings layer.
    #[must_use]
    pub fn overlay(&self, higher: &Self) -> Self {
        let mut extension = self.extension.clone();
        for (key, value) in &higher.extension {
            extension.insert(key.clone(), value.clone());
        }
        Self {
            default_provider: higher
                .default_provider
                .clone()
                .or_else(|| self.default_provider.clone()),
            default_model: higher
                .default_model
                .clone()
                .or_else(|| self.default_model.clone()),
            default_thinking_level: higher
                .default_thinking_level
                .or(self.default_thinking_level),
            default_project_trust: higher.default_project_trust.or(self.default_project_trust),
            steering_mode: higher.steering_mode.or(self.steering_mode),
            follow_up_mode: higher.follow_up_mode.or(self.follow_up_mode),
            compaction: merge_optional(
                self.compaction.as_ref(),
                higher.compaction.as_ref(),
                CompactionSettings::overlay,
            ),
            retry: merge_optional(
                self.retry.as_ref(),
                higher.retry.as_ref(),
                RetrySettings::overlay,
            ),
            resources: self.resources.overlay(&higher.resources),
            packages: higher.packages.clone().or_else(|| self.packages.clone()),
            enable_skill_commands: higher.enable_skill_commands.or(self.enable_skill_commands),
            session_dir: higher
                .session_dir
                .clone()
                .or_else(|| self.session_dir.clone()),
            extension,
        }
    }

    /// Effective compaction defaults.
    pub fn resolved_compaction(&self) -> ResolvedCompactionSettings {
        let settings = self.compaction.as_ref();
        ResolvedCompactionSettings {
            enabled: settings.and_then(|value| value.enabled).unwrap_or(true),
            reserve_tokens: settings
                .and_then(|value| value.reserve_tokens)
                .unwrap_or(16_384),
            keep_recent_tokens: settings
                .and_then(|value| value.keep_recent_tokens)
                .unwrap_or(20_000),
        }
    }

    /// Effective retry defaults.
    pub fn resolved_retry(&self) -> ResolvedRetrySettings {
        let retry = self.retry.as_ref();
        let provider = retry.and_then(|value| value.provider.as_ref());
        ResolvedRetrySettings {
            enabled: retry.and_then(|value| value.enabled).unwrap_or(true),
            max_retries: retry.and_then(|value| value.max_retries).unwrap_or(3),
            base_delay_ms: retry.and_then(|value| value.base_delay_ms).unwrap_or(2_000),
            provider_timeout_ms: provider.and_then(|value| value.timeout_ms),
            provider_max_retries: provider.and_then(|value| value.max_retries),
            provider_max_retry_delay_ms: provider
                .and_then(|value| value.max_retry_delay_ms)
                .unwrap_or(60_000),
        }
    }
}

fn merge_optional<T: Clone>(
    lower: Option<&T>,
    higher: Option<&T>,
    merge: impl FnOnce(&T, &T) -> T,
) -> Option<T> {
    match (lower, higher) {
        (Some(lower), Some(higher)) => Some(merge(lower, higher)),
        (None, Some(higher)) => Some(higher.clone()),
        (Some(lower), None) => Some(lower.clone()),
        (None, None) => None,
    }
}

/// Fully defaulted compaction settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedCompactionSettings {
    /// Whether automatic compaction is enabled.
    pub enabled: bool,
    /// Tokens reserved for the next response.
    pub reserve_tokens: u64,
    /// Recent tokens preserved verbatim.
    pub keep_recent_tokens: u64,
}

/// Fully defaulted retry settings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedRetrySettings {
    /// Whether retries are enabled.
    pub enabled: bool,
    /// Agent-level retry limit.
    pub max_retries: u32,
    /// Initial retry delay in milliseconds.
    pub base_delay_ms: u64,
    /// Optional provider request timeout in milliseconds.
    pub provider_timeout_ms: Option<u64>,
    /// Optional provider-specific retry limit.
    pub provider_max_retries: Option<u32>,
    /// Maximum provider retry delay in milliseconds.
    pub provider_max_retry_delay_ms: u64,
}

/// Durable settings layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SettingsScope {
    /// User-wide settings.
    Global,
    /// Trust-gated project settings.
    Project,
}

/// Storage failure.
#[derive(Debug, Error)]
pub enum SettingsStorageError {
    /// Filesystem operation failed.
    #[error("settings I/O at {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// JSON serialization or deserialization failed.
    #[error("settings JSON at {path}: {source}")]
    Json {
        /// Path being encoded or decoded.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Storage backend returned a custom failure.
    #[error("{0}")]
    Backend(String),
}

/// Backend for global and project settings.
#[async_trait]
pub trait SettingsStorage: Send + Sync {
    /// Load one settings layer, returning defaults when the backend chooses to
    /// represent an absent layer that way.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsStorageError`] when the backend cannot read or decode
    /// the layer.
    async fn load(&self, scope: SettingsScope) -> Result<Settings, SettingsStorageError>;
    /// Persist one complete settings layer.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsStorageError`] when the backend cannot encode or
    /// durably store the layer.
    async fn save(
        &self,
        scope: SettingsScope,
        settings: &Settings,
    ) -> Result<(), SettingsStorageError>;
}

/// Atomic file-backed storage.
#[derive(Debug)]
pub struct FileSettingsStorage {
    global_path: PathBuf,
    project_path: PathBuf,
    io_lock: AsyncMutex<()>,
}

impl FileSettingsStorage {
    /// Global settings are `<agent_dir>/settings.json`; project settings are
    /// `<cwd>/.ri/settings.json`.
    pub fn new(cwd: impl Into<PathBuf>, agent_dir: impl Into<PathBuf>) -> Self {
        Self {
            global_path: agent_dir.into().join("settings.json"),
            project_path: cwd.into().join(".ri").join("settings.json"),
            io_lock: AsyncMutex::new(()),
        }
    }

    fn path(&self, scope: SettingsScope) -> &Path {
        match scope {
            SettingsScope::Global => &self.global_path,
            SettingsScope::Project => &self.project_path,
        }
    }
}

#[async_trait]
impl SettingsStorage for FileSettingsStorage {
    async fn load(&self, scope: SettingsScope) -> Result<Settings, SettingsStorageError> {
        let _guard = self.io_lock.lock().await;
        let path = self.path(scope);
        match tokio::fs::read(path).await {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).map_err(|source| SettingsStorageError::Json {
                    path: path.to_path_buf(),
                    source,
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Settings::default()),
            Err(source) => Err(SettingsStorageError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    async fn save(
        &self,
        scope: SettingsScope,
        settings: &Settings,
    ) -> Result<(), SettingsStorageError> {
        let _guard = self.io_lock.lock().await;
        let path = self.path(scope);
        let bytes =
            serde_json::to_vec_pretty(settings).map_err(|source| SettingsStorageError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        let mut terminated = bytes;
        terminated.push(b'\n');
        atomic_write(path, &terminated, AtomicWriteOptions::default())
            .await
            .map_err(|source| SettingsStorageError::Io {
                path: path.to_path_buf(),
                source,
            })
    }
}

/// In-memory backend useful for embedding and tests.
#[derive(Debug, Default)]
pub struct MemorySettingsStorage {
    values: AsyncMutex<BTreeMap<SettingsScope, Settings>>,
    fail_next: Mutex<Option<String>>,
}

impl MemorySettingsStorage {
    /// Cause the next load or save operation to fail.
    pub fn fail_next(&self, message: impl Into<String>) {
        *mutex_lock(&self.fail_next) = Some(message.into());
    }

    fn take_failure(&self) -> Option<SettingsStorageError> {
        mutex_lock(&self.fail_next)
            .take()
            .map(SettingsStorageError::Backend)
    }
}

#[async_trait]
impl SettingsStorage for MemorySettingsStorage {
    async fn load(&self, scope: SettingsScope) -> Result<Settings, SettingsStorageError> {
        if let Some(error) = self.take_failure() {
            return Err(error);
        }
        Ok(self
            .values
            .lock()
            .await
            .get(&scope)
            .cloned()
            .unwrap_or_default())
    }

    async fn save(
        &self,
        scope: SettingsScope,
        settings: &Settings,
    ) -> Result<(), SettingsStorageError> {
        if let Some(error) = self.take_failure() {
            return Err(error);
        }
        self.values.lock().await.insert(scope, settings.clone());
        Ok(())
    }
}

/// Non-fatal persistence or reload error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SettingsError {
    /// Layer whose operation failed.
    pub scope: SettingsScope,
    /// Human-readable backend error.
    pub message: String,
}

/// Settings manager control failure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SettingsManagerError {
    /// A project-layer mutation was attempted before trust was granted.
    #[error("project is not trusted; refusing to write project settings")]
    UntrustedProject,
    /// The background persistence worker is no longer available.
    #[error("settings persistence worker stopped")]
    WorkerStopped,
}

#[derive(Clone, Debug)]
struct LayerState {
    global: Settings,
    project: Settings,
    effective: Settings,
}

impl LayerState {
    fn new(global: Settings, project: Settings) -> Self {
        let effective = global.overlay(&project);
        Self {
            global,
            project,
            effective,
        }
    }

    fn refresh(&mut self) {
        self.effective = self.global.overlay(&self.project);
    }
}

enum WriteCommand {
    Save {
        scope: SettingsScope,
        settings: Box<Settings>,
    },
    Flush(oneshot::Sender<()>),
}

/// Global/project settings manager.
///
/// Setters update the in-memory effective value before enqueueing a durable
/// write. The single writer preserves call order. [`Self::flush`] is the
/// durability boundary; write failures are collected by [`Self::drain_errors`].
#[derive(Clone)]
pub struct SettingsManager {
    state: Arc<RwLock<LayerState>>,
    project_trusted: Arc<AtomicBool>,
    writer: mpsc::UnboundedSender<WriteCommand>,
    errors: Arc<Mutex<Vec<SettingsError>>>,
    storage: Arc<dyn SettingsStorage>,
}

impl std::fmt::Debug for SettingsManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SettingsManager")
            .field(
                "project_trusted",
                &self.project_trusted.load(Ordering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl SettingsManager {
    /// Open settings and start the serialized persistence worker.
    pub async fn open(storage: Arc<dyn SettingsStorage>, project_trusted: bool) -> Self {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let global =
            load_or_default(storage.as_ref(), SettingsScope::Global, Arc::clone(&errors)).await;
        let project = if project_trusted {
            load_or_default(
                storage.as_ref(),
                SettingsScope::Project,
                Arc::clone(&errors),
            )
            .await
        } else {
            Settings::default()
        };
        let state = Arc::new(RwLock::new(LayerState::new(global, project)));
        let (writer, receiver) = mpsc::unbounded_channel();
        tokio::spawn(settings_writer(
            Arc::clone(&storage),
            receiver,
            Arc::clone(&errors),
        ));
        Self {
            state,
            project_trusted: Arc::new(AtomicBool::new(project_trusted)),
            writer,
            errors,
            storage,
        }
    }

    /// Open conventional file paths.
    pub async fn open_files(
        cwd: impl Into<PathBuf>,
        agent_dir: impl Into<PathBuf>,
        project_trusted: bool,
    ) -> Self {
        Self::open(
            Arc::new(FileSettingsStorage::new(cwd, agent_dir)),
            project_trusted,
        )
        .await
    }

    /// Current merged settings.
    pub async fn effective(&self) -> Settings {
        self.state.read().await.effective.clone()
    }

    /// Current global layer.
    pub async fn global(&self) -> Settings {
        self.state.read().await.global.clone()
    }

    /// Current project layer. It is empty while the project is untrusted.
    pub async fn project(&self) -> Settings {
        self.state.read().await.project.clone()
    }

    /// Whether the project settings layer is currently enabled.
    pub fn is_project_trusted(&self) -> bool {
        self.project_trusted.load(Ordering::Acquire)
    }

    /// Mutate global settings and queue a write.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if persistence can no
    /// longer be queued.
    pub async fn update_global(
        &self,
        update: impl FnOnce(&mut Settings),
    ) -> Result<(), SettingsManagerError> {
        let snapshot = {
            let mut state = self.state.write().await;
            update(&mut state.global);
            state.refresh();
            state.global.clone()
        };
        self.writer
            .send(WriteCommand::Save {
                scope: SettingsScope::Global,
                settings: Box::new(snapshot),
            })
            .map_err(|_| SettingsManagerError::WorkerStopped)
    }

    /// Mutate project settings and queue a write.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::UntrustedProject`] while project trust
    /// is disabled, or [`SettingsManagerError::WorkerStopped`] if persistence
    /// can no longer be queued.
    pub async fn update_project(
        &self,
        update: impl FnOnce(&mut Settings),
    ) -> Result<(), SettingsManagerError> {
        if !self.is_project_trusted() {
            return Err(SettingsManagerError::UntrustedProject);
        }
        let snapshot = {
            let mut state = self.state.write().await;
            update(&mut state.project);
            state.refresh();
            state.project.clone()
        };
        self.writer
            .send(WriteCommand::Save {
                scope: SettingsScope::Project,
                settings: Box::new(snapshot),
            })
            .map_err(|_| SettingsManagerError::WorkerStopped)
    }

    /// Convenience setter for the global provider/model selection.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if persistence can no
    /// longer be queued.
    pub async fn set_default_model(
        &self,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<(), SettingsManagerError> {
        let provider = provider.into();
        let model = model.into();
        self.update_global(move |settings| {
            settings.default_provider = Some(provider);
            settings.default_model = Some(model);
        })
        .await
    }

    /// Replace global package specs.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if persistence can no
    /// longer be queued.
    pub async fn set_packages(
        &self,
        packages: Vec<PackageSpec>,
    ) -> Result<(), SettingsManagerError> {
        self.update_global(move |settings| settings.packages = Some(packages))
            .await
    }

    /// Replace project package specs.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::UntrustedProject`] while project trust
    /// is disabled, or [`SettingsManagerError::WorkerStopped`] if persistence
    /// can no longer be queued.
    pub async fn set_project_packages(
        &self,
        packages: Vec<PackageSpec>,
    ) -> Result<(), SettingsManagerError> {
        self.update_project(move |settings| settings.packages = Some(packages))
            .await
    }

    /// Wait until all previously queued writes have completed.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if the persistence
    /// worker has stopped.
    pub async fn flush(&self) -> Result<(), SettingsManagerError> {
        let (sender, receiver) = oneshot::channel();
        self.writer
            .send(WriteCommand::Flush(sender))
            .map_err(|_| SettingsManagerError::WorkerStopped)?;
        receiver
            .await
            .map_err(|_| SettingsManagerError::WorkerStopped)
    }

    /// Change the trust gate. Enabling loads the project layer; disabling
    /// immediately removes it from effective settings.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if pending writes
    /// cannot be flushed first. Project-load failures are recorded by
    /// [`Self::drain_errors`].
    pub async fn set_project_trusted(&self, trusted: bool) -> Result<(), SettingsManagerError> {
        self.flush().await?;
        let previous = self.project_trusted.swap(trusted, Ordering::AcqRel);
        if previous == trusted {
            return Ok(());
        }
        if trusted {
            match self.storage.load(SettingsScope::Project).await {
                Ok(project) => {
                    let mut state = self.state.write().await;
                    state.project = project;
                    state.refresh();
                }
                Err(error) => {
                    mutex_lock(&self.errors).push(SettingsError {
                        scope: SettingsScope::Project,
                        message: error.to_string(),
                    });
                    let mut state = self.state.write().await;
                    state.project = Settings::default();
                    state.refresh();
                }
            }
        } else {
            let mut state = self.state.write().await;
            state.project = Settings::default();
            state.refresh();
        }
        Ok(())
    }

    /// Flush pending writes, then reload both permitted layers. A failed layer
    /// retains its previous in-memory value and records an error.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsManagerError::WorkerStopped`] if pending writes
    /// cannot be flushed.
    pub async fn reload(&self) -> Result<(), SettingsManagerError> {
        self.flush().await?;
        let global = self.storage.load(SettingsScope::Global).await;
        let project = if self.is_project_trusted() {
            Some(self.storage.load(SettingsScope::Project).await)
        } else {
            None
        };
        let mut state = self.state.write().await;
        match global {
            Ok(global) => state.global = global,
            Err(error) => mutex_lock(&self.errors).push(SettingsError {
                scope: SettingsScope::Global,
                message: error.to_string(),
            }),
        }
        match project {
            Some(Ok(project)) => state.project = project,
            Some(Err(error)) => mutex_lock(&self.errors).push(SettingsError {
                scope: SettingsScope::Project,
                message: error.to_string(),
            }),
            None => state.project = Settings::default(),
        }
        state.refresh();
        Ok(())
    }

    /// Drain persistence and reload failures.
    pub fn drain_errors(&self) -> Vec<SettingsError> {
        std::mem::take(&mut *mutex_lock(&self.errors))
    }
}

async fn load_or_default(
    storage: &dyn SettingsStorage,
    scope: SettingsScope,
    errors: Arc<Mutex<Vec<SettingsError>>>,
) -> Settings {
    match storage.load(scope).await {
        Ok(settings) => settings,
        Err(error) => {
            mutex_lock(&errors).push(SettingsError {
                scope,
                message: error.to_string(),
            });
            Settings::default()
        }
    }
}

async fn settings_writer(
    storage: Arc<dyn SettingsStorage>,
    mut receiver: mpsc::UnboundedReceiver<WriteCommand>,
    errors: Arc<Mutex<Vec<SettingsError>>>,
) {
    while let Some(command) = receiver.recv().await {
        match command {
            WriteCommand::Save { scope, settings } => {
                if let Err(error) = storage.save(scope, &settings).await {
                    mutex_lock(&errors).push(SettingsError {
                        scope,
                        message: error.to_string(),
                    });
                }
            }
            WriteCommand::Flush(sender) => {
                let _ = sender.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn nested_settings_merge_while_arrays_replace() {
        let lower = Settings {
            compaction: Some(CompactionSettings {
                enabled: Some(false),
                reserve_tokens: Some(10),
                keep_recent_tokens: Some(20),
            }),
            resources: ResourceSettings {
                skills: Some(vec![PathBuf::from("global")]),
                ..ResourceSettings::default()
            },
            ..Settings::default()
        };
        let higher = Settings {
            compaction: Some(CompactionSettings {
                enabled: Some(true),
                ..CompactionSettings::default()
            }),
            resources: ResourceSettings {
                skills: Some(vec![PathBuf::from("project")]),
                ..ResourceSettings::default()
            },
            ..Settings::default()
        };
        let merged = lower.overlay(&higher);
        assert_eq!(
            merged.compaction,
            Some(CompactionSettings {
                enabled: Some(true),
                reserve_tokens: Some(10),
                keep_recent_tokens: Some(20),
            })
        );
        assert_eq!(
            merged.resources.skills,
            Some(vec![PathBuf::from("project")])
        );
    }

    #[tokio::test]
    async fn flush_is_a_durability_boundary() {
        let storage = Arc::new(MemorySettingsStorage::default());
        let manager = SettingsManager::open(storage.clone(), true).await;
        manager
            .set_default_model("provider", "model")
            .await
            .expect("queue");
        manager.flush().await.expect("flush");
        let stored = storage
            .load(SettingsScope::Global)
            .await
            .expect("stored settings");
        assert_eq!(stored.default_provider.as_deref(), Some("provider"));
        assert_eq!(stored.default_model.as_deref(), Some("model"));
    }

    #[tokio::test]
    async fn write_failures_are_queued_without_reverting_memory() {
        let storage = Arc::new(MemorySettingsStorage::default());
        let manager = SettingsManager::open(storage.clone(), true).await;
        storage.fail_next("disk full");
        manager
            .set_default_model("provider", "model")
            .await
            .expect("queue");
        manager.flush().await.expect("flush");
        assert_eq!(
            manager.effective().await.default_model.as_deref(),
            Some("model")
        );
        let errors = manager.drain_errors();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("disk full"));
    }

    #[tokio::test]
    async fn untrusted_projects_cannot_read_or_write_project_layer() {
        let storage = Arc::new(MemorySettingsStorage::default());
        storage
            .save(
                SettingsScope::Project,
                &Settings {
                    default_model: Some("unsafe".to_owned()),
                    ..Settings::default()
                },
            )
            .await
            .expect("seed");
        let manager = SettingsManager::open(storage, false).await;
        assert_eq!(manager.effective().await.default_model, None);
        assert_eq!(
            manager
                .update_project(|settings| settings.default_model = Some("x".to_owned()))
                .await,
            Err(SettingsManagerError::UntrustedProject)
        );
    }

    #[tokio::test]
    async fn file_storage_round_trips_both_layers() {
        let temp = tempdir().expect("tempdir");
        let cwd = temp.path().join("project");
        let agent = temp.path().join("agent");
        let manager = SettingsManager::open_files(&cwd, &agent, true).await;
        manager
            .update_global(|settings| settings.default_provider = Some("global".to_owned()))
            .await
            .expect("global");
        manager
            .update_project(|settings| settings.default_model = Some("project".to_owned()))
            .await
            .expect("project");
        manager.flush().await.expect("flush");

        let reopened = SettingsManager::open_files(&cwd, &agent, true).await;
        let effective = reopened.effective().await;
        assert_eq!(effective.default_provider.as_deref(), Some("global"));
        assert_eq!(effective.default_model.as_deref(), Some("project"));
    }
}
