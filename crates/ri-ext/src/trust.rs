//! Project trust storage, resource detection, and ordered resolution.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::atomic::{AtomicWriteOptions, atomic_write};
use crate::extension::{
    ExtensionRunner, ProjectTrustDecision as HookTrustDecision, ProjectTrustResult,
};
use crate::settings::DefaultProjectTrust;

const TRUST_REQUIRING_RI_ENTRIES: &[&str] = &[
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
    "ri-package.toml",
];

/// Saved trust entry inherited by descendants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustEntry {
    /// Exact path at which the inherited decision was stored.
    pub path: PathBuf,
    /// Whether the path is trusted.
    pub trusted: bool,
}

/// One atomic trust-store update. `None` removes an exact entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustUpdate {
    /// Exact project path to update.
    pub path: PathBuf,
    /// Decision to store, or `None` to remove the entry.
    pub decision: Option<bool>,
}

/// Trust-store failure.
#[derive(Debug, Error)]
pub enum TrustStoreError {
    /// Filesystem operation failed.
    #[error("trust-store I/O at {path}: {source}")]
    Io {
        /// Trust-store path being accessed.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// Stored trust data was malformed.
    #[error("invalid trust store at {path}: {message}")]
    Invalid {
        /// Path containing malformed data.
        path: PathBuf,
        /// Validation or decoding failure.
        message: String,
    },
    /// Storage backend returned a custom failure.
    #[error("{0}")]
    Backend(String),
}

/// Durable project trust backend.
#[async_trait]
pub trait TrustStore: Send + Sync {
    /// Return the nearest exact or ancestor entry.
    ///
    /// # Errors
    ///
    /// Returns [`TrustStoreError`] when the backend cannot read or decode its
    /// stored decisions.
    async fn get_entry(&self, cwd: &Path) -> Result<Option<TrustEntry>, TrustStoreError>;
    /// Apply updates atomically.
    ///
    /// # Errors
    ///
    /// Returns [`TrustStoreError`] when the backend cannot persist the update
    /// set.
    async fn set_many(&self, updates: &[TrustUpdate]) -> Result<(), TrustStoreError>;
}

/// JSON trust store at `<agent_dir>/trust.json`.
#[derive(Debug)]
pub struct FileTrustStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl FileTrustStore {
    /// Create a JSON trust store under the supplied user configuration root.
    pub fn new(agent_dir: impl Into<PathBuf>) -> Self {
        Self {
            path: agent_dir.into().join("trust.json"),
            lock: Mutex::new(()),
        }
    }

    async fn read(&self) -> Result<BTreeMap<String, bool>, TrustStoreError> {
        match tokio::fs::read(&self.path).await {
            Ok(bytes) => {
                serde_json::from_slice::<BTreeMap<String, bool>>(&bytes).map_err(|error| {
                    TrustStoreError::Invalid {
                        path: self.path.clone(),
                        message: error.to_string(),
                    }
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(source) => Err(TrustStoreError::Io {
                path: self.path.clone(),
                source,
            }),
        }
    }

    async fn write(&self, values: &BTreeMap<String, bool>) -> Result<(), TrustStoreError> {
        let mut bytes =
            serde_json::to_vec_pretty(values).map_err(|error| TrustStoreError::Invalid {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        bytes.push(b'\n');
        atomic_write(&self.path, &bytes, AtomicWriteOptions::default())
            .await
            .map_err(|source| TrustStoreError::Io {
                path: self.path.clone(),
                source,
            })
    }
}

#[async_trait]
impl TrustStore for FileTrustStore {
    async fn get_entry(&self, cwd: &Path) -> Result<Option<TrustEntry>, TrustStoreError> {
        let _guard = self.lock.lock().await;
        let values = self.read().await?;
        find_nearest(&values, cwd)
    }

    async fn set_many(&self, updates: &[TrustUpdate]) -> Result<(), TrustStoreError> {
        let _guard = self.lock.lock().await;
        let mut values = self.read().await?;
        for update in updates {
            let key = normalize_path(&update.path)?.to_string_lossy().into_owned();
            if let Some(decision) = update.decision {
                values.insert(key, decision);
            } else {
                values.remove(&key);
            }
        }
        self.write(&values).await
    }
}

/// In-memory trust store.
#[derive(Debug, Default)]
pub struct MemoryTrustStore {
    values: Mutex<BTreeMap<String, bool>>,
}

#[async_trait]
impl TrustStore for MemoryTrustStore {
    async fn get_entry(&self, cwd: &Path) -> Result<Option<TrustEntry>, TrustStoreError> {
        let values = self.values.lock().await;
        find_nearest(&values, cwd)
    }

    async fn set_many(&self, updates: &[TrustUpdate]) -> Result<(), TrustStoreError> {
        let mut values = self.values.lock().await;
        for update in updates {
            let key = normalize_path(&update.path)?.to_string_lossy().into_owned();
            if let Some(decision) = update.decision {
                values.insert(key, decision);
            } else {
                values.remove(&key);
            }
        }
        Ok(())
    }
}

fn find_nearest(
    values: &BTreeMap<String, bool>,
    cwd: &Path,
) -> Result<Option<TrustEntry>, TrustStoreError> {
    let mut current = normalize_path(cwd)?;
    loop {
        let key = current.to_string_lossy();
        if let Some(trusted) = values.get(key.as_ref()) {
            return Ok(Some(TrustEntry {
                path: current,
                trusted: *trusted,
            }));
        }
        if !current.pop() {
            return Ok(None);
        }
    }
}

fn normalize_path(path: &Path) -> Result<PathBuf, TrustStoreError> {
    std::fs::canonicalize(path).map_err(|source| TrustStoreError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// User-facing trust choice. Multiple updates allow "trust parent" while
/// clearing a more-specific child decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustPromptChoice {
    /// Decision returned for the current session.
    pub trusted: bool,
    /// Durable trust-store changes selected by the user.
    pub updates: Vec<TrustUpdate>,
}

/// Interactive trust prompt supplied by a TUI/RPC host.
#[async_trait]
pub trait TrustPrompt: Send + Sync {
    /// Ask the user how to handle an undecided project.
    ///
    /// `Ok(None)` represents cancellation and is treated as denial.
    ///
    /// # Errors
    ///
    /// Returns [`TrustResolveError`] when the host cannot present or process
    /// the prompt.
    async fn choose(&self, cwd: &Path) -> Result<Option<TrustPromptChoice>, TrustResolveError>;
}

/// Trust resolution failure.
#[derive(Debug, Error)]
pub enum TrustResolveError {
    /// Reading or writing the trust store failed.
    #[error(transparent)]
    Store(#[from] TrustStoreError),
    /// The interactive host failed to obtain a decision.
    #[error("trust prompt failed: {0}")]
    Prompt(String),
}

/// Inputs controlling one trust decision.
#[derive(Clone, Debug)]
pub struct TrustResolveOptions {
    /// Explicit decision that bypasses every other source.
    pub override_value: Option<bool>,
    /// Non-interactive fallback policy.
    pub default: DefaultProjectTrust,
    /// Whether an interactive prompt can be displayed.
    pub has_ui: bool,
    /// If set, skips filesystem detection. Useful for embedders with virtual
    /// resources.
    pub requires_trust: Option<bool>,
    /// Home directory whose `.agents/skills` is a user resource, not project.
    pub home_dir: Option<PathBuf>,
}

impl Default for TrustResolveOptions {
    fn default() -> Self {
        Self {
            override_value: None,
            default: DefaultProjectTrust::Ask,
            has_ui: false,
            requires_trust: None,
            home_dir: None,
        }
    }
}

/// Ordered project trust resolver.
pub struct TrustResolver {
    store: Arc<dyn TrustStore>,
    extensions: Option<Arc<ExtensionRunner>>,
    prompt: Option<Arc<dyn TrustPrompt>>,
}

impl std::fmt::Debug for TrustResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TrustResolver")
            .field("has_extensions", &self.extensions.is_some())
            .field("has_prompt", &self.prompt.is_some())
            .finish_non_exhaustive()
    }
}

impl TrustResolver {
    /// Create a resolver backed by the supplied durable trust store.
    pub fn new(store: Arc<dyn TrustStore>) -> Self {
        Self {
            store,
            extensions: None,
            prompt: None,
        }
    }

    #[must_use]
    /// Add extension-contributed project-trust hooks.
    pub fn with_extensions(mut self, extensions: Arc<ExtensionRunner>) -> Self {
        self.extensions = Some(extensions);
        self
    }

    #[must_use]
    /// Add the host's interactive trust prompt.
    pub fn with_prompt(mut self, prompt: Arc<dyn TrustPrompt>) -> Self {
        self.prompt = Some(prompt);
        self
    }

    /// Resolve in this exact order:
    ///
    /// explicit override; no-resources fast path; first extension yes/no;
    /// nearest saved decision; configured default; interactive prompt; deny.
    ///
    /// # Errors
    ///
    /// Returns [`TrustResolveError`] when a trust hook requires persistence,
    /// the trust store fails, or the interactive prompt fails.
    pub async fn resolve(
        &self,
        cwd: &Path,
        options: &TrustResolveOptions,
    ) -> Result<bool, TrustResolveError> {
        if let Some(value) = options.override_value {
            return Ok(value);
        }
        let requires_trust = match options.requires_trust {
            Some(value) => value,
            None => has_trust_requiring_project_resources(cwd, options.home_dir.as_deref())?,
        };
        if !requires_trust {
            return Ok(true);
        }
        let extension_decision = match &self.extensions {
            Some(extensions) => extensions.emit_project_trust(cwd).await,
            None => None,
        };
        if let Some(ProjectTrustResult { decision, remember }) = extension_decision {
            let trusted = decision == HookTrustDecision::Yes;
            if remember {
                self.store
                    .set_many(&[TrustUpdate {
                        path: cwd.to_path_buf(),
                        decision: Some(trusted),
                    }])
                    .await?;
            }
            return Ok(trusted);
        }
        if let Some(entry) = self.store.get_entry(cwd).await? {
            return Ok(entry.trusted);
        }
        match options.default {
            DefaultProjectTrust::Always => return Ok(true),
            DefaultProjectTrust::Never => return Ok(false),
            DefaultProjectTrust::Ask => {}
        }
        if !options.has_ui {
            return Ok(false);
        }
        let Some(prompt) = &self.prompt else {
            return Ok(false);
        };
        let Some(choice) = prompt.choose(cwd).await? else {
            return Ok(false);
        };
        if !choice.updates.is_empty() {
            self.store.set_many(&choice.updates).await?;
        }
        Ok(choice.trusted)
    }
}

/// Whether project-local resources require a trust decision.
///
/// # Errors
/// Returns an error when the project or configured home path cannot be canonicalized.
pub fn has_trust_requiring_project_resources(
    cwd: &Path,
    home_dir: Option<&Path>,
) -> Result<bool, TrustStoreError> {
    let cwd = normalize_path(cwd)?;
    let config_dir = cwd.join(".ri");
    if TRUST_REQUIRING_RI_ENTRIES
        .iter()
        .any(|entry| config_dir.join(entry).exists())
    {
        return Ok(true);
    }

    let detected_home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .or_else(|| directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf()));
    let user_home = home_dir
        .or(detected_home.as_deref())
        .map(normalize_path)
        .transpose()?;
    let mut current = cwd;
    loop {
        if user_home.as_ref() == Some(&current) {
            return Ok(false);
        }
        let candidate = current.join(".agents").join("skills");
        if candidate.exists() {
            return Ok(true);
        }
        if !current.pop() {
            return Ok(false);
        }
    }
}

/// Standard trust prompt choices for an exact project, its parent, and
/// session-only decisions.
///
/// # Errors
/// Returns an error when the project path cannot be canonicalized.
pub fn trust_prompt_choices(cwd: &Path) -> Result<Vec<TrustPromptChoice>, TrustStoreError> {
    let cwd = normalize_path(cwd)?;
    let mut choices = vec![TrustPromptChoice {
        trusted: true,
        updates: vec![TrustUpdate {
            path: cwd.clone(),
            decision: Some(true),
        }],
    }];
    if let Some(parent) = cwd.parent() {
        choices.push(TrustPromptChoice {
            trusted: true,
            updates: vec![
                TrustUpdate {
                    path: parent.to_path_buf(),
                    decision: Some(true),
                },
                TrustUpdate {
                    path: cwd.clone(),
                    decision: None,
                },
            ],
        });
    }
    choices.push(TrustPromptChoice {
        trusted: true,
        updates: Vec::new(),
    });
    choices.push(TrustPromptChoice {
        trusted: false,
        updates: vec![TrustUpdate {
            path: cwd,
            decision: Some(false),
        }],
    });
    choices.push(TrustPromptChoice {
        trusted: false,
        updates: Vec::new(),
    });
    Ok(choices)
}

#[derive(Debug, Serialize, Deserialize)]
struct _TrustFileDocumentationOnly(BTreeMap<String, bool>);

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::extension::{
        ContextBinding, ContextFactory, EventBus, EventHook, GenerationClock, HookError,
        HookRegistration, NoopContextActions,
    };
    use crate::source::SourceInfo;

    use super::*;

    #[tokio::test]
    async fn nearest_parent_decision_is_inherited() {
        let temp = tempdir().expect("tempdir");
        let parent = temp.path().join("parent");
        let child = parent.join("child");
        std::fs::create_dir_all(&child).expect("dirs");
        let store = MemoryTrustStore::default();
        store
            .set_many(&[TrustUpdate {
                path: parent.clone(),
                decision: Some(true),
            }])
            .await
            .expect("set");
        assert_eq!(
            store.get_entry(&child).await.expect("get"),
            Some(TrustEntry {
                path: normalize_path(&parent).expect("canonical parent"),
                trusted: true,
            })
        );
    }

    struct TrustHook;

    #[async_trait]
    impl EventHook for TrustHook {
        async fn on_project_trust(
            &self,
            _cwd: &Path,
            _context: &crate::extension::ExtensionContext,
        ) -> Result<Option<ProjectTrustResult>, HookError> {
            Ok(Some(ProjectTrustResult {
                decision: HookTrustDecision::No,
                remember: true,
            }))
        }
    }

    #[tokio::test]
    async fn extension_decision_precedes_saved_and_default() {
        let temp = tempdir().expect("tempdir");
        let contexts = ContextFactory::new(
            GenerationClock::default(),
            ContextBinding::default(),
            Arc::new(NoopContextActions),
            EventBus::default(),
        );
        let runner = Arc::new(ExtensionRunner::new(
            vec![HookRegistration {
                extension_id: "trust".to_owned(),
                source: SourceInfo::inline("trust"),
                hook: Arc::new(TrustHook),
            }],
            contexts,
        ));
        let store = Arc::new(MemoryTrustStore::default());
        store
            .set_many(&[TrustUpdate {
                path: temp.path().to_path_buf(),
                decision: Some(true),
            }])
            .await
            .expect("seed");
        let resolver = TrustResolver::new(store.clone()).with_extensions(runner);
        let trusted = resolver
            .resolve(
                temp.path(),
                &TrustResolveOptions {
                    default: DefaultProjectTrust::Always,
                    requires_trust: Some(true),
                    ..TrustResolveOptions::default()
                },
            )
            .await
            .expect("resolve");
        assert!(!trusted);
        assert!(
            !store
                .get_entry(temp.path())
                .await
                .expect("remember")
                .expect("entry")
                .trusted
        );
    }

    #[tokio::test]
    async fn no_project_resources_skips_prompt_and_defaults_to_trusted() {
        let temp = tempdir().expect("tempdir");
        let resolver = TrustResolver::new(Arc::new(MemoryTrustStore::default()));
        assert!(
            resolver
                .resolve(
                    temp.path(),
                    &TrustResolveOptions {
                        default: DefaultProjectTrust::Never,
                        ..TrustResolveOptions::default()
                    }
                )
                .await
                .expect("resolve")
        );
    }

    #[test]
    fn detects_ri_and_ancestor_agent_skills() {
        let temp = tempdir().expect("tempdir");
        let project = temp.path().join("project");
        let child = project.join("child");
        std::fs::create_dir_all(project.join(".agents").join("skills")).expect("skills");
        std::fs::create_dir_all(&child).expect("child");
        assert!(has_trust_requiring_project_resources(&child, None).expect("detect resources"));
    }

    #[tokio::test]
    async fn file_store_is_sorted_and_round_trips() {
        let temp = tempdir().expect("tempdir");
        let first = temp.path().join("z");
        let second = temp.path().join("a");
        std::fs::create_dir_all(&first).expect("first");
        std::fs::create_dir_all(&second).expect("second");
        let store = FileTrustStore::new(temp.path());
        store
            .set_many(&[
                TrustUpdate {
                    path: first.clone(),
                    decision: Some(true),
                },
                TrustUpdate {
                    path: second,
                    decision: Some(false),
                },
            ])
            .await
            .expect("write");
        assert!(
            store
                .get_entry(&first)
                .await
                .expect("read")
                .expect("entry")
                .trusted
        );
        let content = std::fs::read_to_string(temp.path().join("trust.json")).expect("content");
        let keys = serde_json::from_str::<BTreeMap<String, bool>>(&content)
            .expect("json")
            .into_keys()
            .collect::<Vec<_>>();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
