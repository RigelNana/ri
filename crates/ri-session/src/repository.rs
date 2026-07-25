//! High-level session handles and built-in repositories.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use chrono::SecondsFormat;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::backend::{entries_for_fork, fork_create_options, header_from_options};
use crate::error::{Error, Result};
use crate::model::{
    ActiveToolsChangeEntry, BranchSummaryEntry, CURRENT_SESSION_VERSION, CompactionEntry,
    CreateOptions, CustomEntry, CustomMessageEntry, EntryBase, ForkOptions, LabelEntry, LeafEntry,
    ListOptions, MessageEntry, ModelChangeEntry, SequencedEntry, SessionContext, SessionEntry,
    SessionHeader, SessionInfoEntry, SessionMetadata, SessionStats, ThinkingLevelChangeEntry,
    Usage,
};
use crate::state::{SessionSnapshot, SessionTreeNode};

/// Policy used when a JSONL file contains a malformed entry after a valid header.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MalformedEntryPolicy {
    /// Fail the open operation and preserve the file unchanged.
    #[default]
    Reject,
    /// Ignore malformed records and valid records that depend on them.
    Skip,
}

/// Persistence operations for one already-selected session.
///
/// Implementations must not expose an entry in [`snapshot`](Self::snapshot)
/// unless its durable append has completed successfully.
#[async_trait]
pub trait SessionStore: Send + Sync + fmt::Debug {
    /// Backend-neutral metadata for this session.
    async fn metadata(&self) -> Result<SessionMetadata>;

    /// Return a consistent validated snapshot.
    async fn snapshot(&self) -> Result<SessionSnapshot>;

    /// Atomically append an entry and advance the durable leaf.
    async fn append(&self, entry: SessionEntry) -> Result<SequencedEntry>;
}

/// CRUD and fork operations shared by session repositories.
#[async_trait]
pub trait Repository: Send + Sync + fmt::Debug {
    /// Create an empty session.
    async fn create(&self, options: CreateOptions) -> Result<Session>;

    /// Open a session by stable identifier.
    async fn open(&self, id: &str) -> Result<Session>;

    /// List sessions, newest first.
    async fn list(&self, options: ListOptions) -> Result<Vec<SessionMetadata>>;

    /// Delete a session and all backend-owned records.
    async fn delete(&self, id: &str) -> Result<()>;

    /// Copy all history or one selected path into a new session.
    async fn fork(&self, source_id: &str, options: ForkOptions) -> Result<Session>;
}

/// High-level operations over a backend-specific [`SessionStore`].
#[derive(Clone)]
pub struct Session {
    store: Arc<dyn SessionStore>,
}

impl fmt::Debug for Session {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Session").finish_non_exhaustive()
    }
}

impl Session {
    /// Wrap a backend store in the public session API.
    pub fn from_store(store: Arc<dyn SessionStore>) -> Self {
        Self { store }
    }

    /// Access the backend store for integration code.
    pub fn store(&self) -> &Arc<dyn SessionStore> {
        &self.store
    }

    /// Return backend-neutral session metadata.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide metadata.
    pub async fn metadata(&self) -> Result<SessionMetadata> {
        self.store.metadata().await
    }

    /// Return a consistent validated snapshot.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot be read or contains invalid state.
    pub async fn snapshot(&self) -> Result<SessionSnapshot> {
        self.store.snapshot().await
    }

    /// Return the immutable session header.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn header(&self) -> Result<SessionHeader> {
        Ok(self.snapshot().await?.header().clone())
    }

    /// Return the durable active leaf.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn leaf_id(&self) -> Result<Option<String>> {
        Ok(self.snapshot().await?.leaf_id().map(str::to_owned))
    }

    /// Return one entry by identifier.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn entry(&self, id: &str) -> Result<Option<SequencedEntry>> {
        Ok(self.snapshot().await?.entry(id).cloned())
    }

    /// Return append-ordered entries after an optional cursor.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn entries(
        &self,
        after_sequence: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<SequencedEntry>> {
        Ok(self.snapshot().await?.entries_after(after_sequence, limit))
    }

    /// Return the active root-to-leaf path without compaction filtering.
    ///
    /// # Errors
    /// Returns an error when the snapshot is invalid or references a missing entry.
    pub async fn branch(&self) -> Result<Vec<SequencedEntry>> {
        Ok(self
            .snapshot()
            .await?
            .active_path()?
            .into_iter()
            .cloned()
            .collect())
    }

    /// Return the complete parent tree.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn tree(&self) -> Result<Vec<SessionTreeNode>> {
        Ok(self.snapshot().await?.tree())
    }

    /// Project the active branch into model context.
    ///
    /// # Errors
    /// Returns an error when the active branch is invalid or references a missing entry.
    pub async fn context(&self) -> Result<SessionContext> {
        self.snapshot().await?.context()
    }

    /// Return aggregate committed usage.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn stats(&self) -> Result<SessionStats> {
        Ok(self.snapshot().await?.stats())
    }

    /// Resolve the latest label for an entry.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn label(&self, id: &str) -> Result<Option<String>> {
        Ok(self.snapshot().await?.label(id))
    }

    /// Resolve the latest session display name.
    ///
    /// # Errors
    /// Returns an error when the backing store cannot provide a valid snapshot.
    pub async fn name(&self) -> Result<Option<String>> {
        Ok(self.snapshot().await?.session_name())
    }

    /// Append a fully constructed entry.
    ///
    /// Most callers should use the typed convenience methods, which select the
    /// current leaf and generate a collision-resistant id.
    ///
    /// # Errors
    /// Returns an error when the entry is invalid or cannot be durably appended.
    pub async fn append_entry(&self, entry: SessionEntry) -> Result<SequencedEntry> {
        self.store.append(entry).await
    }

    /// Append an application message.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_message(&self, message: Value) -> Result<String> {
        let entry = SessionEntry::Message(MessageEntry {
            base: self.next_base().await?,
            message,
        });
        self.append_id(entry).await
    }

    /// Append a model selection.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_model_change(
        &self,
        provider: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<String> {
        let entry = SessionEntry::ModelChange(ModelChangeEntry {
            base: self.next_base().await?,
            provider: provider.into(),
            model_id: model_id.into(),
        });
        self.append_id(entry).await
    }

    /// Append a reasoning-level selection.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_thinking_level_change(
        &self,
        thinking_level: impl Into<String>,
    ) -> Result<String> {
        let entry = SessionEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
            base: self.next_base().await?,
            thinking_level: thinking_level.into(),
        });
        self.append_id(entry).await
    }

    /// Append an active-tool selection.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_active_tools_change(
        &self,
        active_tool_names: Vec<String>,
    ) -> Result<String> {
        let entry = SessionEntry::ActiveToolsChange(ActiveToolsChangeEntry {
            base: self.next_base().await?,
            active_tool_names,
        });
        self.append_id(entry).await
    }

    /// Append a legacy-compatible compaction checkpoint.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the checkpoint cannot be appended.
    pub async fn append_compaction(
        &self,
        summary: impl Into<String>,
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
    ) -> Result<String> {
        self.append_compaction_with(
            summary,
            first_kept_entry_id,
            tokens_before,
            None,
            None,
            None,
            None,
        )
        .await
    }

    /// Append a compaction checkpoint with retained messages and accounting.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the checkpoint cannot be appended.
    #[allow(clippy::too_many_arguments)]
    pub async fn append_compaction_with(
        &self,
        summary: impl Into<String>,
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
        retained_tail: Option<Vec<Value>>,
        details: Option<Value>,
        usage: Option<Usage>,
        from_hook: Option<bool>,
    ) -> Result<String> {
        let entry = SessionEntry::Compaction(CompactionEntry {
            base: self.next_base().await?,
            summary: summary.into(),
            first_kept_entry_id,
            tokens_before,
            retained_tail,
            details,
            usage,
            from_hook,
        });
        self.append_id(entry).await
    }

    /// Append a branch summary without moving the leaf.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the summary cannot be appended.
    pub async fn append_branch_summary(
        &self,
        from_id: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<String> {
        self.append_branch_summary_with(from_id, summary, None, None, None)
            .await
    }

    /// Append a branch summary with details and accounting.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the summary cannot be appended.
    pub async fn append_branch_summary_with(
        &self,
        from_id: impl Into<String>,
        summary: impl Into<String>,
        details: Option<Value>,
        usage: Option<Usage>,
        from_hook: Option<bool>,
    ) -> Result<String> {
        let entry = SessionEntry::BranchSummary(BranchSummaryEntry {
            base: self.next_base().await?,
            from_id: from_id.into(),
            summary: summary.into(),
            details,
            usage,
            from_hook,
        });
        self.append_id(entry).await
    }

    /// Append application state omitted from context.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_custom(
        &self,
        custom_type: impl Into<String>,
        data: Option<Value>,
    ) -> Result<String> {
        let entry = SessionEntry::Custom(CustomEntry {
            base: self.next_base().await?,
            custom_type: custom_type.into(),
            data,
        });
        self.append_id(entry).await
    }

    /// Append application content included in context.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the entry cannot be appended.
    pub async fn append_custom_message(
        &self,
        custom_type: impl Into<String>,
        content: Value,
        display: bool,
        details: Option<Value>,
    ) -> Result<String> {
        let entry = SessionEntry::CustomMessage(CustomMessageEntry {
            base: self.next_base().await?,
            custom_type: custom_type.into(),
            content,
            display,
            details,
        });
        self.append_id(entry).await
    }

    /// Set or clear a label.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the label cannot be appended.
    pub async fn append_label(
        &self,
        target_id: impl Into<String>,
        label: Option<String>,
    ) -> Result<String> {
        let entry = SessionEntry::Label(LabelEntry {
            base: self.next_base().await?,
            target_id: target_id.into(),
            label,
        });
        self.append_id(entry).await
    }

    /// Set or clear the display name after replacing line breaks with spaces.
    ///
    /// # Errors
    /// Returns an error when current state cannot be read or the update cannot be appended.
    pub async fn append_session_info(&self, name: Option<String>) -> Result<String> {
        let sanitized = name.map(|name| {
            name.replace(['\r', '\n'], " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        });
        let entry = SessionEntry::SessionInfo(SessionInfoEntry {
            base: self.next_base().await?,
            name: sanitized,
        });
        self.append_id(entry).await
    }

    /// Durably select an earlier entry as the active leaf.
    ///
    /// # Errors
    /// Returns an error if the target does not exist or the leaf update cannot be appended.
    pub async fn move_to(&self, target_id: Option<String>) -> Result<String> {
        let snapshot = self.snapshot().await?;
        if let Some(target_id) = target_id.as_deref()
            && snapshot.entry(target_id).is_none()
        {
            return Err(Error::NotFound(format!(
                "session entry {target_id} was not found"
            )));
        }
        let entry = SessionEntry::Leaf(LeafEntry {
            base: next_base_from_snapshot(&snapshot),
            target_id,
        });
        self.append_id(entry).await
    }

    /// Alias for [`move_to`](Self::move_to) when creating a branch.
    ///
    /// # Errors
    /// Returns an error if the target does not exist or the leaf update cannot be appended.
    pub async fn branch_from(&self, target_id: impl Into<String>) -> Result<String> {
        self.move_to(Some(target_id.into())).await
    }

    /// Durably move before all roots.
    ///
    /// # Errors
    /// Returns an error when the leaf update cannot be durably appended.
    pub async fn reset_leaf(&self) -> Result<String> {
        self.move_to(None).await
    }

    /// Move to an entry and append a summary of the abandoned branch.
    ///
    /// # Errors
    /// Returns an error if navigation fails or the branch summary cannot be appended.
    pub async fn branch_with_summary(
        &self,
        target_id: Option<String>,
        summary: impl Into<String>,
    ) -> Result<String> {
        let old_leaf = self.leaf_id().await?.unwrap_or_else(|| "root".to_owned());
        self.move_to(target_id).await?;
        self.append_branch_summary(old_leaf, summary).await
    }

    async fn append_id(&self, entry: SessionEntry) -> Result<String> {
        Ok(self.store.append(entry).await?.entry.id().to_owned())
    }

    async fn next_base(&self) -> Result<EntryBase> {
        Ok(next_base_from_snapshot(&self.snapshot().await?))
    }
}

/// Process-local repository with the same semantics as durable backends.
#[derive(Debug, Clone, Default)]
pub struct MemoryRepository {
    sessions: Arc<RwLock<HashMap<String, Arc<MemoryStore>>>>,
}

/// Compatibility name for [`MemoryRepository`].
pub type InMemoryRepository = MemoryRepository;

/// Explicit session-oriented compatibility name for [`MemoryRepository`].
pub type InMemorySessionRepository = MemoryRepository;

#[derive(Debug)]
struct MemoryStore {
    metadata: SessionMetadata,
    state: Mutex<SessionSnapshot>,
}

#[async_trait]
impl SessionStore for MemoryStore {
    async fn metadata(&self) -> Result<SessionMetadata> {
        Ok(self.metadata.clone())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        Ok(self.state.lock().await.clone())
    }

    async fn append(&self, entry: SessionEntry) -> Result<SequencedEntry> {
        let mut state = self.state.lock().await;
        let stored = SequencedEntry {
            sequence: state.next_sequence(),
            entry,
        };
        state.push(stored.clone())?;
        Ok(stored)
    }
}

#[async_trait]
impl Repository for MemoryRepository {
    async fn create(&self, options: CreateOptions) -> Result<Session> {
        let header = header_from_options(options)?;
        let id = header.id.clone();
        let state = SessionSnapshot::new(header.clone())?;
        let store = Arc::new(MemoryStore {
            metadata: SessionMetadata::from_header(&header, None),
            state: Mutex::new(state),
        });
        let mut sessions = self.sessions.write().await;
        if sessions.contains_key(&id) {
            return Err(Error::Conflict(format!("session {id} already exists")));
        }
        sessions.insert(id, Arc::clone(&store));
        Ok(Session::from_store(store))
    }

    async fn open(&self, id: &str) -> Result<Session> {
        let store = self
            .sessions
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("session {id} was not found")))?;
        Ok(Session::from_store(store))
    }

    async fn list(&self, options: ListOptions) -> Result<Vec<SessionMetadata>> {
        let stores: Vec<_> = self.sessions.read().await.values().cloned().collect();
        let mut metadata = Vec::with_capacity(stores.len());
        for store in stores {
            let item = store.metadata().await?;
            if options.cwd.as_deref().is_none_or(|cwd| cwd == item.cwd) {
                metadata.push(item);
            }
        }
        metadata.sort_by_key(|item| std::cmp::Reverse(item.created_at));
        Ok(metadata)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        if self.sessions.write().await.remove(id).is_none() {
            return Err(Error::NotFound(format!("session {id} was not found")));
        }
        Ok(())
    }

    async fn fork(&self, source_id: &str, options: ForkOptions) -> Result<Session> {
        let source = self.open(source_id).await?;
        let source_snapshot = source.snapshot().await?;
        let entries = entries_for_fork(&source_snapshot, &options)?;
        let create_options = fork_create_options(&source_snapshot, options);
        let destination = self.create(create_options).await?;
        for stored in entries {
            destination.append_entry(stored.entry).await?;
        }
        Ok(destination)
    }
}

/// Filesystem JSONL repository with one append-only file per session.
#[derive(Debug, Clone)]
pub struct JsonlRepository {
    root: PathBuf,
    malformed_entry_policy: MalformedEntryPolicy,
    stores: Arc<RwLock<HashMap<String, Weak<JsonlStore>>>>,
}

/// Compatibility name for [`JsonlRepository`].
pub type FileRepository = JsonlRepository;

/// Explicit session-oriented compatibility name for [`JsonlRepository`].
pub type JsonlSessionRepository = JsonlRepository;

impl JsonlRepository {
    /// Create a repository rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::with_policy(root, MalformedEntryPolicy::Reject)
    }

    /// Create a repository with an explicit malformed-entry policy.
    pub fn with_policy(
        root: impl Into<PathBuf>,
        malformed_entry_policy: MalformedEntryPolicy,
    ) -> Self {
        Self {
            root: root.into(),
            malformed_entry_policy,
            stores: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Repository root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Open a specific JSONL path and cache it by header id.
    ///
    /// # Errors
    /// Returns an error when the file cannot be read, migrated, or validated.
    pub async fn open_path(&self, path: impl Into<PathBuf>) -> Result<Session> {
        let path = path.into();
        let loaded = load_jsonl(&path, self.malformed_entry_policy).await?;
        let id = loaded.snapshot.header().id.clone();
        if let Some(existing) = self.stores.read().await.get(&id).and_then(Weak::upgrade) {
            return Ok(Session::from_store(existing));
        }
        if loaded.migrated {
            rewrite_jsonl(&path, &loaded.snapshot).await?;
        }
        let metadata = SessionMetadata::from_header(loaded.snapshot.header(), Some(path.clone()));
        let store = Arc::new(JsonlStore {
            path,
            metadata,
            state: Mutex::new(loaded.snapshot),
        });
        let mut stores = self.stores.write().await;
        if let Some(existing) = stores.get(&id).and_then(Weak::upgrade) {
            return Ok(Session::from_store(existing));
        }
        stores.insert(id, Arc::downgrade(&store));
        Ok(Session::from_store(store))
    }

    async fn find_path(&self, id: &str) -> Result<PathBuf> {
        let mut directory = tokio::fs::read_dir(&self.root)
            .await
            .map_err(|source| Error::io(&self.root, source))?;
        while let Some(item) = directory
            .next_entry()
            .await
            .map_err(|source| Error::io(&self.root, source))?
        {
            let path = item.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            match read_header(&path).await {
                Ok(header) if header.id == id => return Ok(path),
                Ok(_) => {}
                Err(error) if self.malformed_entry_policy == MalformedEntryPolicy::Skip => {
                    drop(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(Error::NotFound(format!("session {id} was not found")))
    }
}

#[derive(Debug)]
struct JsonlStore {
    path: PathBuf,
    metadata: SessionMetadata,
    state: Mutex<SessionSnapshot>,
}

#[async_trait]
impl SessionStore for JsonlStore {
    async fn metadata(&self) -> Result<SessionMetadata> {
        Ok(self.metadata.clone())
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        Ok(self.state.lock().await.clone())
    }

    async fn append(&self, entry: SessionEntry) -> Result<SequencedEntry> {
        let mut state = self.state.lock().await;
        let stored = SequencedEntry {
            sequence: state.next_sequence(),
            entry,
        };
        let mut candidate = state.clone();
        candidate.push(stored.clone())?;
        let line = serde_json::to_vec(&stored.entry)?;
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&self.path)
            .await
            .map_err(|source| Error::io(&self.path, source))?;
        file.write_all(&line)
            .await
            .map_err(|source| Error::io(&self.path, source))?;
        file.write_all(b"\n")
            .await
            .map_err(|source| Error::io(&self.path, source))?;
        file.flush()
            .await
            .map_err(|source| Error::io(&self.path, source))?;
        file.sync_data()
            .await
            .map_err(|source| Error::io(&self.path, source))?;
        *state = candidate;
        Ok(stored)
    }
}

#[async_trait]
impl Repository for JsonlRepository {
    async fn create(&self, options: CreateOptions) -> Result<Session> {
        tokio::fs::create_dir_all(&self.root)
            .await
            .map_err(|source| Error::io(&self.root, source))?;
        let header = header_from_options(options)?;
        if self.open(&header.id).await.is_ok() {
            return Err(Error::Conflict(format!(
                "session {} already exists",
                header.id
            )));
        }
        let timestamp = header
            .timestamp
            .to_rfc3339_opts(SecondsFormat::Millis, true)
            .replace([':', '.'], "-");
        let path = self.root.join(format!("{timestamp}_{}.jsonl", header.id));
        let mut file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .await
            .map_err(|source| Error::io(&path, source))?;
        let header_line = serde_json::to_vec(&header)?;
        file.write_all(&header_line)
            .await
            .map_err(|source| Error::io(&path, source))?;
        file.write_all(b"\n")
            .await
            .map_err(|source| Error::io(&path, source))?;
        file.sync_all()
            .await
            .map_err(|source| Error::io(&path, source))?;
        drop(file);

        let snapshot = SessionSnapshot::new(header.clone())?;
        let metadata = SessionMetadata::from_header(&header, Some(path.clone()));
        let store = Arc::new(JsonlStore {
            path,
            metadata,
            state: Mutex::new(snapshot),
        });
        self.stores
            .write()
            .await
            .insert(header.id, Arc::downgrade(&store));
        Ok(Session::from_store(store))
    }

    async fn open(&self, id: &str) -> Result<Session> {
        if let Some(store) = self.stores.read().await.get(id).and_then(Weak::upgrade) {
            return Ok(Session::from_store(store));
        }
        if !tokio::fs::try_exists(&self.root)
            .await
            .map_err(|source| Error::io(&self.root, source))?
        {
            return Err(Error::NotFound(format!("session {id} was not found")));
        }
        self.open_path(self.find_path(id).await?).await
    }

    async fn list(&self, options: ListOptions) -> Result<Vec<SessionMetadata>> {
        if !tokio::fs::try_exists(&self.root)
            .await
            .map_err(|source| Error::io(&self.root, source))?
        {
            return Ok(Vec::new());
        }
        let mut directory = tokio::fs::read_dir(&self.root)
            .await
            .map_err(|source| Error::io(&self.root, source))?;
        let mut sessions = Vec::new();
        while let Some(item) = directory
            .next_entry()
            .await
            .map_err(|source| Error::io(&self.root, source))?
        {
            let path = item.path();
            if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            match read_header(&path).await {
                Ok(header) if options.cwd.as_deref().is_none_or(|cwd| cwd == header.cwd) => {
                    sessions.push(SessionMetadata::from_header(&header, Some(path)));
                }
                Ok(_) => {}
                Err(error) if self.malformed_entry_policy == MalformedEntryPolicy::Skip => {
                    drop(error);
                }
                Err(error) => return Err(error),
            }
        }
        sessions.sort_by_key(|item| std::cmp::Reverse(item.created_at));
        Ok(sessions)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let path = self.find_path(id).await?;
        tokio::fs::remove_file(&path)
            .await
            .map_err(|source| Error::io(&path, source))?;
        self.stores.write().await.remove(id);
        Ok(())
    }

    async fn fork(&self, source_id: &str, options: ForkOptions) -> Result<Session> {
        let source = self.open(source_id).await?;
        let source_snapshot = source.snapshot().await?;
        let entries = entries_for_fork(&source_snapshot, &options)?;
        let create_options = fork_create_options(&source_snapshot, options);
        let destination = self.create(create_options).await?;
        for stored in entries {
            destination.append_entry(stored.entry).await?;
        }
        Ok(destination)
    }
}

fn next_base_from_snapshot(snapshot: &SessionSnapshot) -> EntryBase {
    let id = loop {
        let id = Uuid::new_v4().simple().to_string();
        let short = id[..8].to_owned();
        if snapshot.entry(&short).is_none() {
            break short;
        }
    };
    EntryBase::new(id, snapshot.leaf_id().map(str::to_owned))
}

struct LoadedJsonl {
    snapshot: SessionSnapshot,
    migrated: bool,
}

async fn read_header(path: &Path) -> Result<SessionHeader> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|source| Error::io(path, source))?;
    let line = content
        .lines()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| Error::InvalidSession(format!("{} has no header", path.display())))?;
    let header: SessionHeader = serde_json::from_str(line).map_err(|error| {
        Error::InvalidSession(format!("{} has an invalid header: {error}", path.display()))
    })?;
    if header.version > CURRENT_SESSION_VERSION {
        return Err(Error::InvalidSession(format!(
            "{} uses unsupported version {}",
            path.display(),
            header.version
        )));
    }
    Ok(header)
}

async fn load_jsonl(path: &Path, policy: MalformedEntryPolicy) -> Result<LoadedJsonl> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|source| Error::io(path, source))?;
    parse_jsonl(&content, path, policy)
}

fn parse_jsonl(content: &str, path: &Path, policy: MalformedEntryPolicy) -> Result<LoadedJsonl> {
    let mut non_blank = content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty());
    let Some((header_index, header_line)) = non_blank.next() else {
        return Err(Error::InvalidSession(format!(
            "{} has no session header",
            path.display()
        )));
    };
    let header_value: Value = serde_json::from_str(header_line).map_err(|error| {
        Error::InvalidSession(format!(
            "{} line {} has an invalid header: {error}",
            path.display(),
            header_index + 1
        ))
    })?;
    let mut header: SessionHeader = serde_json::from_value(header_value).map_err(|error| {
        Error::InvalidSession(format!(
            "{} line {} has an invalid header: {error}",
            path.display(),
            header_index + 1
        ))
    })?;
    if header.version > CURRENT_SESSION_VERSION {
        return Err(Error::InvalidSession(format!(
            "{} uses unsupported version {}",
            path.display(),
            header.version
        )));
    }
    let source_version = header.version;
    let migrated = source_version < CURRENT_SESSION_VERSION;
    header.make_current();
    let mut raw_entries: Vec<(usize, u64, Value)> = Vec::new();
    for (record_index, (line_index, line)) in non_blank.enumerate() {
        match serde_json::from_str::<Value>(line) {
            Ok(value) => raw_entries.push((
                record_index + 1,
                u64::try_from(record_index + 1).unwrap_or(u64::MAX),
                value,
            )),
            Err(error) => match policy {
                MalformedEntryPolicy::Reject => {
                    return Err(Error::InvalidEntry(format!(
                        "{} line {} is malformed JSON: {error}",
                        path.display(),
                        line_index + 1
                    )));
                }
                MalformedEntryPolicy::Skip => {}
            },
        }
    }
    crate::legacy::migrate_legacy_entries(
        source_version,
        raw_entries.iter_mut().map(|(_, _, value)| value),
        || Uuid::new_v4().simple().to_string()[..8].to_owned(),
        policy == MalformedEntryPolicy::Reject,
    )
    .map_err(|error| {
        Error::InvalidEntry(format!(
            "{} contains an invalid legacy entry: {error}",
            path.display()
        ))
    })?;

    let mut snapshot = SessionSnapshot::new(header)?;
    for (_, sequence, value) in raw_entries {
        let parsed = serde_json::from_value::<SessionEntry>(value).map_err(Error::from);
        let outcome = parsed.and_then(|entry| snapshot.push(SequencedEntry { sequence, entry }));
        if let Err(error) = outcome
            && policy == MalformedEntryPolicy::Reject
        {
            return Err(Error::InvalidEntry(format!(
                "{} contains an invalid entry at sequence {sequence}: {error}",
                path.display()
            )));
        }
    }
    Ok(LoadedJsonl { snapshot, migrated })
}

async fn rewrite_jsonl(path: &Path, snapshot: &SessionSnapshot) -> Result<()> {
    let mut bytes = serde_json::to_vec(snapshot.header())?;
    bytes.push(b'\n');
    for stored in snapshot.entries() {
        bytes.extend(serde_json::to_vec(&stored.entry)?);
        bytes.push(b'\n');
    }
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .await
        .map_err(|source| Error::io(path, source))?;
    file.write_all(&bytes)
        .await
        .map_err(|source| Error::io(path, source))?;
    file.sync_all()
        .await
        .map_err(|source| Error::io(path, source))
}
