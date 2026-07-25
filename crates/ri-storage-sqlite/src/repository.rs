use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Utc};
use parking_lot::Mutex;
use ri_session::{
    CURRENT_SESSION_VERSION, CreateOptions, Error, ForkOptions, ListOptions, MalformedEntryPolicy,
    Repository, Result, SequencedEntry, Session, SessionEntry, SessionHeader, SessionMetadata,
    SessionSnapshot, SessionStats, SessionStore,
    backend::{entries_for_fork, fork_create_options, header_from_options},
};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::Value;

use crate::migrations::{self, storage_error};

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Effective safety-related `SQLite` pragma values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PragmaSettings {
    /// Journal mode reported by `SQLite`.
    pub journal_mode: String,
    /// Numeric synchronous level (`2` is `FULL`).
    pub synchronous: i64,
    /// Busy timeout in milliseconds.
    pub busy_timeout_ms: i64,
    /// Whether foreign-key enforcement is enabled.
    pub foreign_keys: bool,
}

/// Persisted derived state for one session.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedSession {
    /// Durable active leaf.
    pub active_leaf_id: Option<String>,
    /// Root-to-leaf entry ids in active-branch order.
    pub active_branch: Vec<String>,
    /// Aggregate usage statistics.
    pub stats: SessionStats,
    /// Latest labels by target entry id.
    pub labels: BTreeMap<String, String>,
    /// Latest display name.
    pub name: Option<String>,
    /// Last sequence included in this materialization.
    pub last_sequence: u64,
}

/// `SQLite` implementation of the backend-neutral session repository.
#[derive(Clone)]
pub struct SqliteRepository {
    path: Arc<PathBuf>,
    connection: Arc<Mutex<Connection>>,
    malformed_entry_policy: MalformedEntryPolicy,
}

/// Compatibility name for [`SqliteRepository`].
pub type SqliteSessionRepository = SqliteRepository;

/// Pi-style compatibility name for [`SqliteRepository`].
pub type SqliteSessionRepo = SqliteRepository;

impl fmt::Debug for SqliteRepository {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteRepository")
            .field("path", &self.path)
            .field("malformed_entry_policy", &self.malformed_entry_policy)
            .finish_non_exhaustive()
    }
}

impl SqliteRepository {
    /// Open or create a database with strict malformed-entry recovery.
    ///
    /// # Errors
    ///
    /// Returns an error when the database cannot be opened, configured,
    /// migrated, or recovered.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_policy(path, MalformedEntryPolicy::Reject)
    }

    /// Open or create a database and select reopen behavior for malformed rows.
    ///
    /// # Errors
    ///
    /// Returns an error when the parent directory or database cannot be
    /// created, configured, migrated, or recovered.
    pub fn open_with_policy(
        path: impl Into<PathBuf>,
        malformed_entry_policy: MalformedEntryPolicy,
    ) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut connection = Connection::open(&path).map_err(storage_error)?;
        configure(&mut connection)?;
        migrations::apply(&mut connection)?;
        recover_all(&mut connection, malformed_entry_policy)?;
        Ok(Self {
            path: Arc::new(path),
            connection: Arc::new(Mutex::new(connection)),
            malformed_entry_policy,
        })
    }

    /// Database path used by this repository.
    pub fn path(&self) -> &Path {
        self.path.as_ref()
    }

    /// Applied schema migration version.
    ///
    /// # Errors
    ///
    /// Returns an error when the schema metadata query fails.
    pub fn schema_version(&self) -> Result<u32> {
        migrations::schema_version(&self.connection.lock())
    }

    /// Read effective durability and contention pragmas.
    ///
    /// # Errors
    ///
    /// Returns an error when any pragma query fails.
    pub fn pragma_settings(&self) -> Result<PragmaSettings> {
        let connection = self.connection.lock();
        let journal_mode = connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .map_err(storage_error)?;
        let synchronous = connection
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(storage_error)?;
        let busy_timeout_ms = connection
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .map_err(storage_error)?;
        let foreign_keys: i64 = connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .map_err(storage_error)?;
        Ok(PragmaSettings {
            journal_mode,
            synchronous,
            busy_timeout_ms,
            foreign_keys: foreign_keys != 0,
        })
    }

    /// Run `SQLite`'s integrity checker.
    ///
    /// # Errors
    ///
    /// Returns an error when the check cannot run or reports corruption.
    pub fn integrity_check(&self) -> Result<()> {
        let result: String = self
            .connection
            .lock()
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .map_err(storage_error)?;
        if result == "ok" {
            Ok(())
        } else {
            Err(Error::Storage(format!(
                "SQLite integrity check failed: {result}"
            )))
        }
    }

    /// Read the transactionally maintained active branch and aggregate state.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is missing or its materialized rows
    /// cannot be queried or decoded.
    pub fn materialized(&self, session_id: &str) -> Result<MaterializedSession> {
        let connection = self.connection.lock();
        let row = connection
            .query_row(
                r"
SELECT active_leaf_id, name, message_count, cached_tokens, uncached_tokens,
       total_tokens, cost_total, labels_json, last_sequence
FROM session_materialized
WHERE session_id = ?1
",
                [session_id],
                |row| {
                    Ok(MaterializedRow {
                        active_leaf_id: row.get(0)?,
                        name: row.get(1)?,
                        message_count: row.get(2)?,
                        cached_tokens: row.get(3)?,
                        uncached_tokens: row.get(4)?,
                        total_tokens: row.get(5)?,
                        cost_total: row.get(6)?,
                        labels_json: row.get(7)?,
                        last_sequence: row.get(8)?,
                    })
                },
            )
            .optional()
            .map_err(storage_error)?
            .ok_or_else(|| {
                Error::NotFound(format!("materialized session {session_id} was not found"))
            })?;
        let mut statement = connection
            .prepare("SELECT entry_id FROM active_branch WHERE session_id = ?1 ORDER BY position")
            .map_err(storage_error)?;
        let active_branch = statement
            .query_map([session_id], |row| row.get(0))
            .map_err(storage_error)?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(storage_error)?;
        row.into_public(active_branch)
    }
}

struct SqliteStore {
    path: Arc<PathBuf>,
    connection: Arc<Mutex<Connection>>,
    session_id: String,
    malformed_entry_policy: MalformedEntryPolicy,
}

impl fmt::Debug for SqliteStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteStore")
            .field("path", &self.path)
            .field("session_id", &self.session_id)
            .field("malformed_entry_policy", &self.malformed_entry_policy)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SessionStore for SqliteStore {
    async fn metadata(&self) -> Result<SessionMetadata> {
        let connection = self.connection.lock();
        let header = load_header(&connection, &self.session_id)?;
        Ok(SessionMetadata::from_header(
            &header,
            Some(self.path.as_ref().clone()),
        ))
    }

    async fn snapshot(&self) -> Result<SessionSnapshot> {
        let connection = self.connection.lock();
        Ok(load_snapshot(&connection, &self.session_id, self.malformed_entry_policy)?.snapshot)
    }

    async fn append(&self, entry: SessionEntry) -> Result<SequencedEntry> {
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        let loaded = load_snapshot(&transaction, &self.session_id, self.malformed_entry_policy)?;
        let mut snapshot = loaded.snapshot;
        let sequence_i64: i64 = transaction
            .query_row(
                "SELECT next_sequence FROM sessions WHERE id = ?1",
                [&self.session_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(storage_error)?
            .ok_or_else(|| Error::NotFound(format!("session {} was not found", self.session_id)))?;
        let sequence = u64::try_from(sequence_i64).map_err(|_| {
            Error::InvalidSession(format!(
                "session {} has an invalid next sequence",
                self.session_id
            ))
        })?;
        let stored = SequencedEntry { sequence, entry };
        snapshot.push(stored.clone())?;
        let next_sequence = sequence.checked_add(1).ok_or_else(|| {
            Error::Storage(format!("session {} sequence overflow", self.session_id))
        })?;
        let next_sequence_i64 = i64::try_from(next_sequence).map_err(|_| {
            Error::Storage(format!("session {} sequence overflow", self.session_id))
        })?;
        let entry_json = serde_json::to_string(&stored.entry)?;
        transaction
            .execute(
                r"
INSERT INTO session_entries (
    session_id, entry_sequence, id, parent_id, entry_type, timestamp, entry_json
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
",
                params![
                    self.session_id,
                    sequence_i64,
                    stored.entry.id(),
                    stored.entry.parent_id(),
                    stored.entry.kind(),
                    timestamp_string(stored.entry.base().timestamp),
                    entry_json,
                ],
            )
            .map_err(storage_error)?;
        transaction
            .execute(
                "UPDATE sessions SET next_sequence = ?1, active_leaf_id = ?2 WHERE id = ?3",
                params![next_sequence_i64, snapshot.leaf_id(), self.session_id],
            )
            .map_err(storage_error)?;
        write_materialized(&transaction, &self.session_id, &snapshot)?;
        transaction.commit().map_err(storage_error)?;
        Ok(stored)
    }
}

#[async_trait]
impl Repository for SqliteRepository {
    async fn create(&self, options: CreateOptions) -> Result<Session> {
        let header = header_from_options(options)?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        let exists = transaction
            .query_row("SELECT 1 FROM sessions WHERE id = ?1", [&header.id], |_| {
                Ok(())
            })
            .optional()
            .map_err(storage_error)?
            .is_some();
        if exists {
            return Err(Error::Conflict(format!(
                "session {} already exists",
                header.id
            )));
        }
        let metadata_json = header
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        transaction
            .execute(
                r"
INSERT INTO sessions (
    id, format_version, created_at, cwd, parent_session, metadata_json,
    active_leaf_id, next_sequence
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, 1)
",
                params![
                    header.id,
                    i64::from(header.version),
                    timestamp_string(header.timestamp),
                    header.cwd,
                    header.parent_session,
                    metadata_json,
                ],
            )
            .map_err(storage_error)?;
        let snapshot = SessionSnapshot::new(header.clone())?;
        write_materialized(&transaction, &header.id, &snapshot)?;
        transaction.commit().map_err(storage_error)?;
        drop(connection);
        Ok(Session::from_store(Arc::new(SqliteStore {
            path: Arc::clone(&self.path),
            connection: Arc::clone(&self.connection),
            session_id: header.id,
            malformed_entry_policy: self.malformed_entry_policy,
        })))
    }

    async fn open(&self, id: &str) -> Result<Session> {
        let exists = self
            .connection
            .lock()
            .query_row("SELECT 1 FROM sessions WHERE id = ?1", [id], |_| Ok(()))
            .optional()
            .map_err(storage_error)?
            .is_some();
        if !exists {
            return Err(Error::NotFound(format!("session {id} was not found")));
        }
        Ok(Session::from_store(Arc::new(SqliteStore {
            path: Arc::clone(&self.path),
            connection: Arc::clone(&self.connection),
            session_id: id.to_owned(),
            malformed_entry_policy: self.malformed_entry_policy,
        })))
    }

    async fn list(&self, options: ListOptions) -> Result<Vec<SessionMetadata>> {
        let connection = self.connection.lock();
        let query = if options.cwd.is_some() {
            r"
SELECT id, format_version, created_at, cwd, parent_session, metadata_json
FROM sessions WHERE cwd = ?1 ORDER BY created_at DESC, id
"
        } else {
            r"
SELECT id, format_version, created_at, cwd, parent_session, metadata_json
FROM sessions ORDER BY created_at DESC, id
"
        };
        let mut statement = connection.prepare(query).map_err(storage_error)?;
        let mut metadata = Vec::new();
        if let Some(cwd) = options.cwd {
            let rows = statement
                .query_map(params![cwd], session_row_from_sql)
                .map_err(storage_error)?;
            for row in rows {
                metadata.push(SessionMetadata::from_header(
                    &row.map_err(storage_error)?.into_header()?,
                    Some(self.path.as_ref().clone()),
                ));
            }
        } else {
            let rows = statement
                .query_map([], session_row_from_sql)
                .map_err(storage_error)?;
            for row in rows {
                metadata.push(SessionMetadata::from_header(
                    &row.map_err(storage_error)?.into_header()?,
                    Some(self.path.as_ref().clone()),
                ));
            }
        }
        Ok(metadata)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        let changed = transaction
            .execute("DELETE FROM sessions WHERE id = ?1", [id])
            .map_err(storage_error)?;
        if changed == 0 {
            return Err(Error::NotFound(format!("session {id} was not found")));
        }
        transaction.commit().map_err(storage_error)
    }

    async fn fork(&self, source_id: &str, options: ForkOptions) -> Result<Session> {
        let source = self.open(source_id).await?;
        let source_snapshot = source.snapshot().await?;
        let entries = entries_for_fork(&source_snapshot, &options)?;
        let create_options = fork_create_options(&source_snapshot, options);
        let destination = self.create(create_options).await?;
        let destination_id = destination.metadata().await?.id;
        for stored in entries {
            if let Err(error) = destination.append_entry(stored.entry).await {
                let _ = self.delete(&destination_id).await;
                return Err(error);
            }
        }
        Ok(destination)
    }
}

#[derive(Debug)]
struct SessionRow {
    id: String,
    format_version: i64,
    created_at: String,
    cwd: String,
    parent_session: Option<String>,
    metadata_json: Option<String>,
}

impl SessionRow {
    fn into_header(self) -> Result<SessionHeader> {
        let version = u32::try_from(self.format_version).map_err(|_| {
            Error::InvalidSession(format!("session {} has an invalid format version", self.id))
        })?;
        if version != CURRENT_SESSION_VERSION {
            return Err(Error::InvalidSession(format!(
                "session {} uses unsupported format version {version}",
                self.id
            )));
        }
        let timestamp = DateTime::parse_from_rfc3339(&self.created_at)
            .map_err(|error| {
                Error::InvalidSession(format!(
                    "session {} has an invalid creation time: {error}",
                    self.id
                ))
            })?
            .with_timezone(&Utc);
        let metadata = self
            .metadata_json
            .map(|json| serde_json::from_str::<BTreeMap<String, Value>>(&json))
            .transpose()?;
        let mut header = SessionHeader::new(self.id, self.cwd);
        header.version = version;
        header.timestamp = timestamp;
        header.parent_session = self.parent_session;
        header.metadata = metadata;
        Ok(header)
    }
}

#[derive(Debug)]
struct EntryRow {
    sequence: i64,
    id: String,
    parent_id: Option<String>,
    entry_type: String,
    entry_json: String,
}

struct LoadedSnapshot {
    snapshot: SessionSnapshot,
    maximum_raw_sequence: u64,
}

#[derive(Debug)]
struct MaterializedRow {
    active_leaf_id: Option<String>,
    name: Option<String>,
    message_count: i64,
    cached_tokens: i64,
    uncached_tokens: i64,
    total_tokens: i64,
    cost_total: f64,
    labels_json: String,
    last_sequence: i64,
}

impl MaterializedRow {
    fn into_public(self, active_branch: Vec<String>) -> Result<MaterializedSession> {
        let convert = |value: i64, field: &str| {
            u64::try_from(value)
                .map_err(|_| Error::InvalidSession(format!("materialized {field} is negative")))
        };
        Ok(MaterializedSession {
            active_leaf_id: self.active_leaf_id,
            active_branch,
            stats: SessionStats {
                message_count: convert(self.message_count, "message count")?,
                cached_tokens: convert(self.cached_tokens, "cached tokens")?,
                uncached_tokens: convert(self.uncached_tokens, "uncached tokens")?,
                total_tokens: convert(self.total_tokens, "total tokens")?,
                cost_total: self.cost_total,
            },
            labels: serde_json::from_str(&self.labels_json)?,
            name: self.name,
            last_sequence: convert(self.last_sequence, "last sequence")?,
        })
    }
}

fn configure(connection: &mut Connection) -> Result<()> {
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .map_err(storage_error)?;
    connection
        .execute_batch(
            r"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
",
        )
        .map_err(storage_error)
}

fn recover_all(
    connection: &mut Connection,
    malformed_entry_policy: MalformedEntryPolicy,
) -> Result<()> {
    let session_ids: Vec<String> = {
        let mut statement = connection
            .prepare("SELECT id FROM sessions ORDER BY id")
            .map_err(storage_error)?;
        let rows = statement
            .query_map([], |row| row.get(0))
            .map_err(storage_error)?;
        rows.collect::<std::result::Result<_, _>>()
            .map_err(storage_error)?
    };
    if session_ids.is_empty() {
        return Ok(());
    }
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(storage_error)?;
    for session_id in session_ids {
        let loaded = load_snapshot(&transaction, &session_id, malformed_entry_policy)?;
        let stored_next: i64 = transaction
            .query_row(
                "SELECT next_sequence FROM sessions WHERE id = ?1",
                [&session_id],
                |row| row.get(0),
            )
            .map_err(storage_error)?;
        let recovered_next = loaded.maximum_raw_sequence.checked_add(1).ok_or_else(|| {
            Error::Storage(format!(
                "session {session_id} sequence overflow during recovery"
            ))
        })?;
        let recovered_next = i64::try_from(recovered_next).map_err(|_| {
            Error::Storage(format!(
                "session {session_id} sequence overflow during recovery"
            ))
        })?;
        transaction
            .execute(
                "UPDATE sessions SET next_sequence = ?1, active_leaf_id = ?2 WHERE id = ?3",
                params![
                    stored_next.max(recovered_next),
                    loaded.snapshot.leaf_id(),
                    session_id,
                ],
            )
            .map_err(storage_error)?;
        write_materialized(&transaction, &session_id, &loaded.snapshot)?;
    }
    transaction.commit().map_err(storage_error)
}

fn load_header(connection: &Connection, session_id: &str) -> Result<SessionHeader> {
    connection
        .query_row(
            r"
SELECT id, format_version, created_at, cwd, parent_session, metadata_json
FROM sessions WHERE id = ?1
",
            [session_id],
            session_row_from_sql,
        )
        .optional()
        .map_err(storage_error)?
        .ok_or_else(|| Error::NotFound(format!("session {session_id} was not found")))?
        .into_header()
}

fn load_snapshot(
    connection: &Connection,
    session_id: &str,
    malformed_entry_policy: MalformedEntryPolicy,
) -> Result<LoadedSnapshot> {
    let header = load_header(connection, session_id)?;
    let mut snapshot = SessionSnapshot::new(header)?;
    let mut statement = connection
        .prepare(
            r"
SELECT entry_sequence, id, parent_id, entry_type, entry_json
FROM session_entries WHERE session_id = ?1 ORDER BY entry_sequence
",
        )
        .map_err(storage_error)?;
    let rows = statement
        .query_map([session_id], |row| {
            Ok(EntryRow {
                sequence: row.get(0)?,
                id: row.get(1)?,
                parent_id: row.get(2)?,
                entry_type: row.get(3)?,
                entry_json: row.get(4)?,
            })
        })
        .map_err(storage_error)?;
    let mut maximum_raw_sequence = 0;
    for row in rows {
        let row = row.map_err(storage_error)?;
        if let Ok(sequence) = u64::try_from(row.sequence) {
            maximum_raw_sequence = maximum_raw_sequence.max(sequence);
        }
        let outcome = decode_stored_entry(&row).and_then(|stored| snapshot.push(stored));
        if let Err(error) = outcome
            && malformed_entry_policy == MalformedEntryPolicy::Reject
        {
            return Err(error);
        }
    }
    Ok(LoadedSnapshot {
        snapshot,
        maximum_raw_sequence,
    })
}

fn decode_stored_entry(row: &EntryRow) -> Result<SequencedEntry> {
    let sequence = u64::try_from(row.sequence)
        .map_err(|_| Error::InvalidEntry("SQLite entry has a negative sequence".to_owned()))?;
    let entry: SessionEntry = serde_json::from_str(&row.entry_json)?;
    if entry.id() != row.id
        || entry.parent_id() != row.parent_id.as_deref()
        || entry.kind() != row.entry_type
    {
        return Err(Error::InvalidEntry(format!(
            "SQLite index columns disagree with entry {}",
            row.id
        )));
    }
    Ok(SequencedEntry { sequence, entry })
}

fn write_materialized(
    transaction: &Transaction<'_>,
    session_id: &str,
    snapshot: &SessionSnapshot,
) -> Result<()> {
    let stats = snapshot.stats();
    let labels_json = serde_json::to_string(&snapshot.labels())?;
    let last_sequence = snapshot.entries().last().map_or(0, |entry| entry.sequence);
    transaction
        .execute(
            r"
INSERT INTO session_materialized (
    session_id, active_leaf_id, name, message_count, cached_tokens,
    uncached_tokens, total_tokens, cost_total, labels_json, last_sequence
) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
ON CONFLICT(session_id) DO UPDATE SET
    active_leaf_id = excluded.active_leaf_id,
    name = excluded.name,
    message_count = excluded.message_count,
    cached_tokens = excluded.cached_tokens,
    uncached_tokens = excluded.uncached_tokens,
    total_tokens = excluded.total_tokens,
    cost_total = excluded.cost_total,
    labels_json = excluded.labels_json,
    last_sequence = excluded.last_sequence
",
            params![
                session_id,
                snapshot.leaf_id(),
                snapshot.session_name(),
                to_sql_integer(stats.message_count, "message count")?,
                to_sql_integer(stats.cached_tokens, "cached tokens")?,
                to_sql_integer(stats.uncached_tokens, "uncached tokens")?,
                to_sql_integer(stats.total_tokens, "total tokens")?,
                stats.cost_total,
                labels_json,
                to_sql_integer(last_sequence, "last sequence")?,
            ],
        )
        .map_err(storage_error)?;
    transaction
        .execute(
            "DELETE FROM active_branch WHERE session_id = ?1",
            [session_id],
        )
        .map_err(storage_error)?;
    for (position, stored) in snapshot.active_path()?.into_iter().enumerate() {
        transaction
            .execute(
                r"
INSERT INTO active_branch (session_id, position, entry_id, entry_sequence)
VALUES (?1, ?2, ?3, ?4)
",
                params![
                    session_id,
                    i64::try_from(position).map_err(|_| {
                        Error::Storage(format!("session {session_id} active branch is too large"))
                    })?,
                    stored.entry.id(),
                    to_sql_integer(stored.sequence, "entry sequence")?,
                ],
            )
            .map_err(storage_error)?;
    }
    Ok(())
}

fn session_row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    Ok(SessionRow {
        id: row.get(0)?,
        format_version: row.get(1)?,
        created_at: row.get(2)?,
        cwd: row.get(3)?,
        parent_session: row.get(4)?,
        metadata_json: row.get(5)?,
    })
}

fn timestamp_string(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn to_sql_integer(value: u64, field: &str) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::Storage(format!("{field} exceeds SQLite integer range")))
}
