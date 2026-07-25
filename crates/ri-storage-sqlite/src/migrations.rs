use std::collections::BTreeSet;

use chrono::{SecondsFormat, Utc};
use ri_session::{Error, Result};
use rusqlite::{Connection, TransactionBehavior, params};

pub(crate) const CURRENT_SCHEMA_VERSION: u32 = 2;

const MIGRATIONS: &[(u32, &str)] = &[
    (
        1,
        r"
CREATE TABLE sessions (
    id TEXT PRIMARY KEY NOT NULL,
    format_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    cwd TEXT NOT NULL,
    parent_session TEXT NULL,
    metadata_json TEXT NULL,
    active_leaf_id TEXT NULL,
    next_sequence INTEGER NOT NULL CHECK (next_sequence > 0)
) WITHOUT ROWID;

CREATE INDEX sessions_created_at_idx ON sessions(created_at DESC);
CREATE INDEX sessions_cwd_idx ON sessions(cwd);
CREATE INDEX sessions_parent_idx ON sessions(parent_session);

CREATE TABLE session_entries (
    session_id TEXT NOT NULL,
    entry_sequence INTEGER NOT NULL CHECK (entry_sequence > 0),
    id TEXT NOT NULL,
    parent_id TEXT NULL,
    entry_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    PRIMARY KEY (session_id, entry_sequence),
    UNIQUE (session_id, id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX session_entries_parent_idx
    ON session_entries(session_id, parent_id, entry_sequence);
CREATE INDEX session_entries_type_idx
    ON session_entries(session_id, entry_type, entry_sequence);
",
    ),
    (
        2,
        r"
CREATE TABLE session_materialized (
    session_id TEXT PRIMARY KEY NOT NULL,
    active_leaf_id TEXT NULL,
    name TEXT NULL,
    message_count INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    uncached_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_total REAL NOT NULL,
    labels_json TEXT NOT NULL,
    last_sequence INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE TABLE active_branch (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    entry_id TEXT NOT NULL,
    entry_sequence INTEGER NOT NULL,
    PRIMARY KEY (session_id, position),
    UNIQUE (session_id, entry_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX active_branch_entry_idx ON active_branch(session_id, entry_id);
",
    ),
];

pub(crate) fn apply(conn: &mut Connection) -> Result<()> {
    conn.execute_batch(
        r"
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY NOT NULL,
    applied_at TEXT NOT NULL
) WITHOUT ROWID;
",
    )
    .map_err(storage_error)?;

    let applied: BTreeSet<u32> = {
        let mut statement = conn
            .prepare("SELECT version FROM schema_migrations ORDER BY version")
            .map_err(storage_error)?;
        let rows = statement
            .query_map([], |row| row.get::<_, u32>(0))
            .map_err(storage_error)?;
        rows.collect::<std::result::Result<_, _>>()
            .map_err(storage_error)?
    };
    if let Some(version) = applied.last()
        && *version > CURRENT_SCHEMA_VERSION
    {
        return Err(Error::Storage(format!(
            "SQLite schema version {version} is newer than supported version \
             {CURRENT_SCHEMA_VERSION}"
        )));
    }

    for &(version, sql) in MIGRATIONS {
        if applied.contains(&version) {
            continue;
        }
        let transaction = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        transaction.execute_batch(sql).map_err(storage_error)?;
        transaction
            .execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                params![
                    version,
                    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
                ],
            )
            .map_err(storage_error)?;
        transaction
            .pragma_update(None, "user_version", version)
            .map_err(storage_error)?;
        transaction.commit().map_err(storage_error)?;
    }
    Ok(())
}

pub(crate) fn schema_version(conn: &Connection) -> Result<u32> {
    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
        [],
        |row| row.get(0),
    )
    .map_err(storage_error)
}

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn storage_error(error: rusqlite::Error) -> Error {
    Error::Storage(error.to_string())
}
