use ri_session::{CreateOptions, ForkOptions, ListOptions, Repository, SessionEntry};
use rusqlite::Connection;
use serde_json::json;
use tempfile::tempdir;

use crate::SqliteRepository;

fn create_options(id: &str) -> CreateOptions {
    CreateOptions {
        id: Some(id.to_owned()),
        cwd: "/workspace".to_owned(),
        parent_session: None,
        metadata: None,
    }
}

#[test]
fn configures_durable_pragmas_and_applies_migrations() {
    let directory = tempdir().unwrap();
    let repository = SqliteRepository::open(directory.path().join("sessions.db")).unwrap();
    let pragmas = repository.pragma_settings().unwrap();
    assert_eq!(pragmas.journal_mode.to_ascii_lowercase(), "wal");
    assert_eq!(pragmas.synchronous, 2);
    assert_eq!(pragmas.busy_timeout_ms, 5_000);
    assert!(pragmas.foreign_keys);
    assert_eq!(repository.schema_version().unwrap(), 2);
    repository.integrity_check().unwrap();
}

#[tokio::test]
async fn reopens_tree_leaf_stats_labels_and_materialized_branch() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("sessions.db");
    let repository = SqliteRepository::open(&path).unwrap();
    let session = repository
        .create(create_options("sqlite-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "root"}))
        .await
        .unwrap();
    session
        .append_message(json!({
            "role": "assistant",
            "content": "old",
            "provider": "provider",
            "model": "model",
            "usage": {
                "input": 5,
                "output": 3,
                "cacheRead": 2,
                "cacheWrite": 1,
                "totalTokens": 11,
                "cost": {"total": 0.5}
            }
        }))
        .await
        .unwrap();
    session.branch_from(root.clone()).await.unwrap();
    let alternate = session
        .append_message(json!({"role": "user", "content": "alternate"}))
        .await
        .unwrap();
    let label_entry = session
        .append_label(alternate.clone(), Some("kept".to_owned()))
        .await
        .unwrap();

    let materialized = repository.materialized("sqlite-session").unwrap();
    assert_eq!(
        materialized.active_leaf_id.as_deref(),
        Some(label_entry.as_str())
    );
    assert_eq!(
        materialized.active_branch,
        vec![root.clone(), alternate.clone(), label_entry.clone()]
    );
    assert_eq!(materialized.stats.message_count, 3);
    assert_eq!(materialized.stats.total_tokens, 11);
    assert_eq!(
        materialized.labels.get(&alternate).map(String::as_str),
        Some("kept")
    );
    drop(session);
    drop(repository);

    let reopened_repository = SqliteRepository::open(&path).unwrap();
    let reopened = reopened_repository.open("sqlite-session").await.unwrap();
    assert_eq!(
        reopened.leaf_id().await.unwrap().as_deref(),
        Some(label_entry.as_str())
    );
    assert_eq!(
        reopened.label(&alternate).await.unwrap().as_deref(),
        Some("kept")
    );
    assert_eq!(reopened.context().await.unwrap().messages.len(), 2);
    assert_eq!(reopened.stats().await.unwrap().total_tokens, 11);
    assert_eq!(
        reopened_repository
            .materialized("sqlite-session")
            .unwrap()
            .last_sequence,
        5
    );
}

#[tokio::test]
async fn failed_materialization_rolls_back_entry_sequence_and_leaf() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("sessions.db");
    let repository = SqliteRepository::open(&path).unwrap();
    let session = repository
        .create(create_options("rollback-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "root"}))
        .await
        .unwrap();

    let trigger_connection = Connection::open(&path).unwrap();
    trigger_connection
        .execute_batch(
            r"
CREATE TRIGGER force_materialization_failure
BEFORE UPDATE ON session_materialized
BEGIN
    SELECT RAISE(ABORT, 'forced materialization failure');
END;
",
        )
        .unwrap();
    assert!(
        session
            .append_message(json!({"role": "assistant", "content": "rolled back"}))
            .await
            .is_err()
    );
    trigger_connection
        .execute_batch("DROP TRIGGER force_materialization_failure;")
        .unwrap();
    drop(trigger_connection);

    let snapshot = session.snapshot().await.unwrap();
    assert_eq!(snapshot.entries().len(), 1);
    assert_eq!(snapshot.leaf_id(), Some(root.as_str()));
    let committed = session
        .append_message(json!({"role": "assistant", "content": "committed"}))
        .await
        .unwrap();
    let stored = session.entry(&committed).await.unwrap().unwrap();
    assert_eq!(stored.sequence, 2);
    assert_eq!(
        repository
            .materialized("rollback-session")
            .unwrap()
            .last_sequence,
        2
    );
}

#[tokio::test]
async fn reopen_recovers_corrupt_materialized_state_from_entries() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("sessions.db");
    let repository = SqliteRepository::open(&path).unwrap();
    let session = repository
        .create(create_options("recovery-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "root"}))
        .await
        .unwrap();
    let assistant = session
        .append_message(json!({"role": "assistant", "content": "reply"}))
        .await
        .unwrap();
    drop(session);
    drop(repository);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE session_materialized \
             SET message_count = 999, active_leaf_id = NULL, labels_json = '[]'",
            [],
        )
        .unwrap();
    connection.execute("DELETE FROM active_branch", []).unwrap();
    connection
        .execute(
            "UPDATE sessions SET active_leaf_id = NULL, next_sequence = 1 \
             WHERE id = 'recovery-session'",
            [],
        )
        .unwrap();
    drop(connection);

    let reopened = SqliteRepository::open(&path).unwrap();
    let materialized = reopened.materialized("recovery-session").unwrap();
    assert_eq!(materialized.stats.message_count, 2);
    assert_eq!(
        materialized.active_leaf_id.as_deref(),
        Some(assistant.as_str())
    );
    assert_eq!(materialized.active_branch, vec![root, assistant]);
    let session = reopened.open("recovery-session").await.unwrap();
    let next = session
        .append_message(json!({"role": "user", "content": "after recovery"}))
        .await
        .unwrap();
    assert_eq!(session.entry(&next).await.unwrap().unwrap().sequence, 3);
}

#[test]
fn migrates_a_version_one_database_on_reopen() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("sessions.db");
    {
        let repository = SqliteRepository::open(&path).unwrap();
        assert_eq!(repository.schema_version().unwrap(), 2);
    }
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            r"
DROP TABLE active_branch;
DROP TABLE session_materialized;
DELETE FROM schema_migrations WHERE version = 2;
PRAGMA user_version = 1;
",
        )
        .unwrap();
    drop(connection);

    let migrated = SqliteRepository::open(&path).unwrap();
    assert_eq!(migrated.schema_version().unwrap(), 2);
    migrated.integrity_check().unwrap();
}

#[tokio::test]
async fn supports_fork_list_and_delete() {
    let directory = tempdir().unwrap();
    let repository = SqliteRepository::open(directory.path().join("sessions.db")).unwrap();
    let source = repository.create(create_options("source")).await.unwrap();
    source
        .append_message(json!({"role": "user", "content": "one"}))
        .await
        .unwrap();
    let second = source
        .append_message(json!({"role": "assistant", "content": "two"}))
        .await
        .unwrap();
    let fork = repository
        .fork(
            "source",
            ForkOptions {
                id: Some("fork".to_owned()),
                entry_id: Some(second),
                position: ri_session::ForkPosition::At,
                ..ForkOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(fork.snapshot().await.unwrap().entries().len(), 2);
    assert_eq!(
        repository.list(ListOptions::default()).await.unwrap().len(),
        2
    );
    repository.delete("fork").await.unwrap();
    assert_eq!(
        repository.list(ListOptions::default()).await.unwrap().len(),
        1
    );

    let entries = source.snapshot().await.unwrap();
    assert!(matches!(
        &entries.entries()[0].entry,
        SessionEntry::Message(_)
    ));
}
