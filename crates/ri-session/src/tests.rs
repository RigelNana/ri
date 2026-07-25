use std::collections::BTreeMap;

use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::io::AsyncWriteExt;

use crate::{
    CreateOptions, ForkOptions, ForkPosition, JsonlRepository, ListOptions, MalformedEntryPolicy,
    MemoryRepository, Repository, SessionEntry,
};

fn create_options(id: &str) -> CreateOptions {
    CreateOptions {
        id: Some(id.to_owned()),
        cwd: "/workspace".to_owned(),
        parent_session: None,
        metadata: Some(BTreeMap::from([(
            "owner".to_owned(),
            Value::String("test".to_owned()),
        )])),
    }
}

#[tokio::test]
async fn memory_repository_builds_branches_tree_labels_and_stats() {
    let repository = MemoryRepository::default();
    let session = repository
        .create(create_options("memory-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "root"}))
        .await
        .unwrap();
    let abandoned = session
        .append_message(json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "old"}],
            "provider": "provider",
            "model": "model-a",
            "usage": {
                "input": 4,
                "output": 3,
                "cacheRead": 2,
                "cacheWrite": 1,
                "totalTokens": 10,
                "cost": {"total": 0.25}
            }
        }))
        .await
        .unwrap();
    session.branch_from(root.clone()).await.unwrap();
    let alternate = session
        .append_message(json!({"role": "user", "content": "alternate"}))
        .await
        .unwrap();
    session
        .append_label(alternate.clone(), Some("checkpoint".to_owned()))
        .await
        .unwrap();

    let snapshot = session.snapshot().await.unwrap();
    assert!(snapshot.entries().iter().any(|stored| {
        matches!(
            &stored.entry,
            SessionEntry::Leaf(entry) if entry.target_id.as_deref() == Some(root.as_str())
        )
    }));
    let path: Vec<_> = snapshot
        .path_to(Some(&alternate))
        .unwrap()
        .into_iter()
        .map(|stored| stored.entry.id().to_owned())
        .collect();
    assert_eq!(path, vec![root.clone(), alternate.clone()]);
    assert_eq!(snapshot.label(&alternate).as_deref(), Some("checkpoint"));
    assert_eq!(snapshot.stats().message_count, 3);
    assert_eq!(snapshot.stats().total_tokens, 10);
    assert!((snapshot.stats().cost_total - 0.25).abs() < f64::EPSILON);

    let tree = snapshot.tree();
    assert_eq!(tree.len(), 1);
    let child_ids: Vec<_> = tree[0]
        .children
        .iter()
        .map(|node| node.entry.id())
        .collect();
    assert!(child_ids.contains(&abandoned.as_str()));
    assert!(child_ids.contains(&alternate.as_str()));
}

#[tokio::test]
async fn compaction_context_supports_legacy_boundary_and_retained_tail() {
    let repository = MemoryRepository::default();
    let session = repository
        .create(create_options("context-session"))
        .await
        .unwrap();
    session
        .append_model_change("provider", "model-a")
        .await
        .unwrap();
    session.append_thinking_level_change("high").await.unwrap();
    session
        .append_active_tools_change(vec!["read".to_owned()])
        .await
        .unwrap();
    session
        .append_message(json!({"role": "user", "content": "old"}))
        .await
        .unwrap();
    session
        .append_message(json!({
            "role": "assistant",
            "content": [],
            "provider": "provider",
            "model": "model-a"
        }))
        .await
        .unwrap();
    let kept = session
        .append_message(json!({"role": "user", "content": "kept"}))
        .await
        .unwrap();
    session
        .append_compaction("legacy summary", Some(kept), 50)
        .await
        .unwrap();
    session
        .append_message(json!({"role": "assistant", "content": "after"}))
        .await
        .unwrap();

    let context = session.context().await.unwrap();
    assert_eq!(context.messages[0]["role"], "compactionSummary");
    assert_eq!(context.messages[1]["content"], "kept");
    assert_eq!(context.messages[2]["content"], "after");
    assert_eq!(context.thinking_level, "high");
    assert_eq!(context.model.unwrap().model_id, "model-a");
    assert_eq!(context.active_tool_names.unwrap(), vec!["read"]);

    session
        .append_compaction_with(
            "self contained",
            None,
            25,
            Some(vec![json!({"role": "user", "content": "retained"})]),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    session
        .append_message(json!({"role": "assistant", "content": "new"}))
        .await
        .unwrap();
    let context = session.context().await.unwrap();
    assert_eq!(context.messages.len(), 3);
    assert_eq!(context.messages[0]["summary"], "self contained");
    assert_eq!(context.messages[1]["content"], "retained");
    assert_eq!(context.messages[2]["content"], "new");
}

#[tokio::test]
async fn jsonl_reopens_with_durable_leaf_and_supports_list_delete() {
    let directory = tempdir().unwrap();
    let repository = JsonlRepository::new(directory.path());
    let session = repository
        .create(create_options("jsonl-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "root"}))
        .await
        .unwrap();
    session
        .append_message(json!({"role": "assistant", "content": "abandoned"}))
        .await
        .unwrap();
    session.branch_from(root).await.unwrap();
    let alternate = session
        .append_message(json!({"role": "user", "content": "alternate"}))
        .await
        .unwrap();
    let path = session.metadata().await.unwrap().path.unwrap();
    drop(session);
    drop(repository);

    let reopened_repository = JsonlRepository::new(directory.path());
    let reopened = reopened_repository.open("jsonl-session").await.unwrap();
    assert_eq!(
        reopened.leaf_id().await.unwrap().as_deref(),
        Some(alternate.as_str())
    );
    assert_eq!(reopened.context().await.unwrap().messages.len(), 2);
    assert!(
        tokio::fs::read_to_string(path)
            .await
            .unwrap()
            .lines()
            .count()
            >= 5
    );

    let listed = reopened_repository
        .list(ListOptions {
            cwd: Some("/workspace".to_owned()),
        })
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    reopened_repository.delete("jsonl-session").await.unwrap();
    assert!(
        reopened_repository
            .list(ListOptions::default())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn jsonl_malformed_policy_is_explicit() {
    let directory = tempdir().unwrap();
    let repository = JsonlRepository::new(directory.path());
    let session = repository
        .create(create_options("malformed-session"))
        .await
        .unwrap();
    let root = session
        .append_message(json!({"role": "user", "content": "valid"}))
        .await
        .unwrap();
    let path = session.metadata().await.unwrap().path.unwrap();
    drop(session);
    drop(repository);

    let mut file = tokio::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .await
        .unwrap();
    file.write_all(b"{not-json}\n").await.unwrap();
    file.write_all(
        format!(
            "{}\n",
            json!({
                "type": "message",
                "id": "afterbad",
                "parentId": root,
                "timestamp": "2026-01-01T00:00:00.000Z",
                "message": {"role": "assistant", "content": "still valid"}
            })
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    file.sync_all().await.unwrap();
    drop(file);

    let strict = JsonlRepository::new(directory.path());
    assert!(strict.open("malformed-session").await.is_err());
    let permissive = JsonlRepository::with_policy(directory.path(), MalformedEntryPolicy::Skip);
    let reopened = permissive.open("malformed-session").await.unwrap();
    assert_eq!(reopened.snapshot().await.unwrap().entries().len(), 2);
}

#[tokio::test]
async fn jsonl_migrates_v1_and_v2_records_on_reopen() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("legacy.jsonl");
    let contents = [
        json!({
            "type": "session",
            "id": "legacy-session",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "cwd": "/workspace"
        }),
        json!({
            "type": "message",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "message": {"role": "user", "content": "hello"}
        }),
        json!({
            "type": "message",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "message": {"role": "hookMessage", "content": "legacy hook"}
        }),
    ]
    .into_iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>()
    .join("\n");
    tokio::fs::write(&path, format!("{contents}\n"))
        .await
        .unwrap();

    let repository = JsonlRepository::new(directory.path());
    let session = repository.open("legacy-session").await.unwrap();
    let snapshot = session.snapshot().await.unwrap();
    assert_eq!(snapshot.header().version, 3);
    assert_eq!(snapshot.entries().len(), 2);
    let SessionEntry::Message(second) = &snapshot.entries()[1].entry else {
        panic!("expected migrated message");
    };
    assert_eq!(second.message["role"], "custom");
    let rewritten = tokio::fs::read_to_string(path).await.unwrap();
    assert!(rewritten.lines().next().unwrap().contains("\"version\":3"));
    assert!(rewritten.contains("\"parentId\""));
}

#[tokio::test]
async fn repository_fork_before_user_message_copies_only_prior_path() {
    let repository = MemoryRepository::default();
    let source = repository
        .create(create_options("source-session"))
        .await
        .unwrap();
    source
        .append_message(json!({"role": "user", "content": "first"}))
        .await
        .unwrap();
    source
        .append_message(json!({"role": "assistant", "content": "reply"}))
        .await
        .unwrap();
    let second_user = source
        .append_message(json!({"role": "user", "content": "second"}))
        .await
        .unwrap();

    let fork = repository
        .fork(
            "source-session",
            ForkOptions {
                id: Some("fork-session".to_owned()),
                entry_id: Some(second_user),
                position: ForkPosition::Before,
                ..ForkOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(fork.snapshot().await.unwrap().entries().len(), 2);
    assert_eq!(
        fork.metadata().await.unwrap().parent_session.as_deref(),
        Some("source-session")
    );
}
