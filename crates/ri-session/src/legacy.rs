//! Shared migration rules for Pi-compatible session records predating v3.

use std::collections::HashSet;

use serde_json::Value;

/// Failure while upgrading legacy raw session entries.
#[derive(Debug, thiserror::Error)]
pub enum LegacyMigrationError {
    /// The caller-provided v1 identifier generator produced an invalid value.
    #[error("generated invalid or duplicate entry id `{0}`")]
    InvalidGeneratedId(String),
    /// A record that needs v1 tree fields is not a JSON object.
    #[error("legacy entry at index {index} is not a JSON object")]
    EntryNotObject {
        /// Zero-based entry index, excluding the header.
        index: usize,
    },
    /// A v1 compaction pointer does not select an existing file entry.
    #[error("legacy compaction at index {index} references missing file-entry index {file_index}")]
    InvalidCompactionIndex {
        /// Zero-based entry index, excluding the header.
        index: usize,
        /// One-based file-entry index used by v1.
        file_index: usize,
    },
}

/// Migrates raw v1/v2 entry values to v3 tree and custom-message semantics.
///
/// `strict = false` leaves malformed records for a repository's later skip
/// policy while still migrating every valid object.
///
/// # Errors
///
/// Returns an error for invalid generated identifiers and, in strict mode,
/// malformed objects or invalid v1 compaction indexes.
pub fn migrate_legacy_entries<'a, I, F>(
    version: u32,
    records: I,
    mut next_id: F,
    strict: bool,
) -> Result<(), LegacyMigrationError>
where
    I: IntoIterator<Item = &'a mut Value>,
    F: FnMut() -> String,
{
    let mut records = records.into_iter().collect::<Vec<_>>();
    if version < 2 {
        let mut unique = HashSet::with_capacity(records.len());
        let mut ids = Vec::with_capacity(records.len());
        for _ in 0..records.len() {
            let id = next_id();
            if id.is_empty() || !unique.insert(id.clone()) {
                return Err(LegacyMigrationError::InvalidGeneratedId(id));
            }
            ids.push(id);
        }

        let mut previous_id: Option<String> = None;
        for (index, record) in records.iter_mut().enumerate() {
            let Some(object) = record.as_object_mut() else {
                if strict {
                    return Err(LegacyMigrationError::EntryNotObject { index });
                }
                continue;
            };
            object.insert("id".to_owned(), Value::String(ids[index].clone()));
            object.insert(
                "parentId".to_owned(),
                previous_id.clone().map_or(Value::Null, Value::String),
            );
            previous_id = Some(ids[index].clone());

            if object.get("type").and_then(Value::as_str) == Some("compaction")
                && let Some(pointer) = object.remove("firstKeptEntryIndex")
            {
                let file_index = pointer
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(usize::MAX);
                let target = file_index
                    .checked_sub(1)
                    .and_then(|entry_index| ids.get(entry_index));
                if let Some(target) = target {
                    object.insert("firstKeptEntryId".to_owned(), Value::String(target.clone()));
                } else if strict {
                    return Err(LegacyMigrationError::InvalidCompactionIndex { index, file_index });
                }
            }
        }
    }

    if version < 3 {
        for record in records {
            rename_message_role(record, "hookMessage", "custom");
        }
    }
    Ok(())
}

/// Renames one message role in a raw session entry.
pub fn rename_message_role(record: &mut Value, from: &str, to: &str) {
    let Some(entry) = record.as_object_mut() else {
        return;
    };
    if entry.get("type").and_then(Value::as_str) != Some("message") {
        return;
    }
    let Some(message) = entry.get_mut("message").and_then(Value::as_object_mut) else {
        return;
    };
    if message.get("role").and_then(Value::as_str) == Some(from) {
        message.insert("role".to_owned(), Value::String(to.to_owned()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migration_assigns_tree_fields_compaction_pointer_and_custom_role() {
        let mut records = [
            json!({"type":"message","message":{"role":"user","content":"hi"}}),
            json!({
                "type":"compaction",
                "firstKeptEntryIndex":1,
                "summary":"s",
                "tokensBefore":1,
                "timestamp":"2026-01-01T00:00:00Z"
            }),
            json!({"type":"message","message":{"role":"hookMessage","content":"x"}}),
        ];
        let mut ids = ["a", "b", "c"].into_iter();
        migrate_legacy_entries(
            1,
            records.iter_mut(),
            || ids.next().unwrap().to_owned(),
            true,
        )
        .unwrap();
        assert_eq!(records[1]["parentId"], "a");
        assert_eq!(records[1]["firstKeptEntryId"], "a");
        assert_eq!(records[2]["message"]["role"], "custom");
    }
}
