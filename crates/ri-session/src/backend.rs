//! Backend-neutral repository rules.
//!
//! Persistence implementations should use these helpers so identifiers,
//! headers, and forks have identical behavior across storage backends.

use serde_json::Value;
use uuid::Uuid;

use crate::{
    CreateOptions, Error, ForkOptions, ForkPosition, Result, SequencedEntry, SessionEntry,
    SessionHeader, SessionSnapshot,
};

/// Builds and validates a session header from repository create options.
///
/// # Errors
///
/// Returns [`Error::InvalidSession`] when a caller-selected identifier is not
/// safe to use as a backend key or filename.
pub fn header_from_options(options: CreateOptions) -> Result<SessionHeader> {
    let id = options
        .id
        .unwrap_or_else(|| Uuid::now_v7().hyphenated().to_string());
    validate_session_id(&id)?;
    let mut header = SessionHeader::new(id, options.cwd);
    header.parent_session = options.parent_session;
    header.metadata = options.metadata;
    Ok(header)
}

/// Validates the portable session identifier grammar shared by all backends.
///
/// # Errors
///
/// Returns [`Error::InvalidSession`] when `id` is empty, starts or ends with
/// punctuation, or contains characters outside ASCII alphanumeric, `-`, `_`,
/// and `.`.
pub fn validate_session_id(id: &str) -> Result<()> {
    let valid = id
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric())
        && id.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
        && id
            .chars()
            .next_back()
            .is_some_and(|character| character.is_ascii_alphanumeric());
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidSession(format!(
            "session id {id:?} contains unsafe characters"
        )))
    }
}

/// Selects the append-ordered entries copied by a fork operation.
///
/// # Errors
///
/// Returns [`Error::InvalidForkTarget`] for a missing target or when `Before`
/// does not point to a user message.
pub fn entries_for_fork(
    snapshot: &SessionSnapshot,
    options: &ForkOptions,
) -> Result<Vec<SequencedEntry>> {
    let Some(target_id) = options.entry_id.as_deref() else {
        return Ok(snapshot.entries().to_vec());
    };
    let target = snapshot
        .entry(target_id)
        .ok_or_else(|| Error::InvalidForkTarget(format!("entry {target_id} was not found")))?;
    let effective_id = match options.position {
        ForkPosition::At => Some(target_id),
        ForkPosition::Before => {
            let SessionEntry::Message(message) = &target.entry else {
                return Err(Error::InvalidForkTarget(format!(
                    "entry {target_id} is not a user message"
                )));
            };
            if message.message.get("role").and_then(Value::as_str) != Some("user") {
                return Err(Error::InvalidForkTarget(format!(
                    "entry {target_id} is not a user message"
                )));
            }
            message.base.parent_id.as_deref()
        }
    };
    Ok(snapshot
        .path_to(effective_id)?
        .into_iter()
        .cloned()
        .collect())
}

/// Builds destination create options for a fork, applying source inheritance.
pub fn fork_create_options(source: &SessionSnapshot, options: ForkOptions) -> CreateOptions {
    CreateOptions {
        id: options.id,
        cwd: options.cwd.unwrap_or_else(|| source.header().cwd.clone()),
        parent_session: Some(source.header().id.clone()),
        metadata: options
            .metadata
            .or_else(|| source.header().metadata.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_session_ids_reject_empty_and_punctuation_edges() {
        for valid in ["a", "session-1", "a_b.c"] {
            validate_session_id(valid).expect("valid session id");
        }
        for invalid in ["", "-a", "a-", "a/b", "会话"] {
            assert!(
                validate_session_id(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }
}
