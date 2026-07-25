//! Explicit Pi session JSONL v1-v3 import, migration, and export.

use std::collections::HashMap;

use ri_rpc::{SessionEntry, decode_jsonl};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use uuid::Uuid;

/// Pi session file version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PiSessionVersion {
    /// Legacy linear entries without identifiers.
    V1,
    /// Identifier-linked entry tree with `hookMessage`.
    V2,
    /// Current identifier-linked tree with `custom` messages.
    V3,
}

impl PiSessionVersion {
    /// Numeric header value.
    pub const fn number(self) -> u8 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

/// Normalized Pi session header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiSessionHeader {
    /// Session UUID or caller-selected identifier.
    pub id: String,
    /// ISO-8601 creation timestamp.
    pub timestamp: String,
    /// Working directory recorded by Pi.
    pub cwd: String,
    /// Optional source-session lineage path.
    pub parent_session: Option<String>,
}

/// Imported session normalized to v3 entry semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct PiSession {
    /// Version observed during import.
    pub source_version: PiSessionVersion,
    /// Header metadata.
    pub header: PiSessionHeader,
    /// Append-order entries with v3 messages and tree identifiers.
    pub entries: Vec<SessionEntry>,
}

impl PiSession {
    /// Construct a new normalized session.
    pub const fn new(header: PiSessionHeader, entries: Vec<SessionEntry>) -> Self {
        Self {
            source_version: PiSessionVersion::V3,
            header,
            entries,
        }
    }
}

/// Session compatibility failure.
#[derive(Debug, thiserror::Error)]
pub enum PiSessionError {
    /// Strict JSONL framing or JSON decoding failed.
    #[error(transparent)]
    Jsonl(#[from] ri_rpc::JsonlError),
    /// Session has no header.
    #[error("Pi session is empty")]
    Empty,
    /// First record was not a valid session header.
    #[error("invalid Pi session header: {0}")]
    Header(String),
    /// Header version is unsupported.
    #[error("unsupported Pi session version: {0}")]
    UnsupportedVersion(u64),
    /// A typed entry failed to decode.
    #[error("invalid Pi session entry at line {line}: {source}")]
    Entry {
        /// One-based JSONL line.
        line: usize,
        /// Serde failure.
        #[source]
        source: serde_json::Error,
    },
    /// A record was not a JSON object.
    #[error("invalid Pi session entry at line {line}: entry must be a JSON object")]
    EntryNotObject {
        /// One-based JSONL line.
        line: usize,
    },
    /// Generated migration identifier was empty or duplicated.
    #[error("v1 migration generated invalid or duplicate entry id `{0}`")]
    InvalidGeneratedId(String),
    /// A v1 compaction points outside the input.
    #[error("v1 compaction at line {line} references missing entry index {index}")]
    InvalidCompactionIndex {
        /// One-based JSONL line.
        line: usize,
        /// Legacy file-entry index.
        index: usize,
    },
    /// Tree cannot be represented by v1's single linear sequence.
    #[error("cannot export branched or out-of-order entries as Pi session v1")]
    NonLinearV1,
    /// A compaction pointer cannot be represented in the selected target.
    #[error("compaction references unknown retained entry `{0}`")]
    UnknownCompactionEntry(String),
    /// JSON serialization failed.
    #[error("failed to serialize Pi session: {0}")]
    Serialize(#[from] serde_json::Error),
    /// An internal typed entry did not serialize as an object.
    #[error("typed Pi session entry did not serialize as a JSON object")]
    InvalidSerializedEntry,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WireHeader {
    #[serde(rename = "type")]
    kind: String,
    version: Option<u64>,
    id: String,
    timestamp: String,
    cwd: String,
    parent_session: Option<String>,
}

/// Import and migrate a caller-provided Pi session document.
///
/// This function performs no path discovery or filesystem access.
///
/// # Errors
///
/// Returns an error for invalid JSONL, a missing or unsupported header,
/// malformed entries, or an invalid legacy migration.
pub fn import_session(input: &[u8]) -> Result<PiSession, PiSessionError> {
    import_session_with_ids(input, || {
        Uuid::new_v4().simple().to_string()[..8].to_owned()
    })
}

/// Import a session with an explicit v1 identifier generator.
///
/// The hook makes migrations reproducible in conformance tests and import tools.
///
/// # Errors
///
/// Returns an error for invalid JSONL, a missing or unsupported header,
/// malformed entries, invalid legacy pointers, or empty or duplicate generated
/// identifiers.
pub fn import_session_with_ids<F>(input: &[u8], mut next_id: F) -> Result<PiSession, PiSessionError>
where
    F: FnMut() -> String,
{
    let mut records: Vec<Value> = decode_jsonl(input)?;
    if records.is_empty() {
        return Err(PiSessionError::Empty);
    }
    let header: WireHeader = serde_json::from_value(records.remove(0))
        .map_err(|error| PiSessionError::Header(error.to_string()))?;
    if header.kind != "session" {
        return Err(PiSessionError::Header(format!(
            "expected type `session`, found `{}`",
            header.kind
        )));
    }
    let source_version = match header.version.unwrap_or(1) {
        1 => PiSessionVersion::V1,
        2 => PiSessionVersion::V2,
        3 => PiSessionVersion::V3,
        version => return Err(PiSessionError::UnsupportedVersion(version)),
    };

    ri_session::legacy::migrate_legacy_entries(
        u32::from(source_version.number()),
        records.iter_mut(),
        &mut next_id,
        true,
    )
    .map_err(|error| match error {
        ri_session::legacy::LegacyMigrationError::InvalidGeneratedId(id) => {
            PiSessionError::InvalidGeneratedId(id)
        }
        ri_session::legacy::LegacyMigrationError::EntryNotObject { index } => {
            PiSessionError::EntryNotObject { line: index + 2 }
        }
        ri_session::legacy::LegacyMigrationError::InvalidCompactionIndex { index, file_index } => {
            PiSessionError::InvalidCompactionIndex {
                line: index + 2,
                index: file_index,
            }
        }
    })?;

    let entries = records
        .into_iter()
        .enumerate()
        .map(|(index, record)| {
            serde_json::from_value(record).map_err(|source| PiSessionError::Entry {
                line: index + 2,
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PiSession {
        source_version,
        header: PiSessionHeader {
            id: header.id,
            timestamp: header.timestamp,
            cwd: header.cwd,
            parent_session: header.parent_session,
        },
        entries,
    })
}

/// Export a normalized session to an explicit Pi JSONL version.
///
/// V1 export is rejected unless append order is one strictly linear branch.
///
/// # Errors
///
/// Returns an error if entries cannot be serialized, the target cannot
/// represent the session tree, or a compaction points to an unknown entry.
pub fn export_session(
    session: &PiSession,
    target: PiSessionVersion,
) -> Result<Vec<u8>, PiSessionError> {
    let mut output = Vec::new();
    let mut header = Map::new();
    header.insert("type".to_owned(), Value::String("session".to_owned()));
    if target != PiSessionVersion::V1 {
        header.insert(
            "version".to_owned(),
            Value::Number(Number::from(target.number())),
        );
    }
    header.insert("id".to_owned(), Value::String(session.header.id.clone()));
    header.insert(
        "timestamp".to_owned(),
        Value::String(session.header.timestamp.clone()),
    );
    header.insert("cwd".to_owned(), Value::String(session.header.cwd.clone()));
    if let Some(parent) = &session.header.parent_session {
        header.insert("parentSession".to_owned(), Value::String(parent.clone()));
    }
    append_line(&mut output, &Value::Object(header))?;

    let id_to_file_index = if target == PiSessionVersion::V1 {
        validate_linear_v1(&session.entries)?
    } else {
        HashMap::new()
    };

    for entry in &session.entries {
        let mut value = serde_json::to_value(entry)?;
        if target != PiSessionVersion::V3 {
            ri_session::legacy::rename_message_role(&mut value, "custom", "hookMessage");
        }
        if target == PiSessionVersion::V1 {
            let object = value
                .as_object_mut()
                .ok_or(PiSessionError::InvalidSerializedEntry)?;
            object.remove("id");
            object.remove("parentId");
            if object.get("type").and_then(Value::as_str) == Some("compaction") {
                if let Some(pointer) = object.remove("firstKeptEntryId") {
                    if !pointer.is_null() {
                        let retained_id = pointer.as_str().unwrap_or_default();
                        let Some(index) = id_to_file_index.get(retained_id) else {
                            return Err(PiSessionError::UnknownCompactionEntry(
                                retained_id.to_owned(),
                            ));
                        };
                        object.insert(
                            "firstKeptEntryIndex".to_owned(),
                            Value::Number(Number::from(*index)),
                        );
                    }
                }
            }
        }
        append_line(&mut output, &value)?;
    }
    Ok(output)
}

fn validate_linear_v1(entries: &[SessionEntry]) -> Result<HashMap<String, usize>, PiSessionError> {
    let mut previous: Option<&str> = None;
    let mut indexes = HashMap::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        if entry.parent_id() != previous {
            return Err(PiSessionError::NonLinearV1);
        }
        indexes.insert(entry.id().to_owned(), position + 1);
        previous = Some(entry.id());
    }
    Ok(indexes)
}

fn append_line(output: &mut Vec<u8>, value: &Value) -> Result<(), serde_json::Error> {
    serde_json::to_writer(&mut *output, value)?;
    output.push(b'\n');
    Ok(())
}

/// Serialize a version label for reports and CLIs.
impl Serialize for PiSessionVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(self.number())
    }
}

/// Deserialize numeric version labels.
impl<'de> Deserialize<'de> for PiSessionVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match u8::deserialize(deserializer)? {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            version => Err(serde::de::Error::custom(format!(
                "unsupported Pi session version: {version}"
            ))),
        }
    }
}
