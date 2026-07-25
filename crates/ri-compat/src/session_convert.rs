//! Explicit conversion between native Ri sessions and Pi session wire types.

use chrono::{DateTime, Utc};
use ri_rpc::{SessionEntry as PiEntry, Usage as PiUsage, UsageCost as PiUsageCost};
use ri_session::{
    BranchSummaryEntry, CompactionEntry, CustomEntry, CustomMessageEntry, EntryBase, LabelEntry,
    MessageEntry, ModelChangeEntry, SessionEntry as NativeEntry, SessionHeader, SessionInfoEntry,
    ThinkingLevelChangeEntry, Usage as NativeUsage, UsageCost as NativeUsageCost,
};
use serde_json::Value;

use crate::PiSessionHeader;

/// Failure to represent a session record across the native/Pi boundary.
#[derive(Debug, thiserror::Error)]
pub enum SessionConversionError {
    /// A native field or entry has no Pi session equivalent.
    #[error("native session {0} cannot be represented by the Pi session format")]
    UnsupportedNative(&'static str),
    /// A Pi timestamp is not RFC 3339.
    #[error("invalid Pi timestamp `{value}`: {source}")]
    Timestamp {
        /// Rejected timestamp text.
        value: String,
        /// Parser failure.
        #[source]
        source: chrono::ParseError,
    },
    /// A message value does not conform to the Pi agent-message model.
    #[error("invalid Pi-compatible {context}: {source}")]
    Message {
        /// Value being converted.
        context: &'static str,
        /// JSON value conversion failure.
        #[source]
        source: serde_json::Error,
    },
}

/// Converts a native header to the Pi header model without silently dropping
/// application metadata.
///
/// # Errors
///
/// Returns [`SessionConversionError::UnsupportedNative`] when non-empty native
/// metadata is present because Pi has no corresponding header field.
pub fn native_header_to_pi(
    header: SessionHeader,
) -> Result<PiSessionHeader, SessionConversionError> {
    if header
        .metadata
        .as_ref()
        .is_some_and(|metadata| !metadata.is_empty())
    {
        return Err(SessionConversionError::UnsupportedNative("header metadata"));
    }
    Ok(PiSessionHeader {
        id: header.id,
        timestamp: header.timestamp.to_rfc3339(),
        cwd: header.cwd,
        parent_session: header.parent_session,
    })
}

/// Converts a Pi header to the current native header model.
///
/// # Errors
///
/// Returns an error when the Pi timestamp is not RFC 3339.
pub fn pi_header_to_native(
    header: PiSessionHeader,
) -> Result<SessionHeader, SessionConversionError> {
    let timestamp = parse_timestamp(&header.timestamp)?;
    let mut native = SessionHeader::new(header.id, header.cwd);
    native.timestamp = timestamp;
    native.parent_session = header.parent_session;
    Ok(native)
}

/// Converts one native entry to the Pi session wire model.
///
/// # Errors
///
/// Returns an error for malformed message values or native-only entry kinds.
pub fn native_entry_to_pi(entry: NativeEntry) -> Result<PiEntry, SessionConversionError> {
    match entry {
        NativeEntry::Message(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::Message {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                message: decode(entry.message, "message entry")?,
            })
        }
        NativeEntry::ModelChange(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::ModelChange {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                provider: entry.provider,
                model_id: entry.model_id,
            })
        }
        NativeEntry::ThinkingLevelChange(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::ThinkingLevelChange {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                thinking_level: entry.thinking_level,
            })
        }
        NativeEntry::Compaction(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::Compaction {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                summary: entry.summary,
                first_kept_entry_id: entry.first_kept_entry_id,
                tokens_before: entry.tokens_before,
                retained_tail: entry
                    .retained_tail
                    .unwrap_or_default()
                    .into_iter()
                    .map(|message| decode(message, "compaction retained-tail message"))
                    .collect::<Result<Vec<_>, _>>()?,
                usage: entry.usage.map(native_usage_to_pi),
                details: entry.details,
                from_hook: entry.from_hook.unwrap_or(false),
            })
        }
        NativeEntry::BranchSummary(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::BranchSummary {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                from_id: entry.from_id,
                summary: entry.summary,
                usage: entry.usage.map(native_usage_to_pi),
                details: entry.details,
                from_hook: entry.from_hook.unwrap_or(false),
            })
        }
        NativeEntry::Custom(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::Custom {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                custom_type: entry.custom_type,
                data: entry.data,
            })
        }
        NativeEntry::CustomMessage(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::CustomMessage {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                custom_type: entry.custom_type,
                content: decode(entry.content, "custom-message content")?,
                details: entry.details,
                display: entry.display,
            })
        }
        NativeEntry::Label(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::Label {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                target_id: entry.target_id,
                label: entry.label,
            })
        }
        NativeEntry::SessionInfo(entry) => {
            let base = PiBase::from(entry.base);
            Ok(PiEntry::SessionInfo {
                id: base.id,
                parent_id: base.parent_id,
                timestamp: base.timestamp,
                name: entry.name,
            })
        }
        NativeEntry::ActiveToolsChange(_) => Err(SessionConversionError::UnsupportedNative(
            "active_tools_change entry",
        )),
        NativeEntry::Leaf(_) => Err(SessionConversionError::UnsupportedNative("leaf entry")),
    }
}

/// Converts one Pi session entry to the native session model.
///
/// # Errors
///
/// Returns an error for invalid timestamps or message values that cannot be
/// represented by the native open-JSON message boundary.
pub fn pi_entry_to_native(entry: PiEntry) -> Result<NativeEntry, SessionConversionError> {
    match entry {
        PiEntry::Message {
            id,
            parent_id,
            timestamp,
            message,
        } => Ok(NativeEntry::Message(MessageEntry {
            base: native_base(id, parent_id, &timestamp)?,
            message: encode(message, "message entry")?,
        })),
        PiEntry::ModelChange {
            id,
            parent_id,
            timestamp,
            provider,
            model_id,
        } => Ok(NativeEntry::ModelChange(ModelChangeEntry {
            base: native_base(id, parent_id, &timestamp)?,
            provider,
            model_id,
        })),
        PiEntry::ThinkingLevelChange {
            id,
            parent_id,
            timestamp,
            thinking_level,
        } => Ok(NativeEntry::ThinkingLevelChange(ThinkingLevelChangeEntry {
            base: native_base(id, parent_id, &timestamp)?,
            thinking_level,
        })),
        PiEntry::Compaction {
            id,
            parent_id,
            timestamp,
            summary,
            first_kept_entry_id,
            tokens_before,
            retained_tail,
            usage,
            details,
            from_hook,
        } => Ok(NativeEntry::Compaction(CompactionEntry {
            base: native_base(id, parent_id, &timestamp)?,
            summary,
            first_kept_entry_id,
            tokens_before,
            retained_tail: (!retained_tail.is_empty())
                .then(|| {
                    retained_tail
                        .into_iter()
                        .map(|message| encode(message, "compaction retained-tail message"))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
            details,
            usage: usage.as_ref().map(pi_usage_to_native),
            from_hook: from_hook.then_some(true),
        })),
        PiEntry::BranchSummary {
            id,
            parent_id,
            timestamp,
            from_id,
            summary,
            usage,
            details,
            from_hook,
        } => Ok(NativeEntry::BranchSummary(BranchSummaryEntry {
            base: native_base(id, parent_id, &timestamp)?,
            from_id,
            summary,
            details,
            usage: usage.as_ref().map(pi_usage_to_native),
            from_hook: from_hook.then_some(true),
        })),
        PiEntry::Custom {
            id,
            parent_id,
            timestamp,
            custom_type,
            data,
        } => Ok(NativeEntry::Custom(CustomEntry {
            base: native_base(id, parent_id, &timestamp)?,
            custom_type,
            data,
        })),
        PiEntry::CustomMessage {
            id,
            parent_id,
            timestamp,
            custom_type,
            content,
            details,
            display,
        } => Ok(NativeEntry::CustomMessage(CustomMessageEntry {
            base: native_base(id, parent_id, &timestamp)?,
            custom_type,
            content: encode(content, "custom-message content")?,
            display,
            details,
        })),
        PiEntry::Label {
            id,
            parent_id,
            timestamp,
            target_id,
            label,
        } => Ok(NativeEntry::Label(LabelEntry {
            base: native_base(id, parent_id, &timestamp)?,
            target_id,
            label,
        })),
        PiEntry::SessionInfo {
            id,
            parent_id,
            timestamp,
            name,
        } => Ok(NativeEntry::SessionInfo(SessionInfoEntry {
            base: native_base(id, parent_id, &timestamp)?,
            name,
        })),
    }
}

struct PiBase {
    id: String,
    parent_id: Option<String>,
    timestamp: String,
}

impl From<EntryBase> for PiBase {
    fn from(base: EntryBase) -> Self {
        Self {
            id: base.id,
            parent_id: base.parent_id,
            timestamp: base.timestamp.to_rfc3339(),
        }
    }
}

fn native_base(
    id: String,
    parent_id: Option<String>,
    timestamp: &str,
) -> Result<EntryBase, SessionConversionError> {
    Ok(EntryBase {
        id,
        parent_id,
        timestamp: parse_timestamp(timestamp)?,
    })
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>, SessionConversionError> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|source| SessionConversionError::Timestamp {
            value: value.to_owned(),
            source,
        })
}

fn decode<T: serde::de::DeserializeOwned>(
    value: Value,
    context: &'static str,
) -> Result<T, SessionConversionError> {
    serde_json::from_value(value)
        .map_err(|source| SessionConversionError::Message { context, source })
}

fn encode<T: serde::Serialize>(
    value: T,
    context: &'static str,
) -> Result<Value, SessionConversionError> {
    serde_json::to_value(value)
        .map_err(|source| SessionConversionError::Message { context, source })
}

fn native_usage_to_pi(usage: NativeUsage) -> PiUsage {
    PiUsage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        cache_write1h: None,
        reasoning: None,
        total_tokens: Some(usage.total_tokens),
        cost: Some(PiUsageCost {
            input: usage.cost.input,
            output: usage.cost.output,
            cache_read: usage.cost.cache_read,
            cache_write: usage.cost.cache_write,
            total: usage.cost.total,
        }),
    }
}

fn pi_usage_to_native(usage: &PiUsage) -> NativeUsage {
    let cost = usage
        .cost
        .map_or_else(NativeUsageCost::default, |cost| NativeUsageCost {
            input: cost.input,
            output: cost.output,
            cache_read: cost.cache_read,
            cache_write: cost.cache_write,
            total: cost.total,
        });
    NativeUsage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        total_tokens: usage.total_tokens.unwrap_or_else(|| {
            usage
                .input
                .saturating_add(usage.output)
                .saturating_add(usage.cache_read)
                .saturating_add(usage.cache_write)
        }),
        cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ri_session::{ActiveToolsChangeEntry, LeafEntry};

    fn base(id: &str) -> EntryBase {
        EntryBase {
            id: id.to_owned(),
            parent_id: None,
            timestamp: DateTime::parse_from_rfc3339("2026-07-25T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        }
    }

    #[test]
    fn supported_entries_round_trip_without_json_transcoding_the_entry() {
        let native = NativeEntry::Message(MessageEntry {
            base: base("one"),
            message: serde_json::json!({
                "role": "user",
                "content": "hello",
                "timestamp": 1
            }),
        });
        assert_eq!(
            pi_entry_to_native(native_entry_to_pi(native.clone()).unwrap()).unwrap(),
            native
        );
    }

    #[test]
    fn native_only_entry_kinds_are_rejected_explicitly() {
        let active_tools = NativeEntry::ActiveToolsChange(ActiveToolsChangeEntry {
            base: base("tools"),
            active_tool_names: vec!["read".to_owned()],
        });
        let leaf = NativeEntry::Leaf(LeafEntry {
            base: base("leaf"),
            target_id: None,
        });
        assert!(matches!(
            native_entry_to_pi(active_tools),
            Err(SessionConversionError::UnsupportedNative(
                "active_tools_change entry"
            ))
        ));
        assert!(matches!(
            native_entry_to_pi(leaf),
            Err(SessionConversionError::UnsupportedNative("leaf entry"))
        ));
    }
}
