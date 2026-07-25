//! Backend-neutral validation, traversal, and projection.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::model::{
    CURRENT_SESSION_VERSION, ModelSelection, SequencedEntry, SessionContext, SessionEntry,
    SessionHeader, SessionStats, Usage,
};

/// One defensive tree node returned by [`SessionSnapshot::tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct SessionTreeNode {
    /// Backend append sequence.
    pub sequence: u64,
    /// Immutable entry at this node.
    pub entry: SessionEntry,
    /// Direct children in append order.
    pub children: Vec<Self>,
    /// Latest resolved label for this entry.
    pub label: Option<String>,
    /// Timestamp of the latest effective label update.
    pub label_timestamp: Option<DateTime<Utc>>,
}

/// A validated, immutable view of one committed session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSnapshot {
    header: SessionHeader,
    entries: Vec<SequencedEntry>,
    leaf_id: Option<String>,
}

impl SessionSnapshot {
    /// Create an empty current-version snapshot.
    ///
    /// # Errors
    /// Returns an error when the header version or identifier is invalid.
    pub fn new(header: SessionHeader) -> Result<Self> {
        if header.version != CURRENT_SESSION_VERSION {
            return Err(Error::InvalidSession(format!(
                "unsupported version {}; expected {CURRENT_SESSION_VERSION}",
                header.version
            )));
        }
        if header.id.trim().is_empty() {
            return Err(Error::InvalidSession(
                "session header is missing an id".to_owned(),
            ));
        }
        Ok(Self {
            header,
            entries: Vec::new(),
            leaf_id: None,
        })
    }

    /// Validate and reconstruct a snapshot from ordered backend records.
    ///
    /// # Errors
    /// Returns an error when the header or any ordered entry violates session invariants.
    pub fn from_entries(
        header: SessionHeader,
        entries: impl IntoIterator<Item = SequencedEntry>,
    ) -> Result<Self> {
        let mut snapshot = Self::new(header)?;
        for entry in entries {
            snapshot.push(entry)?;
        }
        Ok(snapshot)
    }

    /// Immutable session header.
    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    /// Entries in backend append order.
    pub fn entries(&self) -> &[SequencedEntry] {
        &self.entries
    }

    /// Durable active leaf reconstructed from all entries.
    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    /// Next sequence after the last committed record.
    pub fn next_sequence(&self) -> u64 {
        self.entries
            .last()
            .map_or(1, |entry| entry.sequence.saturating_add(1))
    }

    /// Validate and apply one backend-assigned record.
    ///
    /// # Errors
    /// Returns an error when sequencing, identifiers, parents, or entry data are invalid.
    pub fn push(&mut self, stored: SequencedEntry) -> Result<()> {
        if stored.sequence == 0 {
            return Err(Error::InvalidEntry(
                "entry sequence must be greater than zero".to_owned(),
            ));
        }
        if let Some(previous) = self.entries.last()
            && stored.sequence <= previous.sequence
        {
            return Err(Error::InvalidEntry(format!(
                "entry sequence {} is not greater than {}",
                stored.sequence, previous.sequence
            )));
        }
        self.validate_entry(&stored.entry)?;
        self.leaf_id = match &stored.entry {
            SessionEntry::Leaf(entry) => entry.target_id.clone(),
            entry => Some(entry.id().to_owned()),
        };
        self.entries.push(stored);
        Ok(())
    }

    /// Find an entry by id.
    pub fn entry(&self, id: &str) -> Option<&SequencedEntry> {
        self.entries.iter().find(|entry| entry.entry.id() == id)
    }

    /// Return entries strictly after a sequence, optionally bounded by a limit.
    pub fn entries_after(
        &self,
        sequence: Option<u64>,
        limit: Option<usize>,
    ) -> Vec<SequencedEntry> {
        let after = sequence.unwrap_or(0);
        self.entries
            .iter()
            .filter(|entry| entry.sequence > after)
            .take(limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    /// Direct children of a parent.  `None` returns tree roots.
    pub fn children(&self, parent_id: Option<&str>) -> Vec<&SequencedEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.entry.parent_id() == parent_id)
            .collect()
    }

    /// Walk from a selected entry to a root in chronological path order.
    ///
    /// # Errors
    /// Returns an error when an entry is missing or the parent graph contains a cycle.
    pub fn path_to(&self, entry_id: Option<&str>) -> Result<Vec<&SequencedEntry>> {
        let Some(entry_id) = entry_id else {
            return Ok(Vec::new());
        };
        let mut current_id = entry_id.to_owned();
        let by_id: HashMap<&str, &SequencedEntry> = self
            .entries
            .iter()
            .map(|entry| (entry.entry.id(), entry))
            .collect();
        let mut reverse_path = Vec::new();
        let mut visited = HashSet::new();
        loop {
            if !visited.insert(current_id.clone()) {
                return Err(Error::InvalidSession(format!(
                    "cycle detected at entry {current_id}"
                )));
            }
            let current = by_id.get(current_id.as_str()).copied().ok_or_else(|| {
                Error::NotFound(format!("session entry {current_id} was not found"))
            })?;
            reverse_path.push(current);
            let Some(parent_id) = current.entry.parent_id() else {
                break;
            };
            current_id = parent_id.to_owned();
        }
        reverse_path.reverse();
        Ok(reverse_path)
    }

    /// Walk from the durable active leaf to a root.
    ///
    /// # Errors
    /// Returns an error when the active path references missing entries or contains a cycle.
    pub fn active_path(&self) -> Result<Vec<&SequencedEntry>> {
        self.path_to(self.leaf_id())
    }

    /// Return the active path transformed by the latest compaction checkpoint.
    ///
    /// # Errors
    /// Returns an error when the active path is invalid.
    pub fn context_entries(&self) -> Result<Vec<&SequencedEntry>> {
        let path = self.active_path()?;
        let Some((compaction_index, compaction)) =
            path.iter()
                .enumerate()
                .rev()
                .find_map(|(index, stored)| match &stored.entry {
                    SessionEntry::Compaction(entry) => Some((index, entry)),
                    _ => None,
                })
        else {
            return Ok(path);
        };

        let mut selected = vec![path[compaction_index]];
        if compaction.retained_tail.is_none()
            && let Some(first_kept_id) = compaction.first_kept_entry_id.as_deref()
            && let Some(first_kept_index) = path[..compaction_index]
                .iter()
                .position(|entry| entry.entry.id() == first_kept_id)
        {
            selected.extend_from_slice(&path[first_kept_index..compaction_index]);
        }
        selected.extend_from_slice(&path[compaction_index + 1..]);
        Ok(selected)
    }

    /// Project the durable active branch into model messages and runtime state.
    ///
    /// # Errors
    /// Returns an error when the active branch is invalid or cannot be projected.
    pub fn context(&self) -> Result<SessionContext> {
        let full_path = self.active_path()?;
        let mut thinking_level = "off".to_owned();
        let mut model = None;
        let mut active_tool_names = None;

        for stored in &full_path {
            match &stored.entry {
                SessionEntry::ThinkingLevelChange(entry) => {
                    thinking_level.clone_from(&entry.thinking_level);
                }
                SessionEntry::ModelChange(entry) => {
                    model = Some(ModelSelection {
                        provider: entry.provider.clone(),
                        model_id: entry.model_id.clone(),
                    });
                }
                SessionEntry::ActiveToolsChange(entry) => {
                    active_tool_names = Some(entry.active_tool_names.clone());
                }
                SessionEntry::Message(entry)
                    if entry.message.get("role").and_then(Value::as_str) == Some("assistant") =>
                {
                    if let (Some(provider), Some(model_id)) = (
                        entry.message.get("provider").and_then(Value::as_str),
                        entry.message.get("model").and_then(Value::as_str),
                    ) {
                        model = Some(ModelSelection {
                            provider: provider.to_owned(),
                            model_id: model_id.to_owned(),
                        });
                    }
                }
                _ => {}
            }
        }

        let mut messages = Vec::new();
        for stored in self.context_entries()? {
            project_entry_messages(&stored.entry, &mut messages);
        }
        Ok(SessionContext {
            messages,
            thinking_level,
            model,
            active_tool_names,
        })
    }

    /// Compute labels after applying all label entries in append order.
    pub fn labels(&self) -> BTreeMap<String, String> {
        self.label_state()
            .into_iter()
            .map(|(id, (label, _))| (id, label))
            .collect()
    }

    /// Resolve the latest label for an entry.
    pub fn label(&self, entry_id: &str) -> Option<String> {
        self.label_state().remove(entry_id).map(|(label, _)| label)
    }

    /// Resolve the latest non-blank session display name.
    pub fn session_name(&self) -> Option<String> {
        for stored in self.entries.iter().rev() {
            if let SessionEntry::SessionInfo(entry) = &stored.entry {
                return entry
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned);
            }
        }
        None
    }

    /// Aggregate message, token, and cost statistics.
    pub fn stats(&self) -> SessionStats {
        let mut stats = SessionStats::default();
        for stored in &self.entries {
            match &stored.entry {
                SessionEntry::Message(entry) => {
                    stats.message_count = stats.message_count.saturating_add(1);
                    if entry.message.get("role").and_then(Value::as_str) == Some("assistant")
                        && let Some(value) = entry.message.get("usage")
                        && let Ok(usage) = serde_json::from_value::<Usage>(value.clone())
                    {
                        apply_usage(&mut stats, usage);
                    }
                }
                SessionEntry::Compaction(entry) => {
                    if let Some(usage) = entry.usage {
                        apply_usage(&mut stats, usage);
                    }
                }
                SessionEntry::BranchSummary(entry) => {
                    if let Some(usage) = entry.usage {
                        apply_usage(&mut stats, usage);
                    }
                }
                _ => {}
            }
        }
        stats
    }

    /// Build the complete parent tree, including durable leaf markers.
    pub fn tree(&self) -> Vec<SessionTreeNode> {
        let labels = self.label_state();
        let by_id: HashMap<&str, usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.entry.id(), index))
            .collect();
        let mut children: HashMap<Option<&str>, Vec<usize>> = HashMap::new();
        for (index, stored) in self.entries.iter().enumerate() {
            let parent = stored
                .entry
                .parent_id()
                .filter(|parent| by_id.contains_key(parent));
            children.entry(parent).or_default().push(index);
        }
        children
            .get(&None)
            .into_iter()
            .flatten()
            .map(|index| self.build_tree_node(*index, &children, &labels))
            .collect()
    }

    fn validate_entry(&self, entry: &SessionEntry) -> Result<()> {
        let base = entry.base();
        if base.id.trim().is_empty() {
            return Err(Error::InvalidEntry("entry is missing an id".to_owned()));
        }
        if self.entry(&base.id).is_some() {
            return Err(Error::Conflict(format!(
                "entry id {} already exists",
                base.id
            )));
        }
        if base.parent_id.as_deref() == Some(base.id.as_str()) {
            return Err(Error::InvalidEntry(format!(
                "entry {} cannot parent itself",
                base.id
            )));
        }
        if let Some(parent_id) = base.parent_id.as_deref()
            && self.entry(parent_id).is_none()
        {
            return Err(Error::InvalidEntry(format!(
                "entry {} references missing parent {parent_id}",
                base.id
            )));
        }

        match entry {
            SessionEntry::Message(message) => {
                if !message.message.is_object() {
                    return Err(Error::InvalidEntry(format!(
                        "message entry {} must contain a JSON object",
                        base.id
                    )));
                }
            }
            SessionEntry::ModelChange(change)
                if change.provider.trim().is_empty() || change.model_id.trim().is_empty() =>
            {
                return Err(Error::InvalidEntry(format!(
                    "model change {} requires provider and model id",
                    base.id
                )));
            }
            SessionEntry::ThinkingLevelChange(change)
                if change.thinking_level.trim().is_empty() =>
            {
                return Err(Error::InvalidEntry(format!(
                    "thinking change {} requires a level",
                    base.id
                )));
            }
            SessionEntry::ActiveToolsChange(change)
                if change
                    .active_tool_names
                    .iter()
                    .any(|name| name.trim().is_empty()) =>
            {
                return Err(Error::InvalidEntry(format!(
                    "active-tools change {} contains a blank name",
                    base.id
                )));
            }
            SessionEntry::Compaction(compaction) => {
                if let Some(first_kept_id) = compaction.first_kept_entry_id.as_deref()
                    && self.entry(first_kept_id).is_none()
                {
                    return Err(Error::InvalidEntry(format!(
                        "compaction {} references missing retained entry {first_kept_id}",
                        base.id
                    )));
                }
            }
            SessionEntry::BranchSummary(summary) => {
                if summary.from_id != "root" && self.entry(&summary.from_id).is_none() {
                    return Err(Error::InvalidEntry(format!(
                        "branch summary {} references missing branch {}",
                        base.id, summary.from_id
                    )));
                }
            }
            SessionEntry::Custom(custom) if custom.custom_type.trim().is_empty() => {
                return Err(Error::InvalidEntry(format!(
                    "custom entry {} requires a custom type",
                    base.id
                )));
            }
            SessionEntry::CustomMessage(custom) => {
                if custom.custom_type.trim().is_empty()
                    || !(custom.content.is_string() || custom.content.is_array())
                {
                    return Err(Error::InvalidEntry(format!(
                        "custom message {} has invalid type or content",
                        base.id
                    )));
                }
            }
            SessionEntry::Label(label) => {
                if self.entry(&label.target_id).is_none() {
                    return Err(Error::InvalidEntry(format!(
                        "label {} references missing entry {}",
                        base.id, label.target_id
                    )));
                }
            }
            SessionEntry::Leaf(leaf) => {
                if let Some(target_id) = leaf.target_id.as_deref()
                    && self.entry(target_id).is_none()
                {
                    return Err(Error::InvalidEntry(format!(
                        "leaf {} references missing entry {target_id}",
                        base.id
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn label_state(&self) -> BTreeMap<String, (String, DateTime<Utc>)> {
        let mut labels = BTreeMap::new();
        for stored in &self.entries {
            let SessionEntry::Label(entry) = &stored.entry else {
                continue;
            };
            let label = entry.label.as_deref().map(str::trim).unwrap_or_default();
            if label.is_empty() {
                labels.remove(&entry.target_id);
            } else {
                labels.insert(
                    entry.target_id.clone(),
                    (label.to_owned(), entry.base.timestamp),
                );
            }
        }
        labels
    }

    fn build_tree_node(
        &self,
        index: usize,
        children: &HashMap<Option<&str>, Vec<usize>>,
        labels: &BTreeMap<String, (String, DateTime<Utc>)>,
    ) -> SessionTreeNode {
        let stored = &self.entries[index];
        let (label, label_timestamp) = labels
            .get(stored.entry.id())
            .map_or((None, None), |(label, timestamp)| {
                (Some(label.clone()), Some(*timestamp))
            });
        let child_nodes = children
            .get(&Some(stored.entry.id()))
            .into_iter()
            .flatten()
            .map(|child| self.build_tree_node(*child, children, labels))
            .collect();
        SessionTreeNode {
            sequence: stored.sequence,
            entry: stored.entry.clone(),
            children: child_nodes,
            label,
            label_timestamp,
        }
    }
}

fn apply_usage(stats: &mut SessionStats, usage: Usage) {
    stats.cached_tokens = stats.cached_tokens.saturating_add(usage.cache_read);
    stats.uncached_tokens = stats
        .uncached_tokens
        .saturating_add(usage.input.saturating_add(usage.cache_write));
    stats.total_tokens = stats.total_tokens.saturating_add(
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write),
    );
    stats.cost_total += usage.cost.total;
}

fn project_entry_messages(entry: &SessionEntry, messages: &mut Vec<Value>) {
    match entry {
        SessionEntry::Message(entry) => {
            let mut message = entry.message.clone();
            if let Some(object) = message.as_object_mut() {
                let role = object.get("role").and_then(Value::as_str);
                if matches!(role, Some("user" | "assistant" | "toolResult"))
                    && object.get("content").is_none_or(Value::is_null)
                {
                    object.insert("content".to_owned(), Value::Array(Vec::new()));
                }
            }
            messages.push(message);
        }
        SessionEntry::Compaction(entry) => {
            messages.push(json!({
                "role": "compactionSummary",
                "summary": entry.summary,
                "tokensBefore": entry.tokens_before,
                "timestamp": format_timestamp(entry.base.timestamp),
            }));
            if let Some(retained_tail) = &entry.retained_tail {
                messages.extend(retained_tail.iter().cloned());
            }
        }
        SessionEntry::BranchSummary(entry) if !entry.summary.is_empty() => {
            messages.push(json!({
                "role": "branchSummary",
                "summary": entry.summary,
                "fromId": entry.from_id,
                "timestamp": format_timestamp(entry.base.timestamp),
            }));
        }
        SessionEntry::CustomMessage(entry) => {
            messages.push(json!({
                "role": "custom",
                "customType": entry.custom_type,
                "content": entry.content,
                "display": entry.display,
                "details": entry.details,
                "timestamp": format_timestamp(entry.base.timestamp),
            }));
        }
        _ => {}
    }
}

fn format_timestamp(timestamp: DateTime<Utc>) -> String {
    timestamp.to_rfc3339_opts(SecondsFormat::Millis, true)
}
