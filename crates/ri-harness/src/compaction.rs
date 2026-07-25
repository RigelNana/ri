//! Deterministic context compaction and branch-summary preparation.

use std::collections::BTreeSet;

use ri_ai::{AssistantMessage, ContentBlock, Message, StopReason, ToolCall, Usage, UserContent};
use ri_session::{SequencedEntry, SessionEntry, SessionSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::projection::{assistant_text, project_values, tool_result_text, user_text};

/// Default tokens reserved for the summary prompt and next response.
pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;
/// Default approximate recent-context budget retained after compaction.
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 20_000;
const ESTIMATED_IMAGE_CHARS: u64 = 4_800;
const TOOL_RESULT_MAX_CHARS: usize = 2_000;

/// Compaction thresholds and retention settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionSettings {
    /// Enable threshold and overflow auto-compaction.
    pub enabled: bool,
    /// Tokens reserved for summary prompting and response generation.
    pub reserve_tokens: u64,
    /// Approximate recent tokens to retain.
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: DEFAULT_RESERVE_TOKENS,
            keep_recent_tokens: DEFAULT_KEEP_RECENT_TOKENS,
        }
    }
}

/// Whether a context estimate crosses the configured threshold.
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: CompactionSettings,
) -> bool {
    settings.enabled && context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

/// Token estimate using the last provider usage plus locally estimated trailing
/// messages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextEstimate {
    /// Estimated total context.
    pub tokens: u64,
    /// Provider-reported context at the last valid assistant message.
    pub usage_tokens: u64,
    /// Estimated messages after that usage report.
    pub trailing_tokens: u64,
    /// Index of the usage-bearing assistant message.
    pub last_usage_index: Option<usize>,
}

/// Estimates one provider-neutral message using a conservative character
/// heuristic.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = match message {
        Message::User(message) => match &message.content {
            UserContent::Text(text) => text.chars().count() as u64,
            UserContent::Blocks(blocks) => blocks
                .iter()
                .map(|block| {
                    let value = serde_json::to_value(block).unwrap_or(Value::Null);
                    if value.get("type").and_then(Value::as_str) == Some("image") {
                        ESTIMATED_IMAGE_CHARS
                    } else {
                        value
                            .get("text")
                            .and_then(Value::as_str)
                            .map_or(0, |text| text.chars().count() as u64)
                    }
                })
                .sum(),
        },
        Message::Assistant(message) => message
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text(text) => text.text.chars().count() as u64,
                ContentBlock::Thinking(thinking) => thinking.thinking.chars().count() as u64,
                ContentBlock::ToolCall(call) => {
                    call.name.chars().count() as u64
                        + serde_json::to_string(&call.arguments)
                            .map_or(0, |value| value.chars().count() as u64)
                }
            })
            .sum(),
        Message::ToolResult(message) => message
            .content
            .iter()
            .map(|block| {
                let value = serde_json::to_value(block).unwrap_or(Value::Null);
                if value.get("type").and_then(Value::as_str) == Some("image") {
                    ESTIMATED_IMAGE_CHARS
                } else {
                    value
                        .get("text")
                        .and_then(Value::as_str)
                        .map_or(0, |text| text.chars().count() as u64)
                }
            })
            .sum(),
    };
    chars.saturating_add(3) / 4
}

/// Estimates a complete message list.
pub fn estimate_context_tokens(messages: &[Message]) -> ContextEstimate {
    let usage = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            let Message::Assistant(message) = message else {
                return None;
            };
            valid_usage(message).map(|tokens| (index, tokens))
        });
    let Some((index, usage_tokens)) = usage else {
        let tokens = messages.iter().map(estimate_message_tokens).sum();
        return ContextEstimate {
            tokens,
            trailing_tokens: tokens,
            ..ContextEstimate::default()
        };
    };
    let trailing_tokens = messages[index + 1..]
        .iter()
        .map(estimate_message_tokens)
        .sum();
    ContextEstimate {
        tokens: usage_tokens.saturating_add(trailing_tokens),
        usage_tokens,
        trailing_tokens,
        last_usage_index: Some(index),
    }
}

fn valid_usage(message: &AssistantMessage) -> Option<u64> {
    if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
        return None;
    }
    let tokens = context_tokens(&message.usage);
    (tokens > 0).then_some(tokens)
}

/// Normalizes provider usage into a context-token count.
pub fn context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    }
}

/// Files observed in summarized history.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FileOperations {
    /// Read paths.
    pub read: BTreeSet<String>,
    /// Created or replaced paths.
    pub written: BTreeSet<String>,
    /// Edited paths.
    pub edited: BTreeSet<String>,
}

impl FileOperations {
    /// Produces sorted read-only and modified path lists.
    pub fn lists(&self) -> FileLists {
        let modified: BTreeSet<_> = self.written.iter().chain(&self.edited).cloned().collect();
        let read_files = self
            .read
            .iter()
            .filter(|path| !modified.contains(*path))
            .cloned()
            .collect();
        FileLists {
            read_files,
            modified_files: modified.into_iter().collect(),
        }
    }
}

/// Persisted cumulative file tracking.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileLists {
    /// Paths only read.
    pub read_files: Vec<String>,
    /// Paths written or edited.
    pub modified_files: Vec<String>,
}

/// Selected compaction boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutPoint {
    /// Index of the first retained entry.
    pub first_kept_entry_index: usize,
    /// Start of a turn split by the cut.
    pub turn_start_index: Option<usize>,
    /// Whether the cut occurs within one turn.
    pub is_split_turn: bool,
}

/// Finds a cut point while keeping approximately `keep_recent_tokens`.
pub fn find_cut_point(
    entries: &[SequencedEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPoint {
    if start_index >= end_index || end_index > entries.len() {
        return CutPoint {
            first_kept_entry_index: start_index.min(entries.len()),
            ..CutPoint::default()
        };
    }
    let cut_points = (start_index..end_index)
        .filter(|index| valid_cut_point(&entries[*index].entry))
        .collect::<Vec<_>>();
    if cut_points.is_empty() {
        return CutPoint {
            first_kept_entry_index: start_index,
            ..CutPoint::default()
        };
    }

    let mut accumulated = 0_u64;
    let mut cut_index = cut_points[0];
    for index in (start_index..end_index).rev() {
        if let Some(message) = entry_messages(&entries[index].entry)
            .ok()
            .and_then(|messages| messages.into_iter().next())
        {
            accumulated = accumulated.saturating_add(estimate_message_tokens(&message));
        }
        if accumulated >= keep_recent_tokens {
            cut_index = cut_points
                .iter()
                .copied()
                .find(|candidate| *candidate >= index)
                .or_else(|| {
                    cut_points
                        .iter()
                        .rev()
                        .copied()
                        .find(|candidate| *candidate <= index)
                })
                .unwrap_or(cut_index);
            break;
        }
    }

    while cut_index > start_index {
        match &entries[cut_index - 1].entry {
            SessionEntry::Compaction(_)
            | SessionEntry::Message(_)
            | SessionEntry::BranchSummary(_)
            | SessionEntry::CustomMessage(_) => break,
            _ => cut_index -= 1,
        }
    }
    let starts_user_turn = is_user_entry(&entries[cut_index].entry);
    let turn_start_index = if starts_user_turn {
        None
    } else {
        (start_index..=cut_index)
            .rev()
            .find(|index| starts_turn(&entries[*index].entry))
    };
    CutPoint {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !starts_user_turn && turn_start_index.is_some(),
    }
}

/// Deterministic input to summary generation.
#[derive(Clone, Debug)]
pub struct CompactionPreparation {
    /// Complete active branch used during preparation.
    pub branch_entries: Vec<SequencedEntry>,
    /// First retained entry id.
    pub first_kept_entry_id: String,
    /// Complete turns summarized into the historical summary.
    pub messages_to_summarize: Vec<Message>,
    /// Prefix summarized separately when a turn is split.
    pub turn_prefix_messages: Vec<Message>,
    /// Exact serialized recent tail persisted on the compaction entry.
    pub retained_tail: Vec<Value>,
    /// Whether one turn was split.
    pub is_split_turn: bool,
    /// Estimated context before compaction.
    pub tokens_before: u64,
    /// Previous summary used for iterative compaction.
    pub previous_summary: Option<String>,
    /// Cumulative file operations.
    pub file_operations: FileOperations,
    /// Settings used for preparation.
    pub settings: CompactionSettings,
}

/// Persisted compaction result.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionResult {
    /// Summary replacing historical context.
    pub summary: String,
    /// First retained entry.
    pub first_kept_entry_id: String,
    /// Context estimate before compaction.
    pub tokens_before: u64,
    /// Estimated rebuilt-context tokens after persistence.
    pub estimated_tokens_after: u64,
    /// Usage spent creating the summary.
    pub usage: Option<Usage>,
    /// Self-contained retained tail.
    pub retained_tail: Vec<Value>,
    /// Extension-owned compaction metadata. Built-in compaction stores
    /// [`FileLists`] serialized as JSON without narrowing hook-provided data.
    pub details: Option<Value>,
    /// Whether a hook supplied the summary.
    pub from_hook: bool,
}

/// Prepares the active branch for compaction.
///
/// # Errors
/// Returns an error when the session branch is invalid or its messages cannot be projected.
pub fn prepare_compaction(
    snapshot: &SessionSnapshot,
    settings: CompactionSettings,
) -> Result<Option<CompactionPreparation>> {
    let path = snapshot
        .active_path()?
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    if path.is_empty()
        || matches!(
            path.last().map(|entry| &entry.entry),
            Some(SessionEntry::Compaction(_))
        )
    {
        return Ok(None);
    }

    let previous = path
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, stored)| match &stored.entry {
            SessionEntry::Compaction(compaction) => Some((index, compaction)),
            _ => None,
        });
    let mut previous_summary = None;
    let mut boundary_start = 0;
    let mut file_operations = FileOperations::default();
    if let Some((index, compaction)) = previous {
        previous_summary = Some(compaction.summary.clone());
        boundary_start = compaction
            .first_kept_entry_id
            .as_deref()
            .and_then(|id| path.iter().position(|entry| entry.entry.id() == id))
            .unwrap_or(index + 1);
        if compaction.from_hook != Some(true)
            && let Some(details) = compaction.details.clone()
            && let Ok(files) = serde_json::from_value::<FileLists>(details)
        {
            file_operations.read.extend(files.read_files);
            file_operations.edited.extend(files.modified_files);
        }
    }

    let projected = snapshot.context()?;
    let messages = project_values(&projected.messages)?;
    let tokens_before = estimate_context_tokens(&messages).tokens;
    let cut = find_cut_point(
        &path,
        boundary_start,
        path.len(),
        settings.keep_recent_tokens,
    );
    let first_kept = path.get(cut.first_kept_entry_index).ok_or_else(|| {
        Error::Compaction("compaction could not select a retained entry".to_owned())
    })?;
    let first_kept_entry_id = first_kept.entry.id().to_owned();
    let history_end = if cut.is_split_turn {
        cut.turn_start_index.unwrap_or(cut.first_kept_entry_index)
    } else {
        cut.first_kept_entry_index
    };

    let messages_to_summarize = messages_from_entries(&path[boundary_start..history_end], false)?;
    let turn_prefix_messages = if cut.is_split_turn {
        messages_from_entries(
            &path[cut.turn_start_index.unwrap_or(history_end)..cut.first_kept_entry_index],
            false,
        )?
    } else {
        Vec::new()
    };
    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return Ok(None);
    }
    let retained_tail = values_from_entries(&path[cut.first_kept_entry_index..], false);

    for message in messages_to_summarize.iter().chain(&turn_prefix_messages) {
        extract_file_operations(message, &mut file_operations);
    }
    for stored in &path[boundary_start..history_end] {
        inherit_summary_files(&stored.entry, &mut file_operations);
    }

    Ok(Some(CompactionPreparation {
        branch_entries: path,
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        retained_tail,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary,
        file_operations,
        settings,
    }))
}

/// Entries and file operations prepared for an abandoned branch summary.
#[derive(Clone, Debug, Default)]
pub struct BranchPreparation {
    /// Messages retained within the summary token budget.
    pub messages: Vec<Message>,
    /// Cumulative file operations from all abandoned entries.
    pub file_operations: FileOperations,
    /// Estimated tokens included.
    pub total_tokens: u64,
}

/// Returns abandoned entries and the deepest common ancestor.
///
/// # Errors
/// Returns an error when either branch path is invalid or references a missing entry.
pub fn collect_abandoned_branch(
    snapshot: &SessionSnapshot,
    old_leaf: Option<&str>,
    target: Option<&str>,
) -> Result<(Vec<SequencedEntry>, Option<String>)> {
    let Some(old_leaf) = old_leaf else {
        return Ok((Vec::new(), None));
    };
    let old_path = snapshot.path_to(Some(old_leaf))?;
    let target_path = snapshot.path_to(target)?;
    let old_ids: BTreeSet<_> = old_path.iter().map(|entry| entry.entry.id()).collect();
    let common = target_path
        .iter()
        .rev()
        .find(|entry| old_ids.contains(entry.entry.id()))
        .map(|entry| entry.entry.id().to_owned());

    let mut entries = Vec::new();
    let mut current = Some(old_leaf);
    while let Some(id) = current {
        if Some(id) == common.as_deref() {
            break;
        }
        let entry = snapshot
            .entry(id)
            .ok_or_else(|| Error::Session(format!("missing branch entry {id}")))?;
        entries.push(entry.clone());
        current = entry.entry.parent_id();
    }
    entries.reverse();
    Ok((entries, common))
}

/// Prepares abandoned branch entries newest-first within a token budget while
/// retaining cumulative file tracking from the complete branch.
///
/// # Errors
/// Returns an error when an entry cannot be converted into provider-neutral messages.
pub fn prepare_branch(entries: &[SequencedEntry], token_budget: u64) -> Result<BranchPreparation> {
    let mut file_operations = FileOperations::default();
    for stored in entries {
        inherit_summary_files(&stored.entry, &mut file_operations);
        for message in entry_messages(&stored.entry)? {
            extract_file_operations(&message, &mut file_operations);
        }
    }

    let mut messages = Vec::new();
    let mut total_tokens = 0_u64;
    for stored in entries.iter().rev() {
        let entry_messages = entry_messages(&stored.entry)?;
        for message in entry_messages
            .into_iter()
            .filter(|message| !matches!(message, Message::ToolResult(_)))
            .rev()
        {
            let tokens = estimate_message_tokens(&message);
            let summary_entry = matches!(
                stored.entry,
                SessionEntry::Compaction(_) | SessionEntry::BranchSummary(_)
            );
            if token_budget > 0 && total_tokens.saturating_add(tokens) > token_budget {
                if summary_entry && total_tokens < token_budget.saturating_mul(9) / 10 {
                    messages.insert(0, message);
                    total_tokens = total_tokens.saturating_add(tokens);
                }
                return Ok(BranchPreparation {
                    messages,
                    file_operations,
                    total_tokens,
                });
            }
            messages.insert(0, message);
            total_tokens = total_tokens.saturating_add(tokens);
        }
    }
    Ok(BranchPreparation {
        messages,
        file_operations,
        total_tokens,
    })
}

/// System instruction shared by compaction and branch summarization.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. \
Read the serialized conversation and produce only the requested structured summary. \
Do not continue the conversation or answer questions from it.";

/// Structured historical summary instruction.
pub const COMPACTION_PROMPT: &str = r"Create a structured context checkpoint summary that another model can use to continue the work.

Use this exact structure:
## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context

Keep each section concise. Preserve exact paths, symbol names, commands, and error messages.";

/// Iterative compaction instruction.
pub const UPDATE_COMPACTION_PROMPT: &str = r"Update the existing structured summary with the new conversation.
Preserve still-relevant goals, constraints, completed work, decisions, paths, symbol names, commands, and errors.
Move completed items out of In Progress and update Next Steps.
Use the same exact section structure as the previous summary.";

/// Split-turn prefix instruction.
pub const TURN_PREFIX_PROMPT: &str = r"This is the prefix of a turn whose recent suffix remains in context.
Summarize only the prefix using:
## Original Request
## Early Progress
## Context for Suffix
Preserve details needed to understand the retained suffix.";

/// Branch-summary instruction.
pub const BRANCH_SUMMARY_PROMPT: &str = r"Create a structured summary of this abandoned conversation branch for context after returning elsewhere.
Use:
## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
Keep it concise and preserve exact paths, symbols, commands, and errors.";

/// Serializes messages as data rather than a conversation to continue.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message {
            Message::User(message) => {
                let text = user_text(message);
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
                }
            }
            Message::Assistant(message) => {
                let thinking = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Thinking(block) => Some(block.thinking.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !thinking.is_empty() {
                    parts.push(format!("[Assistant thinking]: {thinking}"));
                }
                let text = assistant_text(message);
                if !text.is_empty() {
                    parts.push(format!("[Assistant]: {text}"));
                }
                let calls = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(call) => Some(format_tool_call(call)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
                }
            }
            Message::ToolResult(message) => {
                let text = tool_result_text(message);
                if !text.is_empty() {
                    let chars = text.chars().count();
                    let value = if chars <= TOOL_RESULT_MAX_CHARS {
                        text
                    } else {
                        let prefix = text.chars().take(TOOL_RESULT_MAX_CHARS).collect::<String>();
                        format!(
                            "{prefix}\n\n[... {} more characters truncated]",
                            chars - TOOL_RESULT_MAX_CHARS
                        )
                    };
                    parts.push(format!("[Tool result]: {value}"));
                }
            }
        }
    }
    parts.join("\n\n")
}

/// Builds a compaction request prompt.
pub fn compaction_request_text(
    messages: &[Message],
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
) -> String {
    let instruction = if previous_summary.is_some() {
        UPDATE_COMPACTION_PROMPT
    } else {
        COMPACTION_PROMPT
    };
    let mut prompt = format!(
        "<conversation>\n{}\n</conversation>\n\n",
        serialize_conversation(messages)
    );
    if let Some(previous) = previous_summary {
        prompt.push_str("<previous-summary>\n");
        prompt.push_str(previous);
        prompt.push_str("\n</previous-summary>\n\n");
    }
    prompt.push_str(instruction);
    if let Some(custom) = custom_instructions {
        prompt.push_str("\n\nAdditional focus: ");
        prompt.push_str(custom);
    }
    prompt
}

/// Builds a branch summary request prompt.
pub fn branch_request_text(
    messages: &[Message],
    custom_instructions: Option<&str>,
    replace_instructions: bool,
) -> String {
    let instruction = if replace_instructions {
        custom_instructions
            .unwrap_or(BRANCH_SUMMARY_PROMPT)
            .to_owned()
    } else {
        let mut value = BRANCH_SUMMARY_PROMPT.to_owned();
        if let Some(custom) = custom_instructions {
            value.push_str("\n\nAdditional focus: ");
            value.push_str(custom);
        }
        value
    };
    format!(
        "<conversation>\n{}\n</conversation>\n\n{instruction}",
        serialize_conversation(messages)
    )
}

/// Appends deterministic XML file lists to a generated summary.
pub fn append_file_lists(summary: &mut String, files: &FileLists) {
    if files.read_files.is_empty() && files.modified_files.is_empty() {
        return;
    }
    if !files.read_files.is_empty() {
        summary.push_str("\n\n<read-files>\n");
        summary.push_str(&files.read_files.join("\n"));
        summary.push_str("\n</read-files>");
    }
    if !files.modified_files.is_empty() {
        summary.push_str("\n\n<modified-files>\n");
        summary.push_str(&files.modified_files.join("\n"));
        summary.push_str("\n</modified-files>");
    }
}

/// Combines usage from split-turn summary requests.
pub fn combine_usage(first: &Usage, second: &Usage) -> Usage {
    let mut usage = Usage::from_parts(
        first.input.saturating_add(second.input),
        first.output.saturating_add(second.output),
        first.cache_read.saturating_add(second.cache_read),
        first.cache_write.saturating_add(second.cache_write),
    );
    usage.cache_write_1h = match (first.cache_write_1h, second.cache_write_1h) {
        (None, None) => None,
        (left, right) => Some(left.unwrap_or(0).saturating_add(right.unwrap_or(0))),
    };
    usage.reasoning = match (first.reasoning, second.reasoning) {
        (None, None) => None,
        (left, right) => Some(left.unwrap_or(0).saturating_add(right.unwrap_or(0))),
    };
    usage.cost.input = first.cost.input + second.cost.input;
    usage.cost.output = first.cost.output + second.cost.output;
    usage.cost.cache_read = first.cost.cache_read + second.cost.cache_read;
    usage.cost.cache_write = first.cost.cache_write + second.cost.cache_write;
    usage.cost.total = first.cost.total + second.cost.total;
    usage
}

/// Converts AI usage to the session accounting shape.
pub fn session_usage(usage: &Usage) -> ri_session::Usage {
    ri_session::Usage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        total_tokens: usage.total_tokens,
        cost: ri_session::UsageCost {
            input: usage.cost.input,
            output: usage.cost.output,
            cache_read: usage.cost.cache_read,
            cache_write: usage.cost.cache_write,
            total: usage.cost.total,
        },
    }
}

fn valid_cut_point(entry: &SessionEntry) -> bool {
    match entry {
        SessionEntry::Message(message) => matches!(
            message.message.get("role").and_then(Value::as_str),
            Some("user" | "assistant")
        ),
        SessionEntry::BranchSummary(_) | SessionEntry::CustomMessage(_) => true,
        _ => false,
    }
}

fn starts_turn(entry: &SessionEntry) -> bool {
    is_user_entry(entry)
        || matches!(
            entry,
            SessionEntry::BranchSummary(_) | SessionEntry::CustomMessage(_)
        )
}

fn is_user_entry(entry: &SessionEntry) -> bool {
    matches!(
        entry,
        SessionEntry::Message(message)
            if message.message.get("role").and_then(Value::as_str) == Some("user")
    )
}

fn messages_from_entries(
    entries: &[SequencedEntry],
    include_compaction: bool,
) -> Result<Vec<Message>> {
    let values = values_from_entries(entries, include_compaction);
    project_values(&values)
}

fn values_from_entries(entries: &[SequencedEntry], include_compaction: bool) -> Vec<Value> {
    entries
        .iter()
        .flat_map(|stored| entry_values(&stored.entry, include_compaction))
        .collect()
}

fn entry_messages(entry: &SessionEntry) -> Result<Vec<Message>> {
    project_values(&entry_values(entry, true))
}

fn entry_values(entry: &SessionEntry, include_compaction: bool) -> Vec<Value> {
    match entry {
        SessionEntry::Message(entry) => vec![entry.message.clone()],
        SessionEntry::Compaction(entry) if include_compaction => {
            vec![json!({
                "role": "compactionSummary",
                "summary": entry.summary,
                "tokensBefore": entry.tokens_before,
                "timestamp": entry.base.timestamp.to_rfc3339(),
            })]
        }
        SessionEntry::BranchSummary(entry) if !entry.summary.is_empty() => {
            vec![json!({
                "role": "branchSummary",
                "summary": entry.summary,
                "fromId": entry.from_id,
                "timestamp": entry.base.timestamp.to_rfc3339(),
            })]
        }
        SessionEntry::CustomMessage(entry) => vec![json!({
            "role": "custom",
            "customType": entry.custom_type,
            "content": entry.content,
            "display": entry.display,
            "details": entry.details,
            "timestamp": entry.base.timestamp.to_rfc3339(),
        })],
        _ => Vec::new(),
    }
}

fn extract_file_operations(message: &Message, operations: &mut FileOperations) {
    let Message::Assistant(message) = message else {
        return;
    };
    for block in &message.content {
        let ContentBlock::ToolCall(call) = block else {
            continue;
        };
        let Some(path) = call.arguments.get("path").and_then(Value::as_str) else {
            continue;
        };
        match call.name.as_str() {
            "read" => {
                operations.read.insert(path.to_owned());
            }
            "write" => {
                operations.written.insert(path.to_owned());
            }
            "edit" => {
                operations.edited.insert(path.to_owned());
            }
            _ => {}
        }
    }
}

fn inherit_summary_files(entry: &SessionEntry, operations: &mut FileOperations) {
    let details = match entry {
        SessionEntry::Compaction(entry) if entry.from_hook != Some(true) => entry.details.as_ref(),
        SessionEntry::BranchSummary(entry) if entry.from_hook != Some(true) => {
            entry.details.as_ref()
        }
        _ => None,
    };
    let Some(details) = details else {
        return;
    };
    if let Ok(files) = serde_json::from_value::<FileLists>(details.clone()) {
        operations.read.extend(files.read_files);
        operations.edited.extend(files.modified_files);
    }
}

fn format_tool_call(call: &ToolCall) -> String {
    let arguments = call.arguments.as_object().map_or_else(
        || call.arguments.to_string(),
        |arguments| {
            arguments
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    );
    format!("{}({arguments})", call.name)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use ri_ai::{AssistantMessage, ContentBlock, StopReason, TextContent, ToolCall, UserMessage};
    use ri_session::{
        CompactionEntry, CustomMessageEntry, EntryBase, MessageEntry, SequencedEntry, SessionEntry,
        SessionHeader, SessionSnapshot,
    };

    use super::*;

    fn stored(sequence: u64, parent: Option<&str>, id: &str, message: Message) -> SequencedEntry {
        SequencedEntry {
            sequence,
            entry: SessionEntry::Message(MessageEntry {
                base: EntryBase {
                    id: id.to_owned(),
                    parent_id: parent.map(str::to_owned),
                    timestamp: Utc::now(),
                },
                message: serde_json::to_value(message).expect("message"),
            }),
        }
    }

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: UserContent::Text(text.to_owned()),
            timestamp: 1,
        })
    }

    fn assistant(text: &str, tokens: u64) -> Message {
        let mut message = AssistantMessage::empty("test", "test", "model");
        message.content = vec![ContentBlock::Text(TextContent::new(text))];
        message.usage = Usage::from_parts(tokens, 1, 0, 0);
        Message::Assistant(message)
    }

    #[test]
    fn threshold_uses_saturating_window() {
        let settings = CompactionSettings {
            reserve_tokens: 200,
            ..CompactionSettings::default()
        };
        assert!(should_compact(1, 100, settings));
        assert!(!should_compact(
            1,
            100,
            CompactionSettings {
                enabled: false,
                ..settings
            }
        ));
    }

    #[test]
    fn estimate_uses_latest_valid_usage_and_trailing_messages() {
        let messages = vec![user("a"), assistant("b", 100), user("12345678")];
        let estimate = estimate_context_tokens(&messages);
        assert_eq!(estimate.usage_tokens, 101);
        assert_eq!(estimate.trailing_tokens, 2);
        assert_eq!(estimate.tokens, 103);
        assert_eq!(estimate.last_usage_index, Some(1));
    }

    #[test]
    fn cut_point_never_starts_at_tool_result() {
        let user = stored(1, None, "u", user("request"));
        let mut tool_assistant = AssistantMessage::empty("test", "test", "model");
        tool_assistant.stop_reason = StopReason::ToolUse;
        tool_assistant
            .content
            .push(ContentBlock::ToolCall(ToolCall {
                id: "call".into(),
                name: "read".into(),
                arguments: json!({"path": "a.rs"}),
                thought_signature: None,
            }));
        let assistant_entry = stored(2, Some("u"), "a", Message::Assistant(tool_assistant));
        let tool = ri_ai::ToolResultMessage {
            tool_call_id: "call".into(),
            tool_name: "read".into(),
            content: Vec::new(),
            details: None,
            usage: None,
            added_tool_names: Vec::new(),
            is_error: false,
            timestamp: 1,
        };
        let tool = stored(3, Some("a"), "t", Message::ToolResult(tool));
        let final_assistant = stored(4, Some("t"), "f", assistant("done", 10));
        let entries = vec![user, assistant_entry, tool, final_assistant];
        let cut = find_cut_point(&entries, 0, entries.len(), 1);
        assert_ne!(
            entries[cut.first_kept_entry_index].entry.as_message_role(),
            Some("toolResult")
        );
    }

    #[test]
    fn trailing_tool_result_keeps_its_calling_assistant() {
        let user = stored(1, None, "u", user("request"));
        let mut calling = AssistantMessage::empty("test", "test", "model");
        calling.stop_reason = StopReason::ToolUse;
        calling.content.push(ContentBlock::ToolCall(ToolCall {
            id: "call".into(),
            name: "read".into(),
            arguments: json!({"path": "a.rs"}),
            thought_signature: None,
        }));
        let assistant = stored(2, Some("u"), "a", Message::Assistant(calling));
        let tool = stored(
            3,
            Some("a"),
            "t",
            Message::ToolResult(ri_ai::ToolResultMessage {
                tool_call_id: "call".into(),
                tool_name: "read".into(),
                content: vec![ri_ai::message::InputContent::Text(TextContent::new(
                    "x".repeat(1_000),
                ))],
                details: None,
                usage: None,
                added_tool_names: Vec::new(),
                is_error: false,
                timestamp: 1,
            }),
        );
        let entries = vec![user, assistant, tool];
        let cut = find_cut_point(&entries, 0, entries.len(), 1);
        assert_eq!(cut.first_kept_entry_index, 1);
        assert!(matches!(
            entries[cut.first_kept_entry_index].entry,
            SessionEntry::Message(_)
        ));
    }

    #[test]
    fn serialization_truncates_large_tool_results() {
        let message = Message::ToolResult(ri_ai::ToolResultMessage {
            tool_call_id: "call".into(),
            tool_name: "read".into(),
            content: serde_json::from_value(json!([
                {"type": "text", "text": "x".repeat(5_000)}
            ]))
            .expect("content"),
            details: None,
            usage: None,
            added_tool_names: Vec::new(),
            is_error: false,
            timestamp: 1,
        });
        let serialized = serialize_conversation(&[message]);
        assert!(serialized.contains("3000 more characters truncated"));
    }

    #[test]
    fn inherited_file_lists_remain_cumulative() {
        let entry = SessionEntry::Compaction(CompactionEntry {
            base: EntryBase::new("c", None),
            summary: "summary".into(),
            first_kept_entry_id: None,
            tokens_before: 10,
            retained_tail: None,
            details: Some(json!({
                "readFiles": ["old.rs"],
                "modifiedFiles": ["changed.rs"]
            })),
            usage: None,
            from_hook: Some(false),
        });
        let mut operations = FileOperations::default();
        inherit_summary_files(&entry, &mut operations);
        assert!(operations.read.contains("old.rs"));
        assert!(operations.edited.contains("changed.rs"));
    }

    #[test]
    fn preparation_skips_a_noop_compaction() {
        let entries = vec![
            stored(1, None, "u", user("short request")),
            stored(2, Some("u"), "a", assistant("short answer", 10)),
        ];
        let snapshot = SessionSnapshot::from_entries(SessionHeader::new("session", "."), entries)
            .expect("snapshot");
        let prepared = prepare_compaction(
            &snapshot,
            CompactionSettings {
                keep_recent_tokens: 10_000,
                ..CompactionSettings::default()
            },
        )
        .expect("prepare");
        assert!(prepared.is_none());
    }

    #[test]
    fn cut_point_does_not_cross_a_context_visible_custom_message() {
        let entries = vec![
            stored(1, None, "u", user("request")),
            SequencedEntry {
                sequence: 2,
                entry: SessionEntry::CustomMessage(CustomMessageEntry {
                    base: EntryBase::new("c", Some("u".into())),
                    custom_type: "extension".into(),
                    content: Value::String("context".into()),
                    display: false,
                    details: None,
                }),
            },
            stored(3, Some("c"), "a", assistant("done", 10)),
        ];
        let cut = find_cut_point(&entries, 0, entries.len(), 1);
        assert_eq!(cut.first_kept_entry_index, 2);
        assert!(cut.is_split_turn);
        assert_eq!(cut.turn_start_index, Some(1));
    }

    #[test]
    fn branch_summaries_omit_tool_results_but_track_assistant_calls() {
        let mut calling = AssistantMessage::empty("test", "test", "model");
        calling.stop_reason = StopReason::ToolUse;
        calling.content.push(ContentBlock::ToolCall(ToolCall {
            id: "call".into(),
            name: "read".into(),
            arguments: json!({"path": "src/lib.rs"}),
            thought_signature: None,
        }));
        let assistant = stored(1, None, "a", Message::Assistant(calling));
        let tool_result = stored(
            2,
            Some("a"),
            "t",
            Message::ToolResult(ri_ai::ToolResultMessage {
                tool_call_id: "call".into(),
                tool_name: "read".into(),
                content: vec![ri_ai::message::InputContent::Text(TextContent::new(
                    "large result",
                ))],
                details: None,
                usage: None,
                added_tool_names: Vec::new(),
                is_error: false,
                timestamp: 1,
            }),
        );
        let preparation = prepare_branch(&[assistant, tool_result], 0).expect("branch");
        assert_eq!(preparation.messages.len(), 1);
        assert!(matches!(
            preparation.messages.first(),
            Some(Message::Assistant(_))
        ));
        assert!(preparation.file_operations.read.contains("src/lib.rs"));
    }

    trait MessageRole {
        fn as_message_role(&self) -> Option<&str>;
    }

    impl MessageRole for SessionEntry {
        fn as_message_role(&self) -> Option<&str> {
            match self {
                SessionEntry::Message(entry) => entry.message.get("role").and_then(Value::as_str),
                _ => None,
            }
        }
    }
}
