//! Exact and fuzzy text editing with line-ending preservation.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use similar::TextDiff;
use tokio_util::sync::CancellationToken;
use unicode_normalization::UnicodeNormalization;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::error::{EnvError, ToolError};
use crate::mutation::with_file_mutation;
use crate::paths::resolve_path;

/// One targeted replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    /// Text to locate. It must identify one unique, non-overlapping region.
    pub old_text: String,
    /// Replacement text.
    pub new_text: String,
}

/// Input for [`edit`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EditInput {
    /// Relative or absolute target file.
    pub path: PathBuf,
    /// Replacements, all matched against the original file.
    pub edits: Vec<Edit>,
}

/// Structured edit metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EditDetails {
    /// Compact display-oriented diff.
    pub diff: String,
    /// Standard unified patch.
    pub patch: String,
    /// First changed line in the new file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_changed_line: Option<usize>,
}

/// Result of the edit tool.
pub type EditResult = ToolResult<EditDetails>;

/// Preview generated without modifying the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditPreview {
    /// Original LF-normalized content, without a BOM.
    pub base_content: String,
    /// Edited LF-normalized content, without a BOM.
    pub new_content: String,
    /// Unified patch.
    pub patch: String,
    /// First changed line.
    pub first_changed_line: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineEnding {
    Lf,
    CrLf,
}

#[derive(Clone, Debug)]
struct MatchedEdit {
    edit_index: usize,
    start: usize,
    len: usize,
    new_text: String,
}

#[derive(Debug)]
struct ReplacementGroup {
    start_line: usize,
    end_line: usize,
    replacements: Vec<MatchedEdit>,
}

/// Edit one UTF-8 file.
///
/// # Errors
///
/// Returns an error for missing or non-UTF-8 files, cancellation, ambiguous or
/// overlapping matches, unchanged replacements, and environment failures.
pub async fn edit(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: EditInput,
    cancellation: &CancellationToken,
) -> Result<EditResult, ToolError> {
    if input.edits.is_empty() {
        return Err(ToolError::InvalidInput(
            "Edit tool input is invalid. edits must contain at least one replacement.".to_owned(),
        ));
    }
    let absolute = resolve_path(&input.path, cwd);
    let display_path = input.path.to_string_lossy().into_owned();

    with_file_mutation(env, &absolute, || async {
        check_cancelled(cancellation)?;
        env.metadata(&absolute).await.map_err(|error| {
            EnvError::Other(format!(
                "Could not edit file: {display_path}. {}.",
                concise_env_error(&error)
            ))
        })?;
        check_cancelled(cancellation)?;

        let bytes = env.read_file(&absolute).await?;
        let raw = String::from_utf8(bytes)
            .map_err(|_| EnvError::Other(format!("{display_path} is not valid UTF-8")))?;
        check_cancelled(cancellation)?;

        let (bom, content) = strip_bom(&raw);
        let ending = detect_line_ending(content);
        let normalized = normalize_to_lf(content);
        let (base_content, new_content) =
            apply_edits_to_normalized_content(&normalized, &input.edits, &display_path)
                .map_err(|error| EnvError::Other(error.to_string()))?;
        let final_content = format!("{bom}{}", restore_line_endings(&new_content, ending));

        env.write_file(&absolute, final_content.as_bytes()).await?;
        // Do not release the path lock before a cancelled write has settled.
        check_cancelled(cancellation)?;

        let patch = generate_unified_patch(&display_path, &base_content, &new_content, 4);
        let first_changed_line = first_changed_line(&base_content, &new_content);
        let details = EditDetails {
            diff: patch.clone(),
            patch,
            first_changed_line,
        };
        Ok(EditResult::text(
            format!(
                "Successfully replaced {} block(s) in {display_path}.",
                input.edits.len()
            ),
            Some(details),
        ))
    })
    .await
    .map_err(map_edit_env_error)
}

/// Compute an edit without writing the file.
///
/// # Errors
///
/// Returns the same validation and read errors as [`edit`], without performing
/// a mutation.
pub async fn preview_edit(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: &EditInput,
    cancellation: &CancellationToken,
) -> Result<EditPreview, ToolError> {
    if input.edits.is_empty() {
        return Err(ToolError::InvalidInput(
            "Edit tool input is invalid. edits must contain at least one replacement.".to_owned(),
        ));
    }
    if cancellation.is_cancelled() {
        return Err(ToolError::Environment(EnvError::Cancelled));
    }
    let absolute = resolve_path(&input.path, cwd);
    let bytes = env
        .read_file(&absolute)
        .await
        .map_err(ToolError::Environment)?;
    let raw = String::from_utf8(bytes).map_err(|_| ToolError::InvalidUtf8(absolute))?;
    let (_, content) = strip_bom(&raw);
    let normalized = normalize_to_lf(content);
    let display_path = input.path.to_string_lossy();
    let (base_content, new_content) =
        apply_edits_to_normalized_content(&normalized, &input.edits, &display_path)?;
    let patch = generate_unified_patch(&display_path, &base_content, &new_content, 4);
    let changed_line = first_changed_line(&base_content, &new_content);
    Ok(EditPreview {
        base_content,
        new_content,
        patch,
        first_changed_line: changed_line,
    })
}

/// Normalize text for compatibility-oriented edit matching.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let compatibility: String = text.nfkc().collect();
    let without_trailing_space = compatibility
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    without_trailing_space
        .chars()
        .map(|character| match character {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{00a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

/// Normalize CRLF and lone CR line endings to LF.
pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Apply validated edits to LF-normalized content.
///
/// # Errors
///
/// Returns an error when a target is empty, missing, ambiguous, overlapping, or
/// produces no change.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<(String, String), ToolError> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|edit| Edit {
            old_text: normalize_to_lf(&edit.old_text),
            new_text: normalize_to_lf(&edit.new_text),
        })
        .collect();

    for (index, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(ToolError::Edit(edit_error_label(
                path,
                index,
                edits.len(),
                "oldText must not be empty",
            )));
        }
    }

    let fuzzy_content = normalize_for_fuzzy_match(normalized_content);
    let uses_fuzzy = normalized_edits.iter().any(|edit| {
        !normalized_content.contains(&edit.old_text)
            && fuzzy_content.contains(&normalize_for_fuzzy_match(&edit.old_text))
    });
    let replacement_base = if uses_fuzzy {
        fuzzy_content
    } else {
        normalized_content.to_owned()
    };

    let mut matches = Vec::with_capacity(normalized_edits.len());
    for (index, edit) in normalized_edits.iter().enumerate() {
        let target = if uses_fuzzy {
            normalize_for_fuzzy_match(&edit.old_text)
        } else {
            edit.old_text.clone()
        };
        let Some(start) = replacement_base.find(&target) else {
            let message = if edits.len() == 1 {
                format!(
                    "Could not find the exact text in {path}. The old text must match exactly \
                     including all whitespace and newlines."
                )
            } else {
                format!(
                    "Could not find edits[{index}] in {path}. The oldText must match exactly \
                     including all whitespace and newlines."
                )
            };
            return Err(ToolError::Edit(message));
        };
        let occurrences = count_fuzzy_occurrences(&replacement_base, &edit.old_text);
        if occurrences > 1 {
            let message = if edits.len() == 1 {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must be \
                     unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText \
                     must be unique. Please provide more context to make it unique."
                )
            };
            return Err(ToolError::Edit(message));
        }
        matches.push(MatchedEdit {
            edit_index: index,
            start,
            len: target.len(),
            new_text: edit.new_text.clone(),
        });
    }

    matches.sort_by_key(|matched| matched.start);
    for pair in matches.windows(2) {
        if pair[0].start + pair[0].len > pair[1].start {
            return Err(ToolError::Edit(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target \
                 disjoint regions.",
                pair[0].edit_index, pair[1].edit_index
            )));
        }
    }

    let new_content = if uses_fuzzy {
        apply_replacements_preserving_unchanged_lines(
            normalized_content,
            &replacement_base,
            &matches,
        )?
    } else {
        apply_replacements(&replacement_base, &matches, 0)
    };
    if new_content == normalized_content {
        let message = if edits.len() == 1 {
            format!(
                "No changes made to {path}. The replacement produced identical content. This \
                 might indicate an issue with special characters or the text not existing as \
                 expected."
            )
        } else {
            format!("No changes made to {path}. The replacements produced identical content.")
        };
        return Err(ToolError::Edit(message));
    }
    Ok((normalized_content.to_owned(), new_content))
}

/// Generate a standard unified patch.
pub fn generate_unified_patch(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> String {
    TextDiff::from_lines(old_content, new_content)
        .unified_diff()
        .context_radius(context_lines)
        .header(path, path)
        .to_string()
}

fn apply_replacements(content: &str, replacements: &[MatchedEdit], offset: usize) -> String {
    let mut result = content.to_owned();
    for replacement in replacements.iter().rev() {
        let start = replacement.start - offset;
        result.replace_range(start..start + replacement.len, &replacement.new_text);
    }
    result
}

fn apply_replacements_preserving_unchanged_lines(
    original: &str,
    base: &str,
    replacements: &[MatchedEdit],
) -> Result<String, ToolError> {
    let original_lines = split_lines_with_endings(original);
    let base_spans = line_spans(base);
    if original_lines.len() != base_spans.len() {
        return Err(ToolError::Edit(
            "Cannot preserve unchanged lines because normalized content changed line count."
                .to_owned(),
        ));
    }

    let mut groups: Vec<ReplacementGroup> = Vec::new();
    for replacement in replacements {
        let (start_line, end_line) = replacement_line_range(&base_spans, replacement)?;
        if let Some(current) = groups.last_mut()
            && start_line < current.end_line
        {
            current.end_line = current.end_line.max(end_line);
            current.replacements.push(replacement.clone());
            continue;
        }
        groups.push(ReplacementGroup {
            start_line,
            end_line,
            replacements: vec![replacement.clone()],
        });
    }

    let mut result = String::new();
    let mut original_index = 0;
    for group in groups {
        result.push_str(&original_lines[original_index..group.start_line].concat());
        let start = base_spans[group.start_line].0;
        let end = base_spans[group.end_line - 1].1;
        result.push_str(&apply_replacements(
            &base[start..end],
            &group.replacements,
            start,
        ));
        original_index = group.end_line;
    }
    result.push_str(&original_lines[original_index..].concat());
    Ok(result)
}

fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    for (index, character) in content.char_indices() {
        if character == '\n' {
            result.push(&content[start..=index]);
            start = index + 1;
        }
    }
    if start < content.len() {
        result.push(&content[start..]);
    }
    result
}

fn line_spans(content: &str) -> Vec<(usize, usize)> {
    let mut offset = 0;
    split_lines_with_endings(content)
        .into_iter()
        .map(|line| {
            let span = (offset, offset + line.len());
            offset = span.1;
            span
        })
        .collect()
}

fn replacement_line_range(
    spans: &[(usize, usize)],
    replacement: &MatchedEdit,
) -> Result<(usize, usize), ToolError> {
    let Some(start_line) = spans
        .iter()
        .position(|span| replacement.start >= span.0 && replacement.start < span.1)
    else {
        return Err(ToolError::Edit(
            "Replacement range is outside the base content.".to_owned(),
        ));
    };
    let replacement_end = replacement.start + replacement.len;
    let mut end_line = start_line;
    while end_line < spans.len() && spans[end_line].1 < replacement_end {
        end_line += 1;
    }
    if end_line >= spans.len() {
        return Err(ToolError::Edit(
            "Replacement range is outside the base content.".to_owned(),
        ));
    }
    Ok((start_line, end_line + 1))
}

fn count_fuzzy_occurrences(content: &str, target: &str) -> usize {
    let content = normalize_for_fuzzy_match(content);
    let target = normalize_for_fuzzy_match(target);
    if target.is_empty() {
        return 0;
    }
    content.match_indices(&target).count()
}

fn detect_line_ending(content: &str) -> LineEnding {
    let bytes = content.as_bytes();
    for index in 0..bytes.len() {
        if bytes[index] == b'\n' {
            return if index > 0 && bytes[index - 1] == b'\r' {
                LineEnding::CrLf
            } else {
                LineEnding::Lf
            };
        }
    }
    LineEnding::Lf
}

fn restore_line_endings(content: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::Lf => content.to_owned(),
        LineEnding::CrLf => content.replace('\n', "\r\n"),
    }
}

fn strip_bom(content: &str) -> (&str, &str) {
    content
        .strip_prefix('\u{feff}')
        .map_or(("", content), |text| ("\u{feff}", text))
}

fn first_changed_line(old_content: &str, new_content: &str) -> Option<usize> {
    let old_lines: Vec<_> = old_content.split('\n').collect();
    let new_lines: Vec<_> = new_content.split('\n').collect();
    let common = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(old, new)| old == new)
        .count();
    (old_content != new_content).then_some(common + 1)
}

fn edit_error_label(path: &str, index: usize, total: usize, message: &str) -> String {
    if total == 1 {
        format!("{message} in {path}.")
    } else {
        format!("edits[{index}].{message} in {path}.")
    }
}

fn concise_env_error(error: &EnvError) -> String {
    match error {
        EnvError::Io(error) => error.raw_os_error().map_or_else(
            || format!("Error: {error}"),
            |code| format!("OS error code: {code}"),
        ),
        other => other.to_string(),
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), EnvError> {
    if cancellation.is_cancelled() {
        Err(EnvError::Cancelled)
    } else {
        Ok(())
    }
}

fn map_edit_env_error(error: EnvError) -> ToolError {
    match error {
        EnvError::Other(message) if message.contains("Could not edit file:") => {
            ToolError::Edit(message)
        }
        EnvError::Other(message) if message.contains("is not valid UTF-8") => {
            ToolError::Edit(message)
        }
        EnvError::Other(message) => ToolError::Edit(message),
        other => ToolError::Environment(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_edit_preserves_untouched_trailing_space() {
        let input = "keep  \nreplace me   \nafter   \n";
        let edits = vec![Edit {
            old_text: "replace me\n".to_owned(),
            new_text: "changed\n".to_owned(),
        }];
        let (_, result) = apply_edits_to_normalized_content(input, &edits, "test").unwrap();
        assert_eq!(result, "keep  \nchanged\nafter   \n");
    }

    #[test]
    fn rejects_overlapping_edits() {
        let edits = vec![
            Edit {
                old_text: "one\ntwo\n".to_owned(),
                new_text: "x".to_owned(),
            },
            Edit {
                old_text: "two\nthree".to_owned(),
                new_text: "y".to_owned(),
            },
        ];
        let error =
            apply_edits_to_normalized_content("one\ntwo\nthree", &edits, "test").unwrap_err();
        assert!(error.to_string().contains("overlap"));
    }
}
