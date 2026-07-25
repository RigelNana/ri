//! Cancellation-free, synchronous autocomplete contracts.

/// Current editor state supplied to an autocomplete provider.
#[derive(Clone, Copy, Debug)]
pub struct AutocompleteContext<'a> {
    /// Logical editor lines.
    pub lines: &'a [String],
    /// Cursor line.
    pub cursor_line: usize,
    /// Cursor byte offset.
    pub cursor_col: usize,
    /// Whether completion was explicitly requested with Tab.
    pub forced: bool,
}

/// A selectable completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutocompleteItem {
    /// Text inserted by the default completion implementation.
    pub value: String,
    /// Display label.
    pub label: String,
    /// Optional secondary text.
    pub description: Option<String>,
}

impl AutocompleteItem {
    /// Creates an item whose value and label are equal.
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            label: value.clone(),
            value,
            description: None,
        }
    }

    /// Adds a description.
    #[must_use]
    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Provider response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutocompleteResult {
    /// Matching items.
    pub items: Vec<AutocompleteItem>,
    /// Text directly before the cursor to replace.
    pub prefix: String,
}

/// Pluggable editor autocomplete.
pub trait AutocompleteProvider: Send {
    /// Returns suggestions for the current state.
    fn suggestions(&mut self, context: AutocompleteContext<'_>) -> Option<AutocompleteResult>;

    /// Applies one completion. Providers can override for quoting or structured
    /// insertions; the default replaces `prefix` on the current line.
    fn apply(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> (Vec<String>, usize, usize) {
        let mut lines = lines.to_vec();
        let line = lines.get(cursor_line).cloned().unwrap_or_default();
        let end = floor_char_boundary(&line, cursor_col.min(line.len()));
        let start = floor_char_boundary(&line, end.saturating_sub(prefix.len()));
        let mut completed = String::with_capacity(line.len() + item.value.len());
        completed.push_str(&line[..start]);
        completed.push_str(&item.value);
        completed.push_str(&line[end..]);
        if cursor_line >= lines.len() {
            lines.resize(cursor_line + 1, String::new());
        }
        lines[cursor_line] = completed;
        (lines, cursor_line, start + item.value.len())
    }
}

/// In-memory prefix/fuzzy provider useful for commands and tests.
#[derive(Clone, Debug, Default)]
pub struct StaticAutocomplete {
    items: Vec<AutocompleteItem>,
    trigger: Option<char>,
}

impl StaticAutocomplete {
    /// Creates a provider.
    pub fn new(items: impl IntoIterator<Item = AutocompleteItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
            trigger: None,
        }
    }

    /// Restricts completion to tokens beginning with `trigger`.
    #[must_use]
    pub fn with_trigger(mut self, trigger: char) -> Self {
        self.trigger = Some(trigger);
        self
    }
}

impl AutocompleteProvider for StaticAutocomplete {
    fn suggestions(&mut self, context: AutocompleteContext<'_>) -> Option<AutocompleteResult> {
        let line = context.lines.get(context.cursor_line)?;
        let before = &line[..context.cursor_col.min(line.len())];
        let token_start = before
            .char_indices()
            .rev()
            .find_map(|(index, character)| {
                character
                    .is_whitespace()
                    .then_some(index + character.len_utf8())
            })
            .unwrap_or(0);
        let prefix = &before[token_start..];
        if let Some(trigger) = self.trigger {
            if !prefix.starts_with(trigger) {
                return None;
            }
        } else if prefix.is_empty() && !context.forced {
            return None;
        }

        let query = self
            .trigger
            .and_then(|trigger| prefix.strip_prefix(trigger))
            .unwrap_or(prefix)
            .to_ascii_lowercase();
        let mut scored: Vec<(usize, AutocompleteItem)> = self
            .items
            .iter()
            .filter_map(|item| {
                fuzzy_score(&item.value.to_ascii_lowercase(), &query)
                    .map(|score| (score, item.clone()))
            })
            .collect();
        scored.sort_by_key(|(score, item)| (*score, item.label.clone()));
        let items = scored.into_iter().map(|(_, item)| item).collect::<Vec<_>>();
        (!items.is_empty()).then(|| AutocompleteResult {
            items,
            prefix: prefix.to_owned(),
        })
    }

    fn apply(
        &mut self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &AutocompleteItem,
        prefix: &str,
    ) -> (Vec<String>, usize, usize) {
        let mut replacement = item.clone();
        if let Some(trigger) = self.trigger {
            replacement.value = format!("{trigger}{}", item.value);
        }
        let mut lines = lines.to_vec();
        let line = lines.get(cursor_line).cloned().unwrap_or_default();
        let cursor_col = floor_char_boundary(&line, cursor_col.min(line.len()));
        let start = floor_char_boundary(&line, cursor_col.saturating_sub(prefix.len()));
        let mut completed = String::new();
        completed.push_str(&line[..start]);
        completed.push_str(&replacement.value);
        completed.push_str(&line[cursor_col.min(line.len())..]);
        if cursor_line >= lines.len() {
            lines.resize(cursor_line + 1, String::new());
        }
        lines[cursor_line] = completed;
        (lines, cursor_line, start + replacement.value.len())
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn fuzzy_score(candidate: &str, query: &str) -> Option<usize> {
    if query.is_empty() {
        return Some(0);
    }
    if candidate == query {
        return Some(0);
    }
    if candidate.starts_with(query) {
        return Some(1 + candidate.len() - query.len());
    }
    let mut position = 0;
    let mut score = 10;
    for expected in query.chars() {
        let found = candidate[position..].find(expected)?;
        score += found;
        position += found + expected.len_utf8();
    }
    Some(score + candidate.len().saturating_sub(query.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_provider_fuzzy_filters_and_applies_trigger() {
        let mut provider = StaticAutocomplete::new([
            AutocompleteItem::new("help"),
            AutocompleteItem::new("history"),
        ])
        .with_trigger('/');
        let lines = vec!["/hl".to_owned()];
        let result = provider
            .suggestions(AutocompleteContext {
                lines: &lines,
                cursor_line: 0,
                cursor_col: 3,
                forced: false,
            })
            .unwrap();
        assert_eq!(result.items[0].value, "help");
        let (lines, _, cursor) = provider.apply(&lines, 0, 3, &result.items[0], &result.prefix);
        assert_eq!(lines, ["/help"]);
        assert_eq!(cursor, 5);
    }
}
