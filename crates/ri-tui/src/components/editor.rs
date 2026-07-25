use std::collections::BTreeMap;

use unicode_segmentation::UnicodeSegmentation;

use crate::Result;
use crate::ansi::{grapheme_width, truncate_to_width, visible_width};
use crate::autocomplete::{
    AutocompleteContext, AutocompleteItem, AutocompleteProvider, AutocompleteResult,
};
use crate::component::{Component, InputEvent, RenderContext};
use crate::editing::{
    KillRing, UndoStack, next_grapheme_boundary, previous_grapheme_boundary, word_backward,
    word_forward,
};
use crate::keys::{KeyCode, KeyEventKind, Modifiers};
use crate::line::ConstrainedLine;
use crate::renderer::CURSOR_MARKER;
use crate::theme::EditorTheme;

/// Logical editor cursor. `column` is a UTF-8 byte offset at a grapheme boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cursor {
    /// Logical line.
    pub line: usize,
    /// Byte offset within the line.
    pub column: usize,
}

/// Editor viewport state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Viewport {
    /// First visible wrapped line.
    pub offset: usize,
    /// Maximum visible wrapped text lines. Zero selects a responsive height.
    pub height: usize,
}

/// Editor configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditorOptions {
    /// Horizontal padding.
    pub padding_x: usize,
    /// Explicit viewport height. `None` uses 30% of terminal rows, at least 3.
    pub viewport_height: Option<usize>,
    /// Paste line count above which a compact marker is stored.
    pub large_paste_lines: usize,
    /// Paste byte count above which a compact marker is stored.
    pub large_paste_bytes: usize,
    /// Number of autocomplete rows.
    pub autocomplete_height: usize,
}

impl Default for EditorOptions {
    fn default() -> Self {
        Self {
            padding_x: 0,
            viewport_height: None,
            large_paste_lines: 10,
            large_paste_bytes: 1000,
            autocomplete_height: 5,
        }
    }
}

/// One word-wrapped source range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WordWrapChunk {
    /// Source text.
    pub text: String,
    /// Starting byte offset.
    pub start: usize,
    /// Exclusive ending byte offset.
    pub end: usize,
    /// Terminal-cell width.
    pub width: usize,
}

/// Wraps a logical line without splitting grapheme clusters.
pub fn word_wrap_line(line: &str, maximum_width: usize) -> Vec<WordWrapChunk> {
    if line.is_empty() || maximum_width == 0 {
        return vec![WordWrapChunk {
            text: String::new(),
            start: 0,
            end: 0,
            width: 0,
        }];
    }
    if visible_width(line) <= maximum_width {
        return vec![WordWrapChunk {
            text: line.to_owned(),
            start: 0,
            end: line.len(),
            width: visible_width(line),
        }];
    }

    let graphemes = line
        .grapheme_indices(true)
        .map(|(start, grapheme)| {
            let end = start + grapheme.len();
            (start, end, grapheme, grapheme_width(grapheme))
        })
        .collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut start_grapheme = 0;
    while start_grapheme < graphemes.len() {
        let start_byte = graphemes[start_grapheme].0;
        let mut current_width = 0;
        let mut last_break = None;
        let mut end_grapheme = start_grapheme;
        while end_grapheme < graphemes.len() {
            let (_, end, grapheme, width) = graphemes[end_grapheme];
            if current_width + width > maximum_width {
                break;
            }
            current_width += width;
            let next_is_cjk = graphemes
                .get(end_grapheme + 1)
                .is_some_and(|(_, _, next, _)| is_cjk(next));
            if grapheme.chars().all(char::is_whitespace) || is_cjk(grapheme) || next_is_cjk {
                last_break = Some((end_grapheme + 1, end));
            }
            end_grapheme += 1;
        }

        let (next_grapheme, end_byte) = if end_grapheme == graphemes.len() {
            (end_grapheme, line.len())
        } else if let Some((break_grapheme, break_byte)) =
            last_break.filter(|(index, _)| *index > start_grapheme)
        {
            (break_grapheme, break_byte)
        } else if end_grapheme == start_grapheme {
            (start_grapheme + 1, graphemes[start_grapheme].1)
        } else {
            (end_grapheme, graphemes[end_grapheme].0)
        };
        let text = line[start_byte..end_byte].to_owned();
        chunks.push(WordWrapChunk {
            width: visible_width(&text),
            text,
            start: start_byte,
            end: end_byte,
        });
        start_grapheme = next_grapheme;
    }
    chunks
}

fn is_cjk(grapheme: &str) -> bool {
    grapheme.chars().next().is_some_and(|character| {
        matches!(
            character as u32,
            0x2e80..=0x9fff
                | 0xac00..=0xd7af
                | 0x1100..=0x11ff
        )
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EditorState {
    lines: Vec<String>,
    cursor: Cursor,
}

#[derive(Clone, Debug)]
struct Snapshot {
    state: EditorState,
    pastes: BTreeMap<u64, String>,
    paste_counter: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LastAction {
    Kill,
    Yank,
    TypeWord,
}

#[derive(Clone, Debug)]
struct VisualLine {
    logical_line: usize,
    start: usize,
    end: usize,
    text: String,
}

type TextCallback = Box<dyn FnMut(&str) + Send>;

/// Multi-line grapheme-aware text editor.
pub struct Editor {
    state: EditorState,
    options: EditorOptions,
    theme: EditorTheme,
    focused: bool,
    viewport: Viewport,
    last_width: usize,
    preferred_cell_column: Option<usize>,
    undo: UndoStack<Snapshot>,
    kill_ring: KillRing,
    last_action: Option<LastAction>,
    last_yank: Option<(Cursor, Cursor)>,
    history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<EditorState>,
    provider: Option<Box<dyn AutocompleteProvider>>,
    suggestions: Vec<AutocompleteItem>,
    completion_prefix: String,
    selected_suggestion: usize,
    pastes: BTreeMap<u64, String>,
    paste_counter: u64,
    on_change: Option<TextCallback>,
    on_submit: Option<TextCallback>,
}

impl std::fmt::Debug for Editor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Editor")
            .field("state", &self.state)
            .field("options", &self.options)
            .field("focused", &self.focused)
            .field("viewport", &self.viewport)
            .field("suggestions", &self.suggestions)
            .field("pastes", &self.pastes.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for Editor {
    fn default() -> Self {
        Self::new(EditorTheme::default(), EditorOptions::default())
    }
}

impl Editor {
    /// Creates an editor.
    pub fn new(theme: EditorTheme, options: EditorOptions) -> Self {
        Self {
            state: EditorState {
                lines: vec![String::new()],
                cursor: Cursor::default(),
            },
            options,
            theme,
            focused: false,
            viewport: Viewport::default(),
            last_width: 80,
            preferred_cell_column: None,
            undo: UndoStack::default(),
            kill_ring: KillRing::default(),
            last_action: None,
            last_yank: None,
            history: Vec::new(),
            history_index: None,
            history_draft: None,
            provider: None,
            suggestions: Vec::new(),
            completion_prefix: String::new(),
            selected_suggestion: 0,
            pastes: BTreeMap::new(),
            paste_counter: 0,
            on_change: None,
            on_submit: None,
        }
    }

    /// Current logical text.
    pub fn text(&self) -> String {
        self.state.lines.join("\n")
    }

    /// Text with compact paste markers expanded.
    pub fn expanded_text(&self) -> String {
        let mut text = self.text();
        for (id, content) in &self.pastes {
            let prefix = format!("[paste #{id} ");
            while let Some(start) = text.find(&prefix) {
                let Some(relative_end) = text[start..].find(']') else {
                    break;
                };
                text.replace_range(start..=start + relative_end, content);
            }
        }
        text
    }

    /// Defensive copy of logical lines.
    pub fn lines(&self) -> Vec<String> {
        self.state.lines.clone()
    }

    /// Current cursor.
    pub const fn cursor(&self) -> Cursor {
        self.state.cursor
    }

    /// Current viewport.
    pub const fn viewport(&self) -> Viewport {
        self.viewport
    }

    /// Sets explicit visible text rows. Zero restores responsive sizing.
    pub fn set_viewport_height(&mut self, height: usize) {
        self.viewport.height = height;
    }

    /// Replaces text and moves to its end.
    pub fn set_text(&mut self, text: impl AsRef<str>) {
        let normalized = normalize_text(text.as_ref());
        if self.text() != normalized {
            self.push_undo();
        }
        self.state.lines = normalized.split('\n').map(ToOwned::to_owned).collect();
        if self.state.lines.is_empty() {
            self.state.lines.push(String::new());
        }
        self.state.cursor.line = self.state.lines.len() - 1;
        self.state.cursor.column = self.state.lines.last().map_or(0, String::len);
        self.pastes.clear();
        self.paste_counter = 0;
        self.exit_history();
        self.clear_completion();
        self.notify_change();
    }

    /// Sets the cursor after clamping to a UTF-8 and grapheme boundary.
    pub fn set_cursor(&mut self, cursor: Cursor) {
        let line = cursor.line.min(self.state.lines.len().saturating_sub(1));
        let requested = cursor.column.min(self.state.lines[line].len());
        let column = floor_grapheme_boundary(&self.state.lines[line], requested);
        self.state.cursor = Cursor { line, column };
        self.preferred_cell_column = None;
    }

    /// Inserts text atomically.
    pub fn insert_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.push_undo();
        self.insert_internal(&normalize_text(text));
        self.last_action = None;
        self.refresh_completion(false);
        self.notify_change();
    }

    /// Configures autocomplete.
    pub fn set_autocomplete_provider(&mut self, provider: impl AutocompleteProvider + 'static) {
        self.provider = Some(Box::new(provider));
        self.clear_completion();
    }

    /// Current suggestions.
    pub fn suggestions(&self) -> &[AutocompleteItem] {
        &self.suggestions
    }

    /// Adds a non-empty, non-consecutive history entry.
    pub fn add_history(&mut self, text: impl Into<String>) {
        let text = text.into();
        let text = text.trim();
        if text.is_empty() || self.history.first().is_some_and(|entry| entry == text) {
            return;
        }
        self.history.insert(0, text.to_owned());
        self.history.truncate(100);
    }

    /// Registers text change handling.
    pub fn on_change(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        self.on_change = Some(Box::new(callback));
    }

    /// Registers submit handling.
    pub fn on_submit(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        self.on_submit = Some(Box::new(callback));
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            state: self.state.clone(),
            pastes: self.pastes.clone(),
            paste_counter: self.paste_counter,
        }
    }

    fn push_undo(&mut self) {
        let snapshot = self.snapshot();
        self.undo.push(&snapshot);
    }

    fn undo(&mut self) {
        if let Some(snapshot) = self.undo.pop() {
            self.state = snapshot.state;
            self.pastes = snapshot.pastes;
            self.paste_counter = snapshot.paste_counter;
            self.last_action = None;
            self.last_yank = None;
            self.preferred_cell_column = None;
            self.clear_completion();
            self.notify_change();
        }
    }

    fn notify_change(&mut self) {
        let text = self.text();
        if let Some(callback) = &mut self.on_change {
            callback(&text);
        }
    }

    fn insert_character(&mut self, text: &str) {
        let word = !text.chars().any(char::is_whitespace);
        if !word || self.last_action != Some(LastAction::TypeWord) {
            self.push_undo();
        }
        self.exit_history();
        self.insert_internal(text);
        self.last_action = Some(LastAction::TypeWord);
        self.last_yank = None;
        self.refresh_completion(false);
        self.notify_change();
    }

    fn insert_internal(&mut self, text: &str) {
        let inserted = text.split('\n').collect::<Vec<_>>();
        let cursor = self.state.cursor;
        let current = self.state.lines[cursor.line].clone();
        let before = &current[..cursor.column];
        let after = &current[cursor.column..];
        if inserted.len() == 1 {
            self.state.lines[cursor.line] = format!("{before}{text}{after}");
            self.state.cursor.column += text.len();
        } else {
            let mut replacement = Vec::with_capacity(inserted.len());
            replacement.push(format!("{before}{}", inserted[0]));
            replacement.extend(
                inserted[1..inserted.len() - 1]
                    .iter()
                    .map(|line| (*line).to_owned()),
            );
            replacement.push(format!("{}{after}", inserted.last().unwrap_or(&"")));
            let inserted_count = replacement.len();
            self.state
                .lines
                .splice(cursor.line..=cursor.line, replacement);
            self.state.cursor.line += inserted_count - 1;
            self.state.cursor.column = inserted.last().map_or(0, |line| line.len());
        }
        self.preferred_cell_column = None;
    }

    fn add_newline(&mut self) {
        self.split_line(true);
    }

    fn split_line(&mut self, snapshot: bool) {
        if snapshot {
            self.push_undo();
        }
        self.exit_history();
        let cursor = self.state.cursor;
        let current = self.state.lines[cursor.line].clone();
        let before = current[..cursor.column].to_owned();
        let after = current[cursor.column..].to_owned();
        self.state.lines[cursor.line] = before;
        self.state.lines.insert(cursor.line + 1, after);
        self.state.cursor = Cursor {
            line: cursor.line + 1,
            column: 0,
        };
        self.last_action = None;
        self.clear_completion();
        self.notify_change();
    }

    fn handle_paste(&mut self, text: &str) {
        let mut text = normalize_text(text)
            .chars()
            .filter(|character| *character == '\n' || !character.is_control())
            .collect::<String>();
        if text.is_empty() {
            return;
        }
        if matches!(text.chars().next(), Some('/' | '~' | '.')) {
            let line = &self.state.lines[self.state.cursor.line];
            if self.state.cursor.column > 0
                && line[..self.state.cursor.column]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_alphanumeric)
            {
                text.insert(0, ' ');
            }
        }
        self.push_undo();
        let line_count = text.lines().count().max(1);
        if line_count > self.options.large_paste_lines
            || text.len() > self.options.large_paste_bytes
        {
            self.paste_counter += 1;
            self.pastes.insert(self.paste_counter, text.clone());
            let marker = if line_count > self.options.large_paste_lines {
                format!("[paste #{} +{} lines]", self.paste_counter, line_count)
            } else {
                format!("[paste #{} {} bytes]", self.paste_counter, text.len())
            };
            self.insert_internal(&marker);
        } else {
            self.insert_internal(&text);
        }
        self.last_action = None;
        self.clear_completion();
        self.notify_change();
    }

    fn backspace(&mut self) {
        let cursor = self.state.cursor;
        if cursor.column > 0 {
            let start = previous_grapheme_boundary(&self.state.lines[cursor.line], cursor.column);
            self.delete_on_line(start, cursor.column, None);
        } else if cursor.line > 0 {
            self.push_undo();
            let current = self.state.lines.remove(cursor.line);
            self.state.cursor.line -= 1;
            self.state.cursor.column = self.state.lines[self.state.cursor.line].len();
            self.state.lines[self.state.cursor.line].push_str(&current);
            self.last_action = None;
            self.notify_change();
        }
    }

    fn delete_forward(&mut self) {
        let cursor = self.state.cursor;
        let line_len = self.state.lines[cursor.line].len();
        if cursor.column < line_len {
            let end = next_grapheme_boundary(&self.state.lines[cursor.line], cursor.column);
            self.delete_on_line(cursor.column, end, None);
        } else if cursor.line + 1 < self.state.lines.len() {
            self.push_undo();
            let next = self.state.lines.remove(cursor.line + 1);
            self.state.lines[cursor.line].push_str(&next);
            self.last_action = None;
            self.notify_change();
        }
    }

    fn delete_on_line(&mut self, start: usize, end: usize, kill_prepend: Option<bool>) {
        if start >= end {
            return;
        }
        self.push_undo();
        let deleted = self.state.lines[self.state.cursor.line][start..end].to_owned();
        if let Some(prepend) = kill_prepend {
            self.kill_ring
                .push(deleted, prepend, self.last_action == Some(LastAction::Kill));
            self.last_action = Some(LastAction::Kill);
        } else {
            self.last_action = None;
        }
        self.state.lines[self.state.cursor.line].replace_range(start..end, "");
        self.state.cursor.column = start;
        self.last_yank = None;
        self.clear_completion();
        self.notify_change();
    }

    fn kill_backward_word(&mut self) {
        let cursor = self.state.cursor;
        if cursor.column == 0 {
            if cursor.line > 0 {
                self.kill_newline(true);
            }
            return;
        }
        let start = word_backward(&self.state.lines[cursor.line], cursor.column);
        self.delete_on_line(start, cursor.column, Some(true));
    }

    fn kill_forward_word(&mut self) {
        let cursor = self.state.cursor;
        if cursor.column == self.state.lines[cursor.line].len() {
            if cursor.line + 1 < self.state.lines.len() {
                self.kill_newline(false);
            }
            return;
        }
        let end = word_forward(&self.state.lines[cursor.line], cursor.column);
        self.delete_on_line(cursor.column, end, Some(false));
    }

    fn kill_newline(&mut self, backward: bool) {
        self.push_undo();
        self.kill_ring
            .push("\n", backward, self.last_action == Some(LastAction::Kill));
        if backward {
            let current = self.state.lines.remove(self.state.cursor.line);
            self.state.cursor.line -= 1;
            self.state.cursor.column = self.state.lines[self.state.cursor.line].len();
            self.state.lines[self.state.cursor.line].push_str(&current);
        } else {
            let next = self.state.lines.remove(self.state.cursor.line + 1);
            self.state.lines[self.state.cursor.line].push_str(&next);
        }
        self.last_action = Some(LastAction::Kill);
        self.notify_change();
    }

    fn yank(&mut self) {
        let Some(text) = self.kill_ring.peek().map(ToOwned::to_owned) else {
            return;
        };
        self.push_undo();
        let start = self.state.cursor;
        self.insert_internal(&text);
        self.last_yank = Some((start, self.state.cursor));
        self.last_action = Some(LastAction::Yank);
        self.notify_change();
    }

    fn yank_pop(&mut self) {
        if self.last_action != Some(LastAction::Yank) || self.kill_ring.len() < 2 {
            return;
        }
        let Some((start, end)) = self.last_yank else {
            return;
        };
        self.push_undo();
        self.remove_range(start, end);
        self.kill_ring.rotate();
        let replacement = self.kill_ring.peek().unwrap_or_default().to_owned();
        self.insert_internal(&replacement);
        self.last_yank = Some((start, self.state.cursor));
        self.last_action = Some(LastAction::Yank);
        self.notify_change();
    }

    fn remove_range(&mut self, start: Cursor, end: Cursor) {
        if start.line == end.line {
            self.state.lines[start.line].replace_range(start.column..end.column, "");
        } else {
            let suffix = self.state.lines[end.line][end.column..].to_owned();
            self.state.lines[start.line].truncate(start.column);
            self.state.lines[start.line].push_str(&suffix);
            self.state.lines.drain(start.line + 1..=end.line);
        }
        self.state.cursor = start;
    }

    fn move_horizontal(&mut self, direction: i32) {
        let cursor = self.state.cursor;
        if direction < 0 {
            if cursor.column > 0 {
                self.state.cursor.column =
                    previous_grapheme_boundary(&self.state.lines[cursor.line], cursor.column);
            } else if cursor.line > 0 {
                self.state.cursor.line -= 1;
                self.state.cursor.column = self.state.lines[self.state.cursor.line].len();
            }
        } else if cursor.column < self.state.lines[cursor.line].len() {
            self.state.cursor.column =
                next_grapheme_boundary(&self.state.lines[cursor.line], cursor.column);
        } else if cursor.line + 1 < self.state.lines.len() {
            self.state.cursor.line += 1;
            self.state.cursor.column = 0;
        }
        self.preferred_cell_column = None;
        self.last_action = None;
        self.refresh_completion(false);
    }

    fn move_word(&mut self, direction: i32) {
        let cursor = self.state.cursor;
        if direction < 0 {
            if cursor.column == 0 && cursor.line > 0 {
                self.state.cursor.line -= 1;
                self.state.cursor.column = self.state.lines[self.state.cursor.line].len();
            } else {
                self.state.cursor.column =
                    word_backward(&self.state.lines[cursor.line], cursor.column);
            }
        } else if cursor.column == self.state.lines[cursor.line].len()
            && cursor.line + 1 < self.state.lines.len()
        {
            self.state.cursor.line += 1;
            self.state.cursor.column = 0;
        } else {
            self.state.cursor.column = word_forward(&self.state.lines[cursor.line], cursor.column);
        }
        self.preferred_cell_column = None;
        self.last_action = None;
    }

    fn visual_lines(&self, width: usize) -> Vec<VisualLine> {
        let mut output = Vec::new();
        for (logical_line, line) in self.state.lines.iter().enumerate() {
            for chunk in word_wrap_line(line, width.max(1)) {
                output.push(VisualLine {
                    logical_line,
                    start: chunk.start,
                    end: chunk.end,
                    text: chunk.text,
                });
            }
        }
        output
    }

    fn current_visual_index(&self, lines: &[VisualLine]) -> usize {
        lines
            .iter()
            .enumerate()
            .find_map(|(index, line)| {
                if line.logical_line != self.state.cursor.line {
                    return None;
                }
                let last = lines
                    .get(index + 1)
                    .is_none_or(|next| next.logical_line != line.logical_line);
                (self.state.cursor.column >= line.start
                    && (self.state.cursor.column < line.end
                        || (last && self.state.cursor.column == line.end)))
                    .then_some(index)
            })
            .unwrap_or_else(|| lines.len().saturating_sub(1))
    }

    fn move_vertical(&mut self, delta: i32) {
        let lines = self.visual_lines(self.last_width.max(1));
        let current = self.current_visual_index(&lines);
        let distance = usize::try_from(delta.unsigned_abs()).unwrap_or(usize::MAX);
        let target = if delta < 0 {
            current.saturating_sub(distance)
        } else {
            current
                .saturating_add(distance)
                .min(lines.len().saturating_sub(1))
        };
        if current == target {
            return;
        }
        let current_line = &lines[current];
        let current_column = visible_width(
            &self.state.lines[current_line.logical_line]
                [current_line.start..self.state.cursor.column],
        );
        let wanted = self.preferred_cell_column.unwrap_or(current_column);
        self.preferred_cell_column = Some(wanted);
        let target_line = &lines[target];
        let byte = byte_at_cell(&target_line.text, wanted);
        self.state.cursor = Cursor {
            line: target_line.logical_line,
            column: target_line.start + byte,
        };
        self.last_action = None;
        self.refresh_completion(false);
    }

    fn browse_history(&mut self, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let next = match (self.history_index, older) {
            (None, true) => 0,
            (Some(index), true) => (index + 1).min(self.history.len() - 1),
            (Some(0), false) => {
                self.history_index = None;
                if let Some(draft) = self.history_draft.take() {
                    self.state = draft;
                    self.notify_change();
                }
                return;
            }
            (Some(index), false) => index - 1,
            (None, false) => return,
        };
        if self.history_index.is_none() {
            self.history_draft = Some(self.state.clone());
        }
        self.history_index = Some(next);
        self.state.lines = self.history[next]
            .split('\n')
            .map(ToOwned::to_owned)
            .collect();
        self.state.cursor = if older {
            Cursor::default()
        } else {
            Cursor {
                line: self.state.lines.len() - 1,
                column: self.state.lines.last().map_or(0, String::len),
            }
        };
        self.notify_change();
    }

    fn exit_history(&mut self) {
        self.history_index = None;
        self.history_draft = None;
    }

    fn refresh_completion(&mut self, forced: bool) {
        let Some(provider) = &mut self.provider else {
            return;
        };
        let result = provider.suggestions(AutocompleteContext {
            lines: &self.state.lines,
            cursor_line: self.state.cursor.line,
            cursor_col: self.state.cursor.column,
            forced,
        });
        if let Some(AutocompleteResult { items, prefix }) = result {
            self.suggestions = items;
            self.completion_prefix = prefix;
            self.selected_suggestion = self
                .selected_suggestion
                .min(self.suggestions.len().saturating_sub(1));
        } else {
            self.clear_completion();
        }
    }

    fn clear_completion(&mut self) {
        self.suggestions.clear();
        self.completion_prefix.clear();
        self.selected_suggestion = 0;
    }

    fn apply_completion(&mut self) -> bool {
        let Some(item) = self.suggestions.get(self.selected_suggestion).cloned() else {
            return false;
        };
        if self.provider.is_none() {
            return false;
        }
        self.push_undo();
        let (mut lines, line, column) = self
            .provider
            .as_mut()
            .expect("provider checked above")
            .apply(
                &self.state.lines,
                self.state.cursor.line,
                self.state.cursor.column,
                &item,
                &self.completion_prefix,
            );
        for line in &mut lines {
            *line = sanitize_logical_line(line);
        }
        if lines.is_empty() {
            lines.push(String::new());
        }
        let line = line.min(lines.len() - 1);
        let column = floor_grapheme_boundary(&lines[line], column.min(lines[line].len()));
        self.state.lines = lines;
        self.state.cursor = Cursor { line, column };
        self.clear_completion();
        self.notify_change();
        true
    }

    fn submit(&mut self) {
        let submitted = self.expanded_text().trim().to_owned();
        self.state = EditorState {
            lines: vec![String::new()],
            cursor: Cursor::default(),
        };
        self.pastes.clear();
        self.paste_counter = 0;
        self.undo.clear();
        self.viewport.offset = 0;
        self.exit_history();
        self.clear_completion();
        self.last_action = None;
        self.last_yank = None;
        self.notify_change();
        if let Some(callback) = &mut self.on_submit {
            callback(&submitted);
        }
    }

    fn render_border(&self, width: usize, above: usize, below: usize) -> String {
        let indicator = if above > 0 {
            format!("─── ↑ {above} more ")
        } else if below > 0 {
            format!("─── ↓ {below} more ")
        } else {
            String::new()
        };
        let border = if indicator.is_empty() {
            "─".repeat(width)
        } else {
            let clipped = truncate_to_width(&indicator, width, "...", false);
            let used = visible_width(&clipped);
            format!("{clipped}{}", "─".repeat(width.saturating_sub(used)))
        };
        self.theme.border.paint(border)
    }
}

impl Component for Editor {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        let padding = self
            .options
            .padding_x
            .min(context.width.saturating_sub(1) / 2);
        let content_width = context.width.saturating_sub(padding * 2).max(1);
        self.last_width = content_width;
        let visual = self.visual_lines(content_width);
        let cursor_index = self.current_visual_index(&visual);
        let responsive_height = (context.height * 3 / 10).max(3);
        let maximum = if self.viewport.height > 0 {
            self.viewport.height
        } else {
            self.options.viewport_height.unwrap_or(responsive_height)
        }
        .max(1);
        if cursor_index < self.viewport.offset {
            self.viewport.offset = cursor_index;
        } else if cursor_index >= self.viewport.offset + maximum {
            self.viewport.offset = cursor_index + 1 - maximum;
        }
        self.viewport.offset = self
            .viewport
            .offset
            .min(visual.len().saturating_sub(maximum));
        let end = (self.viewport.offset + maximum).min(visual.len());
        let below = visual.len().saturating_sub(end);
        let margin = " ".repeat(padding);
        let mut output = vec![ConstrainedLine::new(
            self.render_border(context.width, self.viewport.offset, 0),
            context.width,
        )?];

        for (index, visual_line) in visual[self.viewport.offset..end].iter().enumerate() {
            let visual_index = self.viewport.offset + index;
            let over_wide = visible_width(&visual_line.text) > content_width;
            let mut text = if over_wide {
                "�".to_owned()
            } else {
                visual_line.text.clone()
            };
            if visual_index == cursor_index {
                let cursor_local = if over_wide {
                    if self.state.cursor.column > visual_line.start {
                        text.len()
                    } else {
                        0
                    }
                } else {
                    self.state
                        .cursor
                        .column
                        .saturating_sub(visual_line.start)
                        .min(text.len())
                };
                let cursor_local = floor_boundary(&text, cursor_local);
                let after = &text[cursor_local..];
                let end = next_grapheme_boundary(after, 0);
                let cursor_text = if end == 0 { " " } else { &after[..end] };
                let rest = if end == 0 { after } else { &after[end..] };
                let marker = if context.focused || self.focused {
                    CURSOR_MARKER
                } else {
                    ""
                };
                text = format!(
                    "{}{marker}{}{rest}",
                    &text[..cursor_local],
                    self.theme.cursor.paint(cursor_text)
                );
            }
            let used = visible_width(&text);
            let line = format!(
                "{margin}{text}{}{margin}",
                " ".repeat(content_width.saturating_sub(used))
            );
            output.push(ConstrainedLine::new(line, context.width)?);
        }
        output.push(ConstrainedLine::new(
            self.render_border(context.width, 0, below),
            context.width,
        )?);

        if !self.suggestions.is_empty() {
            let shown = self
                .suggestions
                .len()
                .min(self.options.autocomplete_height.max(1));
            let start = self
                .selected_suggestion
                .saturating_sub(shown / 2)
                .min(self.suggestions.len() - shown);
            for index in start..start + shown {
                let item = &self.suggestions[index];
                let prefix = if index == self.selected_suggestion {
                    self.theme.autocomplete.selected_prefix.as_str()
                } else {
                    "  "
                };
                let available = content_width.saturating_sub(visible_width(prefix));
                let row = format!(
                    "{prefix}{}",
                    truncate_to_width(&item.label, available, "", false)
                );
                let row = if index == self.selected_suggestion {
                    self.theme.autocomplete.selected.paint(row)
                } else {
                    row
                };
                let used = visible_width(&row);
                output.push(ConstrainedLine::new(
                    format!(
                        "{margin}{row}{}{margin}",
                        " ".repeat(content_width.saturating_sub(used))
                    ),
                    context.width,
                )?);
            }
        }
        Ok(output)
    }

    fn handle_event(&mut self, event: &InputEvent) -> bool {
        if let InputEvent::Paste(text) = event {
            self.handle_paste(text);
            return true;
        }
        let InputEvent::Key(key) = event else {
            return false;
        };
        if key.kind == KeyEventKind::Release {
            return false;
        }

        if !self.suggestions.is_empty() {
            if key.matches("escape") {
                self.clear_completion();
                return true;
            }
            if matches!(key.code, KeyCode::Up) {
                self.selected_suggestion = self
                    .selected_suggestion
                    .checked_sub(1)
                    .unwrap_or(self.suggestions.len() - 1);
                return true;
            }
            if matches!(key.code, KeyCode::Down) {
                self.selected_suggestion = (self.selected_suggestion + 1) % self.suggestions.len();
                return true;
            }
            if matches!(key.code, KeyCode::Tab | KeyCode::Enter) && self.apply_completion() {
                return true;
            }
        }
        if matches!(key.code, KeyCode::Tab) {
            self.refresh_completion(true);
            if self.suggestions.len() == 1 {
                self.apply_completion();
            }
            return true;
        }
        if key.matches("ctrl+z") {
            self.undo();
            return true;
        }
        if key.matches("ctrl+y") {
            self.yank();
            return true;
        }
        if key.matches("alt+y") {
            self.yank_pop();
            return true;
        }
        if key.matches("ctrl+w") || key.matches("alt+backspace") {
            self.kill_backward_word();
            return true;
        }
        if key.matches("alt+d") || key.matches("alt+delete") {
            self.kill_forward_word();
            return true;
        }
        if key.matches("ctrl+u") {
            let end = self.state.cursor.column;
            self.delete_on_line(0, end, Some(true));
            return true;
        }
        if key.matches("ctrl+k") {
            let start = self.state.cursor.column;
            let end = self.state.lines[self.state.cursor.line].len();
            if start == end && self.state.cursor.line + 1 < self.state.lines.len() {
                self.kill_newline(false);
            } else {
                self.delete_on_line(start, end, Some(false));
            }
            return true;
        }
        if matches!(key.code, KeyCode::Backspace) {
            self.backspace();
            return true;
        }
        if matches!(key.code, KeyCode::Delete) {
            self.delete_forward();
            return true;
        }
        if key.matches("ctrl+a") || matches!(key.code, KeyCode::Home) {
            self.state.cursor.column = 0;
            self.preferred_cell_column = None;
            return true;
        }
        if key.matches("ctrl+e") || matches!(key.code, KeyCode::End) {
            self.state.cursor.column = self.state.lines[self.state.cursor.line].len();
            self.preferred_cell_column = None;
            return true;
        }
        if matches!(key.code, KeyCode::Left)
            && (key.modifiers.contains(Modifiers::CTRL) || key.modifiers.contains(Modifiers::ALT))
        {
            self.move_word(-1);
            return true;
        }
        if matches!(key.code, KeyCode::Right)
            && (key.modifiers.contains(Modifiers::CTRL) || key.modifiers.contains(Modifiers::ALT))
        {
            self.move_word(1);
            return true;
        }
        if matches!(key.code, KeyCode::Left) {
            self.move_horizontal(-1);
            return true;
        }
        if matches!(key.code, KeyCode::Right) {
            self.move_horizontal(1);
            return true;
        }
        if matches!(key.code, KeyCode::Up) {
            let visual = self.visual_lines(self.last_width);
            if self.current_visual_index(&visual) == 0
                && (self.text().is_empty() || self.state.cursor.column == 0)
            {
                self.browse_history(true);
            } else {
                self.move_vertical(-1);
            }
            return true;
        }
        if matches!(key.code, KeyCode::Down) {
            let visual = self.visual_lines(self.last_width);
            if self.history_index.is_some()
                && self.current_visual_index(&visual) + 1 == visual.len()
            {
                self.browse_history(false);
            } else {
                self.move_vertical(1);
            }
            return true;
        }
        if matches!(key.code, KeyCode::PageUp) {
            let distance = i32::try_from(self.viewport.height.max(3)).unwrap_or(i32::MAX);
            self.move_vertical(-distance);
            return true;
        }
        if matches!(key.code, KeyCode::PageDown) {
            let distance = i32::try_from(self.viewport.height.max(3)).unwrap_or(i32::MAX);
            self.move_vertical(distance);
            return true;
        }
        if matches!(key.code, KeyCode::Enter) {
            let modified = key.modifiers.contains(Modifiers::SHIFT)
                || key.modifiers.contains(Modifiers::CTRL)
                || key.modifiers.contains(Modifiers::ALT);
            let current = &self.state.lines[self.state.cursor.line];
            if modified
                || (self.state.cursor.column > 0
                    && current[..self.state.cursor.column].ends_with('\\'))
            {
                if modified {
                    self.add_newline();
                } else {
                    self.push_undo();
                    let end = self.state.cursor.column;
                    let start =
                        previous_grapheme_boundary(&self.state.lines[self.state.cursor.line], end);
                    self.state.lines[self.state.cursor.line].replace_range(start..end, "");
                    self.state.cursor.column = start;
                    self.split_line(false);
                }
            } else {
                self.submit();
            }
            return true;
        }
        if let Some(text) = key.printable() {
            self.insert_character(text);
            return true;
        }
        false
    }

    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    fn focusable(&self) -> bool {
        true
    }
}

fn normalize_text(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut output = String::with_capacity(normalized.len());
    for character in normalized.chars() {
        match character {
            '\n' => output.push('\n'),
            '\t' => output.push_str("    "),
            character if character.is_control() => {}
            character => output.push(character),
        }
    }
    output
}

fn sanitize_logical_line(text: &str) -> String {
    normalize_text(text).replace('\n', "")
}

fn byte_at_cell(text: &str, target: usize) -> usize {
    let mut width = 0;
    for (index, grapheme) in text.grapheme_indices(true) {
        let next = width + grapheme_width(grapheme);
        if next > target {
            return index;
        }
        width = next;
    }
    text.len()
}

fn floor_boundary(text: &str, mut offset: usize) -> usize {
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn floor_grapheme_boundary(text: &str, offset: usize) -> usize {
    text.grapheme_indices(true)
        .map(|(index, _)| index)
        .take_while(|index| *index <= offset)
        .last()
        .unwrap_or(0)
        .max(if offset == text.len() { text.len() } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{KeyParseMode, parse_key};

    fn key(sequence: &str) -> InputEvent {
        InputEvent::Key(parse_key(sequence, KeyParseMode::Legacy).unwrap())
    }

    #[test]
    fn wraps_cjk_and_words_by_cells() {
        let chunks = word_wrap_line("hello world", 6);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<Vec<_>>(),
            ["hello ", "world"]
        );
        let cjk = word_wrap_line("你好世界", 4);
        assert_eq!(cjk.len(), 2);
        assert!(cjk.iter().all(|chunk| chunk.width == 4));
    }

    #[test]
    fn edits_graphemes_multiline_and_undoes() {
        let mut editor = Editor::default();
        editor.insert_text("a👨‍👩‍👧‍👦b\nsecond");
        editor.handle_event(&key("\x7f"));
        assert_eq!(editor.text(), "a👨‍👩‍👧‍👦b\nsecon");
        editor.handle_event(&key("\x1a"));
        assert_eq!(editor.text(), "a👨‍👩‍👧‍👦b\nsecond");
    }

    #[test]
    fn large_paste_is_reversible_and_expandable() {
        let mut editor = Editor::default();
        let paste = (0..12)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        editor.handle_event(&InputEvent::Paste(paste.clone()));
        assert!(editor.text().starts_with("[paste #1 +12 lines]"));
        assert_eq!(editor.expanded_text(), paste);
        editor.handle_event(&key("\x1a"));
        assert_eq!(editor.text(), "");
    }
}
