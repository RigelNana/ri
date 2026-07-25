use crate::Result;
use crate::ansi::{slice_columns, visible_width};
use crate::component::{Component, InputEvent, RenderContext};
use crate::editing::{
    KillRing, UndoStack, next_grapheme_boundary, previous_grapheme_boundary, word_backward,
    word_forward,
};
use crate::keys::{KeyCode, KeyEventKind};
use crate::line::ConstrainedLine;
use crate::renderer::CURSOR_MARKER;
use crate::theme::Style;

#[derive(Clone, Debug, Eq, PartialEq)]
struct InputState {
    value: String,
    cursor: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LastAction {
    Kill,
    Yank,
    TypeWord,
}

type TextCallback = Box<dyn FnMut(&str) + Send>;

/// Single-line Unicode input with horizontal cell scrolling.
pub struct Input {
    state: InputState,
    prompt: String,
    focused: bool,
    cursor_style: Style,
    undo: UndoStack<InputState>,
    kill_ring: KillRing,
    last_action: Option<LastAction>,
    on_submit: Option<TextCallback>,
    on_escape: Option<Box<dyn FnMut() + Send>>,
}

impl std::fmt::Debug for Input {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Input")
            .field("state", &self.state)
            .field("prompt", &self.prompt)
            .field("focused", &self.focused)
            .finish_non_exhaustive()
    }
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Input {
    /// Creates an empty input.
    pub fn new() -> Self {
        Self {
            state: InputState {
                value: String::new(),
                cursor: 0,
            },
            prompt: "> ".to_owned(),
            focused: false,
            cursor_style: Style {
                inverse: true,
                ..Style::default()
            },
            undo: UndoStack::default(),
            kill_ring: KillRing::default(),
            last_action: None,
            on_submit: None,
            on_escape: None,
        }
    }

    /// Changes the prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Returns the value.
    pub fn value(&self) -> &str {
        &self.state.value
    }

    /// Replaces the value and moves the cursor to its end.
    pub fn set_value(&mut self, value: impl Into<String>) {
        self.push_undo();
        self.state.value = sanitize_single_line(&value.into());
        self.state.cursor = self.state.value.len();
        self.last_action = None;
    }

    /// Current cursor byte offset.
    pub const fn cursor(&self) -> usize {
        self.state.cursor
    }

    /// Registers submit handling.
    pub fn on_submit(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        self.on_submit = Some(Box::new(callback));
    }

    /// Registers Escape handling.
    pub fn on_escape(&mut self, callback: impl FnMut() + Send + 'static) {
        self.on_escape = Some(Box::new(callback));
    }

    fn push_undo(&mut self) {
        self.undo.push(&self.state);
    }

    fn insert(&mut self, text: &str) {
        let text = sanitize_single_line(text);
        if text.is_empty() {
            return;
        }
        let word_typing = !text.chars().any(char::is_whitespace);
        if !word_typing || self.last_action != Some(LastAction::TypeWord) {
            self.push_undo();
        }
        self.state.value.insert_str(self.state.cursor, &text);
        self.state.cursor += text.len();
        self.last_action = Some(LastAction::TypeWord);
    }

    fn delete_range(&mut self, start: usize, end: usize, kill: Option<bool>) {
        if start >= end || end > self.state.value.len() {
            return;
        }
        self.push_undo();
        let deleted = self.state.value[start..end].to_owned();
        if let Some(prepend) = kill {
            self.kill_ring
                .push(deleted, prepend, self.last_action == Some(LastAction::Kill));
            self.last_action = Some(LastAction::Kill);
        } else {
            self.last_action = None;
        }
        self.state.value.replace_range(start..end, "");
        self.state.cursor = start;
    }

    fn yank(&mut self) {
        let Some(text) = self.kill_ring.peek().map(ToOwned::to_owned) else {
            return;
        };
        self.push_undo();
        self.state.value.insert_str(self.state.cursor, &text);
        self.state.cursor += text.len();
        self.last_action = Some(LastAction::Yank);
    }

    fn yank_pop(&mut self) {
        if self.last_action != Some(LastAction::Yank) || self.kill_ring.len() < 2 {
            return;
        }
        let previous = self.kill_ring.peek().unwrap_or_default().to_owned();
        if self.state.cursor < previous.len() {
            return;
        }
        self.push_undo();
        let start = self.state.cursor - previous.len();
        self.state.value.replace_range(start..self.state.cursor, "");
        self.state.cursor = start;
        self.kill_ring.rotate();
        let replacement = self.kill_ring.peek().unwrap_or_default().to_owned();
        self.state.value.insert_str(self.state.cursor, &replacement);
        self.state.cursor += replacement.len();
        self.last_action = Some(LastAction::Yank);
    }

    fn undo(&mut self) {
        if let Some(state) = self.undo.pop() {
            self.state = state;
            self.last_action = None;
        }
    }
}

impl Component for Input {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        let prompt = slice_columns(&self.prompt, 0, context.width, true);
        let available = context.width.saturating_sub(prompt.width);
        if available == 0 {
            return Ok(vec![ConstrainedLine::new(prompt.text, context.width)?]);
        }

        let total_width = visible_width(&self.state.value);
        let cursor_column = visible_width(&self.state.value[..self.state.cursor]);
        let display_width = if self.state.cursor == self.state.value.len() {
            available.saturating_sub(1)
        } else {
            available
        };
        let start_column = if total_width <= display_width {
            0
        } else {
            let half = display_width / 2;
            cursor_column
                .saturating_sub(half)
                .min(total_width.saturating_sub(display_width))
        };
        let visible = slice_columns(&self.state.value, start_column, display_width, true);
        let before = slice_columns(
            &self.state.value,
            start_column,
            cursor_column.saturating_sub(start_column),
            true,
        );
        let cursor_byte = before.text.len().min(visible.text.len());
        let after = &visible.text[cursor_byte..];
        let cursor_end = next_grapheme_boundary(after, 0);
        let at_cursor = if cursor_end == 0 {
            " "
        } else {
            &after[..cursor_end]
        };
        let remainder = if cursor_end == 0 {
            after
        } else {
            &after[cursor_end..]
        };
        let marker = if context.focused || self.focused {
            CURSOR_MARKER
        } else {
            ""
        };
        let mut line = prompt.text;
        line.push_str(&visible.text[..cursor_byte]);
        line.push_str(marker);
        line.push_str(&self.cursor_style.paint(at_cursor));
        line.push_str(remainder);
        let line_width = visible_width(&line);
        line.push_str(&" ".repeat(context.width.saturating_sub(line_width)));
        Ok(vec![ConstrainedLine::new(line, context.width)?])
    }

    fn handle_event(&mut self, event: &InputEvent) -> bool {
        if let InputEvent::Paste(text) = event {
            self.last_action = None;
            self.insert(text);
            return true;
        }
        let InputEvent::Key(key) = event else {
            return false;
        };
        if key.kind == KeyEventKind::Release {
            return false;
        }
        if key.matches("escape") {
            if let Some(callback) = &mut self.on_escape {
                callback();
            }
            return true;
        }
        if key.matches("enter") {
            if let Some(callback) = &mut self.on_submit {
                callback(&self.state.value);
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
        if key.matches("backspace") {
            let start = previous_grapheme_boundary(&self.state.value, self.state.cursor);
            self.delete_range(start, self.state.cursor, None);
            return true;
        }
        if key.matches("delete") {
            let end = next_grapheme_boundary(&self.state.value, self.state.cursor);
            self.delete_range(self.state.cursor, end, None);
            return true;
        }
        if key.matches("ctrl+w") || key.matches("alt+backspace") {
            let start = word_backward(&self.state.value, self.state.cursor);
            self.delete_range(start, self.state.cursor, Some(true));
            return true;
        }
        if key.matches("alt+d") || key.matches("alt+delete") {
            let end = word_forward(&self.state.value, self.state.cursor);
            self.delete_range(self.state.cursor, end, Some(false));
            return true;
        }
        if key.matches("ctrl+u") {
            self.delete_range(0, self.state.cursor, Some(true));
            return true;
        }
        if key.matches("ctrl+k") {
            let end = self.state.value.len();
            self.delete_range(self.state.cursor, end, Some(false));
            return true;
        }
        if matches!(key.code, KeyCode::Home) || key.matches("ctrl+a") {
            self.state.cursor = 0;
            self.last_action = None;
            return true;
        }
        if matches!(key.code, KeyCode::End) || key.matches("ctrl+e") {
            self.state.cursor = self.state.value.len();
            self.last_action = None;
            return true;
        }
        if matches!(key.code, KeyCode::Left) {
            self.state.cursor = if key.modifiers.contains(crate::keys::Modifiers::CTRL)
                || key.modifiers.contains(crate::keys::Modifiers::ALT)
            {
                word_backward(&self.state.value, self.state.cursor)
            } else {
                previous_grapheme_boundary(&self.state.value, self.state.cursor)
            };
            self.last_action = None;
            return true;
        }
        if matches!(key.code, KeyCode::Right) {
            self.state.cursor = if key.modifiers.contains(crate::keys::Modifiers::CTRL)
                || key.modifiers.contains(crate::keys::Modifiers::ALT)
            {
                word_forward(&self.state.value, self.state.cursor)
            } else {
                next_grapheme_boundary(&self.state.value, self.state.cursor)
            };
            self.last_action = None;
            return true;
        }
        if let Some(text) = key.printable() {
            self.insert(text);
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

fn sanitize_single_line(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\t' => output.push_str("    "),
            character if character.is_control() => {}
            character => output.push(character),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::{KeyParseMode, parse_key};

    #[test]
    fn deletes_extended_graphemes_and_undoes() {
        let mut input = Input::new();
        input.handle_event(&InputEvent::Paste("a👨‍👩‍👧‍👦".to_owned()));
        input.handle_event(&InputEvent::Key(
            parse_key("\x7f", KeyParseMode::Legacy).unwrap(),
        ));
        assert_eq!(input.value(), "a");
        input.handle_event(&InputEvent::Key(
            parse_key("\x1a", KeyParseMode::Legacy).unwrap(),
        ));
        assert_eq!(input.value(), "a👨‍👩‍👧‍👦");
    }

    #[test]
    fn horizontally_scrolls_by_terminal_cells() {
        let mut input = Input::new();
        input.set_value("ab界cd界ef");
        let line = input
            .render(RenderContext {
                width: 8,
                height: 1,
                focused: true,
            })
            .unwrap()
            .remove(0);
        assert!(line.width() <= 8);
        assert!(line.as_str().contains(CURSOR_MARKER));
    }
}
