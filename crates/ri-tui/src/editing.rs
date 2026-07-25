//! Reusable grapheme navigation, word movement, undo, and kill-ring state.

use std::collections::VecDeque;

use unicode_segmentation::UnicodeSegmentation;

/// Returns the preceding extended-grapheme boundary.
pub fn previous_grapheme_boundary(text: &str, cursor: usize) -> usize {
    let cursor = floor_char_boundary(text, cursor.min(text.len()));
    text[..cursor]
        .grapheme_indices(true)
        .next_back()
        .map_or(0, |(index, _)| index)
}

/// Returns the next extended-grapheme boundary.
pub fn next_grapheme_boundary(text: &str, cursor: usize) -> usize {
    let cursor = ceil_char_boundary(text, cursor.min(text.len()));
    text[cursor..]
        .graphemes(true)
        .next()
        .map_or(text.len(), |grapheme| cursor + grapheme.len())
}

/// Finds the previous word or punctuation boundary.
pub fn word_backward(text: &str, cursor: usize) -> usize {
    let mut cursor = floor_char_boundary(text, cursor.min(text.len()));
    while cursor > 0 {
        let previous = previous_grapheme_boundary(text, cursor);
        if !text[previous..cursor].chars().all(char::is_whitespace) {
            break;
        }
        cursor = previous;
    }
    if cursor == 0 {
        return 0;
    }

    let previous = previous_grapheme_boundary(text, cursor);
    let punctuation = text[previous..cursor].chars().all(is_punctuation);
    cursor = previous;
    while cursor > 0 {
        let prior = previous_grapheme_boundary(text, cursor);
        let grapheme = &text[prior..cursor];
        if grapheme.chars().all(char::is_whitespace)
            || grapheme.chars().all(is_punctuation) != punctuation
        {
            break;
        }
        cursor = prior;
    }
    cursor
}

/// Finds the next word or punctuation boundary.
pub fn word_forward(text: &str, cursor: usize) -> usize {
    let mut cursor = ceil_char_boundary(text, cursor.min(text.len()));
    while cursor < text.len() {
        let next = next_grapheme_boundary(text, cursor);
        if !text[cursor..next].chars().all(char::is_whitespace) {
            break;
        }
        cursor = next;
    }
    if cursor == text.len() {
        return cursor;
    }

    let next = next_grapheme_boundary(text, cursor);
    let punctuation = text[cursor..next].chars().all(is_punctuation);
    cursor = next;
    while cursor < text.len() {
        let following = next_grapheme_boundary(text, cursor);
        let grapheme = &text[cursor..following];
        if grapheme.chars().all(char::is_whitespace)
            || grapheme.chars().all(is_punctuation) != punctuation
        {
            break;
        }
        cursor = following;
    }
    cursor
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn is_punctuation(character: char) -> bool {
    "(){}[]<>.,;:'\"!?+-=*/\\|&%^$#@~`".contains(character)
}

/// Clone-on-push undo history with a bounded memory footprint.
#[derive(Clone, Debug)]
pub struct UndoStack<T> {
    entries: VecDeque<T>,
    capacity: usize,
}

impl<T> Default for UndoStack<T> {
    fn default() -> Self {
        Self::new(256)
    }
}

impl<T> UndoStack<T> {
    /// Creates a bounded stack. A zero capacity disables snapshots.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Removes all snapshots.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of available snapshots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no snapshot is available.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pops the most recent detached snapshot.
    pub fn pop(&mut self) -> Option<T> {
        self.entries.pop_back()
    }
}

impl<T: Clone> UndoStack<T> {
    /// Stores a clone of `state`.
    pub fn push(&mut self, state: &T) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(state.clone());
    }
}

/// Emacs-style ring of killed text.
#[derive(Clone, Debug)]
pub struct KillRing {
    entries: VecDeque<String>,
    capacity: usize,
}

impl Default for KillRing {
    fn default() -> Self {
        Self::new(60)
    }
}

impl KillRing {
    /// Creates a ring with the given entry capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds killed text, optionally joining the newest entry.
    pub fn push(&mut self, text: impl Into<String>, prepend: bool, accumulate: bool) {
        let text = text.into();
        if text.is_empty() || self.capacity == 0 {
            return;
        }
        let latest = if accumulate {
            self.entries.back_mut()
        } else {
            None
        };
        if let Some(latest) = latest {
            if prepend {
                latest.insert_str(0, &text);
            } else {
                latest.push_str(&text);
            }
            return;
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(text);
    }

    /// Most recent killed text.
    pub fn peek(&self) -> Option<&str> {
        self.entries.back().map(String::as_str)
    }

    /// Cycles the newest entry to the oldest position.
    pub fn rotate(&mut self) {
        if self.entries.len() <= 1 {
            return;
        }
        if let Some(entry) = self.entries.pop_back() {
            self.entries.push_front(entry);
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grapheme_navigation_keeps_emoji_atomic() {
        let text = "a👨‍👩‍👧‍👦b";
        let after_emoji = text.len() - 1;
        assert_eq!(previous_grapheme_boundary(text, after_emoji), 1);
        assert_eq!(next_grapheme_boundary(text, 1), after_emoji);
    }

    #[test]
    fn word_navigation_distinguishes_punctuation() {
        assert_eq!(word_backward("hello.world", 11), 6);
        assert_eq!(word_backward("hello.world", 6), 5);
        assert_eq!(word_forward("hello.world", 0), 5);
        assert_eq!(word_forward("hello.world", 5), 6);
    }

    #[test]
    fn kill_ring_accumulates_in_both_directions() {
        let mut ring = KillRing::default();
        ring.push("world", false, false);
        ring.push("hello ", true, true);
        assert_eq!(ring.peek(), Some("hello world"));
    }
}
