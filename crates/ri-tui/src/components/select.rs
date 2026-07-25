use crate::Result;
use crate::ansi::{truncate_to_width, visible_width};
use crate::component::{Component, InputEvent, RenderContext};
use crate::keys::KeyCode;
use crate::line::ConstrainedLine;
use crate::theme::SelectTheme;

/// Selection-list item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectItem {
    /// Application value.
    pub value: String,
    /// Primary label.
    pub label: String,
    /// Optional secondary description.
    pub description: Option<String>,
}

impl SelectItem {
    /// Creates an item with equal value and label.
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        Self {
            label: value.clone(),
            value,
            description: None,
        }
    }

    /// Adds a display description.
    #[must_use]
    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

type SelectCallback = Box<dyn FnMut(&SelectItem) + Send>;

/// Filterable keyboard-driven list.
pub struct SelectList {
    items: Vec<SelectItem>,
    filtered: Vec<usize>,
    selected: usize,
    max_visible: usize,
    theme: SelectTheme,
    on_select: Option<SelectCallback>,
    on_cancel: Option<Box<dyn FnMut() + Send>>,
}

impl std::fmt::Debug for SelectList {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectList")
            .field("items", &self.items)
            .field("filtered", &self.filtered)
            .field("selected", &self.selected)
            .field("max_visible", &self.max_visible)
            .finish_non_exhaustive()
    }
}

impl SelectList {
    /// Creates a selection list.
    pub fn new(
        items: impl IntoIterator<Item = SelectItem>,
        max_visible: usize,
        theme: SelectTheme,
    ) -> Self {
        let items = items.into_iter().collect::<Vec<_>>();
        let filtered = (0..items.len()).collect();
        Self {
            items,
            filtered,
            selected: 0,
            max_visible: max_visible.max(1),
            theme,
            on_select: None,
            on_cancel: None,
        }
    }

    /// Filters by a case-insensitive value prefix.
    pub fn set_filter(&mut self, filter: &str) {
        let filter = filter.to_ascii_lowercase();
        self.filtered = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                item.value
                    .to_ascii_lowercase()
                    .starts_with(&filter)
                    .then_some(index)
            })
            .collect();
        self.selected = 0;
    }

    /// Selects a filtered index, clamped to available items.
    pub fn set_selected_index(&mut self, index: usize) {
        self.selected = index.min(self.filtered.len().saturating_sub(1));
    }

    /// Current item.
    pub fn selected_item(&self) -> Option<&SelectItem> {
        self.filtered
            .get(self.selected)
            .and_then(|index| self.items.get(*index))
    }

    /// Registers a confirmation callback.
    pub fn on_select(&mut self, callback: impl FnMut(&SelectItem) + Send + 'static) {
        self.on_select = Some(Box::new(callback));
    }

    /// Registers a cancellation callback.
    pub fn on_cancel(&mut self, callback: impl FnMut() + Send + 'static) {
        self.on_cancel = Some(Box::new(callback));
    }

    fn move_selection(&mut self, delta: i32) {
        if self.filtered.is_empty() {
            return;
        }
        self.selected = if delta < 0 {
            self.selected
                .checked_sub(1)
                .unwrap_or(self.filtered.len() - 1)
        } else {
            (self.selected + 1) % self.filtered.len()
        };
    }
}

impl Component for SelectList {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        if self.filtered.is_empty() {
            return Ok(vec![ConstrainedLine::new(
                self.theme.no_match.paint(truncate_to_width(
                    "  No matching items",
                    context.width,
                    "",
                    false,
                )),
                context.width,
            )?]);
        }

        let start = self
            .selected
            .saturating_sub(self.max_visible / 2)
            .min(self.filtered.len().saturating_sub(self.max_visible));
        let end = (start + self.max_visible).min(self.filtered.len());
        let primary_width = self.filtered[start..end]
            .iter()
            .filter_map(|index| self.items.get(*index))
            .map(|item| visible_width(&item.label))
            .max()
            .unwrap_or(1)
            .clamp(1, 32);
        let mut lines = Vec::new();
        for filtered_index in start..end {
            let item = &self.items[self.filtered[filtered_index]];
            let selected = filtered_index == self.selected;
            let prefix = if selected {
                self.theme.selected_prefix.as_str()
            } else {
                "  "
            };
            let prefix_width = visible_width(prefix);
            let mut line = String::from(prefix);
            let available = context.width.saturating_sub(prefix_width);
            if let Some(description) = &item.description {
                if context.width > 40 && available > primary_width + 3 {
                    let label = truncate_to_width(&item.label, primary_width, "", true);
                    let description_width = available.saturating_sub(primary_width + 2);
                    line.push_str(&label);
                    line.push_str("  ");
                    line.push_str(&self.theme.description.paint(truncate_to_width(
                        &description.replace(['\r', '\n'], " "),
                        description_width,
                        "",
                        false,
                    )));
                } else {
                    line.push_str(&truncate_to_width(&item.label, available, "", false));
                }
            } else {
                line.push_str(&truncate_to_width(&item.label, available, "", false));
            }
            let line = if selected {
                self.theme.selected.paint(line)
            } else {
                line
            };
            lines.push(ConstrainedLine::new(line, context.width)?);
        }
        if start > 0 || end < self.filtered.len() {
            let status = format!("  ({}/{})", self.selected + 1, self.filtered.len());
            lines.push(ConstrainedLine::new(
                self.theme
                    .scroll_info
                    .paint(truncate_to_width(&status, context.width, "", false)),
                context.width,
            )?);
        }
        Ok(lines)
    }

    fn handle_event(&mut self, event: &InputEvent) -> bool {
        let InputEvent::Key(key) = event else {
            return false;
        };
        match key.code {
            KeyCode::Up => {
                self.move_selection(-1);
                true
            }
            KeyCode::Down => {
                self.move_selection(1);
                true
            }
            KeyCode::Enter => {
                let selected = self.selected_item().cloned();
                if let (Some(item), Some(callback)) = (selected.as_ref(), self.on_select.as_mut()) {
                    callback(item);
                }
                true
            }
            KeyCode::Escape => {
                if let Some(callback) = &mut self.on_cancel {
                    callback();
                }
                true
            }
            _ => false,
        }
    }

    fn focusable(&self) -> bool {
        true
    }
}
