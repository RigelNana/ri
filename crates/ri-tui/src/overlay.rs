//! Responsive overlay layout, compositing, and focus restoration.

use crate::Result;
use crate::ansi::{extract_segments, slice_columns, visible_width};
use crate::component::{Component, ComponentId, InputEvent, RenderContext};

const SEGMENT_RESET: &str = "\x1b[0m\x1b]8;;\x07";

/// Absolute or percentage-based layout value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SizeValue {
    /// Terminal cells.
    Cells(u16),
    /// Hundredths of one percent (`5000` means 50%).
    Percent(u16),
}

impl SizeValue {
    /// Creates a percentage, clamped to 0–100%.
    pub fn percent(value: f32) -> Self {
        // Finite input is clamped to the exact representable basis-point range.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let basis_points = if value.is_finite() {
            (value.clamp(0.0, 100.0) * 100.0).round() as u16
        } else {
            0
        };
        Self::Percent(basis_points)
    }

    fn resolve(self, reference: usize) -> usize {
        match self {
            Self::Cells(cells) => usize::from(cells),
            Self::Percent(basis_points) => {
                reference.saturating_mul(usize::from(basis_points)) / 10_000
            }
        }
    }
}

impl From<u16> for SizeValue {
    fn from(value: u16) -> Self {
        Self::Cells(value)
    }
}

/// Overlay anchor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OverlayAnchor {
    /// Centered both ways.
    #[default]
    Center,
    /// Top left.
    TopLeft,
    /// Top center.
    TopCenter,
    /// Top right.
    TopRight,
    /// Left center.
    LeftCenter,
    /// Right center.
    RightCenter,
    /// Bottom left.
    BottomLeft,
    /// Bottom center.
    BottomCenter,
    /// Bottom right.
    BottomRight,
}

/// Insets from terminal edges.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Margins {
    /// Top rows.
    pub top: u16,
    /// Right columns.
    pub right: u16,
    /// Bottom rows.
    pub bottom: u16,
    /// Left columns.
    pub left: u16,
}

impl Margins {
    /// Equal inset on every edge.
    pub const fn uniform(value: u16) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }
}

/// Terminal-size visibility bounds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResponsiveVisibility {
    /// Hide below this width.
    pub min_width: Option<u16>,
    /// Hide above this width.
    pub max_width: Option<u16>,
    /// Hide below this height.
    pub min_height: Option<u16>,
    /// Hide above this height.
    pub max_height: Option<u16>,
}

impl ResponsiveVisibility {
    fn visible(self, width: u16, height: u16) -> bool {
        self.min_width.is_none_or(|minimum| width >= minimum)
            && self.max_width.is_none_or(|maximum| width <= maximum)
            && self.min_height.is_none_or(|minimum| height >= minimum)
            && self.max_height.is_none_or(|maximum| height <= maximum)
    }
}

/// Overlay sizing, positioning, and focus behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlayOptions {
    /// Width or percentage of terminal width.
    pub width: Option<SizeValue>,
    /// Width floor.
    pub min_width: Option<u16>,
    /// Height cap.
    pub max_height: Option<SizeValue>,
    /// Position anchor.
    pub anchor: OverlayAnchor,
    /// Horizontal anchor offset.
    pub offset_x: i16,
    /// Vertical anchor offset.
    pub offset_y: i16,
    /// Absolute or percentage row; overrides anchor vertically.
    pub row: Option<SizeValue>,
    /// Absolute or percentage column; overrides anchor horizontally.
    pub column: Option<SizeValue>,
    /// Terminal-edge margins.
    pub margins: Margins,
    /// Responsive visibility bounds.
    pub responsive: ResponsiveVisibility,
    /// Capture keyboard focus when visible.
    pub capturing: bool,
}

impl Default for OverlayOptions {
    fn default() -> Self {
        Self {
            width: None,
            min_width: None,
            max_height: None,
            anchor: OverlayAnchor::Center,
            offset_x: 0,
            offset_y: 0,
            row: None,
            column: None,
            margins: Margins::default(),
            responsive: ResponsiveVisibility::default(),
            capturing: true,
        }
    }
}

/// Resolved overlay rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OverlayPosition {
    /// Viewport row.
    pub row: usize,
    /// Viewport column.
    pub column: usize,
    /// Width.
    pub width: usize,
    /// Effective height.
    pub height: usize,
}

/// Handle identity for a mounted overlay.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OverlayId(ComponentId);

impl OverlayId {
    /// Component identity used for focus routing.
    pub const fn component_id(self) -> ComponentId {
        self.0
    }
}

struct OverlayEntry {
    id: OverlayId,
    component: Box<dyn Component>,
    options: OverlayOptions,
    pre_focus: Option<ComponentId>,
    hidden: bool,
    focus_order: u64,
    was_visible: bool,
}

struct RenderedOverlay {
    lines: Vec<String>,
    position: OverlayPosition,
}

/// Overlay stack used internally by [`crate::Tui`].
#[derive(Default)]
pub(crate) struct OverlayManager {
    entries: Vec<OverlayEntry>,
    focus_counter: u64,
}

impl std::fmt::Debug for OverlayManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OverlayManager")
            .field(
                "entries",
                &self
                    .entries
                    .iter()
                    .map(|entry| entry.id)
                    .collect::<Vec<_>>(),
            )
            .field("focus_counter", &self.focus_counter)
            .finish()
    }
}

impl OverlayManager {
    pub(crate) fn show(
        &mut self,
        component: Box<dyn Component>,
        options: OverlayOptions,
        current_focus: Option<ComponentId>,
        size: (u16, u16),
    ) -> (OverlayId, Option<ComponentId>) {
        self.focus_counter += 1;
        let id = OverlayId(ComponentId::allocate());
        let visible = options.responsive.visible(size.0, size.1);
        let capturing = options.capturing;
        self.entries.push(OverlayEntry {
            id,
            component,
            options,
            pre_focus: current_focus,
            hidden: false,
            focus_order: self.focus_counter,
            was_visible: visible,
        });
        (
            id,
            if visible && capturing {
                Some(id.0)
            } else {
                current_focus
            },
        )
    }

    pub(crate) fn remove(
        &mut self,
        id: OverlayId,
        focused: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        let index = self.entries.iter().position(|entry| entry.id == id)?;
        let entry = self.entries.remove(index);
        for other in &mut self.entries {
            if other.pre_focus == Some(id.0) {
                other.pre_focus = entry.pre_focus;
            }
        }
        if focused == Some(id.0) {
            self.top_capturing(size).map_or_else(
                || self.restore_target(entry.pre_focus, size),
                |entry| Some(entry.id.0),
            )
        } else {
            focused
        }
    }

    pub(crate) fn set_hidden(
        &mut self,
        id: OverlayId,
        hidden: bool,
        focused: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return focused;
        };
        if self.entries[index].hidden == hidden {
            return focused;
        }
        self.entries[index].hidden = hidden;
        if hidden && focused == Some(id.0) {
            let pre_focus = self.entries[index].pre_focus;
            return self.top_capturing(size).map_or_else(
                || self.restore_target(pre_focus, size),
                |entry| Some(entry.id.0),
            );
        }
        if !hidden && self.entries[index].options.capturing && self.visible_at(index, size) {
            self.focus_counter += 1;
            self.entries[index].focus_order = self.focus_counter;
            return Some(id.0);
        }
        focused
    }

    pub(crate) fn focus(
        &mut self,
        id: OverlayId,
        focused: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return focused;
        };
        if !self.visible_at(index, size) {
            return focused;
        }
        self.focus_counter += 1;
        self.entries[index].focus_order = self.focus_counter;
        Some(id.0)
    }

    pub(crate) fn unfocus(
        &self,
        id: OverlayId,
        focused: Option<ComponentId>,
        explicit: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        if focused != Some(id.0) && explicit.is_none() {
            return focused;
        }
        if let Some(target) = explicit {
            return Some(target);
        }
        let entry = self.entries.iter().find(|entry| entry.id == id);
        self.top_capturing_except(id, size).map_or_else(
            || self.restore_target(entry.and_then(|entry| entry.pre_focus), size),
            |entry| Some(entry.id.0),
        )
    }

    pub(crate) fn reconcile_focus(
        &mut self,
        focused: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        let mut focus = focused;
        for index in 0..self.entries.len() {
            let visible = self.visible_at(index, size);
            let was_visible = self.entries[index].was_visible;
            self.entries[index].was_visible = visible;
            if was_visible && !visible && focus == Some(self.entries[index].id.0) {
                focus = self.top_capturing(size).map_or_else(
                    || self.restore_target(self.entries[index].pre_focus, size),
                    |entry| Some(entry.id.0),
                );
            } else if !was_visible && visible && self.entries[index].options.capturing {
                self.focus_counter += 1;
                self.entries[index].focus_order = self.focus_counter;
                focus = Some(self.entries[index].id.0);
            }
        }
        focus
    }

    pub(crate) fn contains(&self, id: ComponentId) -> bool {
        self.entries.iter().any(|entry| entry.id.0 == id)
    }

    pub(crate) fn component_mut(&mut self, id: ComponentId) -> Option<&mut (dyn Component + '_)> {
        for entry in &mut self.entries {
            if entry.id.0 == id {
                return Some(entry.component.as_mut());
            }
        }
        None
    }

    pub(crate) fn set_focused(&mut self, id: ComponentId, focused: bool) {
        if let Some(component) = self.component_mut(id) {
            component.set_focused(focused);
        }
    }

    pub(crate) fn dispatch(&mut self, id: ComponentId, event: &InputEvent) -> bool {
        self.component_mut(id)
            .is_some_and(|component| component.handle_event(event))
    }

    pub(crate) fn invalidate(&mut self) {
        for entry in &mut self.entries {
            entry.component.invalidate();
        }
    }

    pub(crate) fn composite(
        &mut self,
        mut base: Vec<String>,
        terminal_width: usize,
        terminal_height: usize,
        focused: Option<ComponentId>,
    ) -> Result<Vec<String>> {
        let size = (
            u16::try_from(terminal_width).unwrap_or(u16::MAX),
            u16::try_from(terminal_height).unwrap_or(u16::MAX),
        );
        let mut indices = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, _)| self.visible_at(index, size).then_some(index))
            .collect::<Vec<_>>();
        indices.sort_by_key(|index| self.entries[*index].focus_order);

        let mut rendered = Vec::new();
        let mut minimum_lines = base.len();
        for index in indices {
            let options = self.entries[index].options.clone();
            let preliminary =
                resolve_overlay_layout(&options, terminal_height, terminal_width, terminal_height);
            let overlay_id = self.entries[index].id.0;
            let mut lines = self.entries[index]
                .component
                .render(RenderContext {
                    width: preliminary.width,
                    height: preliminary.height,
                    focused: focused == Some(overlay_id),
                })?
                .into_iter()
                .map(crate::line::ConstrainedLine::into_string)
                .collect::<Vec<_>>();
            if let Some(maximum) = options
                .max_height
                .map(|value| value.resolve(terminal_height).max(1))
            {
                lines.truncate(maximum);
            }
            let position =
                resolve_overlay_layout(&options, lines.len(), terminal_width, terminal_height);
            minimum_lines = minimum_lines.max(position.row + lines.len());
            rendered.push(RenderedOverlay { lines, position });
        }
        if rendered.is_empty() {
            return Ok(base);
        }

        let working_height = base.len().max(terminal_height).max(minimum_lines);
        base.resize(working_height, String::new());
        let viewport_start = working_height.saturating_sub(terminal_height);
        for overlay in rendered {
            for (line_index, line) in overlay.lines.iter().enumerate() {
                let target = viewport_start + overlay.position.row + line_index;
                if target >= base.len() {
                    continue;
                }
                let clipped = if visible_width(line) > overlay.position.width {
                    slice_columns(line, 0, overlay.position.width, true).text
                } else {
                    line.clone()
                };
                base[target] = composite_line(
                    &base[target],
                    &clipped,
                    overlay.position.column,
                    overlay.position.width,
                    terminal_width,
                );
            }
        }
        Ok(base)
    }

    fn visible_at(&self, index: usize, size: (u16, u16)) -> bool {
        let entry = &self.entries[index];
        !entry.hidden && entry.options.responsive.visible(size.0, size.1)
    }

    fn top_capturing(&self, size: (u16, u16)) -> Option<&OverlayEntry> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(index, entry)| entry.options.capturing && self.visible_at(*index, size))
            .map(|(_, entry)| entry)
            .max_by_key(|entry| entry.focus_order)
    }

    fn top_capturing_except(&self, excluded: OverlayId, size: (u16, u16)) -> Option<&OverlayEntry> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(index, entry)| {
                entry.id != excluded && entry.options.capturing && self.visible_at(*index, size)
            })
            .map(|(_, entry)| entry)
            .max_by_key(|entry| entry.focus_order)
    }

    fn restore_target(
        &self,
        mut candidate: Option<ComponentId>,
        size: (u16, u16),
    ) -> Option<ComponentId> {
        let mut remaining = self.entries.len();
        while let Some(id) = candidate {
            let Some((index, entry)) = self
                .entries
                .iter()
                .enumerate()
                .find(|(_, entry)| entry.id.0 == id)
            else {
                return Some(id);
            };
            if self.visible_at(index, size) {
                return Some(id);
            }
            candidate = entry.pre_focus;
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
        }
        None
    }
}

/// Resolves an overlay rectangle in viewport coordinates.
pub fn resolve_overlay_layout(
    options: &OverlayOptions,
    content_height: usize,
    terminal_width: usize,
    terminal_height: usize,
) -> OverlayPosition {
    let left = usize::from(options.margins.left).min(terminal_width);
    let right = usize::from(options.margins.right).min(terminal_width.saturating_sub(left));
    let top = usize::from(options.margins.top).min(terminal_height);
    let bottom = usize::from(options.margins.bottom).min(terminal_height.saturating_sub(top));
    let available_width = terminal_width.saturating_sub(left + right).max(1);
    let available_height = terminal_height.saturating_sub(top + bottom).max(1);

    let mut width = options.width.map_or(available_width.min(80), |value| {
        value.resolve(terminal_width)
    });
    width = width.max(options.min_width.map_or(1, usize::from));
    width = width.clamp(1, available_width);

    let height_limit = options
        .max_height
        .map_or(available_height, |value| {
            value.resolve(terminal_height).max(1)
        })
        .min(available_height);
    let height = content_height.min(height_limit);
    let effective_height = height.max(1);

    let anchored_row = match options.anchor {
        OverlayAnchor::TopLeft | OverlayAnchor::TopCenter | OverlayAnchor::TopRight => top,
        OverlayAnchor::BottomLeft | OverlayAnchor::BottomCenter | OverlayAnchor::BottomRight => {
            top + available_height.saturating_sub(effective_height)
        }
        OverlayAnchor::Center | OverlayAnchor::LeftCenter | OverlayAnchor::RightCenter => {
            top + available_height.saturating_sub(effective_height) / 2
        }
    };
    let anchored_column = match options.anchor {
        OverlayAnchor::TopLeft | OverlayAnchor::LeftCenter | OverlayAnchor::BottomLeft => left,
        OverlayAnchor::TopRight | OverlayAnchor::RightCenter | OverlayAnchor::BottomRight => {
            left + available_width.saturating_sub(width)
        }
        OverlayAnchor::Center | OverlayAnchor::TopCenter | OverlayAnchor::BottomCenter => {
            left + available_width.saturating_sub(width) / 2
        }
    };

    let row = options.row.map_or(anchored_row, |value| match value {
        SizeValue::Cells(value) => usize::from(value),
        SizeValue::Percent(_) => {
            top + value.resolve(available_height.saturating_sub(effective_height))
        }
    });
    let column = options.column.map_or(anchored_column, |value| match value {
        SizeValue::Cells(value) => usize::from(value),
        SizeValue::Percent(_) => left + value.resolve(available_width.saturating_sub(width)),
    });
    let row = apply_offset(row, options.offset_y).clamp(
        top,
        terminal_height
            .saturating_sub(bottom + effective_height)
            .max(top),
    );
    let column = apply_offset(column, options.offset_x)
        .clamp(left, terminal_width.saturating_sub(right + width).max(left));

    OverlayPosition {
        row,
        column,
        width,
        height,
    }
}

fn apply_offset(value: usize, offset: i16) -> usize {
    if offset >= 0 {
        value.saturating_add(offset.unsigned_abs().into())
    } else {
        value.saturating_sub(offset.unsigned_abs().into())
    }
}

/// Replaces a cell range in a base line without splitting wide graphemes.
pub fn composite_line(
    base: &str,
    overlay: &str,
    start_column: usize,
    overlay_width: usize,
    total_width: usize,
) -> String {
    let after_start = start_column.saturating_add(overlay_width);
    let base_segments = extract_segments(
        base,
        start_column,
        after_start,
        total_width.saturating_sub(after_start),
        true,
    );
    let overlay = slice_columns(overlay, 0, overlay_width, true);
    let before_padding = start_column.saturating_sub(base_segments.before_width);
    let overlay_padding = overlay_width.saturating_sub(overlay.width);
    let after_target = total_width.saturating_sub(start_column + overlay_width);
    let after_padding = after_target.saturating_sub(base_segments.after_width);

    let mut output = String::new();
    output.push_str(&base_segments.before);
    output.push_str(&" ".repeat(before_padding));
    output.push_str(SEGMENT_RESET);
    output.push_str(&overlay.text);
    output.push_str(&" ".repeat(overlay_padding));
    output.push_str(SEGMENT_RESET);
    output.push_str(&base_segments.after);
    output.push_str(&" ".repeat(after_padding));
    if visible_width(&output) > total_width {
        slice_columns(&output, 0, total_width, true).text
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_percent_and_margins() {
        let options = OverlayOptions {
            width: Some(SizeValue::percent(50.0)),
            row: Some(SizeValue::percent(100.0)),
            column: Some(SizeValue::percent(100.0)),
            margins: Margins::uniform(2),
            ..OverlayOptions::default()
        };
        assert_eq!(
            resolve_overlay_layout(&options, 3, 100, 30),
            OverlayPosition {
                row: 25,
                column: 48,
                width: 50,
                height: 3
            }
        );
    }

    #[test]
    fn composition_excludes_cjk_crossing_boundary() {
        let output = composite_line("abcd让EFGH", "│XX│", 5, 4, 20);
        assert!(!output.contains('让'));
        assert_eq!(visible_width(&output), 20);
        assert!(slice_columns(&output, 5, 4, true).text.contains("│XX│"));
    }
}
