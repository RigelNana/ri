use crate::Result;
use crate::ansi::{truncate_to_width, visible_width, wrap_text_ansi};
use crate::component::{Component, RenderContext};
use crate::line::ConstrainedLine;
use crate::theme::Style;

/// Wrapped multi-line text.
#[derive(Clone, Debug)]
pub struct Text {
    content: String,
    padding_x: usize,
    padding_y: usize,
    style: Style,
    cache: Option<(usize, String, Vec<ConstrainedLine>)>,
}

impl Text {
    /// Creates unpadded text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: text.into(),
            padding_x: 0,
            padding_y: 0,
            style: Style::default(),
            cache: None,
        }
    }

    /// Sets horizontal and vertical padding.
    #[must_use]
    pub fn with_padding(mut self, horizontal: usize, vertical: usize) -> Self {
        self.padding_x = horizontal;
        self.padding_y = vertical;
        self.cache = None;
        self
    }

    /// Sets the line style.
    #[must_use]
    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self.cache = None;
        self
    }

    /// Replaces text.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.content = text.into();
        self.cache = None;
    }

    /// Current source text.
    pub fn text(&self) -> &str {
        &self.content
    }
}

impl Component for Text {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        if let Some((_, _, lines)) = self
            .cache
            .as_ref()
            .filter(|(width, text, _)| *width == context.width && text == &self.content)
        {
            return Ok(lines.clone());
        }
        let padding_x = self.padding_x.min(context.width / 2);
        let content_width = context.width.saturating_sub(padding_x * 2).max(1);
        let margin = " ".repeat(padding_x);
        let mut lines = Vec::new();
        for _ in 0..self.padding_y {
            lines.push(ConstrainedLine::new(
                self.style.paint(" ".repeat(context.width)),
                context.width,
            )?);
        }
        for wrapped in wrap_text_ansi(&self.content, content_width) {
            let wrapped_width = visible_width(&wrapped);
            let padding = " ".repeat(content_width.saturating_sub(wrapped_width));
            let full = format!("{margin}{wrapped}{padding}{margin}");
            lines.push(ConstrainedLine::new(self.style.paint(full), context.width)?);
        }
        for _ in 0..self.padding_y {
            lines.push(ConstrainedLine::new(
                self.style.paint(" ".repeat(context.width)),
                context.width,
            )?);
        }
        self.cache = Some((context.width, self.content.clone(), lines.clone()));
        Ok(lines)
    }

    fn invalidate(&mut self) {
        self.cache = None;
    }
}

/// Single-line text clipped to the viewport.
#[derive(Clone, Debug)]
pub struct TruncatedText {
    text: String,
    ellipsis: String,
    padding_x: usize,
    style: Style,
}

impl TruncatedText {
    /// Creates a line using `...` when clipped.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ellipsis: "...".to_owned(),
            padding_x: 0,
            style: Style::default(),
        }
    }

    /// Changes the clipping marker.
    #[must_use]
    pub fn with_ellipsis(mut self, ellipsis: impl Into<String>) -> Self {
        self.ellipsis = ellipsis.into();
        self
    }

    /// Sets horizontal padding.
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding_x = padding;
        self
    }

    /// Sets style.
    #[must_use]
    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

impl Component for TruncatedText {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        let padding = self.padding_x.min(context.width / 2);
        let content_width = context.width.saturating_sub(padding * 2);
        let clipped = truncate_to_width(&self.text, content_width, &self.ellipsis, true);
        let line = format!("{}{}{}", " ".repeat(padding), clipped, " ".repeat(padding));
        Ok(vec![ConstrainedLine::new(
            self.style.paint(line),
            context.width,
        )?])
    }
}

/// Fixed vertical spacing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spacer {
    lines: usize,
}

impl Spacer {
    /// Creates the requested number of empty lines.
    pub const fn new(lines: usize) -> Self {
        Self { lines }
    }
}

impl Default for Spacer {
    fn default() -> Self {
        Self { lines: 1 }
    }
}

impl Component for Spacer {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        Ok((0..self.lines)
            .map(|_| ConstrainedLine::empty(context.width))
            .collect())
    }
}
