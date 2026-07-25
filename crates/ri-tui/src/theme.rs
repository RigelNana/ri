//! Typed terminal themes.

/// A terminal foreground or background color.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    /// Terminal default.
    Default,
    /// One of the basic ANSI colors, numbered 0 through 15.
    Ansi(u8),
    /// A 256-color palette index.
    Indexed(u8),
    /// A 24-bit color.
    Rgb(u8, u8, u8),
}

impl Color {
    fn foreground(self) -> String {
        match self {
            Self::Default => "39".to_owned(),
            Self::Ansi(index @ 0..=7) => (30 + index).to_string(),
            Self::Ansi(index @ 8..=15) => (90 + index - 8).to_string(),
            Self::Ansi(index) | Self::Indexed(index) => format!("38;5;{index}"),
            Self::Rgb(red, green, blue) => format!("38;2;{red};{green};{blue}"),
        }
    }

    fn background(self) -> String {
        match self {
            Self::Default => "49".to_owned(),
            Self::Ansi(index @ 0..=7) => (40 + index).to_string(),
            Self::Ansi(index @ 8..=15) => (100 + index - 8).to_string(),
            Self::Ansi(index) | Self::Indexed(index) => format!("48;5;{index}"),
            Self::Rgb(red, green, blue) => format!("48;2;{red};{green};{blue}"),
        }
    }
}

/// Composable SGR text style.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
// These independent booleans intentionally mirror orthogonal SGR attributes
// and preserve the ergonomic public struct-literal API.
#[allow(clippy::struct_excessive_bools)]
pub struct Style {
    /// Foreground color.
    pub foreground: Option<Color>,
    /// Background color.
    pub background: Option<Color>,
    /// Bold text.
    pub bold: bool,
    /// Dim text.
    pub dim: bool,
    /// Italic text.
    pub italic: bool,
    /// Underlined text.
    pub underline: bool,
    /// Struck-through text.
    pub strikethrough: bool,
    /// Inverse video.
    pub inverse: bool,
}

impl Style {
    /// Styles text and restores all SGR attributes afterward.
    pub fn paint(self, text: impl AsRef<str>) -> String {
        let mut codes = Vec::new();
        if self.bold {
            codes.push("1".to_owned());
        }
        if self.dim {
            codes.push("2".to_owned());
        }
        if self.italic {
            codes.push("3".to_owned());
        }
        if self.underline {
            codes.push("4".to_owned());
        }
        if self.inverse {
            codes.push("7".to_owned());
        }
        if self.strikethrough {
            codes.push("9".to_owned());
        }
        if let Some(color) = self.foreground {
            codes.push(color.foreground());
        }
        if let Some(color) = self.background {
            codes.push(color.background());
        }
        if codes.is_empty() {
            return text.as_ref().to_owned();
        }
        format!("\x1b[{}m{}\x1b[0m", codes.join(";"), text.as_ref())
    }
}

/// Editor-specific theme.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditorTheme {
    /// Horizontal border.
    pub border: Style,
    /// Fake cursor cell.
    pub cursor: Style,
    /// Scroll indicators.
    pub scroll_indicator: Style,
    /// Autocomplete theme.
    pub autocomplete: SelectTheme,
}

impl Default for EditorTheme {
    fn default() -> Self {
        Self {
            border: Style {
                foreground: Some(Color::Ansi(8)),
                ..Style::default()
            },
            cursor: Style {
                inverse: true,
                ..Style::default()
            },
            scroll_indicator: Style {
                foreground: Some(Color::Ansi(8)),
                dim: true,
                ..Style::default()
            },
            autocomplete: SelectTheme::default(),
        }
    }
}

/// Selection-list theme.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectTheme {
    /// Selected row.
    pub selected: Style,
    /// Secondary description.
    pub description: Style,
    /// Scroll status.
    pub scroll_info: Style,
    /// Empty-result message.
    pub no_match: Style,
    /// Prefix displayed for a selected row.
    pub selected_prefix: String,
}

impl Default for SelectTheme {
    fn default() -> Self {
        Self {
            selected: Style {
                inverse: true,
                ..Style::default()
            },
            description: Style {
                foreground: Some(Color::Ansi(8)),
                ..Style::default()
            },
            scroll_info: Style {
                dim: true,
                ..Style::default()
            },
            no_match: Style {
                dim: true,
                ..Style::default()
            },
            selected_prefix: "→ ".to_owned(),
        }
    }
}

/// Markdown element styles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkdownTheme {
    /// Headings.
    pub heading: Style,
    /// Links.
    pub link: Style,
    /// Inline code.
    pub code: Style,
    /// Fenced code blocks.
    pub code_block: Style,
    /// Blockquote text.
    pub quote: Style,
    /// Blockquote marker.
    pub quote_border: Style,
    /// List marker.
    pub list_bullet: Style,
    /// Horizontal rule.
    pub horizontal_rule: Style,
    /// Strong emphasis.
    pub strong: Style,
    /// Emphasis.
    pub emphasis: Style,
    /// Deleted text.
    pub deleted: Style,
}

impl Default for MarkdownTheme {
    fn default() -> Self {
        Self {
            heading: Style {
                bold: true,
                foreground: Some(Color::Ansi(14)),
                ..Style::default()
            },
            link: Style {
                underline: true,
                foreground: Some(Color::Ansi(12)),
                ..Style::default()
            },
            code: Style {
                foreground: Some(Color::Ansi(11)),
                ..Style::default()
            },
            code_block: Style {
                foreground: Some(Color::Ansi(10)),
                ..Style::default()
            },
            quote: Style {
                italic: true,
                foreground: Some(Color::Ansi(8)),
                ..Style::default()
            },
            quote_border: Style {
                foreground: Some(Color::Ansi(8)),
                ..Style::default()
            },
            list_bullet: Style {
                foreground: Some(Color::Ansi(14)),
                ..Style::default()
            },
            horizontal_rule: Style {
                foreground: Some(Color::Ansi(8)),
                ..Style::default()
            },
            strong: Style {
                bold: true,
                ..Style::default()
            },
            emphasis: Style {
                italic: true,
                ..Style::default()
            },
            deleted: Style {
                strikethrough: true,
                ..Style::default()
            },
        }
    }
}

/// Complete library theme.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Theme {
    /// Editor styles.
    pub editor: EditorTheme,
    /// Markdown styles.
    pub markdown: MarkdownTheme,
    /// Selection-list styles.
    pub select: SelectTheme,
}
