use std::fmt::Write as _;

use crate::Result;
use crate::ansi::{visible_width, wrap_text_ansi};
use crate::component::{Component, RenderContext};
use crate::line::ConstrainedLine;
use crate::theme::MarkdownTheme;

/// ANSI markdown renderer with cell-aware wrapping.
#[derive(Clone, Debug)]
pub struct Markdown {
    source: String,
    padding_x: usize,
    padding_y: usize,
    theme: MarkdownTheme,
    cache: Option<(usize, String, Vec<ConstrainedLine>)>,
}

impl Markdown {
    /// Creates markdown content.
    pub fn new(source: impl Into<String>, theme: MarkdownTheme) -> Self {
        Self {
            source: source.into(),
            padding_x: 0,
            padding_y: 0,
            theme,
            cache: None,
        }
    }

    /// Adds horizontal and vertical padding.
    #[must_use]
    pub fn with_padding(mut self, horizontal: usize, vertical: usize) -> Self {
        self.padding_x = horizontal;
        self.padding_y = vertical;
        self
    }

    /// Replaces source.
    pub fn set_source(&mut self, source: impl Into<String>) {
        self.source = source.into();
        self.cache = None;
    }

    /// Returns markdown source.
    pub fn source(&self) -> &str {
        &self.source
    }

    fn render_blocks(&self, width: usize) -> Vec<String> {
        let normalized = self.source.replace("\r\n", "\n").replace('\r', "\n");
        let source_lines = normalized.lines().collect::<Vec<_>>();
        let mut output = Vec::new();
        let mut index = 0;
        let mut in_fence = false;
        let mut fence_language = String::new();
        let mut code: Vec<String> = Vec::new();

        while index < source_lines.len() {
            let line = source_lines[index];
            let trimmed = line.trim();
            if let Some(language) = trimmed.strip_prefix("```") {
                if in_fence {
                    for code_line in &code {
                        let highlighted = highlight_code(code_line, &fence_language, &self.theme);
                        output.extend(wrap_with_prefix(
                            &highlighted,
                            "  ",
                            width,
                            &self.theme.code_block,
                        ));
                    }
                    code.clear();
                    fence_language.clear();
                    in_fence = false;
                } else {
                    in_fence = true;
                    fence_language = language.trim().to_ascii_lowercase();
                }
                index += 1;
                continue;
            }
            if in_fence {
                code.push(line.to_owned());
                index += 1;
                continue;
            }
            if trimmed.is_empty() {
                if output.last().is_some_and(|line| !line.is_empty()) {
                    output.push(String::new());
                }
                index += 1;
                continue;
            }
            if trimmed.len() >= 3
                && trimmed
                    .chars()
                    .all(|character| matches!(character, '-' | '_' | '*'))
            {
                output.push(self.theme.horizontal_rule.paint("─".repeat(width)));
                index += 1;
                continue;
            }
            if let Some(heading) = trimmed.strip_prefix('#') {
                let heading = heading.trim_start_matches('#').trim();
                let rendered = self
                    .theme
                    .heading
                    .paint(render_inline(heading, &self.theme));
                output.extend(wrap_text_ansi(&rendered, width));
                index += 1;
                continue;
            }
            if let Some(quote) = trimmed.strip_prefix('>') {
                let prefix = self.theme.quote_border.paint("│ ");
                let body = self
                    .theme
                    .quote
                    .paint(render_inline(quote.trim(), &self.theme));
                output.extend(wrap_prefixed(&body, &prefix, width));
                index += 1;
                continue;
            }
            if let Some(item) = list_item(trimmed) {
                let prefix = self.theme.list_bullet.paint(format!("{} ", item.0));
                let body = render_inline(item.1, &self.theme);
                output.extend(wrap_prefixed(&body, &prefix, width));
                index += 1;
                continue;
            }

            let mut paragraph = trimmed.to_owned();
            index += 1;
            while index < source_lines.len() {
                let next = source_lines[index].trim();
                if next.is_empty()
                    || next.starts_with('#')
                    || next.starts_with('>')
                    || next.starts_with("```")
                    || list_item(next).is_some()
                {
                    break;
                }
                paragraph.push(' ');
                paragraph.push_str(next);
                index += 1;
            }
            output.extend(wrap_text_ansi(
                &render_inline(&paragraph, &self.theme),
                width,
            ));
        }
        if in_fence {
            for code_line in &code {
                output.extend(wrap_with_prefix(
                    &highlight_code(code_line, &fence_language, &self.theme),
                    "  ",
                    width,
                    &self.theme.code_block,
                ));
            }
        }
        while output.last().is_some_and(String::is_empty) {
            output.pop();
        }
        output
    }
}

impl Component for Markdown {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        if let Some((_, _, lines)) = self
            .cache
            .as_ref()
            .filter(|(width, source, _)| *width == context.width && source == &self.source)
        {
            return Ok(lines.clone());
        }
        let padding = self.padding_x.min(context.width / 2);
        let content_width = context.width.saturating_sub(padding * 2).max(1);
        let margin = " ".repeat(padding);
        let mut output = Vec::new();
        for _ in 0..self.padding_y {
            output.push(ConstrainedLine::empty(context.width));
        }
        for line in self.render_blocks(content_width) {
            output.push(ConstrainedLine::new(
                format!("{margin}{line}{margin}"),
                context.width,
            )?);
        }
        for _ in 0..self.padding_y {
            output.push(ConstrainedLine::empty(context.width));
        }
        self.cache = Some((context.width, self.source.clone(), output.clone()));
        Ok(output)
    }

    fn invalidate(&mut self) {
        self.cache = None;
    }
}

fn list_item(line: &str) -> Option<(String, &str)> {
    if let Some(body) = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .or_else(|| line.strip_prefix("+ "))
    {
        return Some(("•".to_owned(), body));
    }
    let (number, body) = line.split_once(". ")?;
    number
        .chars()
        .all(|character| character.is_ascii_digit())
        .then(|| (format!("{number}."), body))
}

fn wrap_prefixed(body: &str, prefix: &str, width: usize) -> Vec<String> {
    let prefix_width = visible_width(prefix);
    let body_width = width.saturating_sub(prefix_width).max(1);
    wrap_text_ansi(body, body_width)
        .into_iter()
        .enumerate()
        .map(|(index, line)| {
            if index == 0 {
                format!("{prefix}{line}")
            } else {
                format!("{}{line}", " ".repeat(prefix_width))
            }
        })
        .collect()
}

fn wrap_with_prefix(
    body: &str,
    prefix: &str,
    width: usize,
    style: &crate::theme::Style,
) -> Vec<String> {
    let prefix_width = visible_width(prefix);
    let body_width = width.saturating_sub(prefix_width).max(1);
    wrap_text_ansi(body, body_width)
        .into_iter()
        .map(|line| format!("{prefix}{}", style.paint(line)))
        .collect()
}

fn render_inline(source: &str, theme: &MarkdownTheme) -> String {
    let mut output = String::new();
    let mut remaining = source;
    while !remaining.is_empty() {
        if let Some((rest, character)) = remaining
            .strip_prefix('\\')
            .and_then(|rest| rest.chars().next().map(|character| (rest, character)))
        {
            output.push(character);
            remaining = &rest[character.len_utf8()..];
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "**", "**") {
            output.push_str(&theme.strong.paint(render_inline(body, theme)));
            remaining = tail;
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "__", "__") {
            output.push_str(&theme.strong.paint(render_inline(body, theme)));
            remaining = tail;
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "~~", "~~") {
            output.push_str(&theme.deleted.paint(render_inline(body, theme)));
            remaining = tail;
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "`", "`") {
            output.push_str(&theme.code.paint(body));
            remaining = tail;
            continue;
        }
        if let Some((label, url, tail)) = markdown_link(remaining) {
            let safe_url = url
                .chars()
                .filter(|character| !character.is_control())
                .collect::<String>();
            write!(
                output,
                "\x1b]8;;{safe_url}\x07{}\x1b]8;;\x07",
                theme.link.paint(label)
            )
            .expect("writing to a String cannot fail");
            remaining = tail;
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "*", "*") {
            output.push_str(&theme.emphasis.paint(render_inline(body, theme)));
            remaining = tail;
            continue;
        }
        if let Some((body, tail)) = delimited(remaining, "_", "_") {
            output.push_str(&theme.emphasis.paint(render_inline(body, theme)));
            remaining = tail;
            continue;
        }
        let character = remaining.chars().next().expect("non-empty text");
        output.push(character);
        remaining = &remaining[character.len_utf8()..];
    }
    output
}

fn delimited<'a>(source: &'a str, opener: &str, closer: &str) -> Option<(&'a str, &'a str)> {
    let rest = source.strip_prefix(opener)?;
    let end = rest.find(closer)?;
    Some((&rest[..end], &rest[end + closer.len()..]))
}

fn markdown_link(source: &str) -> Option<(&str, &str, &str)> {
    let rest = source.strip_prefix('[')?;
    let label_end = rest.find("](")?;
    let url_start = label_end + 2;
    let url_end = rest[url_start..].find(')')?;
    Some((
        &rest[..label_end],
        &rest[url_start..url_start + url_end],
        &rest[url_start + url_end + 1..],
    ))
}

fn highlight_code(line: &str, language: &str, theme: &MarkdownTheme) -> String {
    let language_known = matches!(
        language,
        "rust" | "rs" | "javascript" | "js" | "typescript" | "ts" | "python" | "py" | "json"
    );
    if !language_known {
        return line.to_owned();
    }
    let comment = if matches!(language, "python" | "py") {
        line.find('#')
    } else {
        line.find("//")
    };
    if let Some(index) = comment {
        return format!("{}{}", &line[..index], theme.quote.paint(&line[index..]));
    }
    let mut output = String::new();
    let mut token = String::new();
    for character in line.chars() {
        if character.is_alphanumeric() || character == '_' {
            token.push(character);
        } else {
            flush_code_token(&mut output, &mut token, theme);
            output.push(character);
        }
    }
    flush_code_token(&mut output, &mut token, theme);
    output
}

fn flush_code_token(output: &mut String, token: &mut String, theme: &MarkdownTheme) {
    if token.is_empty() {
        return;
    }
    if matches!(
        token.as_str(),
        "fn" | "let"
            | "mut"
            | "pub"
            | "impl"
            | "struct"
            | "enum"
            | "const"
            | "function"
            | "class"
            | "return"
            | "import"
            | "from"
            | "def"
            | "true"
            | "false"
            | "null"
    ) {
        output.push_str(&theme.strong.paint(&*token));
    } else {
        output.push_str(token);
    }
    token.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::RenderContext;

    #[test]
    fn renders_blocks_and_links_with_width_limits() {
        let mut markdown = Markdown::new(
            "# Title\n\n- **bold** [link](https://example.test)\n\n```rust\nlet x = 1;\n```",
            MarkdownTheme::default(),
        );
        let lines = markdown
            .render(RenderContext {
                width: 20,
                height: 20,
                focused: false,
            })
            .unwrap();
        assert!(lines.iter().all(|line| line.width() <= 20));
        assert!(lines.iter().any(|line| line.as_str().contains("\x1b]8;;")));
    }
}
