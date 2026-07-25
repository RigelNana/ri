//! ANSI-aware Unicode cell measurement and slicing.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Full SGR reset.
pub const SGR_RESET: &str = "\x1b[0m";
/// OSC 8 hyperlink reset using BEL.
pub const OSC8_RESET: &str = "\x1b]8;;\x07";

#[derive(Clone, Copy, Debug)]
struct Segment<'a> {
    text: &'a str,
    width: usize,
    control: bool,
}

fn sequence_len(input: &str, offset: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    if bytes.get(offset) != Some(&0x1b) {
        return None;
    }
    let second = *bytes.get(offset + 1)?;
    match second {
        b'[' => {
            let mut index = offset + 2;
            while let Some(byte) = bytes.get(index) {
                if (0x40..=0x7e).contains(byte) {
                    return Some(index + 1 - offset);
                }
                index += 1;
            }
            Some(bytes.len() - offset)
        }
        b']' | b'P' | b'_' | b'^' => {
            let allow_bel = matches!(second, b']' | b'_');
            let mut index = offset + 2;
            while let Some(byte) = bytes.get(index) {
                if allow_bel && *byte == 0x07 {
                    return Some(index + 1 - offset);
                }
                if *byte == 0x1b && bytes.get(index + 1) == Some(&b'\\') {
                    return Some(index + 2 - offset);
                }
                index += 1;
            }
            Some(bytes.len() - offset)
        }
        b'O' => {
            if bytes.get(offset + 2).is_some_and(u8::is_ascii) {
                Some(3)
            } else {
                Some(2)
            }
        }
        _ if second.is_ascii() => Some(2),
        _ => Some(1),
    }
}

fn segments(input: &str) -> Vec<Segment<'_>> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < input.len() {
        if let Some(length) = sequence_len(input, offset) {
            result.push(Segment {
                text: &input[offset..offset + length],
                width: 0,
                control: true,
            });
            offset += length;
            continue;
        }
        let grapheme = input[offset..]
            .graphemes(true)
            .next()
            .expect("offset is inside a valid non-empty string");
        result.push(Segment {
            text: grapheme,
            width: grapheme_width(grapheme),
            control: false,
        });
        offset += grapheme.len();
    }
    result
}

/// Returns the terminal-cell width of one extended grapheme cluster.
pub fn grapheme_width(grapheme: &str) -> usize {
    if grapheme == "\t" {
        return 3;
    }
    if grapheme.chars().all(char::is_control) {
        return 0;
    }

    let first = grapheme
        .chars()
        .find(|character| !character.is_control() && !is_combining_or_format(*character));
    let Some(first) = first else {
        return 0;
    };
    let codepoint = first as u32;

    // Terminals consistently reserve two cells for flags and emoji-presentation
    // clusters, including incomplete streamed regional indicators.
    if (0x1f1e6..=0x1f1ff).contains(&codepoint)
        || grapheme.contains('\u{fe0f}')
        || grapheme.contains('\u{200d}')
        || is_emoji_codepoint(codepoint)
    {
        return 2;
    }

    UnicodeWidthStr::width(grapheme)
}

fn is_combining_or_format(character: char) -> bool {
    matches!(
        character as u32,
        0x0300..=0x036f
            | 0x1ab0..=0x1aff
            | 0x1dc0..=0x1dff
            | 0x20d0..=0x20ff
            | 0xfe00..=0xfe0f
            | 0xfe20..=0xfe2f
            | 0x200b..=0x200f
            | 0x202a..=0x202e
            | 0x2060..=0x206f
    )
}

fn is_emoji_codepoint(codepoint: u32) -> bool {
    matches!(
        codepoint,
        0x1f300..=0x1faff
            | 0x2300..=0x23ff
            | 0x2600..=0x27bf
            | 0x2b50..=0x2b55
    )
}

/// Measures visible terminal cells, ignoring CSI/OSC/DCS/APC sequences.
pub fn visible_width(input: &str) -> usize {
    segments(input).iter().map(|segment| segment.width).sum()
}

/// Removes terminal control sequences while preserving visible text.
pub fn strip_ansi(input: &str) -> String {
    segments(input)
        .into_iter()
        .filter(|segment| !segment.control)
        .map(|segment| segment.text)
        .collect()
}

/// Normalizes text before terminal output.
///
/// Tabs are expanded to three cells without touching tabs embedded in control
/// strings. Thai/Lao AM vowels are decomposed to avoid stale-cell bugs in a
/// number of terminal emulators.
pub fn normalize_terminal_output(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for segment in segments(input) {
        if segment.control {
            output.push_str(segment.text);
            continue;
        }
        for character in segment.text.chars() {
            match character {
                '\t' => output.push_str("   "),
                '\u{0e33}' => output.push_str("\u{0e4d}\u{0e32}"),
                '\u{0eb3}' => output.push_str("\u{0ecd}\u{0eb2}"),
                _ => output.push(character),
            }
        }
    }
    output
}

/// Result of an ANSI-aware column slice.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ColumnSlice {
    /// Sliced text including relevant control sequences.
    pub text: String,
    /// Actual visible width of `text`.
    pub width: usize,
}

/// Extracts a range of terminal columns.
///
/// A grapheme beginning before `start` is never split. With `strict`, a wide
/// grapheme crossing the right boundary is excluded too.
pub fn slice_columns(input: &str, start: usize, length: usize, strict: bool) -> ColumnSlice {
    if length == 0 {
        return ColumnSlice::default();
    }
    let end = start.saturating_add(length);
    let mut output = String::new();
    let mut width = 0;
    let mut column = 0;
    let mut pending_controls = String::new();
    let mut emitted_visible = false;

    for segment in segments(input) {
        if segment.control {
            if column < start && !emitted_visible {
                pending_controls.push_str(segment.text);
            } else if column < end {
                output.push_str(segment.text);
            }
            continue;
        }

        let segment_end = column.saturating_add(segment.width);
        let starts_in_range = column >= start && column < end;
        let fits = !strict || segment_end <= end;
        if starts_in_range && fits {
            if !emitted_visible {
                output.push_str(&pending_controls);
                emitted_visible = true;
            }
            output.push_str(segment.text);
            width += segment.width;
        }
        column = segment_end;
        if column >= end {
            break;
        }
    }

    ColumnSlice {
        text: output,
        width,
    }
}

/// ANSI-preserving segments used to composite overlays.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExtractedSegments {
    /// Content before an overlay.
    pub before: String,
    /// Width of `before`.
    pub before_width: usize,
    /// Content after an overlay.
    pub after: String,
    /// Width of `after`.
    pub after_width: usize,
}

/// Extracts both sides of a replaced terminal-cell range.
pub fn extract_segments(
    input: &str,
    before_end: usize,
    after_start: usize,
    after_length: usize,
    strict_after: bool,
) -> ExtractedSegments {
    let before = slice_columns(input, 0, before_end, true);
    let after = slice_columns(input, after_start, after_length, strict_after);
    ExtractedSegments {
        before: before.text,
        before_width: before.width,
        after: after.text,
        after_width: after.width,
    }
}

/// Truncates a line to a terminal-cell width and optionally pads it.
pub fn truncate_to_width(input: &str, max_width: usize, ellipsis: &str, pad: bool) -> String {
    if max_width == 0 {
        return String::new();
    }
    let input_width = visible_width(input);
    if input_width <= max_width {
        let mut result = input.to_owned();
        if pad {
            result.push_str(&" ".repeat(max_width - input_width));
        }
        return result;
    }

    let clipped_ellipsis = slice_columns(ellipsis, 0, max_width, true);
    let prefix_width = max_width.saturating_sub(clipped_ellipsis.width);
    let prefix = slice_columns(input, 0, prefix_width, true);
    let mut result = prefix.text;
    result.push_str(SGR_RESET);
    result.push_str(&clipped_ellipsis.text);
    result.push_str(SGR_RESET);
    if pad {
        let used = prefix.width + clipped_ellipsis.width;
        result.push_str(&" ".repeat(max_width.saturating_sub(used)));
    }
    result
}

#[derive(Debug, Default)]
struct ContinuationStyle {
    sgr: String,
    hyperlink: Option<String>,
}

impl ContinuationStyle {
    fn observe(&mut self, code: &str) {
        if code.starts_with("\x1b[") && code.ends_with('m') {
            let body = &code[2..code.len() - 1];
            if body.is_empty() || body.split(';').any(|part| part == "0") {
                self.sgr.clear();
            } else {
                self.sgr.push_str(code);
            }
            return;
        }
        if code.starts_with("\x1b]8;") {
            let body = code
                .strip_suffix('\x07')
                .or_else(|| code.strip_suffix("\x1b\\"))
                .unwrap_or(code);
            let is_close = body.ends_with(';');
            self.hyperlink = (!is_close).then(|| code.to_owned());
        }
    }

    fn prefix(&self) -> String {
        let mut prefix = self.sgr.clone();
        if let Some(hyperlink) = &self.hyperlink {
            prefix.push_str(hyperlink);
        }
        prefix
    }

    fn close_for_wrap(&self) -> &'static str {
        if self.hyperlink.is_some() {
            "\x1b]8;;\x07\x1b[0m"
        } else if self.sgr.is_empty() {
            ""
        } else {
            SGR_RESET
        }
    }
}

/// Word-wraps text while preserving ANSI styles across physical lines.
pub fn wrap_text_ansi(input: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if input.is_empty() {
        return vec![String::new()];
    }

    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut result = Vec::new();
    let mut style = ContinuationStyle::default();
    for (logical_index, logical) in normalized.split('\n').enumerate() {
        if logical_index > 0 && logical.is_empty() {
            result.push(style.prefix());
            continue;
        }
        wrap_logical_line(logical, width, &mut style, &mut result);
    }
    if result.is_empty() {
        result.push(String::new());
    }
    result
}

fn wrap_logical_line(
    logical: &str,
    width: usize,
    style: &mut ContinuationStyle,
    output: &mut Vec<String>,
) {
    let mut line = style.prefix();
    let mut line_width = 0;
    let mut pending_space = String::new();
    let mut pending_space_width = 0;

    for segment in segments(logical) {
        if segment.control {
            style.observe(segment.text);
            if pending_space.is_empty() {
                line.push_str(segment.text);
            } else {
                pending_space.push_str(segment.text);
            }
            continue;
        }

        let whitespace = segment.text.chars().all(char::is_whitespace);
        if whitespace {
            pending_space.push_str(segment.text);
            pending_space_width += segment.width;
            continue;
        }

        let needed = pending_space_width + segment.width;
        if line_width > 0 && line_width + needed > width {
            line.push_str(style.close_for_wrap());
            output.push(line.trim_end_matches(' ').to_owned());
            line = style.prefix();
            line_width = 0;
            pending_space.clear();
            pending_space_width = 0;
        }

        if segment.width > width {
            // Zero room exists for an over-wide atomic grapheme. Keeping it
            // would violate the line contract, so render a replacement cell.
            line.push('�');
            line_width += 1;
            continue;
        }

        if line_width + pending_space_width + segment.width > width {
            line.push_str(style.close_for_wrap());
            output.push(line.trim_end_matches(' ').to_owned());
            line = style.prefix();
            line_width = 0;
            pending_space.clear();
            pending_space_width = 0;
        }

        line.push_str(&pending_space);
        line_width += pending_space_width;
        pending_space.clear();
        pending_space_width = 0;
        line.push_str(segment.text);
        line_width += segment.width;
    }

    output.push(line.trim_end_matches(' ').to_owned());
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn measures_ansi_cjk_and_emoji() {
        assert_eq!(visible_width("\x1b[31mA界👨‍👩‍👧‍👦\x1b[0m"), 5);
        assert_eq!(visible_width("🇺"), 2);
        assert_eq!(visible_width("e\u{301}"), 1);
    }

    #[test]
    fn strict_slice_never_splits_wide_cells() {
        assert_eq!(slice_columns("abcd让EF", 0, 5, true).text, "abcd");
        assert_eq!(slice_columns("abcd让EF", 4, 2, true).text, "让");
        assert_eq!(slice_columns("abcd让EF", 5, 2, true).text, "E");
    }

    #[test]
    fn wraps_and_reopens_styles() {
        let lines = wrap_text_ansi("\x1b[31mhello world\x1b[0m", 5);
        assert_eq!(lines.len(), 2);
        assert_eq!(visible_width(&lines[0]), 5);
        assert!(lines[1].starts_with("\x1b[31m"));
    }

    proptest! {
        #[test]
        fn truncation_and_wrapping_never_exceed_contract(
            characters in proptest::collection::vec(any::<char>(), 0..80),
            width in 1_usize..40,
        ) {
            let text = characters.into_iter().collect::<String>();
            prop_assert!(visible_width(&truncate_to_width(&text, width, "…", false)) <= width);
            for line in wrap_text_ansi(&text, width) {
                prop_assert!(visible_width(&line) <= width);
            }
        }
    }
}
