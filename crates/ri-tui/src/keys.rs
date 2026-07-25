//! Terminal key decoding for legacy, Kitty CSI-u, and modifyOtherKeys input.

use std::fmt;
use std::ops::{BitOr, BitOrAssign};
use std::str::FromStr;

/// Keyboard modifier bitset.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Modifiers(u8);

impl Modifiers {
    /// No modifiers.
    pub const NONE: Self = Self(0);
    /// Shift.
    pub const SHIFT: Self = Self(1);
    /// Alt/Option.
    pub const ALT: Self = Self(2);
    /// Control.
    pub const CTRL: Self = Self(4);
    /// Super/Command/Windows.
    pub const SUPER: Self = Self(8);

    const LOCK_MASK: u8 = 64 | 128;

    /// Tests whether all flags in `other` are present.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the raw Kitty-compatible mask.
    pub const fn bits(self) -> u8 {
        self.0
    }

    fn without_locks(self) -> Self {
        Self(self.0 & !Self::LOCK_MASK)
    }

    fn from_kitty(value: u16) -> Option<Self> {
        let zero_based = value.checked_sub(1)?;
        u8::try_from(zero_based).ok().map(Self)
    }
}

impl BitOr for Modifiers {
    type Output = Self;

    fn bitor(self, right: Self) -> Self::Output {
        Self(self.0 | right.0)
    }
}

impl BitOrAssign for Modifiers {
    fn bitor_assign(&mut self, right: Self) {
        self.0 |= right.0;
    }
}

/// Logical key identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KeyCode {
    /// Printable Unicode scalar.
    Char(char),
    /// Escape.
    Escape,
    /// Enter or numpad enter.
    Enter,
    /// Tab.
    Tab,
    /// Space.
    Space,
    /// Backspace.
    Backspace,
    /// Delete.
    Delete,
    /// Insert.
    Insert,
    /// Clear.
    Clear,
    /// Home.
    Home,
    /// End.
    End,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Cursor up.
    Up,
    /// Cursor down.
    Down,
    /// Cursor left.
    Left,
    /// Cursor right.
    Right,
    /// Function key 1 through 24.
    Function(u8),
}

impl fmt::Display for KeyCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Char(character) => write!(formatter, "{character}"),
            Self::Escape => formatter.write_str("escape"),
            Self::Enter => formatter.write_str("enter"),
            Self::Tab => formatter.write_str("tab"),
            Self::Space => formatter.write_str("space"),
            Self::Backspace => formatter.write_str("backspace"),
            Self::Delete => formatter.write_str("delete"),
            Self::Insert => formatter.write_str("insert"),
            Self::Clear => formatter.write_str("clear"),
            Self::Home => formatter.write_str("home"),
            Self::End => formatter.write_str("end"),
            Self::PageUp => formatter.write_str("pageup"),
            Self::PageDown => formatter.write_str("pagedown"),
            Self::Up => formatter.write_str("up"),
            Self::Down => formatter.write_str("down"),
            Self::Left => formatter.write_str("left"),
            Self::Right => formatter.write_str("right"),
            Self::Function(number) => write!(formatter, "f{number}"),
        }
    }
}

/// Kitty keyboard event phase.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum KeyEventKind {
    /// Initial press.
    #[default]
    Press,
    /// Auto-repeat.
    Repeat,
    /// Release.
    Release,
}

/// A decoded terminal key event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyEvent {
    /// Logical identity.
    pub code: KeyCode,
    /// Active modifiers.
    pub modifiers: Modifiers,
    /// Event phase.
    pub kind: KeyEventKind,
    /// Printable text represented by this event, when insertion is appropriate.
    pub text: Option<String>,
    /// Base-layout key reported by Kitty flag 4.
    pub base_layout: Option<char>,
}

impl KeyEvent {
    /// Constructs a key press without printable insertion text.
    pub fn press(code: KeyCode) -> Self {
        Self {
            code,
            modifiers: Modifiers::NONE,
            kind: KeyEventKind::Press,
            text: None,
            base_layout: None,
        }
    }

    /// Returns true when the event matches a key chord such as `ctrl+enter`.
    pub fn matches(&self, chord: &str) -> bool {
        KeyChord::from_str(chord).is_ok_and(|expected| expected.matches(self))
    }

    /// Returns printable text for unmodified or shift-only presses/repeats.
    pub fn printable(&self) -> Option<&str> {
        (self.kind != KeyEventKind::Release
            && self.modifiers.without_locks().bits() & !(Modifiers::SHIFT.bits()) == 0)
            .then_some(self.text.as_deref())
            .flatten()
    }
}

/// Interpretation mode for ambiguous legacy encodings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KeyParseMode {
    /// No Kitty protocol was negotiated.
    #[default]
    Legacy,
    /// Kitty progressive keyboard protocol is active.
    Kitty,
}

/// Stateful convenience parser.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyParser {
    mode: KeyParseMode,
}

impl KeyParser {
    /// Creates a parser.
    pub const fn new(mode: KeyParseMode) -> Self {
        Self { mode }
    }

    /// Changes how ambiguous legacy encodings are interpreted.
    pub fn set_mode(&mut self, mode: KeyParseMode) {
        self.mode = mode;
    }

    /// Decodes one framed input sequence.
    pub fn parse(&self, input: &str) -> Option<KeyEvent> {
        parse_key(input, self.mode)
    }
}

#[derive(Clone, Copy, Debug)]
struct KittyFields {
    codepoint: i32,
    shifted: Option<u32>,
    base: Option<u32>,
    modifiers: Modifiers,
    kind: KeyEventKind,
}

/// Decodes one complete terminal input sequence.
pub fn parse_key(input: &str, mode: KeyParseMode) -> Option<KeyEvent> {
    if input.contains("\x1b[200~") {
        return None;
    }

    if let Some(fields) = parse_kitty(input) {
        return event_from_kitty(fields);
    }
    if let Some((codepoint, modifiers)) = parse_modify_other_keys(input) {
        return event_from_codepoint(
            i32::try_from(codepoint).ok()?,
            modifiers,
            KeyEventKind::Press,
            None,
            None,
        );
    }

    parse_legacy(input, mode)
}

fn parse_event_kind(value: Option<&str>) -> KeyEventKind {
    match value.and_then(|value| value.parse::<u8>().ok()) {
        Some(2) => KeyEventKind::Repeat,
        Some(3) => KeyEventKind::Release,
        _ => KeyEventKind::Press,
    }
}

fn parse_kitty(input: &str) -> Option<KittyFields> {
    let body = input.strip_prefix("\x1b[")?;

    if let Some(body) = body.strip_suffix('u') {
        let (keys, event) = body
            .split_once(';')
            .map_or((body, None), |(a, b)| (a, Some(b)));
        let mut key_parts = keys.split(':');
        let codepoint = key_parts.next()?.parse::<i32>().ok()?;
        let shifted = key_parts
            .next()
            .filter(|part| !part.is_empty())
            .and_then(|part| part.parse().ok());
        let base = key_parts
            .next()
            .filter(|part| !part.is_empty())
            .and_then(|part| part.parse().ok());
        let (modifier_text, kind_text) = event
            .and_then(|event| event.split_once(':'))
            .map_or((event.unwrap_or("1"), None), |(modifier, kind)| {
                (modifier, Some(kind))
            });
        return Some(KittyFields {
            codepoint,
            shifted,
            base,
            modifiers: Modifiers::from_kitty(modifier_text.parse().ok()?)?,
            kind: parse_event_kind(kind_text),
        });
    }

    let final_byte = body.chars().last()?;
    let params = &body[..body.len() - final_byte.len_utf8()];
    if matches!(final_byte, 'A' | 'B' | 'C' | 'D' | 'H' | 'F') {
        let mut parts = params.split(';');
        let first = parts.next()?;
        if first != "1" {
            return None;
        }
        let modifier_and_kind = parts.next()?;
        let (modifier, kind) = split_modifier_event(modifier_and_kind)?;
        let codepoint = match final_byte {
            'A' => -1,
            'B' => -2,
            'C' => -3,
            'D' => -4,
            'H' => -14,
            'F' => -15,
            _ => return None,
        };
        return Some(KittyFields {
            codepoint,
            shifted: None,
            base: None,
            modifiers: modifier,
            kind,
        });
    }

    if final_byte == '~' {
        let mut parts = params.split(';');
        let number = parts.next()?.parse::<u16>().ok()?;
        if number == 27 {
            return None;
        }
        let codepoint = match number {
            2 => -11,
            3 => -10,
            5 => -12,
            6 => -13,
            7 => -14,
            8 => -15,
            _ => return None,
        };
        let (modifiers, kind) = parts
            .next()
            .and_then(split_modifier_event)
            .unwrap_or((Modifiers::NONE, KeyEventKind::Press));
        return Some(KittyFields {
            codepoint,
            shifted: None,
            base: None,
            modifiers,
            kind,
        });
    }

    None
}

fn split_modifier_event(input: &str) -> Option<(Modifiers, KeyEventKind)> {
    let (modifier, event) = input
        .split_once(':')
        .map_or((input, None), |(modifier, event)| (modifier, Some(event)));
    Some((
        Modifiers::from_kitty(modifier.parse().ok()?)?,
        parse_event_kind(event),
    ))
}

fn parse_modify_other_keys(input: &str) -> Option<(u32, Modifiers)> {
    let body = input.strip_prefix("\x1b[27;")?.strip_suffix('~')?;
    let (modifier, codepoint) = body.split_once(';')?;
    Some((
        codepoint.parse().ok()?,
        Modifiers::from_kitty(modifier.parse().ok()?)?,
    ))
}

fn event_from_kitty(fields: KittyFields) -> Option<KeyEvent> {
    event_from_codepoint(
        normalize_keypad(fields.codepoint),
        fields.modifiers,
        fields.kind,
        fields.shifted,
        fields.base,
    )
}

fn normalize_keypad(codepoint: i32) -> i32 {
    match codepoint {
        57399..=57408 => 48 + (codepoint - 57399),
        57409 => 46,
        57410 => 47,
        57411 => 42,
        57412 => 45,
        57413 => 43,
        57414 => 13,
        57415 => 61,
        57416 => 44,
        57417 => -4,
        57418 => -3,
        57419 => -1,
        57420 => -2,
        57421 => -12,
        57422 => -13,
        57423 => -14,
        57424 => -15,
        57425 => -11,
        57426 => -10,
        _ => codepoint,
    }
}

fn event_from_codepoint(
    codepoint: i32,
    modifiers: Modifiers,
    kind: KeyEventKind,
    shifted: Option<u32>,
    base: Option<u32>,
) -> Option<KeyEvent> {
    let mut text = None;
    let code = match codepoint {
        -1 => KeyCode::Up,
        -2 => KeyCode::Down,
        -3 => KeyCode::Right,
        -4 => KeyCode::Left,
        -10 => KeyCode::Delete,
        -11 => KeyCode::Insert,
        -12 => KeyCode::PageUp,
        -13 => KeyCode::PageDown,
        -14 => KeyCode::Home,
        -15 => KeyCode::End,
        27 => KeyCode::Escape,
        9 => KeyCode::Tab,
        13 | 57414 => KeyCode::Enter,
        32 => {
            text = Some(" ".to_owned());
            KeyCode::Space
        }
        127 => KeyCode::Backspace,
        codepoint if codepoint >= 0 => {
            let original = char::from_u32(u32::try_from(codepoint).ok()?)?;
            let identity = if modifiers.contains(Modifiers::SHIFT) && original.is_ascii_uppercase()
            {
                original.to_ascii_lowercase()
            } else {
                original
            };
            let printable = if modifiers.contains(Modifiers::SHIFT) {
                shifted.and_then(char::from_u32).unwrap_or(original)
            } else {
                original
            };
            if !printable.is_control() {
                text = Some(printable.to_string());
            }
            KeyCode::Char(identity)
        }
        _ => return None,
    };

    Some(KeyEvent {
        code,
        modifiers,
        kind,
        text,
        base_layout: base.and_then(char::from_u32),
    })
}

fn parse_legacy(input: &str, mode: KeyParseMode) -> Option<KeyEvent> {
    let simple = |code| Some(KeyEvent::press(code));
    let modified = |code, modifiers| Some(modified_event(code, modifiers));
    match input {
        "\x1b" => return simple(KeyCode::Escape),
        "\t" => return simple(KeyCode::Tab),
        "\r" | "\x1bOM" => return simple(KeyCode::Enter),
        "\n" if mode == KeyParseMode::Legacy => return simple(KeyCode::Enter),
        "\n" => {
            let mut event = KeyEvent::press(KeyCode::Enter);
            event.modifiers = Modifiers::SHIFT;
            return Some(event);
        }
        " " => {
            let mut event = KeyEvent::press(KeyCode::Space);
            event.text = Some(" ".to_owned());
            return Some(event);
        }
        "\x7f" | "\x08" => return simple(KeyCode::Backspace),
        "\x1b[Z" => return modified(KeyCode::Tab, Modifiers::SHIFT),
        "\x1b\x7f" | "\x1b\x08" => return modified(KeyCode::Backspace, Modifiers::ALT),
        "\x1b\r" if mode == KeyParseMode::Legacy => {
            return modified(KeyCode::Enter, Modifiers::ALT);
        }
        "\x1b\r" => return modified(KeyCode::Enter, Modifiers::SHIFT),
        "\0" => return modified(KeyCode::Space, Modifiers::CTRL),
        "\x1b[A" | "\x1bOA" => return simple(KeyCode::Up),
        "\x1b[B" | "\x1bOB" => return simple(KeyCode::Down),
        "\x1b[C" | "\x1bOC" => return simple(KeyCode::Right),
        "\x1b[D" | "\x1bOD" => return simple(KeyCode::Left),
        "\x1b[H" | "\x1bOH" | "\x1b[1~" | "\x1b[7~" => return simple(KeyCode::Home),
        "\x1b[F" | "\x1bOF" | "\x1b[4~" | "\x1b[8~" => return simple(KeyCode::End),
        "\x1b[2~" => return simple(KeyCode::Insert),
        "\x1b[3~" => return simple(KeyCode::Delete),
        "\x1b[5~" | "\x1b[[5~" => return simple(KeyCode::PageUp),
        "\x1b[6~" | "\x1b[[6~" => return simple(KeyCode::PageDown),
        "\x1b[E" | "\x1bOE" => return simple(KeyCode::Clear),
        "\x1bb" => return modified(KeyCode::Left, Modifiers::ALT),
        "\x1bf" => return modified(KeyCode::Right, Modifiers::ALT),
        "\x1bp" => return modified(KeyCode::Up, Modifiers::ALT),
        "\x1bn" => return modified(KeyCode::Down, Modifiers::ALT),
        _ => {}
    }

    if let Some(event) = parse_legacy_function(input) {
        return Some(event);
    }
    if let Some(event) = parse_legacy_modified_navigation(input) {
        return Some(event);
    }

    let bytes = input.as_bytes();
    if bytes.len() == 1 && (1..=26).contains(&bytes[0]) {
        return modified(
            KeyCode::Char(char::from(b'a' + bytes[0] - 1)),
            Modifiers::CTRL,
        );
    }
    if bytes.len() == 1 {
        let (code, modifiers) = match bytes[0] {
            28 => (KeyCode::Char('\\'), Modifiers::CTRL),
            29 => (KeyCode::Char(']'), Modifiers::CTRL),
            31 => (KeyCode::Char('-'), Modifiers::CTRL),
            _ => return parse_plain_text(input),
        };
        return modified(code, modifiers);
    }

    if mode != KeyParseMode::Legacy {
        return None;
    }
    let rest = input.strip_prefix('\x1b')?;
    let rest_bytes = rest.as_bytes();
    if rest_bytes.len() == 1 && (1..=26).contains(&rest_bytes[0]) {
        return modified(
            KeyCode::Char(char::from(b'a' + rest_bytes[0] - 1)),
            Modifiers::CTRL | Modifiers::ALT,
        );
    }
    if rest.chars().count() != 1 {
        return None;
    }
    let character = rest.chars().next()?;
    if character.is_control() {
        return None;
    }
    let mut event = modified_event(KeyCode::Char(character), Modifiers::ALT);
    event.text = None;
    Some(event)
}

fn parse_plain_text(input: &str) -> Option<KeyEvent> {
    let mut characters = input.chars();
    let character = characters.next()?;
    if character.is_control() {
        return None;
    }
    let code = if character == ' ' {
        KeyCode::Space
    } else {
        KeyCode::Char(character)
    };
    Some(KeyEvent {
        code,
        modifiers: Modifiers::NONE,
        kind: KeyEventKind::Press,
        text: Some(input.to_owned()),
        base_layout: None,
    })
}

fn modified_event(code: KeyCode, modifiers: Modifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        text: None,
        base_layout: None,
    }
}

fn parse_legacy_function(input: &str) -> Option<KeyEvent> {
    let number = match input {
        "\x1bOP" | "\x1b[11~" | "\x1b[[A" => 1,
        "\x1bOQ" | "\x1b[12~" | "\x1b[[B" => 2,
        "\x1bOR" | "\x1b[13~" | "\x1b[[C" => 3,
        "\x1bOS" | "\x1b[14~" | "\x1b[[D" => 4,
        "\x1b[15~" | "\x1b[[E" => 5,
        "\x1b[17~" => 6,
        "\x1b[18~" => 7,
        "\x1b[19~" => 8,
        "\x1b[20~" => 9,
        "\x1b[21~" => 10,
        "\x1b[23~" => 11,
        "\x1b[24~" => 12,
        _ => return None,
    };
    Some(KeyEvent::press(KeyCode::Function(number)))
}

fn parse_legacy_modified_navigation(input: &str) -> Option<KeyEvent> {
    let (code, modifiers) = match input {
        "\x1b[1;2A" | "\x1b[a" => (KeyCode::Up, Modifiers::SHIFT),
        "\x1b[1;2B" | "\x1b[b" => (KeyCode::Down, Modifiers::SHIFT),
        "\x1b[1;2C" | "\x1b[c" => (KeyCode::Right, Modifiers::SHIFT),
        "\x1b[1;2D" | "\x1b[d" => (KeyCode::Left, Modifiers::SHIFT),
        "\x1b[1;3A" => (KeyCode::Up, Modifiers::ALT),
        "\x1b[1;3B" => (KeyCode::Down, Modifiers::ALT),
        "\x1b[1;3C" => (KeyCode::Right, Modifiers::ALT),
        "\x1b[1;3D" => (KeyCode::Left, Modifiers::ALT),
        "\x1b[1;5A" | "\x1bOa" => (KeyCode::Up, Modifiers::CTRL),
        "\x1b[1;5B" | "\x1bOb" => (KeyCode::Down, Modifiers::CTRL),
        "\x1b[1;5C" | "\x1bOc" => (KeyCode::Right, Modifiers::CTRL),
        "\x1b[1;5D" | "\x1bOd" => (KeyCode::Left, Modifiers::CTRL),
        "\x1b[2$" => (KeyCode::Insert, Modifiers::SHIFT),
        "\x1b[2^" => (KeyCode::Insert, Modifiers::CTRL),
        "\x1b[3$" => (KeyCode::Delete, Modifiers::SHIFT),
        "\x1b[3^" => (KeyCode::Delete, Modifiers::CTRL),
        "\x1b[5$" => (KeyCode::PageUp, Modifiers::SHIFT),
        "\x1b[6$" => (KeyCode::PageDown, Modifiers::SHIFT),
        "\x1b[7$" => (KeyCode::Home, Modifiers::SHIFT),
        "\x1b[8$" => (KeyCode::End, Modifiers::SHIFT),
        "\x1b[5^" => (KeyCode::PageUp, Modifiers::CTRL),
        "\x1b[6^" => (KeyCode::PageDown, Modifiers::CTRL),
        "\x1b[7^" => (KeyCode::Home, Modifiers::CTRL),
        "\x1b[8^" => (KeyCode::End, Modifiers::CTRL),
        _ => return None,
    };
    Some(modified_event(code, modifiers))
}

#[derive(Debug)]
struct KeyChord {
    code: KeyCode,
    modifiers: Modifiers,
}

impl KeyChord {
    fn matches(&self, event: &KeyEvent) -> bool {
        let modifiers_match = self.modifiers.without_locks() == event.modifiers.without_locks();
        if !modifiers_match {
            return false;
        }
        if self.code == event.code {
            return true;
        }
        matches!(
            (&self.code, event.base_layout),
            (KeyCode::Char(expected), Some(base))
                if !event.code_is_authoritative_latin_or_symbol() && *expected == base
        )
    }
}

impl KeyEvent {
    fn code_is_authoritative_latin_or_symbol(&self) -> bool {
        match self.code {
            KeyCode::Char(character) => character.is_ascii_alphanumeric() || is_symbol(character),
            _ => false,
        }
    }
}

impl FromStr for KeyChord {
    type Err = ();

    fn from_str(chord: &str) -> Result<Self, Self::Err> {
        let mut modifiers = Modifiers::NONE;
        let mut key = None;
        for part in chord.to_ascii_lowercase().split('+') {
            match part {
                "shift" => modifiers |= Modifiers::SHIFT,
                "alt" | "meta" => modifiers |= Modifiers::ALT,
                "ctrl" | "control" => modifiers |= Modifiers::CTRL,
                "super" | "cmd" | "command" => modifiers |= Modifiers::SUPER,
                value if key.is_none() => key = Some(value.to_owned()),
                _ => return Err(()),
            }
        }
        let key = key.ok_or(())?;
        let code = match key.as_str() {
            "escape" | "esc" => KeyCode::Escape,
            "enter" | "return" => KeyCode::Enter,
            "tab" => KeyCode::Tab,
            "space" => KeyCode::Space,
            "backspace" => KeyCode::Backspace,
            "delete" => KeyCode::Delete,
            "insert" => KeyCode::Insert,
            "clear" => KeyCode::Clear,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,
            "pageup" => KeyCode::PageUp,
            "pagedown" => KeyCode::PageDown,
            "up" => KeyCode::Up,
            "down" => KeyCode::Down,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            value if value.starts_with('f') => {
                let number = value[1..].parse::<u8>().map_err(|_| ())?;
                if !(1..=24).contains(&number) {
                    return Err(());
                }
                KeyCode::Function(number)
            }
            value if value.chars().count() == 1 => KeyCode::Char(value.chars().next().ok_or(())?),
            _ => return Err(()),
        };
        Ok(Self { code, modifiers })
    }
}

fn is_symbol(character: char) -> bool {
    "`-=[]\\;',./!@#$%^&*()_+|~{}:<>?".contains(character)
}

/// Decodes printable text from Kitty/modifyOtherKeys without accepting Ctrl,
/// Alt, or Super combinations.
pub fn decode_printable(input: &str) -> Option<String> {
    parse_key(input, KeyParseMode::Kitty)?
        .printable()
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kitty_alternate_layout_and_events() {
        let event = parse_key("\x1b[1089::99;5:3u", KeyParseMode::Kitty).unwrap();
        assert_eq!(event.kind, KeyEventKind::Release);
        assert!(event.matches("ctrl+c"));
        assert!(!event.matches("ctrl+d"));
    }

    #[test]
    fn parses_modify_other_keys() {
        let event = parse_key("\x1b[27;6;69~", KeyParseMode::Legacy).unwrap();
        assert!(event.matches("ctrl+shift+e"));
        assert_eq!(event.printable(), None);
    }

    #[test]
    fn decodes_shifted_printable() {
        let event = parse_key("\x1b[69;2u", KeyParseMode::Kitty).unwrap();
        assert_eq!(event.code, KeyCode::Char('e'));
        assert_eq!(event.printable(), Some("E"));
    }
}
