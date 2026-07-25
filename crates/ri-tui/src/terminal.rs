//! Terminal abstraction and crossterm backend.

use std::io::{self, Stdout, Write};
use std::time::Duration;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode as CrosstermKeyCode,
    KeyEventKind as CrosstermKeyEventKind, KeyModifiers,
};
use crossterm::{execute, queue, terminal as crossterm_terminal};

use crate::keys::{KeyCode, KeyEvent, KeyEventKind, Modifiers};

/// Input or resize event produced by a terminal backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminalEvent {
    /// Decoded keyboard event.
    Key(KeyEvent),
    /// Complete bracketed paste.
    Paste(String),
    /// Terminal dimensions changed.
    Resize {
        /// Columns.
        columns: u16,
        /// Rows.
        rows: u16,
    },
    /// Focus entered the terminal.
    FocusGained,
    /// Focus left the terminal.
    FocusLost,
}

/// Minimal synchronous terminal contract.
pub trait Terminal: Send {
    /// Enables interactive terminal modes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal mode setup fails.
    fn start(&mut self) -> io::Result<()>;

    /// Restores modes changed by `start`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal mode restoration fails.
    fn stop(&mut self) -> io::Result<()>;

    /// Writes one already-framed output batch.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the output cannot be written.
    fn write(&mut self, data: &str) -> io::Result<()>;

    /// Flushes pending output.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when pending output cannot be flushed.
    fn flush(&mut self) -> io::Result<()>;

    /// Current `(columns, rows)`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal dimensions cannot be queried.
    fn size(&self) -> io::Result<(u16, u16)>;

    /// Waits up to `timeout` for one event.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when polling or reading terminal input fails.
    fn read_event(&mut self, timeout: Duration) -> io::Result<Option<TerminalEvent>>;

    /// Whether Kitty progressive keyboard reporting is active.
    fn kitty_protocol_active(&self) -> bool {
        false
    }

    /// Moves the cursor vertically.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the movement sequence cannot be written.
    fn move_by(&mut self, rows: i32) -> io::Result<()> {
        match rows.cmp(&0) {
            std::cmp::Ordering::Greater => self.write(&format!("\x1b[{rows}B")),
            std::cmp::Ordering::Less => self.write(&format!("\x1b[{}A", rows.unsigned_abs())),
            std::cmp::Ordering::Equal => Ok(()),
        }
    }

    /// Hides the hardware cursor.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the control sequence cannot be written.
    fn hide_cursor(&mut self) -> io::Result<()> {
        self.write("\x1b[?25l")
    }

    /// Shows the hardware cursor.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the control sequence cannot be written.
    fn show_cursor(&mut self) -> io::Result<()> {
        self.write("\x1b[?25h")
    }

    /// Clears the current line.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the control sequence cannot be written.
    fn clear_line(&mut self) -> io::Result<()> {
        self.write("\x1b[2K")
    }

    /// Clears from the cursor to the end of the display.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the control sequence cannot be written.
    fn clear_from_cursor(&mut self) -> io::Result<()> {
        self.write("\x1b[J")
    }

    /// Clears the display and homes the cursor.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the control sequence cannot be written.
    fn clear_screen(&mut self) -> io::Result<()> {
        self.write("\x1b[2J\x1b[H")
    }

    /// Sets the terminal title, stripping control terminators.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the title sequence cannot be written.
    fn set_title(&mut self, title: &str) -> io::Result<()> {
        let safe = title.replace(['\x1b', '\x07'], "");
        self.write(&format!("\x1b]0;{safe}\x07"))
    }

    /// Sets or clears the OSC 9;4 indeterminate progress state.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the progress sequence cannot be written.
    fn set_progress(&mut self, active: bool) -> io::Result<()> {
        self.write(if active {
            "\x1b]9;4;3\x07"
        } else {
            "\x1b]9;4;0;\x07"
        })
    }
}

/// Real terminal backend implemented with crossterm.
#[derive(Debug)]
pub struct CrosstermTerminal<W = Stdout> {
    writer: W,
    started: bool,
    kitty_protocol_active: bool,
}

impl CrosstermTerminal<Stdout> {
    /// Uses process stdout.
    pub fn stdout() -> Self {
        Self::new(io::stdout())
    }
}

impl<W> CrosstermTerminal<W> {
    /// Creates a terminal around a writer.
    pub const fn new(writer: W) -> Self {
        Self {
            writer,
            started: false,
            kitty_protocol_active: false,
        }
    }

    /// Records the result of keyboard-protocol negotiation.
    pub fn set_kitty_protocol_active(&mut self, active: bool) {
        self.kitty_protocol_active = active;
    }

    /// Borrows the writer.
    pub const fn writer(&self) -> &W {
        &self.writer
    }

    /// Mutably borrows the writer.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Consumes the backend and returns the writer.
    pub fn into_writer(self) -> W {
        self.writer
    }
}

impl<W: Write + Send> Terminal for CrosstermTerminal<W> {
    fn start(&mut self) -> io::Result<()> {
        if self.started {
            return Ok(());
        }
        crossterm_terminal::enable_raw_mode()?;
        if let Err(error) = (|| {
            queue!(self.writer, EnableBracketedPaste, Hide)?;
            // Request Kitty disambiguation, event types, and alternate keys,
            // followed by a status and DA query for negotiation.
            self.writer.write_all(b"\x1b[>7u\x1b[?u\x1b[c")?;
            self.writer.flush()
        })() {
            let _ = crossterm_terminal::disable_raw_mode();
            return Err(error);
        }
        self.started = true;
        Ok(())
    }

    fn stop(&mut self) -> io::Result<()> {
        if !self.started {
            return Ok(());
        }
        let output_result = (|| {
            self.writer.write_all(b"\x1b[<u\x1b[>4;0m\x1b[?2004l")?;
            execute!(self.writer, DisableBracketedPaste, Show)?;
            self.writer.flush()
        })();
        let raw_result = crossterm_terminal::disable_raw_mode();
        self.started = false;
        self.kitty_protocol_active = false;
        output_result.and(raw_result)
    }

    fn write(&mut self, data: &str) -> io::Result<()> {
        self.writer.write_all(data.as_bytes())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    fn size(&self) -> io::Result<(u16, u16)> {
        crossterm_terminal::size()
    }

    fn read_event(&mut self, timeout: Duration) -> io::Result<Option<TerminalEvent>> {
        if !event::poll(timeout)? {
            return Ok(None);
        }
        loop {
            match event::read()? {
                Event::Key(key) => {
                    if let Some(key) = convert_key_event(key) {
                        return Ok(Some(TerminalEvent::Key(key)));
                    }
                }
                Event::Paste(text) => return Ok(Some(TerminalEvent::Paste(text))),
                Event::Resize(columns, rows) => {
                    return Ok(Some(TerminalEvent::Resize { columns, rows }));
                }
                Event::FocusGained => return Ok(Some(TerminalEvent::FocusGained)),
                Event::FocusLost => return Ok(Some(TerminalEvent::FocusLost)),
                Event::Mouse(_) => {}
            }
            if !event::poll(Duration::ZERO)? {
                return Ok(None);
            }
        }
    }

    fn kitty_protocol_active(&self) -> bool {
        self.kitty_protocol_active
    }
}

fn convert_key_event(event: crossterm::event::KeyEvent) -> Option<KeyEvent> {
    let modifiers = convert_modifiers(event.modifiers);
    let kind = match event.kind {
        CrosstermKeyEventKind::Press => KeyEventKind::Press,
        CrosstermKeyEventKind::Repeat => KeyEventKind::Repeat,
        CrosstermKeyEventKind::Release => KeyEventKind::Release,
    };
    let mut text = None;
    let code = match event.code {
        CrosstermKeyCode::Backspace => KeyCode::Backspace,
        CrosstermKeyCode::Enter => KeyCode::Enter,
        CrosstermKeyCode::Left => KeyCode::Left,
        CrosstermKeyCode::Right => KeyCode::Right,
        CrosstermKeyCode::Up => KeyCode::Up,
        CrosstermKeyCode::Down => KeyCode::Down,
        CrosstermKeyCode::Home => KeyCode::Home,
        CrosstermKeyCode::End => KeyCode::End,
        CrosstermKeyCode::PageUp => KeyCode::PageUp,
        CrosstermKeyCode::PageDown => KeyCode::PageDown,
        CrosstermKeyCode::Tab => KeyCode::Tab,
        CrosstermKeyCode::BackTab => {
            return Some(KeyEvent {
                code: KeyCode::Tab,
                modifiers: modifiers | Modifiers::SHIFT,
                kind,
                text: None,
                base_layout: None,
            });
        }
        CrosstermKeyCode::Delete => KeyCode::Delete,
        CrosstermKeyCode::Insert => KeyCode::Insert,
        CrosstermKeyCode::F(number) => KeyCode::Function(number),
        CrosstermKeyCode::Char(character) => {
            if modifiers.bits() & !(Modifiers::SHIFT.bits()) == 0 && !character.is_control() {
                text = Some(character.to_string());
            }
            if character == ' ' {
                KeyCode::Space
            } else if modifiers.contains(Modifiers::SHIFT) && character.is_ascii_uppercase() {
                KeyCode::Char(character.to_ascii_lowercase())
            } else {
                KeyCode::Char(character)
            }
        }
        CrosstermKeyCode::Null => KeyCode::Space,
        CrosstermKeyCode::Esc => KeyCode::Escape,
        CrosstermKeyCode::CapsLock
        | CrosstermKeyCode::ScrollLock
        | CrosstermKeyCode::NumLock
        | CrosstermKeyCode::PrintScreen
        | CrosstermKeyCode::Pause
        | CrosstermKeyCode::Menu
        | CrosstermKeyCode::KeypadBegin
        | CrosstermKeyCode::Media(_)
        | CrosstermKeyCode::Modifier(_) => return None,
    };
    Some(KeyEvent {
        code,
        modifiers,
        kind,
        text,
        base_layout: None,
    })
}

fn convert_modifiers(modifiers: KeyModifiers) -> Modifiers {
    let mut result = Modifiers::NONE;
    if modifiers.contains(KeyModifiers::SHIFT) {
        result |= Modifiers::SHIFT;
    }
    if modifiers.contains(KeyModifiers::ALT) {
        result |= Modifiers::ALT;
    }
    if modifiers.contains(KeyModifiers::CONTROL) {
        result |= Modifiers::CTRL;
    }
    if modifiers.contains(KeyModifiers::SUPER)
        || modifiers.contains(KeyModifiers::HYPER)
        || modifiers.contains(KeyModifiers::META)
    {
        result |= Modifiers::SUPER;
    }
    result
}
