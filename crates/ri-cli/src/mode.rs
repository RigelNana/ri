//! Deterministic selection of interactive and headless modes.

use serde::{Deserialize, Serialize};

use crate::cli::ModeOption;
use crate::error::{CliError, Result};

/// The frontend bound to a shared session runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    /// Full-screen terminal interaction.
    Interactive,
    /// Print only final assistant text.
    Text,
    /// Stream runtime events as JSON lines.
    Json,
    /// Strict command/response JSONL protocol.
    Rpc,
}

/// TTY properties used to choose a default mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoCapabilities {
    /// Standard input is attached to a terminal.
    pub stdin_tty: bool,
    /// Standard output is attached to a terminal.
    pub stdout_tty: bool,
}

impl IoCapabilities {
    /// Construct terminal capabilities.
    pub const fn new(stdin_tty: bool, stdout_tty: bool) -> Self {
        Self {
            stdin_tty,
            stdout_tty,
        }
    }
}

/// How standard input is consumed after mode selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdinUse {
    /// The terminal UI owns standard input.
    Interactive,
    /// Read standard input to EOF and merge it into the initial prompt.
    Prompt,
    /// Decode standard input as strict LF-delimited RPC records.
    Rpc,
    /// No standard-input read is required.
    None,
}

/// Select a run mode using Pi-compatible explicit-mode, print, and TTY order.
///
/// Explicit JSON and RPC modes always remain headless. Without an explicit
/// mode, redirected stdin *or* stdout selects text mode so terminal control
/// sequences are never emitted into a pipe.
///
/// # Errors
///
/// Returns an argument error when print and interactive/RPC options conflict,
/// or when interactive mode does not own terminal stdin and stdout.
pub fn select_mode(
    explicit: Option<ModeOption>,
    print: bool,
    io: IoCapabilities,
) -> Result<RunMode> {
    if print && matches!(explicit, Some(ModeOption::Interactive | ModeOption::Rpc)) {
        return Err(CliError::InvalidArguments(
            "--print cannot be combined with interactive or RPC mode".to_owned(),
        ));
    }

    let selected = match explicit {
        Some(ModeOption::Interactive) => {
            if !io.stdin_tty || !io.stdout_tty {
                return Err(CliError::InvalidArguments(
                    "interactive mode requires terminal stdin and stdout".to_owned(),
                ));
            }
            RunMode::Interactive
        }
        Some(ModeOption::Text) => RunMode::Text,
        Some(ModeOption::Json) => RunMode::Json,
        Some(ModeOption::Rpc) => RunMode::Rpc,
        None if print || !io.stdin_tty || !io.stdout_tty => RunMode::Text,
        None => RunMode::Interactive,
    };
    Ok(selected)
}

/// Determine which subsystem owns standard input.
pub const fn stdin_use(mode: RunMode, io: IoCapabilities) -> StdinUse {
    match mode {
        RunMode::Interactive => StdinUse::Interactive,
        RunMode::Rpc => StdinUse::Rpc,
        RunMode::Text | RunMode::Json if !io.stdin_tty => StdinUse::Prompt,
        RunMode::Text | RunMode::Json => StdinUse::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTY: IoCapabilities = IoCapabilities::new(true, true);
    const PIPED_STDIN: IoCapabilities = IoCapabilities::new(false, true);
    const PIPED_STDOUT: IoCapabilities = IoCapabilities::new(true, false);

    #[test]
    fn defaults_to_interactive_only_with_two_ttys() {
        assert_eq!(select_mode(None, false, TTY).unwrap(), RunMode::Interactive);
        assert_eq!(
            select_mode(None, false, PIPED_STDIN).unwrap(),
            RunMode::Text
        );
        assert_eq!(
            select_mode(None, false, PIPED_STDOUT).unwrap(),
            RunMode::Text
        );
    }

    #[test]
    fn print_forces_text_mode() {
        assert_eq!(select_mode(None, true, TTY).unwrap(), RunMode::Text);
        assert_eq!(
            select_mode(Some(ModeOption::Text), true, TTY).unwrap(),
            RunMode::Text
        );
        assert_eq!(
            select_mode(Some(ModeOption::Json), true, TTY).unwrap(),
            RunMode::Json
        );
    }

    #[test]
    fn explicit_headless_modes_ignore_tty_shape() {
        for io in [TTY, PIPED_STDIN, PIPED_STDOUT] {
            assert_eq!(
                select_mode(Some(ModeOption::Json), false, io).unwrap(),
                RunMode::Json
            );
            assert_eq!(
                select_mode(Some(ModeOption::Rpc), false, io).unwrap(),
                RunMode::Rpc
            );
        }
    }

    #[test]
    fn rejects_impossible_interactive_request() {
        assert!(select_mode(Some(ModeOption::Interactive), false, PIPED_STDIN).is_err());
        assert!(select_mode(Some(ModeOption::Interactive), false, PIPED_STDOUT).is_err());
        assert!(select_mode(Some(ModeOption::Interactive), true, TTY).is_err());
    }

    #[test]
    fn rpc_stdin_is_never_consumed_as_a_prompt() {
        assert_eq!(stdin_use(RunMode::Rpc, PIPED_STDIN), StdinUse::Rpc);
        assert_eq!(stdin_use(RunMode::Text, PIPED_STDIN), StdinUse::Prompt);
        assert_eq!(stdin_use(RunMode::Json, PIPED_STDIN), StdinUse::Prompt);
        assert_eq!(stdin_use(RunMode::Text, TTY), StdinUse::None);
    }
}
