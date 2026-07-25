//! Streaming shell command tool.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::env::{OutputChunk, OutputSink, ProcessRequest};
use crate::error::{EnvError, ToolError};
use crate::output::{OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot};
use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationResult, format_size,
};

const MAX_TIMEOUT_SECONDS: f64 = 2_147_483.647;
const UPDATE_INTERVAL: Duration = Duration::from_millis(100);

/// Input for [`bash`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BashInput {
    /// Bash source to execute.
    pub command: String,
    /// Optional timeout in seconds. There is no default timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

impl BashInput {
    /// Construct a command without a timeout.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            timeout: None,
        }
    }
}

/// Runtime options for the shell tool.
#[derive(Clone, Debug, Default)]
pub struct BashOptions {
    /// Explicit Bash executable.
    pub shell_path: Option<PathBuf>,
    /// Source prepended to every command.
    pub command_prefix: Option<String>,
    /// Environment overrides.
    pub env: BTreeMap<String, String>,
}

/// Structured shell result metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BashDetails {
    /// Output truncation details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    /// File containing complete raw output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<PathBuf>,
    /// Process exit code.
    pub exit_code: i32,
}

/// Result of the shell tool.
pub type BashResult = ToolResult<BashDetails>;

/// Callback for throttled partial shell output.
pub type BashUpdate = Arc<dyn Fn(BashResult) + Send + Sync + 'static>;

/// Execute Bash source and return a tail-truncated result.
///
/// # Errors
///
/// Returns an error for invalid timeouts, shell resolution or process failures,
/// non-zero exit codes, timeout, cancellation, and output spill failures.
pub async fn bash(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: BashInput,
    options: &BashOptions,
    cancellation: &CancellationToken,
    on_update: Option<BashUpdate>,
) -> Result<BashResult, ToolError> {
    let timeout = resolve_timeout(input.timeout)?;
    let shell = resolve_shell(env, options.shell_path.as_deref()).await?;
    let command = options
        .command_prefix
        .as_ref()
        .map_or(input.command.clone(), |prefix| {
            format!("{prefix}\n{}", input.command)
        });
    let legacy_stdin = is_legacy_wsl_bash(&shell);
    let request = ProcessRequest {
        program: shell,
        args: if legacy_stdin {
            vec!["-s".to_owned()]
        } else {
            vec!["-c".to_owned(), command.clone()]
        },
        cwd: cwd.to_owned(),
        stdin: legacy_stdin.then(|| command.into_bytes()),
        env: options.env.clone(),
        timeout,
    };

    let accumulator = Arc::new(Mutex::new(OutputAccumulator::new(
        OutputAccumulatorOptions {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
            temp_file_prefix: "ri-bash".to_owned(),
        },
    )));
    let append_error = Arc::new(Mutex::new(None::<std::io::Error>));
    let last_update = Arc::new(Mutex::new(None::<Instant>));
    let accepting_output = Arc::new(AtomicBool::new(true));
    let sink: OutputSink = {
        let accumulator = Arc::clone(&accumulator);
        let append_error = Arc::clone(&append_error);
        let last_update = Arc::clone(&last_update);
        let accepting_output = Arc::clone(&accepting_output);
        let on_update = on_update.clone();
        Arc::new(move |chunk: OutputChunk| {
            if !accepting_output.load(Ordering::Acquire) {
                return;
            }
            let Some(callback) = &on_update else {
                let append_result = accumulator
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .append(&chunk.data);
                if let Err(error) = append_result {
                    *append_error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                }
                return;
            };
            let now = Instant::now();
            let should_update = last_update
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_none_or(|last| now.duration_since(last) >= UPDATE_INTERVAL);
            let update = {
                let mut collector = accumulator
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Err(error) = collector.append(&chunk.data) {
                    *append_error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
                    return;
                }
                should_update
                    .then(|| collector.snapshot(true).ok())
                    .flatten()
                    .map(|snapshot| result_from_snapshot(snapshot, 0, false))
            };
            if let Some(update) = update {
                *last_update
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(now);
                callback(update);
            }
        })
    };

    let process_result = env
        .execute_process(request, sink, cancellation.clone())
        .await;
    accepting_output.store(false, Ordering::Release);
    let append_failure = append_error
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(error) = append_failure {
        return Err(ToolError::io("capture process output", cwd, error));
    }

    let snapshot = {
        let mut collector = accumulator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        collector
            .finish()
            .and_then(|()| collector.snapshot(true))
            .map_err(|error| ToolError::io("persist process output", cwd, error))?
    };

    let exit_code = process_result
        .as_ref()
        .ok()
        .and_then(|exit| exit.code)
        .unwrap_or(-1);
    if let Some(callback) = on_update {
        callback(result_from_snapshot(snapshot.clone(), exit_code, true));
    }
    let formatted_output = format_snapshot(&snapshot, true);

    match process_result {
        Ok(exit) => {
            let code = exit.code.unwrap_or(-1);
            if code != 0 {
                return Err(ToolError::CommandFailed {
                    code,
                    output: formatted_output,
                });
            }
            Ok(result_from_snapshot(snapshot, code, true))
        }
        Err(EnvError::Cancelled) => Err(ToolError::CommandCancelled {
            output: formatted_output,
        }),
        Err(EnvError::TimedOut(duration)) => Err(ToolError::CommandTimedOut {
            seconds: duration.as_secs_f64(),
            output: formatted_output,
        }),
        Err(error) => Err(ToolError::Environment(error)),
    }
}

fn result_from_snapshot(
    snapshot: OutputSnapshot,
    exit_code: i32,
    final_result: bool,
) -> BashResult {
    let text = format_snapshot(&snapshot, final_result);
    let details = BashDetails {
        truncation: snapshot.truncation.truncated.then_some(snapshot.truncation),
        full_output_path: snapshot.full_output_path,
        exit_code,
    };
    BashResult::text(text, Some(details))
}

fn format_snapshot(snapshot: &OutputSnapshot, include_notice: bool) -> String {
    let mut text = sanitize_output(&snapshot.content);
    if text.is_empty() {
        "(no output)".clone_into(&mut text);
    }
    if include_notice && snapshot.truncation.truncated {
        let path = snapshot.full_output_path.as_ref().map_or_else(
            || "(unavailable)".to_owned(),
            |path| path.display().to_string(),
        );
        let total = snapshot.truncation.total_lines;
        let end_line = total;
        if snapshot.truncation.last_line_partial {
            let _ = write!(
                text,
                "\n\n[Showing last {} of line {end_line} (line is {}). Full output: {path}]",
                format_size(snapshot.truncation.output_bytes),
                format_size(snapshot.last_line_bytes)
            );
        } else {
            let start_line = total
                .saturating_sub(snapshot.truncation.output_lines)
                .saturating_add(1);
            if snapshot.truncation.truncated_by == Some(TruncatedBy::Lines) {
                let _ = write!(
                    text,
                    "\n\n[Showing lines {start_line}-{end_line} of {total}. Full output: {path}]"
                );
            } else {
                let _ = write!(
                    text,
                    "\n\n[Showing lines {start_line}-{end_line} of {total} ({} limit). \
                     Full output: {path}]",
                    format_size(DEFAULT_MAX_BYTES)
                );
            }
        }
    }
    text
}

fn sanitize_output(value: &str) -> String {
    let ansi = Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]").expect("valid ANSI expression");
    let stripped = ansi.replace_all(value, "");
    stripped
        .replace("\r\n", "\n")
        .chars()
        .filter(|character| matches!(*character, '\t' | '\n' | '\r') || !character.is_control())
        .collect()
}

fn resolve_timeout(timeout: Option<f64>) -> Result<Option<Duration>, ToolError> {
    let Some(seconds) = timeout else {
        return Ok(None);
    };
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(ToolError::InvalidInput(
            "Invalid timeout: must be a finite number of seconds".to_owned(),
        ));
    }
    if seconds > MAX_TIMEOUT_SECONDS {
        return Err(ToolError::InvalidInput(format!(
            "Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"
        )));
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}

async fn resolve_shell(
    env: &dyn ExecutionEnv,
    preferred: Option<&Path>,
) -> Result<PathBuf, ToolError> {
    if let Some(preferred) = preferred {
        let value = preferred.to_string_lossy();
        return env.which(&value).await?.ok_or_else(|| {
            ToolError::Environment(EnvError::ExecutableNotFound(value.into_owned()))
        });
    }

    #[cfg(windows)]
    {
        for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Some(root) = std::env::var_os(variable) {
                let candidate = PathBuf::from(root).join("Git").join("bin").join("bash.exe");
                if env.metadata(&candidate).await.is_ok() {
                    return Ok(candidate);
                }
            }
        }
        return env.which("bash.exe").await?.ok_or_else(|| {
            ToolError::Environment(EnvError::ExecutableNotFound("bash.exe".to_owned()))
        });
    }

    #[cfg(not(windows))]
    {
        let standard = Path::new("/bin/bash");
        if env.metadata(standard).await.is_ok() {
            return Ok(standard.to_owned());
        }
        if let Some(shell) = env.which("bash").await? {
            return Ok(shell);
        }
        env.which("sh")
            .await?
            .ok_or_else(|| ToolError::Environment(EnvError::ExecutableNotFound("bash".to_owned())))
    }
}

fn is_legacy_wsl_bash(path: &Path) -> bool {
    #[cfg(windows)]
    {
        let normalized = path.to_string_lossy().replace('/', "\\").to_lowercase();
        let suffixes = [
            "\\windows\\system32\\bash.exe",
            "\\windows\\sysnative\\bash.exe",
        ];
        suffixes.iter().any(|suffix| normalized.ends_with(suffix))
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        false
    }
}
