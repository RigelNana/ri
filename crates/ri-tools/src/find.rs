//! `fd`-compatible path search with an ignore-aware Rust fallback.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use globset::{GlobBuilder, GlobMatcher};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::env::{OutputSink, OutputStream, ProcessRequest, WalkOptions};
use crate::error::{EnvError, ToolError};
use crate::paths::{relative_display, resolve_path};
use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, TruncationOptions, TruncationResult,
    format_size, truncate_head,
};

const DEFAULT_LIMIT: usize = 1_000;

/// Input for [`find`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FindInput {
    /// Glob expression, such as `*.rs` or `src/**/*.test.rs`.
    pub pattern: String,
    /// Directory to search. Defaults to the working directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Global result limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

impl FindInput {
    /// Construct a search rooted at the working directory.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            path: None,
            limit: None,
        }
    }
}

/// Find backend selected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum FindBackend {
    /// External `fd`/`fdfind`.
    Fd,
    /// Ignore-aware Rust walker.
    PureRust,
}

/// Structured find metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FindDetails {
    /// Byte truncation details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    /// Result limit that stopped collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_limit_reached: Option<usize>,
    /// Search backend used.
    pub backend: FindBackend,
}

/// Result of the find tool.
pub type FindResult = ToolResult<FindDetails>;

/// Find paths by glob.
///
/// # Errors
///
/// Returns an error for invalid globs, missing roots, cancellation, or
/// execution-environment failures.
pub async fn find(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: FindInput,
    cancellation: &CancellationToken,
) -> Result<FindResult, ToolError> {
    if cancellation.is_cancelled() {
        return Err(ToolError::Environment(EnvError::Cancelled));
    }
    // Parse before selecting a backend so invalid expressions behave consistently.
    let matcher = build_matcher(&input.pattern)?;
    let root = resolve_path(input.path.as_deref().unwrap_or_else(|| Path::new(".")), cwd);
    let metadata = env.metadata(&root).await.map_err(|error| match error {
        EnvError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
            ToolError::PathNotFound(root.clone())
        }
        other => ToolError::Environment(other),
    })?;
    if !metadata.is_dir {
        return Err(ToolError::NotDirectory(root));
    }
    let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);

    let fd = match env.which("fd").await? {
        Some(fd) => Some(fd),
        None => env.which("fdfind").await?,
    };
    let external = if let Some(fd) = fd {
        match find_with_fd(env, fd, &root, &input.pattern, limit, cancellation).await {
            Ok(paths) => Some((paths, FindBackend::Fd)),
            Err(ToolError::Environment(
                EnvError::Unsupported(_) | EnvError::ExecutableNotFound(_),
            )) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };

    let (mut paths, backend) = if let Some(paths) = external {
        paths
    } else {
        (
            find_pure(env, &root, &input.pattern, &matcher, limit, cancellation).await?,
            FindBackend::PureRust,
        )
    };
    paths.sort();
    paths.dedup();
    let limit_reached = paths.len() >= limit;
    paths.truncate(limit);
    if paths.is_empty() {
        return Ok(FindResult::text(
            "No files found matching pattern",
            Some(FindDetails {
                truncation: None,
                result_limit_reached: None,
                backend,
            }),
        ));
    }

    let truncation = truncate_head(&paths.join("\n"), TruncationOptions::default());
    let mut output = truncation.content.clone();
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} results limit reached. Use limit={} for more, or refine pattern",
            limit.saturating_mul(2)
        ));
    }
    if truncation.truncated {
        notices.push(match truncation.truncated_by {
            Some(TruncatedBy::Lines) => format!("{DEFAULT_MAX_LINES} line limit reached"),
            _ => format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)),
        });
    }
    if !notices.is_empty() {
        let _ = write!(output, "\n\n[{}]", notices.join(". "));
    }
    Ok(FindResult::text(
        output,
        Some(FindDetails {
            truncation: truncation.truncated.then_some(truncation),
            result_limit_reached: limit_reached.then_some(limit),
            backend,
        }),
    ))
}

async fn find_with_fd(
    env: &dyn ExecutionEnv,
    fd: PathBuf,
    root: &Path,
    pattern: &str,
    limit: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<String>, ToolError> {
    let stdout = Arc::new(Mutex::new(Vec::new()));
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let sink: OutputSink = {
        let stdout = Arc::clone(&stdout);
        let stderr = Arc::clone(&stderr);
        Arc::new(move |chunk| {
            let target = if chunk.stream == OutputStream::Stdout {
                &stdout
            } else {
                &stderr
            };
            target
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend_from_slice(&chunk.data);
        })
    };

    let mut args = vec![
        "--glob".to_owned(),
        "--color=never".to_owned(),
        "--hidden".to_owned(),
    ];
    if !inside_git_repository(env, root).await {
        args.push("--no-require-git".to_owned());
    }
    args.extend(["--max-results".to_owned(), limit.to_string()]);
    let mut effective_pattern = pattern.to_owned();
    if pattern.contains('/') {
        args.push("--full-path".to_owned());
        if !pattern.starts_with('/') && !pattern.starts_with("**/") && pattern != "**" {
            effective_pattern = format!("**/{pattern}");
        }
    }
    args.extend([
        "--".to_owned(),
        effective_pattern,
        root.to_string_lossy().into_owned(),
    ]);
    let exit = env
        .execute_process(
            ProcessRequest {
                program: fd,
                args,
                cwd: root.to_owned(),
                stdin: None,
                env: BTreeMap::default(),
                timeout: None,
            },
            sink,
            cancellation.clone(),
        )
        .await?;
    let bytes = stdout
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if exit.code != Some(0) && bytes.is_empty() {
        let error_bytes = stderr
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let message = String::from_utf8_lossy(&error_bytes).trim().to_owned();
        return Err(ToolError::InvalidGlob(if message.is_empty() {
            format!("fd exited with code {:?}", exit.code)
        } else {
            message
        }));
    }
    let output = String::from_utf8_lossy(&bytes);
    let mut paths = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let had_separator = line.ends_with('/') || line.ends_with('\\');
        let candidate = PathBuf::from(line);
        let mut relative = relative_display(&candidate, root);
        if had_separator && !relative.ends_with('/') {
            relative.push('/');
        }
        paths.push(relative);
    }
    Ok(paths)
}

async fn find_pure(
    env: &dyn ExecutionEnv,
    root: &Path,
    pattern: &str,
    matcher: &GlobMatcher,
    limit: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<String>, ToolError> {
    let entries = env
        .walk(
            root,
            WalkOptions {
                include_directories: true,
                follow_links: false,
            },
        )
        .await?;
    let path_pattern = pattern.contains('/');
    let mut paths = Vec::new();
    for entry in entries {
        if cancellation.is_cancelled() {
            return Err(ToolError::Environment(EnvError::Cancelled));
        }
        let relative = relative_display(&entry.path, root);
        let candidate = if path_pattern {
            relative.as_str()
        } else {
            entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&relative)
        };
        if matcher.is_match(candidate) {
            paths.push(if entry.is_dir {
                format!("{relative}/")
            } else {
                relative
            });
            if paths.len() >= limit {
                break;
            }
        }
    }
    Ok(paths)
}

fn build_matcher(pattern: &str) -> Result<GlobMatcher, ToolError> {
    GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map(|glob| glob.compile_matcher())
        .map_err(|error| ToolError::InvalidGlob(error.to_string()))
}

async fn inside_git_repository(env: &dyn ExecutionEnv, root: &Path) -> bool {
    let mut current = Some(root);
    while let Some(path) = current {
        if env.metadata(&path.join(".git")).await.is_ok() {
            return true;
        }
        current = path.parent();
    }
    false
}
