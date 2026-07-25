//! Ripgrep-compatible content search with a pure Rust fallback.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use globset::{GlobBuilder, GlobMatcher};
use regex::{Regex, RegexBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::ExecutionEnv;
use crate::common::ToolResult;
use crate::env::{OutputSink, OutputStream, ProcessRequest, WalkOptions};
use crate::error::{EnvError, ToolError};
use crate::paths::{relative_display, resolve_path};
use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH, TruncatedBy, TruncationOptions,
    TruncationResult, format_size, truncate_head, truncate_line,
};

const DEFAULT_LIMIT: usize = 100;

/// Input for [`grep`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GrepInput {
    /// Regular expression or literal search text.
    pub pattern: String,
    /// File or directory to search. Defaults to the working directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Optional file glob.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glob: Option<String>,
    /// Match without case sensitivity.
    #[serde(default)]
    pub ignore_case: bool,
    /// Treat `pattern` as literal text.
    #[serde(default)]
    pub literal: bool,
    /// Lines of leading and trailing context.
    #[serde(default)]
    pub context: usize,
    /// Global matching-line limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

impl GrepInput {
    /// Construct a regular-expression search rooted at the working directory.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            path: None,
            glob: None,
            ignore_case: false,
            literal: false,
            context: 0,
            limit: None,
        }
    }
}

/// Structured grep metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GrepDetails {
    /// Byte truncation details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    /// Global match limit that stopped collection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_limit_reached: Option<usize>,
    /// At least one source line exceeded 500 characters.
    pub lines_truncated: bool,
    /// Search backend used.
    pub backend: SearchBackend,
}

/// Search implementation selected at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum SearchBackend {
    /// External ripgrep.
    Ripgrep,
    /// Ignore-aware Rust walker and regex engine.
    PureRust,
}

/// Result of the grep tool.
pub type GrepResult = ToolResult<GrepDetails>;

#[derive(Clone, Debug)]
struct Match {
    path: PathBuf,
    line_number: usize,
    line_text: Option<String>,
}

#[derive(Debug)]
struct RipgrepCollector {
    pending: Vec<u8>,
    matches: Vec<Match>,
    root: PathBuf,
    glob: Option<GlobMatcher>,
}

impl RipgrepCollector {
    fn new(root: &Path, glob: Option<GlobMatcher>) -> Self {
        Self {
            pending: Vec::new(),
            matches: Vec::new(),
            root: root.to_owned(),
            glob,
        }
    }

    fn append(&mut self, data: &[u8], limit: usize) -> bool {
        self.pending.extend_from_slice(data);
        while let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=newline).collect();
            if parse_ripgrep_match(&line[..line.len() - 1])
                .is_some_and(|found| self.push(found, limit))
            {
                return true;
            }
        }
        false
    }

    fn push(&mut self, found: Match, limit: usize) -> bool {
        let relative = relative_display(&found.path, &self.root);
        if self.glob.as_ref().is_some_and(|glob| {
            !glob.is_match(&relative)
                && !found
                    .path
                    .file_name()
                    .is_some_and(|name| glob.is_match(name))
        }) {
            return false;
        }
        self.matches.push(found);
        self.matches.len() >= limit
    }
}

/// Search file contents.
///
/// # Errors
///
/// Returns an error for invalid expressions, missing paths, cancellation, or
/// execution-environment failures.
pub async fn grep(
    env: &dyn ExecutionEnv,
    cwd: &Path,
    input: GrepInput,
    cancellation: &CancellationToken,
) -> Result<GrepResult, ToolError> {
    if cancellation.is_cancelled() {
        return Err(ToolError::Environment(EnvError::Cancelled));
    }
    let search_path = resolve_path(input.path.as_deref().unwrap_or_else(|| Path::new(".")), cwd);
    let metadata = env
        .metadata(&search_path)
        .await
        .map_err(|error| match error {
            EnvError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
                ToolError::PathNotFound(search_path.clone())
            }
            other => ToolError::Environment(other),
        })?;
    let is_directory = metadata.is_dir;
    let limit = input.limit.unwrap_or(DEFAULT_LIMIT).max(1);

    let external = if let Some(rg) = env.which("rg").await? {
        match grep_with_rg(env, rg, &search_path, &input, limit, cancellation).await {
            Ok(matches) => Some((matches, SearchBackend::Ripgrep)),
            Err(ToolError::Environment(
                EnvError::Unsupported(_) | EnvError::ExecutableNotFound(_),
            )) => None,
            Err(error) => return Err(error),
        }
    } else {
        None
    };

    let (mut matches, backend) = if let Some(result) = external {
        result
    } else {
        (
            grep_pure(env, &search_path, is_directory, &input, limit, cancellation).await?,
            SearchBackend::PureRust,
        )
    };
    let limit_reached = matches.len() >= limit;
    matches.truncate(limit);
    if matches.is_empty() {
        return Ok(GrepResult::text(
            "No matches found",
            Some(GrepDetails {
                truncation: None,
                match_limit_reached: None,
                lines_truncated: false,
                backend,
            }),
        ));
    }

    let mut cache = HashMap::<PathBuf, Vec<String>>::new();
    let mut output_lines = Vec::new();
    let mut lines_truncated = false;
    for matched in matches {
        if cancellation.is_cancelled() {
            return Err(ToolError::Environment(EnvError::Cancelled));
        }
        let display_path = if is_directory {
            relative_display(&matched.path, &search_path)
        } else {
            matched.path.file_name().map_or_else(
                || relative_display(&matched.path, &search_path),
                |name| name.to_string_lossy().into_owned(),
            )
        };
        if input.context == 0 {
            let line = if let Some(line) = matched.line_text {
                line.trim_end_matches(['\r', '\n']).to_owned()
            } else {
                let lines = load_lines(env, &matched.path, &mut cache).await;
                lines
                    .get(matched.line_number.saturating_sub(1))
                    .cloned()
                    .unwrap_or_else(|| "(unable to read file)".to_owned())
            };
            let (line, truncated) = truncate_line(&line, GREP_MAX_LINE_LENGTH);
            lines_truncated |= truncated;
            output_lines.push(format!("{display_path}:{}: {line}", matched.line_number));
            continue;
        }

        let lines = load_lines(env, &matched.path, &mut cache).await;
        if lines.is_empty() {
            output_lines.push(format!(
                "{display_path}:{}: (unable to read file)",
                matched.line_number
            ));
            continue;
        }
        let start = matched.line_number.saturating_sub(input.context).max(1);
        let end = matched
            .line_number
            .saturating_add(input.context)
            .min(lines.len());
        for current in start..=end {
            let line = lines.get(current - 1).map_or("", String::as_str);
            let (line, truncated) =
                truncate_line(line.trim_end_matches('\r'), GREP_MAX_LINE_LENGTH);
            lines_truncated |= truncated;
            if current == matched.line_number {
                output_lines.push(format!("{display_path}:{current}: {line}"));
            } else {
                output_lines.push(format!("{display_path}-{current}- {line}"));
            }
        }
    }

    let truncation = truncate_head(&output_lines.join("\n"), TruncationOptions::default());
    let mut output = truncation.content.clone();
    let mut notices = Vec::new();
    if limit_reached {
        notices.push(format!(
            "{limit} matches limit reached. Use limit={} for more, or refine pattern",
            limit.saturating_mul(2)
        ));
    }
    if truncation.truncated {
        notices.push(match truncation.truncated_by {
            Some(TruncatedBy::Lines) => format!("{DEFAULT_MAX_LINES} line limit reached"),
            _ => format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)),
        });
    }
    if lines_truncated {
        notices.push(format!(
            "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
        ));
    }
    if !notices.is_empty() {
        let _ = write!(output, "\n\n[{}]", notices.join(". "));
    }
    Ok(GrepResult::text(
        output,
        Some(GrepDetails {
            truncation: truncation.truncated.then_some(truncation),
            match_limit_reached: limit_reached.then_some(limit),
            lines_truncated,
            backend,
        }),
    ))
}

async fn grep_with_rg(
    env: &dyn ExecutionEnv,
    rg: PathBuf,
    search_path: &Path,
    input: &GrepInput,
    limit: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<Match>, ToolError> {
    let glob = input.glob.as_deref().map(build_glob).transpose()?;
    let collector = Arc::new(Mutex::new(RipgrepCollector::new(search_path, glob)));
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let process_cancellation = CancellationToken::new();
    let sink: OutputSink = {
        let collector = Arc::clone(&collector);
        let stderr = Arc::clone(&stderr);
        let process_cancellation = process_cancellation.clone();
        Arc::new(move |chunk| {
            if chunk.stream == OutputStream::Stdout {
                let reached_limit = collector
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .append(&chunk.data, limit);
                if reached_limit {
                    process_cancellation.cancel();
                }
            } else {
                let mut error = stderr
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let remaining = DEFAULT_MAX_BYTES.saturating_sub(error.len());
                error.extend_from_slice(&chunk.data[..chunk.data.len().min(remaining)]);
            }
        })
    };
    let mut args = vec![
        "--json".to_owned(),
        "--line-number".to_owned(),
        "--color=never".to_owned(),
        "--hidden".to_owned(),
        "--no-require-git".to_owned(),
    ];
    if input.ignore_case {
        args.push("--ignore-case".to_owned());
    }
    if input.literal {
        args.push("--fixed-strings".to_owned());
    }
    args.extend([
        "--".to_owned(),
        input.pattern.clone(),
        search_path.to_string_lossy().into_owned(),
    ]);
    let parent_cancellation = cancellation.clone();
    let forwarded_cancellation = process_cancellation.clone();
    let cancellation_task = tokio::spawn(async move {
        parent_cancellation.cancelled().await;
        forwarded_cancellation.cancel();
    });
    let execution = env
        .execute_process(
            ProcessRequest {
                program: rg,
                args,
                cwd: search_path
                    .parent()
                    .map_or_else(|| search_path.to_owned(), Path::to_owned),
                stdin: None,
                env: BTreeMap::default(),
                timeout: None,
            },
            sink,
            process_cancellation,
        )
        .await;
    cancellation_task.abort();
    let mut collector = collector
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !collector.pending.is_empty() && collector.matches.len() < limit {
        let pending = std::mem::take(&mut collector.pending);
        if let Some(found) = parse_ripgrep_match(&pending) {
            collector.push(found, limit);
        }
    }
    let reached_limit = collector.matches.len() >= limit;
    if cancellation.is_cancelled() {
        return Err(ToolError::Environment(EnvError::Cancelled));
    }
    let exit = match execution {
        Ok(exit) => Some(exit),
        Err(EnvError::Cancelled) if reached_limit => None,
        Err(error) => return Err(ToolError::Environment(error)),
    };
    if exit.is_some_and(|exit| !matches!(exit.code, Some(0 | 1))) {
        let message = String::from_utf8_lossy(
            &stderr
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
        .trim()
        .to_owned();
        return Err(ToolError::InvalidRegex(if message.is_empty() {
            format!(
                "ripgrep exited with code {:?}",
                exit.and_then(|status| status.code)
            )
        } else {
            message
        }));
    }
    Ok(std::mem::take(&mut collector.matches))
}

fn parse_ripgrep_match(line: &[u8]) -> Option<Match> {
    let event = serde_json::from_slice::<serde_json::Value>(line).ok()?;
    if event.get("type").and_then(serde_json::Value::as_str) != Some("match") {
        return None;
    }
    let path = event
        .pointer("/data/path/text")
        .and_then(serde_json::Value::as_str)?;
    let line_number = event
        .pointer("/data/line_number")
        .and_then(serde_json::Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())?;
    let line_text = event
        .pointer("/data/lines/text")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Some(Match {
        path: PathBuf::from(path),
        line_number,
        line_text,
    })
}

async fn grep_pure(
    env: &dyn ExecutionEnv,
    search_path: &Path,
    is_directory: bool,
    input: &GrepInput,
    limit: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<Match>, ToolError> {
    let expression = build_regex(input)?;
    let glob = input.glob.as_deref().map(build_glob).transpose()?;
    let files = if is_directory {
        env.walk(search_path, WalkOptions::default()).await?
    } else {
        vec![crate::WalkEntry {
            path: search_path.to_owned(),
            is_dir: false,
        }]
    };
    let mut matches = Vec::new();
    for entry in files {
        if cancellation.is_cancelled() {
            return Err(ToolError::Environment(EnvError::Cancelled));
        }
        if entry.is_dir {
            continue;
        }
        let relative = relative_display(&entry.path, search_path);
        if glob.as_ref().is_some_and(|glob| {
            !glob.is_match(&relative)
                && !entry
                    .path
                    .file_name()
                    .is_some_and(|name| glob.is_match(name))
        }) {
            continue;
        }
        let Ok(bytes) = env.read_file(&entry.path).await else {
            continue;
        };
        if bytes.contains(&0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes)
            .replace("\r\n", "\n")
            .replace('\r', "\n");
        for (index, line) in text.split('\n').enumerate() {
            if expression.is_match(line) {
                matches.push(Match {
                    path: entry.path.clone(),
                    line_number: index + 1,
                    line_text: Some(line.to_owned()),
                });
                if matches.len() >= limit {
                    return Ok(matches);
                }
            }
        }
    }
    Ok(matches)
}

fn build_regex(input: &GrepInput) -> Result<Regex, ToolError> {
    let pattern = if input.literal {
        regex::escape(&input.pattern)
    } else {
        input.pattern.clone()
    };
    RegexBuilder::new(&pattern)
        .case_insensitive(input.ignore_case)
        .build()
        .map_err(|error| ToolError::InvalidRegex(error.to_string()))
}

fn build_glob(pattern: &str) -> Result<GlobMatcher, ToolError> {
    GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .map(|glob| glob.compile_matcher())
        .map_err(|error| ToolError::InvalidGlob(error.to_string()))
}

async fn load_lines(
    env: &dyn ExecutionEnv,
    path: &Path,
    cache: &mut HashMap<PathBuf, Vec<String>>,
) -> Vec<String> {
    if let Some(lines) = cache.get(path) {
        return lines.clone();
    }
    let lines = env.read_file(path).await.map_or_else(
        |_| Vec::new(),
        |bytes| {
            String::from_utf8_lossy(&bytes)
                .replace("\r\n", "\n")
                .replace('\r', "\n")
                .split('\n')
                .map(str::to_owned)
                .collect()
        },
    );
    cache.insert(path.to_owned(), lines.clone());
    lines
}
