//! Replaceable asynchronous execution environment.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use ignore::WalkBuilder;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::error::EnvError;

/// Lightweight metadata needed by the tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvMetadata {
    /// Whether the entry is a regular file.
    pub is_file: bool,
    /// Whether the entry is a directory.
    pub is_dir: bool,
    /// File size in bytes where the environment can provide it.
    pub len: u64,
}

/// One direct child returned by [`ExecutionEnv::read_dir`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvDirEntry {
    /// File name without its parent path.
    pub name: String,
    /// Full path in the execution environment.
    pub path: PathBuf,
    /// Whether the entry is a directory, if known cheaply.
    pub is_dir: Option<bool>,
}

/// Options for a recursive environment walk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkOptions {
    /// Include directories in addition to regular files.
    pub include_directories: bool,
    /// Follow symbolic links.
    pub follow_links: bool,
}

/// One entry returned by [`ExecutionEnv::walk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkEntry {
    /// Full path.
    pub path: PathBuf,
    /// Whether the entry is a directory.
    pub is_dir: bool,
}

/// The source of a process-output chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputStream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

/// A process-output chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputChunk {
    /// Source stream.
    pub stream: OutputStream,
    /// Raw bytes, unmodified and in arrival-sized chunks.
    pub data: Vec<u8>,
}

/// Thread-safe process output callback.
pub type OutputSink = Arc<dyn Fn(OutputChunk) + Send + Sync + 'static>;

/// A program invocation performed by an [`ExecutionEnv`].
#[derive(Clone, Debug)]
pub struct ProcessRequest {
    /// Executable path or name.
    pub program: PathBuf,
    /// Argument vector. The executable is not included.
    pub args: Vec<String>,
    /// Working directory.
    pub cwd: PathBuf,
    /// Bytes sent to standard input before it is closed.
    pub stdin: Option<Vec<u8>>,
    /// Environment overrides. The inherited environment remains available.
    pub env: BTreeMap<String, String>,
    /// Optional wall-clock timeout.
    pub timeout: Option<Duration>,
}

/// Terminal status of a child process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessExit {
    /// Numeric exit code, or `None` when terminated by a signal.
    pub code: Option<i32>,
}

enum ProcessCompletion {
    Exited(io::Result<std::process::ExitStatus>),
    Cancelled,
    TimedOut,
}

#[derive(Clone, Copy)]
enum ProcessInterruption {
    Cancelled,
    TimedOut,
}

/// Async, replaceable host operations used by every built-in tool.
///
/// Implementations may target the local machine, an SSH host, a container, or
/// an in-memory test environment. Search tools use `walk` as their pure-Rust
/// fallback and may use `which` plus `execute_process` for `rg`/`fd`.
#[async_trait]
pub trait ExecutionEnv: Send + Sync {
    /// Read all bytes from a file.
    async fn read_file(&self, path: &Path) -> Result<Vec<u8>, EnvError>;

    /// Replace a file with the supplied bytes.
    async fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), EnvError>;

    /// Recursively create a directory.
    async fn create_dir_all(&self, path: &Path) -> Result<(), EnvError>;

    /// Query metadata while following symbolic links.
    async fn metadata(&self, path: &Path) -> Result<EnvMetadata, EnvError>;

    /// Read direct directory children.
    async fn read_dir(&self, path: &Path) -> Result<Vec<EnvDirEntry>, EnvError>;

    /// Resolve a path through symbolic links.
    async fn canonicalize(&self, path: &Path) -> Result<PathBuf, EnvError>;

    /// Recursively walk a tree while honoring ignore files.
    async fn walk(&self, root: &Path, options: WalkOptions) -> Result<Vec<WalkEntry>, EnvError>;

    /// Resolve an executable using this environment's command search rules.
    async fn which(&self, executable: &str) -> Result<Option<PathBuf>, EnvError>;

    /// Execute a program and stream its raw standard output and error.
    async fn execute_process(
        &self,
        request: ProcessRequest,
        output: OutputSink,
        cancellation: CancellationToken,
    ) -> Result<ProcessExit, EnvError>;
}

/// Local Tokio-backed execution environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalExecutionEnv;

#[async_trait]
impl ExecutionEnv for LocalExecutionEnv {
    async fn read_file(&self, path: &Path) -> Result<Vec<u8>, EnvError> {
        Ok(tokio::fs::read(path).await?)
    }

    async fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), EnvError> {
        tokio::fs::write(path, bytes).await?;
        Ok(())
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), EnvError> {
        tokio::fs::create_dir_all(path).await?;
        Ok(())
    }

    async fn metadata(&self, path: &Path) -> Result<EnvMetadata, EnvError> {
        let metadata = tokio::fs::metadata(path).await?;
        Ok(EnvMetadata {
            is_file: metadata.is_file(),
            is_dir: metadata.is_dir(),
            len: metadata.len(),
        })
    }

    async fn read_dir(&self, path: &Path) -> Result<Vec<EnvDirEntry>, EnvError> {
        let mut directory = tokio::fs::read_dir(path).await?;
        let mut entries = Vec::new();
        while let Some(entry) = directory.next_entry().await? {
            let is_dir = entry.file_type().await.ok().map(|kind| kind.is_dir());
            entries.push(EnvDirEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                path: entry.path(),
                is_dir,
            });
        }
        Ok(entries)
    }

    async fn canonicalize(&self, path: &Path) -> Result<PathBuf, EnvError> {
        Ok(tokio::fs::canonicalize(path).await?)
    }

    async fn walk(&self, root: &Path, options: WalkOptions) -> Result<Vec<WalkEntry>, EnvError> {
        let root = root.to_owned();
        tokio::task::spawn_blocking(move || {
            let mut builder = WalkBuilder::new(&root);
            builder
                .hidden(false)
                .follow_links(options.follow_links)
                .git_ignore(true)
                .git_exclude(true)
                .parents(true)
                .ignore(true)
                .require_git(false);
            builder.filter_entry(|entry| {
                if entry.depth() == 0 {
                    return true;
                }
                let name = entry.file_name();
                name != OsStr::new(".git") && name != OsStr::new("node_modules")
            });

            let mut entries = Vec::new();
            for item in builder.build() {
                let entry = item.map_err(|error| io::Error::other(error.to_string()))?;
                if entry.depth() == 0 {
                    continue;
                }
                let Some(kind) = entry.file_type() else {
                    continue;
                };
                let is_dir = kind.is_dir();
                if kind.is_file() || (options.include_directories && is_dir) {
                    entries.push(WalkEntry {
                        path: entry.into_path(),
                        is_dir,
                    });
                }
            }
            entries.sort_by(|left, right| left.path.cmp(&right.path));
            Ok::<_, io::Error>(entries)
        })
        .await
        .map_err(|error| EnvError::Other(format!("filesystem walk task failed: {error}")))?
        .map_err(EnvError::Io)
    }

    async fn which(&self, executable: &str) -> Result<Option<PathBuf>, EnvError> {
        let candidate = Path::new(executable);
        if candidate.components().count() > 1 || candidate.is_absolute() {
            return Ok(is_executable_file(candidate)
                .await
                .then(|| candidate.to_owned()));
        }

        let Some(path) = std::env::var_os("PATH") else {
            return Ok(None);
        };
        #[cfg(windows)]
        let names = {
            let mut names = vec![executable.to_owned()];
            if Path::new(executable).extension().is_none() {
                let extensions =
                    std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
                names.extend(
                    extensions
                        .split(';')
                        .filter(|extension| !extension.is_empty())
                        .map(|extension| format!("{executable}{extension}")),
                );
            }
            names
        };
        #[cfg(not(windows))]
        let names = vec![executable.to_owned()];

        for directory in std::env::split_paths(&path) {
            for name in &names {
                let full_path = directory.join(name);
                if is_executable_file(&full_path).await {
                    return Ok(Some(full_path));
                }
            }
        }
        Ok(None)
    }

    async fn execute_process(
        &self,
        request: ProcessRequest,
        output: OutputSink,
        cancellation: CancellationToken,
    ) -> Result<ProcessExit, EnvError> {
        if cancellation.is_cancelled() {
            return Err(EnvError::Cancelled);
        }

        let metadata = tokio::fs::metadata(&request.cwd).await.map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "working directory does not exist: {}: {error}",
                    request.cwd.display()
                ),
            )
        })?;
        if !metadata.is_dir() {
            return Err(EnvError::Io(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!(
                    "working directory is not a directory: {}",
                    request.cwd.display()
                ),
            )));
        }

        let mut command = Command::new(&request.program);
        command
            .args(&request.args)
            .current_dir(&request.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(if request.stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .kill_on_drop(true);
        command.envs(&request.env);

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.as_std_mut().process_group(0);
        }

        let mut child = command.spawn()?;
        let pid = child.id();
        let accepting = Arc::new(AtomicBool::new(true));
        let stdout_task = child.stdout.take().map(|stdout| {
            pump_output(
                stdout,
                OutputStream::Stdout,
                Arc::clone(&output),
                Arc::clone(&accepting),
            )
        });
        let stderr_task = child.stderr.take().map(|stderr| {
            pump_output(
                stderr,
                OutputStream::Stderr,
                Arc::clone(&output),
                Arc::clone(&accepting),
            )
        });
        let stdin_task = child
            .stdin
            .take()
            .zip(request.stdin)
            .map(|(mut stdin, bytes)| {
                tokio::spawn(async move {
                    let _ = stdin.write_all(&bytes).await;
                    let _ = stdin.shutdown().await;
                })
            });

        let completion = if let Some(timeout) = request.timeout {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => ProcessCompletion::Cancelled,
                result = child.wait() => ProcessCompletion::Exited(result),
                () = tokio::time::sleep(timeout) => ProcessCompletion::TimedOut,
            }
        } else {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => ProcessCompletion::Cancelled,
                result = child.wait() => ProcessCompletion::Exited(result),
            }
        };

        let (result, interrupted) = match completion {
            ProcessCompletion::Exited(status) => (
                status.map(|status| ProcessExit {
                    code: status.code(),
                }),
                None,
            ),
            ProcessCompletion::Cancelled => {
                terminate_process_tree(pid, &mut child).await;
                (
                    Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "operation cancelled",
                    )),
                    Some(ProcessInterruption::Cancelled),
                )
            }
            ProcessCompletion::TimedOut => {
                terminate_process_tree(pid, &mut child).await;
                (
                    Err(io::Error::new(io::ErrorKind::TimedOut, "process timed out")),
                    Some(ProcessInterruption::TimedOut),
                )
            }
        };

        if interrupted.is_some() {
            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
        }

        finish_pump(stdout_task, &accepting).await;
        finish_pump(stderr_task, &accepting).await;
        if let Some(task) = stdin_task {
            task.abort();
        }
        accepting.store(false, Ordering::Release);

        match interrupted {
            Some(ProcessInterruption::Cancelled) => Err(EnvError::Cancelled),
            Some(ProcessInterruption::TimedOut) => Err(EnvError::TimedOut(
                request.timeout.unwrap_or(Duration::ZERO),
            )),
            None => result.map_err(EnvError::Io),
        }
    }
}

async fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn pump_output<R>(
    mut reader: R,
    stream: OutputStream,
    sink: OutputSink,
    accepting: Arc<AtomicBool>,
) -> JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buffer = vec![0_u8; 8192];
        loop {
            let count = match reader.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(count) => count,
            };
            if accepting.load(Ordering::Acquire) {
                sink(OutputChunk {
                    stream,
                    data: buffer[..count].to_vec(),
                });
            }
        }
    })
}

async fn finish_pump(task: Option<JoinHandle<()>>, accepting: &Arc<AtomicBool>) {
    let Some(mut task) = task else {
        return;
    };
    if tokio::time::timeout(Duration::from_millis(100), &mut task)
        .await
        .is_err()
    {
        accepting.store(false, Ordering::Release);
        task.abort();
    }
}

async fn terminate_process_tree(pid: Option<u32>, child: &mut Child) {
    let Some(pid) = pid else {
        let _ = child.start_kill();
        return;
    };

    #[cfg(windows)]
    {
        let mut taskkill = Command::new("taskkill");
        taskkill
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Ok(mut process) = taskkill.spawn() {
            let _ = tokio::time::timeout(Duration::from_secs(1), process.wait()).await;
        }
    }

    #[cfg(unix)]
    {
        let mut kill = Command::new("kill");
        kill.args(["-KILL", "--", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Ok(mut process) = kill.spawn() {
            let _ = tokio::time::timeout(Duration::from_secs(2), process.wait()).await;
        }
    }

    let _ = child.start_kill();
}
