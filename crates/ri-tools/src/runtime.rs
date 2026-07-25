//! Convenient typed facade over all built-in tools.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::bash::{BashInput, BashOptions, BashResult, BashUpdate, bash};
use crate::edit::{EditInput, EditResult, edit};
use crate::find::{FindInput, FindResult, find};
use crate::grep::{GrepInput, GrepResult, grep};
use crate::ls::{LsInput, LsResult, ls};
use crate::read::{ReadInput, ReadResult, read};
use crate::write::{WriteInput, WriteResult, write};
use crate::{ExecutionEnv, LocalExecutionEnv, ToolError};

/// Typed facade for the seven built-in coding tools.
#[derive(Clone)]
pub struct Tools {
    env: Arc<dyn ExecutionEnv>,
    cwd: PathBuf,
    bash_options: BashOptions,
}

#[allow(clippy::missing_errors_doc)]
impl Tools {
    /// Construct local tools rooted at `cwd`.
    pub fn local(cwd: impl Into<PathBuf>) -> Self {
        Self {
            env: Arc::new(LocalExecutionEnv),
            cwd: cwd.into(),
            bash_options: BashOptions::default(),
        }
    }

    /// Construct tools backed by a replaceable execution environment.
    pub fn new(cwd: impl Into<PathBuf>, env: Arc<dyn ExecutionEnv>) -> Self {
        Self {
            env,
            cwd: cwd.into(),
            bash_options: BashOptions::default(),
        }
    }

    /// Replace shell-specific options.
    #[must_use]
    pub fn with_bash_options(mut self, options: BashOptions) -> Self {
        self.bash_options = options;
        self
    }

    /// Working directory used to resolve relative paths.
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Underlying execution environment.
    pub fn environment(&self) -> &Arc<dyn ExecutionEnv> {
        &self.env
    }

    /// Read without external cancellation.
    pub async fn read(&self, input: ReadInput) -> Result<ReadResult, ToolError> {
        self.read_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// Read with cancellation.
    pub async fn read_with_cancellation(
        &self,
        input: ReadInput,
        cancellation: &CancellationToken,
    ) -> Result<ReadResult, ToolError> {
        read(self.env.as_ref(), &self.cwd, input, cancellation).await
    }

    /// Write without external cancellation.
    pub async fn write(&self, input: WriteInput) -> Result<WriteResult, ToolError> {
        self.write_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// Write with cancellation.
    pub async fn write_with_cancellation(
        &self,
        input: WriteInput,
        cancellation: &CancellationToken,
    ) -> Result<WriteResult, ToolError> {
        write(self.env.as_ref(), &self.cwd, input, cancellation).await
    }

    /// Edit without external cancellation.
    pub async fn edit(&self, input: EditInput) -> Result<EditResult, ToolError> {
        self.edit_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// Edit with cancellation.
    pub async fn edit_with_cancellation(
        &self,
        input: EditInput,
        cancellation: &CancellationToken,
    ) -> Result<EditResult, ToolError> {
        edit(self.env.as_ref(), &self.cwd, input, cancellation).await
    }

    /// Execute Bash without external cancellation or updates.
    pub async fn bash(&self, input: BashInput) -> Result<BashResult, ToolError> {
        self.bash_with_cancellation(input, &CancellationToken::new(), None)
            .await
    }

    /// Execute Bash with cancellation and optional streaming updates.
    pub async fn bash_with_cancellation(
        &self,
        input: BashInput,
        cancellation: &CancellationToken,
        on_update: Option<BashUpdate>,
    ) -> Result<BashResult, ToolError> {
        bash(
            self.env.as_ref(),
            &self.cwd,
            input,
            &self.bash_options,
            cancellation,
            on_update,
        )
        .await
    }

    /// Grep without external cancellation.
    pub async fn grep(&self, input: GrepInput) -> Result<GrepResult, ToolError> {
        self.grep_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// Grep with cancellation.
    pub async fn grep_with_cancellation(
        &self,
        input: GrepInput,
        cancellation: &CancellationToken,
    ) -> Result<GrepResult, ToolError> {
        grep(self.env.as_ref(), &self.cwd, input, cancellation).await
    }

    /// Find without external cancellation.
    pub async fn find(&self, input: FindInput) -> Result<FindResult, ToolError> {
        self.find_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// Find with cancellation.
    pub async fn find_with_cancellation(
        &self,
        input: FindInput,
        cancellation: &CancellationToken,
    ) -> Result<FindResult, ToolError> {
        find(self.env.as_ref(), &self.cwd, input, cancellation).await
    }

    /// List without external cancellation.
    pub async fn ls(&self, input: LsInput) -> Result<LsResult, ToolError> {
        self.ls_with_cancellation(input, &CancellationToken::new())
            .await
    }

    /// List with cancellation.
    pub async fn ls_with_cancellation(
        &self,
        input: LsInput,
        cancellation: &CancellationToken,
    ) -> Result<LsResult, ToolError> {
        ls(self.env.as_ref(), &self.cwd, input, cancellation).await
    }
}

impl fmt::Debug for Tools {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tools")
            .field("cwd", &self.cwd)
            .field("bash_options", &self.bash_options)
            .finish_non_exhaustive()
    }
}
