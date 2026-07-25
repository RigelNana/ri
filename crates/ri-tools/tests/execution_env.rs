//! Contract test for a replaceable asynchronous execution environment.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ri_tools::{
    BashInput, BashOptions, EnvDirEntry, EnvError, EnvMetadata, ExecutionEnv, LocalExecutionEnv,
    OutputChunk, OutputSink, OutputStream, ProcessExit, ProcessRequest, Tools, WalkEntry,
    WalkOptions,
};
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Default)]
struct DelegatingEnv {
    local: LocalExecutionEnv,
}

#[async_trait]
impl ExecutionEnv for DelegatingEnv {
    async fn read_file(&self, path: &Path) -> Result<Vec<u8>, EnvError> {
        self.local.read_file(path).await
    }

    async fn write_file(&self, path: &Path, bytes: &[u8]) -> Result<(), EnvError> {
        self.local.write_file(path, bytes).await
    }

    async fn create_dir_all(&self, path: &Path) -> Result<(), EnvError> {
        self.local.create_dir_all(path).await
    }

    async fn metadata(&self, path: &Path) -> Result<EnvMetadata, EnvError> {
        self.local.metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> Result<Vec<EnvDirEntry>, EnvError> {
        self.local.read_dir(path).await
    }

    async fn canonicalize(&self, path: &Path) -> Result<PathBuf, EnvError> {
        self.local.canonicalize(path).await
    }

    async fn walk(&self, root: &Path, options: WalkOptions) -> Result<Vec<WalkEntry>, EnvError> {
        self.local.walk(root, options).await
    }

    async fn which(&self, executable: &str) -> Result<Option<PathBuf>, EnvError> {
        Ok(Some(PathBuf::from(executable)))
    }

    async fn execute_process(
        &self,
        request: ProcessRequest,
        output: OutputSink,
        cancellation: CancellationToken,
    ) -> Result<ProcessExit, EnvError> {
        assert_eq!(request.program, Path::new("virtual-bash"));
        if cancellation.is_cancelled() {
            return Err(EnvError::Cancelled);
        }
        output(OutputChunk {
            stream: OutputStream::Stdout,
            data: b"split: \xe2".to_vec(),
        });
        output(OutputChunk {
            stream: OutputStream::Stdout,
            data: b"\x82\xac\n".to_vec(),
        });
        let late_output = Arc::clone(&output);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            late_output(OutputChunk {
                stream: OutputStream::Stdout,
                data: b"late\n".to_vec(),
            });
        });
        Ok(ProcessExit { code: Some(0) })
    }
}

#[tokio::test]
async fn custom_environment_streams_without_local_process_fallback() {
    let directory = tempdir().unwrap();
    let tools = Tools::new(directory.path(), Arc::new(DelegatingEnv::default())).with_bash_options(
        BashOptions {
            shell_path: Some("virtual-bash".into()),
            ..BashOptions::default()
        },
    );
    let result = tools.bash(BashInput::new("ignored")).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(result.text_content(), "split: €\n");
}

#[cfg(unix)]
#[tokio::test]
async fn local_which_requires_an_executable_permission_bit() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempdir().unwrap();
    let program = directory.path().join("program");
    tokio::fs::write(&program, b"#!/bin/sh\n").await.unwrap();

    let mut permissions = tokio::fs::metadata(&program).await.unwrap().permissions();
    permissions.set_mode(0o644);
    tokio::fs::set_permissions(&program, permissions.clone())
        .await
        .unwrap();
    assert_eq!(
        LocalExecutionEnv
            .which(program.to_str().unwrap())
            .await
            .unwrap(),
        None
    );

    permissions.set_mode(0o755);
    tokio::fs::set_permissions(&program, permissions)
        .await
        .unwrap();
    assert_eq!(
        LocalExecutionEnv
            .which(program.to_str().unwrap())
            .await
            .unwrap(),
        Some(program)
    );
}
