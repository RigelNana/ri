//! RPC server binding over the shared CLI runtime.

#![cfg(feature = "rpc")]

use std::sync::Arc;

use async_trait::async_trait;
use ri_rpc::{DispatchContext, DispatchError, Request, ResponsePayload, RpcDispatch, RpcServer};

use crate::error::{CliError, Result};
use crate::rpc_io::stdio_transport;
use crate::runtime::CliRuntime;

#[derive(Clone)]
struct RuntimeDispatch {
    runtime: Arc<dyn CliRuntime>,
}

impl std::fmt::Debug for RuntimeDispatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeDispatch")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl RpcDispatch for RuntimeDispatch {
    async fn dispatch(
        &self,
        request: Request,
        context: DispatchContext,
    ) -> std::result::Result<ResponsePayload, DispatchError> {
        self.runtime
            .rpc(request, context)
            .await
            .map_err(|error| DispatchError::new(error.to_string()))
    }
}

/// Serve strict typed JSONL until stdin reaches EOF.
///
/// # Errors
///
/// Returns a transport, protocol, runtime, or interruption error.
pub async fn run_rpc(runtime: Arc<dyn CliRuntime>) -> Result<()> {
    let dispatcher = Arc::new(RuntimeDispatch {
        runtime: Arc::clone(&runtime),
    });
    tokio::select! {
        result = RpcServer::new(stdio_transport(), dispatcher).run() => {
            result.map_err(|error| CliError::Rpc(error.to_string()))
        }
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(|source| CliError::Io {
                operation: "listen for Ctrl+C",
                source,
            })?;
            runtime.abort().await?;
            Err(CliError::Interrupted)
        }
    }
}
