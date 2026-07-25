//! Multiplexed asynchronous typed RPC client.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::protocol::{
    ClientFrame, Command, ExtensionUiResponse, Request, RequestId, Response, ResponsePayload,
    ServerFrame,
};
use crate::transport::{RpcTransport, TransportError};

/// Client-side protocol or transport failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClientError {
    /// The background driver is no longer available.
    #[error("RPC client is closed")]
    Closed,
    /// The peer closed its output unexpectedly.
    #[error("RPC peer disconnected")]
    Disconnected,
    /// Transport failed.
    #[error("RPC transport failed: {0}")]
    Transport(String),
    /// The caller attempted to reuse an in-flight identifier.
    #[error("duplicate in-flight RPC request id: {0}")]
    DuplicateRequestId(RequestId),
    /// No response arrived before the request deadline.
    #[error("timed out waiting for {command} response ({id})")]
    Timeout {
        /// Request identifier.
        id: RequestId,
        /// Command name.
        command: String,
    },
    /// The remote dispatcher rejected the command.
    #[error("remote {command} failed: {error}")]
    Remote {
        /// Known or unknown remote command name.
        command: String,
        /// Remote failure text.
        error: String,
    },
    /// Background task failed.
    #[error("RPC client task failed: {0}")]
    Driver(String),
}

impl From<TransportError> for ClientError {
    fn from(value: TransportError) -> Self {
        Self::Transport(value.to_string())
    }
}

enum DriverCommand {
    Request {
        request: Request,
        reply: oneshot::Sender<Result<Response, ClientError>>,
    },
    ExtensionUiResponse {
        response: ExtensionUiResponse,
        reply: oneshot::Sender<Result<(), ClientError>>,
    },
    Cancel {
        id: RequestId,
    },
    Shutdown {
        reply: oneshot::Sender<Result<(), ClientError>>,
    },
}

/// Cloneable handle to a running RPC client driver.
#[derive(Debug, Clone)]
pub struct RpcClient {
    commands: mpsc::Sender<DriverCommand>,
    notifications: broadcast::Sender<ServerFrame>,
    next_id: Arc<AtomicU64>,
    timeout: Duration,
}

impl RpcClient {
    /// Start a client driver over a typed transport.
    pub fn spawn<T>(transport: T) -> (Self, ClientDriver)
    where
        T: RpcTransport<Incoming = ServerFrame, Outgoing = ClientFrame> + 'static,
    {
        Self::spawn_with_capacity(transport, 256)
    }

    /// Start a client with explicit command/notification queue capacity.
    pub fn spawn_with_capacity<T>(transport: T, capacity: usize) -> (Self, ClientDriver)
    where
        T: RpcTransport<Incoming = ServerFrame, Outgoing = ClientFrame> + 'static,
    {
        let capacity = capacity.max(1);
        let (commands, receiver) = mpsc::channel(capacity);
        let (notifications, _) = broadcast::channel(capacity);
        let driver_notifications = notifications.clone();
        let task = tokio::spawn(run_driver(transport, receiver, driver_notifications));
        (
            Self {
                commands,
                notifications,
                next_id: Arc::new(AtomicU64::new(0)),
                timeout: Duration::from_secs(30),
            },
            ClientDriver { task },
        )
    }

    /// Set the per-request response deadline for this handle.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Subscribe to events, extension-UI requests, and unmatched responses.
    pub fn subscribe(&self) -> broadcast::Receiver<ServerFrame> {
        self.notifications.subscribe()
    }

    /// Send a command using a generated `req_N` identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver is closed, the transport fails, or the
    /// peer does not respond before the configured deadline.
    pub async fn request(&self, command: Command) -> Result<Response, ClientError> {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = RequestId::new(format!("req_{sequence}"));
        self.request_with_id(id, command).await
    }

    /// Send a command using a caller-selected identifier.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is already in flight, the driver is closed,
    /// the transport fails, or the response deadline expires.
    pub async fn request_with_id(
        &self,
        id: RequestId,
        command: Command,
    ) -> Result<Response, ClientError> {
        let command_name = command.name().to_string();
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(DriverCommand::Request {
                request: Request::correlated(id.clone(), command),
                reply,
            })
            .await
            .map_err(|_| ClientError::Closed)?;

        match tokio::time::timeout(self.timeout, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(ClientError::Closed),
            Err(_) => {
                let _ = self
                    .commands
                    .send(DriverCommand::Cancel { id: id.clone() })
                    .await;
                Err(ClientError::Timeout {
                    id,
                    command: command_name,
                })
            }
        }
    }

    /// Send a command and return its typed successful payload.
    ///
    /// # Errors
    ///
    /// Returns any request error, or [`ClientError::Remote`] when the peer
    /// returns an unsuccessful response.
    pub async fn call(&self, command: Command) -> Result<ResponsePayload, ClientError> {
        match self.request(command).await? {
            Response::Success(success) => Ok(success.payload),
            Response::Error(error) => Err(ClientError::Remote {
                command: error.command,
                error: error.error,
            }),
        }
    }

    /// Answer an extension-UI dialog request.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver is closed or cannot send the response.
    pub async fn respond_to_ui(&self, response: ExtensionUiResponse) -> Result<(), ClientError> {
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(DriverCommand::ExtensionUiResponse { response, reply })
            .await
            .map_err(|_| ClientError::Closed)?;
        receiver.await.map_err(|_| ClientError::Closed)?
    }

    /// Flush and stop the background driver.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver is closed or closing its transport
    /// fails.
    pub async fn shutdown(&self) -> Result<(), ClientError> {
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(DriverCommand::Shutdown { reply })
            .await
            .map_err(|_| ClientError::Closed)?;
        receiver.await.map_err(|_| ClientError::Closed)?
    }
}

/// Join handle for the client transport loop.
#[derive(Debug)]
pub struct ClientDriver {
    task: JoinHandle<Result<(), ClientError>>,
}

impl ClientDriver {
    /// Wait for the driver to finish.
    ///
    /// # Errors
    ///
    /// Returns an error if the transport loop fails or its task is cancelled
    /// or panics.
    pub async fn wait(self) -> Result<(), ClientError> {
        self.task
            .await
            .map_err(|error| ClientError::Driver(error.to_string()))?
    }

    /// Abort the driver immediately.
    pub fn abort(&self) {
        self.task.abort();
    }
}

async fn run_driver<T>(
    mut transport: T,
    mut commands: mpsc::Receiver<DriverCommand>,
    notifications: broadcast::Sender<ServerFrame>,
) -> Result<(), ClientError>
where
    T: RpcTransport<Incoming = ServerFrame, Outgoing = ClientFrame> + 'static,
{
    let mut pending: HashMap<RequestId, oneshot::Sender<Result<Response, ClientError>>> =
        HashMap::new();

    let terminal = loop {
        tokio::select! {
            incoming = transport.receive() => {
                match incoming {
                    Ok(Some(ServerFrame::Response(response))) => {
                        let matched = response
                            .request_id()
                            .and_then(|id| pending.remove(id))
                            .is_some_and(|reply| reply.send(Ok(response.clone())).is_ok());
                        if !matched {
                            let _ = notifications.send(ServerFrame::Response(response));
                        }
                    }
                    Ok(Some(frame)) => {
                        let _ = notifications.send(frame);
                    }
                    Ok(None) => break Err(ClientError::Disconnected),
                    Err(error) => break Err(ClientError::from(error)),
                }
            }
            command = commands.recv() => {
                match command {
                    Some(DriverCommand::Request { request, reply }) => {
                        let Some(id) = request.id.clone() else {
                            let _ = reply.send(Err(ClientError::Driver(
                                "client driver received an uncorrelated request".to_owned(),
                            )));
                            continue;
                        };
                        if pending.contains_key(&id) {
                            let _ = reply.send(Err(ClientError::DuplicateRequestId(id)));
                            continue;
                        }
                        pending.insert(id.clone(), reply);
                        if let Err(error) = transport.send(ClientFrame::Request(request)).await {
                            if let Some(reply) = pending.remove(&id) {
                                let _ = reply.send(Err(ClientError::from(error)));
                            }
                        }
                    }
                    Some(DriverCommand::ExtensionUiResponse { response, reply }) => {
                        let result = transport
                            .send(ClientFrame::ExtensionUiResponse(response))
                            .await
                            .map_err(ClientError::from);
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        if failed {
                            break Err(ClientError::Disconnected);
                        }
                    }
                    Some(DriverCommand::Cancel { id }) => {
                        pending.remove(&id);
                    }
                    Some(DriverCommand::Shutdown { reply }) => {
                        let result = transport.close().await.map_err(ClientError::from);
                        let failed = result.is_err();
                        let _ = reply.send(result.clone());
                        break if failed {
                            result
                        } else {
                            Ok(())
                        };
                    }
                    None => {
                        break transport.close().await.map_err(ClientError::from);
                    }
                }
            }
        }
    };

    let pending_error = terminal.clone().err().unwrap_or(ClientError::Closed);
    for (_, reply) in pending {
        let _ = reply.send(Err(pending_error.clone()));
    }
    terminal
}
