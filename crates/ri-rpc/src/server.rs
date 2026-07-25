//! Concurrent RPC server and runtime dispatch boundary.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::protocol::{
    ClientFrame, CommandName, Event, ExtensionUiAction, ExtensionUiRequest, ExtensionUiResponse,
    ExtensionUiResult, NotifyType, Request, Response, ResponsePayload, ServerFrame, UiRequestId,
    WidgetPlacement,
};
use crate::transport::{RpcTransport, TransportError};

/// Runtime command failure returned to an RPC client.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct DispatchError {
    message: String,
}

impl DispatchError {
    /// Construct a dispatch failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Consume the error into its client-visible text.
    pub fn into_message(self) -> String {
        self.message
    }
}

impl From<String> for DispatchError {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for DispatchError {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Boundary implemented by a future SDK/runtime adapter.
///
/// Returning means the command was authoritatively accepted or rejected.
/// In particular, a prompt implementation should return success after
/// preflight acceptance, not after the later provider run finishes.
#[async_trait]
pub trait RpcDispatch: Send + Sync + 'static {
    /// Execute one typed request and return its matching success payload.
    async fn dispatch(
        &self,
        request: Request,
        context: DispatchContext,
    ) -> Result<ResponsePayload, DispatchError>;
}

/// Event and extension-UI facilities supplied to a dispatcher.
#[derive(Debug, Clone)]
pub struct DispatchContext {
    outgoing: mpsc::Sender<ServerFrame>,
    ui: ExtensionUi,
}

impl DispatchContext {
    /// Emit one asynchronous agent event.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn emit(&self, event: Event) -> Result<(), DispatchError> {
        self.outgoing
            .send(ServerFrame::Event(event))
            .await
            .map_err(|_| DispatchError::new("RPC peer disconnected"))
    }

    /// Access the extension-UI bridge.
    pub const fn ui(&self) -> &ExtensionUi {
        &self.ui
    }
}

type PendingUi = Arc<Mutex<HashMap<UiRequestId, oneshot::Sender<ExtensionUiResponse>>>>;

/// Extension UI bridge backed by the RPC request/response sub-protocol.
#[derive(Debug, Clone)]
pub struct ExtensionUi {
    outgoing: mpsc::Sender<ServerFrame>,
    pending: PendingUi,
}

/// Extension UI bridge failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExtensionUiError {
    /// The RPC peer disconnected.
    #[error("RPC peer disconnected")]
    Disconnected,
    /// The client sent a result shape that does not match the dialog.
    #[error("unexpected extension UI response for {method}")]
    UnexpectedResponse {
        /// Dialog method.
        method: &'static str,
    },
}

impl ExtensionUi {
    fn new(outgoing: mpsc::Sender<ServerFrame>, pending: PendingUi) -> Self {
        Self { outgoing, pending }
    }

    async fn send_request(&self, request: ExtensionUiRequest) -> Result<(), ExtensionUiError> {
        self.outgoing
            .send(ServerFrame::ExtensionUiRequest(request))
            .await
            .map_err(|_| ExtensionUiError::Disconnected)
    }

    async fn dialog(
        &self,
        action: ExtensionUiAction,
    ) -> Result<ExtensionUiResult, ExtensionUiError> {
        debug_assert!(action.expects_response());
        let timeout = action.timeout_ms().map(Duration::from_millis);
        let id = UiRequestId::new(Uuid::new_v4().to_string());
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(id.clone(), sender);

        if let Err(error) = self
            .send_request(ExtensionUiRequest::new(id.clone(), action))
            .await
        {
            self.pending.lock().await.remove(&id);
            return Err(error);
        }

        let wait_for_reply = async {
            receiver
                .await
                .map(|response| response.result)
                .map_err(|_| ExtensionUiError::Disconnected)
        };
        let result = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, wait_for_reply).await {
                Ok(result) => result.map(Some),
                Err(_) => Ok(None),
            }
        } else {
            wait_for_reply.await.map(Some)
        };
        self.pending.lock().await.remove(&id);
        result.map(|result| result.unwrap_or_else(ExtensionUiResult::cancelled))
    }

    /// Request a selection. Cancellation and timeout return `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer disconnects or sends a non-value result.
    pub async fn select(
        &self,
        title: impl Into<String>,
        options: Vec<String>,
        timeout_ms: Option<u64>,
    ) -> Result<Option<String>, ExtensionUiError> {
        match self
            .dialog(ExtensionUiAction::Select {
                title: title.into(),
                options,
                timeout: timeout_ms,
            })
            .await?
        {
            ExtensionUiResult::Value { value } => Ok(Some(value)),
            ExtensionUiResult::Cancelled { .. } => Ok(None),
            ExtensionUiResult::Confirmation { .. } => {
                Err(ExtensionUiError::UnexpectedResponse { method: "select" })
            }
        }
    }

    /// Request confirmation. Cancellation and timeout return `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer disconnects or sends a non-confirmation
    /// result.
    pub async fn confirm(
        &self,
        title: impl Into<String>,
        message: impl Into<String>,
        timeout_ms: Option<u64>,
    ) -> Result<bool, ExtensionUiError> {
        match self
            .dialog(ExtensionUiAction::Confirm {
                title: title.into(),
                message: message.into(),
                timeout: timeout_ms,
            })
            .await?
        {
            ExtensionUiResult::Confirmation { confirmed } => Ok(confirmed),
            ExtensionUiResult::Cancelled { .. } => Ok(false),
            ExtensionUiResult::Value { .. } => {
                Err(ExtensionUiError::UnexpectedResponse { method: "confirm" })
            }
        }
    }

    /// Request one-line text. Cancellation and timeout return `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer disconnects or sends a non-value result.
    pub async fn input(
        &self,
        title: impl Into<String>,
        placeholder: Option<String>,
        timeout_ms: Option<u64>,
    ) -> Result<Option<String>, ExtensionUiError> {
        match self
            .dialog(ExtensionUiAction::Input {
                title: title.into(),
                placeholder,
                timeout: timeout_ms,
            })
            .await?
        {
            ExtensionUiResult::Value { value } => Ok(Some(value)),
            ExtensionUiResult::Cancelled { .. } => Ok(None),
            ExtensionUiResult::Confirmation { .. } => {
                Err(ExtensionUiError::UnexpectedResponse { method: "input" })
            }
        }
    }

    /// Request multiline text. Cancellation returns `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer disconnects or sends a non-value result.
    pub async fn editor(
        &self,
        title: impl Into<String>,
        prefill: Option<String>,
    ) -> Result<Option<String>, ExtensionUiError> {
        match self
            .dialog(ExtensionUiAction::Editor {
                title: title.into(),
                prefill,
            })
            .await?
        {
            ExtensionUiResult::Value { value } => Ok(Some(value)),
            ExtensionUiResult::Cancelled { .. } => Ok(None),
            ExtensionUiResult::Confirmation { .. } => {
                Err(ExtensionUiError::UnexpectedResponse { method: "editor" })
            }
        }
    }

    /// Emit a fire-and-forget notification.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn notify(
        &self,
        message: impl Into<String>,
        notify_type: Option<NotifyType>,
    ) -> Result<(), ExtensionUiError> {
        self.fire_and_forget(ExtensionUiAction::Notify {
            message: message.into(),
            notify_type,
        })
        .await
    }

    /// Set or clear a fire-and-forget status entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn set_status(
        &self,
        key: impl Into<String>,
        text: Option<String>,
    ) -> Result<(), ExtensionUiError> {
        self.fire_and_forget(ExtensionUiAction::SetStatus {
            status_key: key.into(),
            status_text: text,
        })
        .await
    }

    /// Set or clear a fire-and-forget widget.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn set_widget(
        &self,
        key: impl Into<String>,
        lines: Option<Vec<String>>,
        placement: Option<WidgetPlacement>,
    ) -> Result<(), ExtensionUiError> {
        self.fire_and_forget(ExtensionUiAction::SetWidget {
            widget_key: key.into(),
            widget_lines: lines,
            widget_placement: placement,
        })
        .await
    }

    /// Set a fire-and-forget terminal title.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn set_title(&self, title: impl Into<String>) -> Result<(), ExtensionUiError> {
        self.fire_and_forget(ExtensionUiAction::SetTitle {
            title: title.into(),
        })
        .await
    }

    /// Replace client editor text.
    ///
    /// # Errors
    ///
    /// Returns an error if the peer's output queue is closed.
    pub async fn set_editor_text(&self, text: impl Into<String>) -> Result<(), ExtensionUiError> {
        self.fire_and_forget(ExtensionUiAction::SetEditorText { text: text.into() })
            .await
    }

    async fn fire_and_forget(&self, action: ExtensionUiAction) -> Result<(), ExtensionUiError> {
        debug_assert!(!action.expects_response());
        let id = UiRequestId::new(Uuid::new_v4().to_string());
        self.send_request(ExtensionUiRequest::new(id, action)).await
    }

    async fn resolve(&self, response: ExtensionUiResponse) -> bool {
        let sender = self.pending.lock().await.remove(&response.id);
        sender.is_some_and(|sender| sender.send(response).is_ok())
    }
}

/// RPC server failure.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Transport failed.
    #[error(transparent)]
    Transport(#[from] TransportError),
    /// A spawned dispatcher task panicked or was cancelled.
    #[error("RPC dispatch task failed: {0}")]
    DispatchTask(#[from] tokio::task::JoinError),
}

/// Concurrent typed RPC server.
#[derive(Debug)]
pub struct RpcServer<T, D> {
    transport: T,
    dispatcher: Arc<D>,
    output_capacity: usize,
}

impl<T, D> RpcServer<T, D> {
    /// Construct a server with a 256-record output queue.
    pub fn new(transport: T, dispatcher: Arc<D>) -> Self {
        Self {
            transport,
            dispatcher,
            output_capacity: 256,
        }
    }

    /// Set the bounded output queue capacity.
    #[must_use]
    pub fn with_output_capacity(mut self, capacity: usize) -> Self {
        self.output_capacity = capacity.max(1);
        self
    }
}

impl<T, D> RpcServer<T, D>
where
    T: RpcTransport<Incoming = ClientFrame, Outgoing = ServerFrame> + 'static,
    D: RpcDispatch,
{
    /// Serve until clean input EOF or a fatal transport failure.
    ///
    /// # Errors
    ///
    /// Returns an error when transport I/O fails or a dispatcher task is
    /// cancelled or panics.
    pub async fn run(mut self) -> Result<(), ServerError> {
        let (outgoing, mut output) = mpsc::channel(self.output_capacity);
        let pending_ui: PendingUi = Arc::new(Mutex::new(HashMap::new()));
        let ui = ExtensionUi::new(outgoing.clone(), Arc::clone(&pending_ui));
        let mut tasks = JoinSet::new();

        loop {
            tokio::select! {
                incoming = self.transport.receive() => {
                    match incoming {
                        Ok(Some(ClientFrame::Request(request))) => {
                            let expected = request.command.name();
                            let id = request.id.clone();
                            let dispatcher = Arc::clone(&self.dispatcher);
                            let context = DispatchContext {
                                outgoing: outgoing.clone(),
                                ui: ui.clone(),
                            };
                            let response_sender = outgoing.clone();
                            tasks.spawn(async move {
                                let response = match dispatcher.dispatch(request, context).await {
                                    Ok(payload) if payload.command() == expected => {
                                        Response::success(id, payload)
                                    }
                                    Ok(payload) => Response::error(
                                        id,
                                        expected.as_str(),
                                        format!(
                                            "dispatcher returned {} payload for {}",
                                            payload.command(),
                                            expected
                                        ),
                                    ),
                                    Err(error) => {
                                        Response::error(id, expected.as_str(), error.into_message())
                                    }
                                };
                                let _ = response_sender.send(ServerFrame::Response(response)).await;
                            });
                        }
                        Ok(Some(ClientFrame::ExtensionUiResponse(response))) => {
                            let _ = ui.resolve(response).await;
                        }
                        Ok(Some(ClientFrame::Invalid(invalid))) => {
                            self.transport
                                .send(ServerFrame::Response(Response::error(
                                    invalid.id,
                                    invalid.command,
                                    invalid.error,
                                )))
                                .await?;
                        }
                        Ok(None) => break,
                        Err(error) if error.is_recoverable_record_error() => {
                            self.transport
                                .send(ServerFrame::Response(Response::error(
                                    None,
                                    "parse",
                                    format!("Failed to parse command: {error}"),
                                )))
                                .await?;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                Some(frame) = output.recv() => {
                    self.transport.send(frame).await?;
                }
                joined = tasks.join_next(), if !tasks.is_empty() => {
                    if let Some(result) = joined {
                        result?;
                    }
                }
            }
        }

        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        pending_ui.lock().await.clear();
        self.transport.close().await?;
        Ok(())
    }
}

/// Ensure callers can refer to command names from dispatch implementations.
pub const fn command_name(command: &crate::protocol::Command) -> CommandName {
    command.name()
}
