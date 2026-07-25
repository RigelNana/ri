//! End-to-end asynchronous client/server and extension-UI coverage.

use std::sync::Arc;

use async_trait::async_trait;
use ri_rpc::{
    ClientFrame, Command, DispatchContext, DispatchError, Event, ExtensionUiResponse,
    ExtensionUiResult, Request, ResponsePayload, RpcClient, RpcDispatch, RpcServer, ServerFrame,
    channel_transport_pair,
};

#[derive(Debug)]
struct TestDispatch;

#[async_trait]
impl RpcDispatch for TestDispatch {
    async fn dispatch(
        &self,
        request: Request,
        context: DispatchContext,
    ) -> Result<ResponsePayload, DispatchError> {
        match request.command {
            Command::Abort => Ok(ResponsePayload::Abort),
            Command::Prompt { .. } => {
                let selected = context
                    .ui()
                    .select(
                        "Continue?",
                        vec!["Allow".to_owned(), "Block".to_owned()],
                        None,
                    )
                    .await
                    .map_err(|error| DispatchError::new(error.to_string()))?;
                if selected.as_deref() != Some("Allow") {
                    return Err(DispatchError::new("blocked"));
                }
                context
                    .emit(Event::QueueUpdate {
                        steering: vec!["accepted".to_owned()],
                        follow_up: Vec::new(),
                    })
                    .await?;
                Ok(ResponsePayload::Prompt)
            }
            command => Err(DispatchError::new(format!(
                "{} is not implemented by the test runtime",
                command.name()
            ))),
        }
    }
}

#[tokio::test]
async fn client_server_multiplex_responses_events_and_extension_ui() {
    let (client_transport, server_transport) =
        channel_transport_pair::<ClientFrame, ServerFrame>(32);
    let server = RpcServer::new(server_transport, Arc::new(TestDispatch));
    let server_task = tokio::spawn(server.run());
    let (client, driver) = RpcClient::spawn(client_transport);
    let mut notifications = client.subscribe();

    let caller = client.clone();
    let call = tokio::spawn(async move {
        caller
            .call(Command::Prompt {
                message: "hello".to_owned(),
                images: Vec::new(),
                streaming_behavior: None,
            })
            .await
    });

    let ui_request = loop {
        if let ServerFrame::ExtensionUiRequest(request) = notifications.recv().await.unwrap() {
            break request;
        }
    };
    client
        .respond_to_ui(ExtensionUiResponse::new(
            ui_request.id,
            ExtensionUiResult::Value {
                value: "Allow".to_owned(),
            },
        ))
        .await
        .unwrap();

    assert_eq!(call.await.unwrap().unwrap(), ResponsePayload::Prompt);
    let event = loop {
        if let ServerFrame::Event(event) = notifications.recv().await.unwrap() {
            break event;
        }
    };
    assert_eq!(
        event,
        Event::QueueUpdate {
            steering: vec!["accepted".to_owned()],
            follow_up: Vec::new(),
        }
    );

    assert_eq!(
        client.call(Command::Abort).await.unwrap(),
        ResponsePayload::Abort
    );
    client.shutdown().await.unwrap();
    driver.wait().await.unwrap();
    server_task.await.unwrap().unwrap();
}
