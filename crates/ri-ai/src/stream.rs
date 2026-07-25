//! Async assistant event stream with an independently awaitable final result.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use tokio::sync::{mpsc, watch};

use crate::{
    error::AiError,
    message::{AssistantMessage, AssistantMessageEvent},
};

/// Producer half of an [`AssistantEventStream`].
pub struct AssistantEventSender {
    events: Option<mpsc::UnboundedSender<AssistantMessageEvent>>,
    result: watch::Sender<Option<Result<AssistantMessage, AiError>>>,
    terminal: bool,
}

impl std::fmt::Debug for AssistantEventSender {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssistantEventSender")
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

impl AssistantEventSender {
    /// Pushes an event. Events after the first terminal event are ignored.
    ///
    /// Returns `false` when consumers have gone away or the stream was already
    /// terminal.
    pub fn send(&mut self, event: AssistantMessageEvent) -> bool {
        if self.terminal {
            return false;
        }
        if let Some(message) = event.final_message().cloned() {
            self.terminal = true;
            self.result.send_replace(Some(Ok(message)));
        }
        self.events
            .as_ref()
            .is_some_and(|sender| sender.send(event).is_ok())
    }

    /// Terminates without a protocol terminal event.
    ///
    /// Adapters normally send an `error` event instead. This method is reserved
    /// for failures that occur before a provider-neutral message can be built.
    pub fn fail(&mut self, error: AiError) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        self.result.send_replace(Some(Err(error)));
        self.events.take();
    }

    /// Closes event iteration after a terminal event.
    pub fn close(&mut self) {
        if !self.terminal {
            self.fail(AiError::Stream(
                "event stream closed without a terminal event".into(),
            ));
            return;
        }
        self.events.take();
    }

    /// Whether a terminal event or error has been recorded.
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }
}

impl Drop for AssistantEventSender {
    fn drop(&mut self) {
        if !self.terminal {
            self.result.send_replace(Some(Err(AiError::Stream(
                "event stream producer dropped without a terminal event".into(),
            ))));
        }
    }
}

/// Consumer half of an asynchronous assistant event stream.
///
/// [`Self::result`] can be awaited before, during, or after event iteration.
/// Error terminal events resolve to their partial assistant message, matching
/// Pi's stream contract; only protocol failures that lack a terminal message
/// return `Err`.
#[derive(Debug)]
pub struct AssistantEventStream {
    events: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    result: watch::Receiver<Option<Result<AssistantMessage, AiError>>>,
}

impl AssistantEventStream {
    /// Waits for the terminal assistant message.
    ///
    /// # Errors
    ///
    /// Returns a stream error if the producer terminates without sending a
    /// terminal assistant event.
    pub async fn result(&self) -> Result<AssistantMessage, AiError> {
        let mut receiver = self.result.clone();
        loop {
            if let Some(result) = receiver.borrow().clone() {
                return result;
            }
            if receiver.changed().await.is_err() {
                return Err(AiError::Stream(
                    "event stream ended before producing a final result".into(),
                ));
            }
        }
    }

    /// Returns a completed stream with one terminal event.
    pub fn completed(event: AssistantMessageEvent) -> Self {
        let (mut sender, stream) = create_assistant_message_event_stream();
        if event.final_message().is_none() {
            sender.fail(AiError::Stream(
                "completed stream requires a terminal event".into(),
            ));
        } else {
            sender.send(event);
            sender.close();
        }
        stream
    }
}

impl Stream for AssistantEventStream {
    type Item = AssistantMessageEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(context)
    }
}

/// Creates producer and consumer halves of an assistant event stream.
pub fn create_assistant_message_event_stream() -> (AssistantEventSender, AssistantEventStream) {
    let (event_sender, event_receiver) = mpsc::unbounded_channel();
    let (result_sender, result_receiver) = watch::channel(None);
    (
        AssistantEventSender {
            events: Some(event_sender),
            result: result_sender,
            terminal: false,
        },
        AssistantEventStream {
            events: event_receiver,
            result: result_receiver,
        },
    )
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::message::{AssistantMessage, StopReason};

    #[tokio::test]
    async fn events_and_result_can_be_consumed_independently() {
        let (mut sender, mut stream) = create_assistant_message_event_stream();
        let partial = AssistantMessage::empty("test", "test", "model");
        sender.send(AssistantMessageEvent::Start {
            partial: partial.clone(),
        });
        sender.send(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: partial.clone(),
        });
        sender.close();

        assert_eq!(
            stream.next().await,
            Some(AssistantMessageEvent::Start {
                partial: partial.clone()
            })
        );
        assert!(matches!(
            stream.next().await,
            Some(AssistantMessageEvent::Done { .. })
        ));
        assert!(stream.next().await.is_none());
        assert_eq!(stream.result().await.expect("final result"), partial);
    }

    #[tokio::test]
    async fn error_event_resolves_to_partial_message() {
        let (mut sender, stream) = create_assistant_message_event_stream();
        let mut partial = AssistantMessage::empty("test", "test", "model");
        partial.stop_reason = StopReason::Aborted;
        partial.error_message = Some("cancelled".into());
        sender.send(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: partial.clone(),
        });
        sender.close();
        assert_eq!(stream.result().await.expect("partial result"), partial);
    }

    #[tokio::test]
    async fn producer_drop_is_not_silently_successful() {
        let (sender, stream) = create_assistant_message_event_stream();
        drop(sender);
        assert!(matches!(stream.result().await, Err(AiError::Stream(_))));
    }

    #[tokio::test]
    async fn ignores_events_after_terminal() {
        let (mut sender, mut stream) = create_assistant_message_event_stream();
        let message = AssistantMessage::empty("test", "test", "model");
        assert!(sender.send(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message,
        }));
        assert!(!sender.send(AssistantMessageEvent::Start {
            partial: AssistantMessage::empty("test", "test", "model")
        }));
        sender.close();
        assert!(matches!(
            stream.next().await,
            Some(AssistantMessageEvent::Done { .. })
        ));
        assert!(stream.next().await.is_none());
    }
}
