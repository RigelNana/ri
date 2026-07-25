//! Asynchronous RPC transport abstractions.

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;
use tokio_util::codec::Framed;

use crate::codec::{JsonlCodec, JsonlError};

/// Transport-layer failure.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// JSONL framing or payload failure.
    #[error(transparent)]
    Jsonl(#[from] JsonlError),
    /// The peer or in-memory transport closed while sending.
    #[error("RPC transport is closed")]
    Closed,
}

impl TransportError {
    /// Whether a malformed consumed JSONL record can be reported and skipped.
    pub const fn is_recoverable_record_error(&self) -> bool {
        match self {
            Self::Jsonl(error) => error.is_recoverable_record_error(),
            Self::Closed => false,
        }
    }
}

/// Bidirectional asynchronous transport used by clients and servers.
#[async_trait]
pub trait RpcTransport: Send {
    /// Message received from the peer.
    type Incoming: Send;
    /// Message sent to the peer.
    type Outgoing: Send;

    /// Receive the next message, or `None` after clean EOF.
    async fn receive(&mut self) -> Result<Option<Self::Incoming>, TransportError>;

    /// Send one message.
    async fn send(&mut self, message: Self::Outgoing) -> Result<(), TransportError>;

    /// Flush and close the outgoing side.
    async fn close(&mut self) -> Result<(), TransportError>;
}

/// Strict JSONL transport over any Tokio byte stream.
#[derive(Debug)]
pub struct JsonlTransport<IO, Incoming, Outgoing> {
    framed: Framed<IO, JsonlCodec<Incoming, Outgoing>>,
}

impl<IO, Incoming, Outgoing> JsonlTransport<IO, Incoming, Outgoing> {
    /// Wrap a byte stream with the default strict JSONL codec.
    pub fn new(io: IO) -> Self {
        Self {
            framed: Framed::new(io, JsonlCodec::new()),
        }
    }

    /// Wrap a byte stream with an explicit record-size limit.
    pub fn with_max_frame_len(io: IO, max_frame_len: usize) -> Self {
        Self {
            framed: Framed::new(io, JsonlCodec::with_max_frame_len(max_frame_len)),
        }
    }

    /// Consume the wrapper and return the byte stream.
    pub fn into_inner(self) -> IO {
        self.framed.into_inner()
    }
}

#[async_trait]
impl<IO, Incoming, Outgoing> RpcTransport for JsonlTransport<IO, Incoming, Outgoing>
where
    IO: AsyncRead + AsyncWrite + Unpin + Send,
    Incoming: DeserializeOwned + Send,
    Outgoing: Serialize + Send,
{
    type Incoming = Incoming;
    type Outgoing = Outgoing;

    async fn receive(&mut self) -> Result<Option<Self::Incoming>, TransportError> {
        self.framed
            .next()
            .await
            .transpose()
            .map_err(TransportError::from)
    }

    async fn send(&mut self, message: Self::Outgoing) -> Result<(), TransportError> {
        self.framed
            .send(message)
            .await
            .map_err(TransportError::from)
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        SinkExt::close(&mut self.framed)
            .await
            .map_err(TransportError::from)
    }
}

/// Typed in-memory transport useful for embedding and deterministic tests.
#[derive(Debug)]
pub struct ChannelTransport<Incoming, Outgoing> {
    incoming: mpsc::Receiver<Incoming>,
    outgoing: Option<mpsc::Sender<Outgoing>>,
}

#[async_trait]
impl<Incoming, Outgoing> RpcTransport for ChannelTransport<Incoming, Outgoing>
where
    Incoming: Send + 'static,
    Outgoing: Send + 'static,
{
    type Incoming = Incoming;
    type Outgoing = Outgoing;

    async fn receive(&mut self) -> Result<Option<Self::Incoming>, TransportError> {
        Ok(self.incoming.recv().await)
    }

    async fn send(&mut self, message: Self::Outgoing) -> Result<(), TransportError> {
        self.outgoing
            .as_ref()
            .ok_or(TransportError::Closed)?
            .send(message)
            .await
            .map_err(|_| TransportError::Closed)
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        self.outgoing = None;
        Ok(())
    }
}

/// Create two connected typed in-memory transports.
pub fn channel_transport_pair<A, B>(
    capacity: usize,
) -> (ChannelTransport<B, A>, ChannelTransport<A, B>)
where
    A: Send + 'static,
    B: Send + 'static,
{
    let (a_to_b_tx, a_to_b_rx) = mpsc::channel(capacity);
    let (b_to_a_tx, b_to_a_rx) = mpsc::channel(capacity);
    (
        ChannelTransport {
            incoming: b_to_a_rx,
            outgoing: Some(a_to_b_tx),
        },
        ChannelTransport {
            incoming: a_to_b_rx,
            outgoing: Some(b_to_a_tx),
        },
    )
}
