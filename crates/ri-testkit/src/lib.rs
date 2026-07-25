//! Test-only support for protocol, persistence, and concurrency contracts.
//!
//! This crate deliberately contains no production fallback provider. Its loopback
//! server makes exact HTTP/SSE chunk boundaries observable to integration tests.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::Duration;

use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Errors emitted by the loopback wire server.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Socket I/O failed.
    #[error("loopback server I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// An HTTP request was malformed.
    #[error("malformed HTTP request: {0}")]
    Request(String),
    /// The request capture channel closed.
    #[error("request capture channel closed")]
    Closed,
    /// JSON response serialization failed.
    #[error("response JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    /// Header construction failed.
    #[error("invalid test header: {0}")]
    Header(String),
}

/// Captured HTTP request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireRequest {
    /// Request method.
    pub method: Method,
    /// Origin-form request target.
    pub target: String,
    /// Parsed headers.
    pub headers: HeaderMap,
    /// Exact request body.
    pub body: Bytes,
}

/// Scripted HTTP response, including exact body chunk boundaries.
#[derive(Clone, Debug)]
pub struct WireResponse {
    /// Response status.
    pub status: StatusCode,
    /// Response headers.
    pub headers: HeaderMap,
    /// Chunks written in order.
    pub chunks: Vec<Bytes>,
    /// Delay between chunks.
    pub chunk_delay: Duration,
}

impl WireResponse {
    /// Build a JSON response.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] if `value` cannot be serialized.
    pub fn json(status: StatusCode, value: &impl Serialize) -> Result<Self, Error> {
        let body = serde_json::to_vec(value)?;
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(Self {
            status,
            headers,
            chunks: vec![Bytes::from(body)],
            chunk_delay: Duration::ZERO,
        })
    }

    /// Build an SSE response from exact transport chunks.
    pub fn sse(chunks: impl IntoIterator<Item = impl Into<Bytes>>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        Self {
            status: StatusCode::OK,
            headers,
            chunks: chunks.into_iter().map(Into::into).collect(),
            chunk_delay: Duration::ZERO,
        }
    }

    /// Add a response header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Header`] if `name` or `value` is not a valid HTTP header.
    pub fn header(mut self, name: &str, value: &str) -> Result<Self, Error> {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| Error::Header(error.to_string()))?;
        let value =
            HeaderValue::from_str(value).map_err(|error| Error::Header(error.to_string()))?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// Set the delay between response chunks.
    #[must_use]
    pub fn delay(mut self, delay: Duration) -> Self {
        self.chunk_delay = delay;
        self
    }
}

/// A local scripted HTTP server.
#[derive(Debug)]
pub struct WireServer {
    address: SocketAddr,
    requests: mpsc::Receiver<WireRequest>,
    cancel: CancellationToken,
    task: Option<JoinHandle<Result<(), Error>>>,
}

impl WireServer {
    /// Bind to localhost and serve scripted responses in FIFO order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the listener cannot bind or report its local address.
    pub async fn start(responses: impl IntoIterator<Item = WireResponse>) -> Result<Self, Error> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let address = listener.local_addr()?;
        let (request_tx, requests) = mpsc::channel(32);
        let cancel = CancellationToken::new();
        let child_cancel = cancel.clone();
        let mut responses: VecDeque<_> = responses.into_iter().collect();
        let task = tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    () = child_cancel.cancelled() => return Ok(()),
                    accepted = listener.accept() => accepted,
                };
                let (mut stream, _) = accepted?;
                let request = read_request(&mut stream).await?;
                request_tx.send(request).await.map_err(|_| Error::Closed)?;
                let response = responses.pop_front().unwrap_or_else(|| WireResponse {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    headers: HeaderMap::new(),
                    chunks: vec![Bytes::from_static(b"script exhausted")],
                    chunk_delay: Duration::ZERO,
                });
                write_response(&mut stream, &response).await?;
            }
        });
        Ok(Self {
            address,
            requests,
            cancel,
            task: Some(task),
        })
    }

    /// Server socket address.
    pub const fn address(&self) -> SocketAddr {
        self.address
    }

    /// HTTP base URL.
    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    /// Receive the next captured request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] if the request capture channel closes.
    pub async fn next(&mut self) -> Result<WireRequest, Error> {
        self.requests.recv().await.ok_or(Error::Closed)
    }

    /// Stop the server and wait for its task.
    ///
    /// # Errors
    ///
    /// Returns an error if the server task fails or cannot be joined.
    ///
    /// # Panics
    ///
    /// Panics if the internal server task handle is unexpectedly absent.
    pub async fn close(mut self) -> Result<(), Error> {
        self.cancel.cancel();
        self.task
            .take()
            .expect("server task is present until close")
            .await
            .map_err(|error| {
                Error::Io(std::io::Error::other(format!(
                    "server task failed: {error}"
                )))
            })?
    }
}

impl Drop for WireServer {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn read_request(stream: &mut TcpStream) -> Result<WireRequest, Error> {
    const MAX_HEADER: usize = 64 * 1024;
    let mut bytes = Vec::new();
    let header_end = loop {
        if bytes.len() >= MAX_HEADER {
            return Err(Error::Request("headers exceed 64 KiB".to_owned()));
        }
        let mut buffer = [0_u8; 4096];
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            return Err(Error::Request(
                "connection closed before headers".to_owned(),
            ));
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let head = std::str::from_utf8(&bytes[..header_end])
        .map_err(|error| Error::Request(error.to_string()))?;
    let mut lines = head.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| Error::Request("missing request line".to_owned()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| Error::Request("missing method".to_owned()))?
        .parse::<Method>()
        .map_err(|error| Error::Request(error.to_string()))?;
    let target = parts
        .next()
        .ok_or_else(|| Error::Request("missing target".to_owned()))?
        .to_owned();
    let mut headers = HeaderMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| Error::Request(format!("invalid header `{line}`")))?;
        let name = HeaderName::from_bytes(name.trim().as_bytes())
            .map_err(|error| Error::Request(error.to_string()))?;
        let value = HeaderValue::from_str(value.trim())
            .map_err(|error| Error::Request(error.to_string()))?;
        headers.append(name, value);
    }
    let content_length = headers
        .get(http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .map(str::parse::<usize>)
        .transpose()
        .map_err(|error| Error::Request(error.to_string()))?
        .unwrap_or(0);
    let mut body = bytes[header_end..].to_vec();
    if body.len() < content_length {
        body.resize(content_length, 0);
        stream
            .read_exact(&mut body[bytes.len() - header_end..])
            .await?;
    }
    body.truncate(content_length);
    Ok(WireRequest {
        method,
        target,
        headers,
        body: Bytes::from(body),
    })
}

async fn write_response(stream: &mut TcpStream, response: &WireResponse) -> Result<(), Error> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\n",
        response.status.as_u16(),
        response.status.canonical_reason().unwrap_or("")
    );
    for (name, value) in &response.headers {
        let value = value
            .to_str()
            .map_err(|error| Error::Header(error.to_string()))?;
        head.push_str(name.as_str());
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
    stream.write_all(head.as_bytes()).await?;
    for chunk in &response.chunks {
        stream
            .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
            .await?;
        stream.write_all(chunk).await?;
        stream.write_all(b"\r\n").await?;
        stream.flush().await?;
        if !response.chunk_delay.is_zero() {
            tokio::time::sleep(response.chunk_delay).await;
        }
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await?;
    Ok(())
}

/// A deterministic one-shot barrier used to expose race ordering in tests.
#[derive(Debug)]
pub struct Barrier {
    entered: Option<oneshot::Receiver<()>>,
    release: Option<oneshot::Sender<()>>,
}

/// Handle held by the operation under test.
#[derive(Debug)]
pub struct BarrierPoint {
    entered: Option<oneshot::Sender<()>>,
    release: Option<oneshot::Receiver<()>>,
}

impl Barrier {
    /// Create a controller and operation-side point.
    pub fn pair() -> (Self, BarrierPoint) {
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        (
            Self {
                entered: Some(entered_rx),
                release: Some(release_tx),
            },
            BarrierPoint {
                entered: Some(entered_tx),
                release: Some(release_rx),
            },
        )
    }

    /// Wait until the operation reaches the point.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Request`] if entry was already observed, or [`Error::Closed`] if the
    /// operation-side point is dropped before entering.
    pub async fn entered(&mut self) -> Result<(), Error> {
        self.entered
            .take()
            .ok_or_else(|| Error::Request("barrier already observed".to_owned()))?
            .await
            .map_err(|_| Error::Closed)
    }

    /// Release the operation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Request`] if the operation was already released, or [`Error::Closed`] if
    /// the operation-side point was dropped.
    pub fn release(&mut self) -> Result<(), Error> {
        self.release
            .take()
            .ok_or_else(|| Error::Request("barrier already released".to_owned()))?
            .send(())
            .map_err(|()| Error::Closed)
    }
}

impl BarrierPoint {
    /// Enter the point and wait for the controller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Request`] if either side of the point was already used, or
    /// [`Error::Closed`] if the controller is dropped before entry or release.
    pub async fn wait(mut self) -> Result<(), Error> {
        self.entered
            .take()
            .ok_or_else(|| Error::Request("barrier point already entered".to_owned()))?
            .send(())
            .map_err(|()| Error::Closed)?;
        self.release
            .take()
            .ok_or_else(|| Error::Request("barrier point already released".to_owned()))?
            .await
            .map_err(|_| Error::Closed)
    }
}

#[cfg(test)]
mod tests {
    use http::StatusCode;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    use super::{Barrier, WireResponse, WireServer};

    #[tokio::test]
    async fn captures_requests_and_preserves_chunks() {
        let mut server = WireServer::start([WireResponse::sse(["data: {\"a\":", "1}\n\n"])])
            .await
            .unwrap();
        let mut stream = TcpStream::connect(server.address()).await.unwrap();
        stream
            .write_all(
                b"POST /v1/messages HTTP/1.1\r\nHost: local\r\nContent-Length: 7\r\n\r\n{\"x\":1}",
            )
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        assert!(
            response
                .windows(b"text/event-stream".len())
                .any(|part| part == b"text/event-stream")
        );
        let request = server.next().await.unwrap();
        assert_eq!(request.target, "/v1/messages");
        assert_eq!(request.body, "{\"x\":1}");
        server.close().await.unwrap();
    }

    #[tokio::test]
    async fn returns_json() {
        let response = WireResponse::json(StatusCode::OK, &json!({"ok": true})).unwrap();
        assert_eq!(response.chunks[0], r#"{"ok":true}"#);
    }

    #[tokio::test]
    async fn barrier_exposes_order() {
        let (mut controller, point) = Barrier::pair();
        let task = tokio::spawn(point.wait());
        controller.entered().await.unwrap();
        assert!(!task.is_finished());
        controller.release().unwrap();
        task.await.unwrap().unwrap();
    }
}
