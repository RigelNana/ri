//! Injectable HTTP transport and standards-compliant SSE decoding.

use std::{collections::BTreeMap, pin::Pin, str::FromStr, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use http::Method;
use indexmap::IndexMap;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::error::AiError;

/// Case-normalized HTTP headers.
pub type HttpHeaders = BTreeMap<String, String>;

/// Transport-neutral HTTP request.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: Method,
    /// Absolute URL.
    pub url: Url,
    /// Request headers.
    pub headers: HttpHeaders,
    /// Optional request body.
    pub body: Bytes,
    /// Whole-request timeout.
    pub timeout: Option<Duration>,
    /// Cooperative cancellation.
    pub cancellation: Option<CancellationToken>,
}

impl HttpRequest {
    /// Creates a JSON POST request.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the body cannot be serialized as JSON.
    pub fn json(url: Url, body: &serde_json::Value) -> Result<Self, AiError> {
        Ok(Self {
            method: Method::POST,
            url,
            headers: BTreeMap::from([
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ]),
            body: Bytes::from(
                serde_json::to_vec(body).map_err(|error| AiError::Validation(error.to_string()))?,
            ),
            timeout: None,
            cancellation: None,
        })
    }
}

/// Buffered HTTP response.
#[derive(Clone, Debug)]
pub struct HttpResponse {
    /// HTTP status.
    pub status: u16,
    /// Response headers.
    pub headers: HttpHeaders,
    /// Response bytes.
    pub body: Bytes,
}

impl HttpResponse {
    /// Parses the response body as JSON.
    ///
    /// # Errors
    ///
    /// Returns a stream error when the buffered body is not valid JSON for
    /// the requested type.
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, AiError> {
        serde_json::from_slice(&self.body)
            .map_err(|error| AiError::Stream(format!("invalid JSON response: {error}")))
    }
}

/// One Server-Sent Event frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseFrame {
    /// Optional event name.
    pub event: Option<String>,
    /// Newline-joined `data:` fields.
    pub data: String,
    /// Last event id.
    pub id: Option<String>,
    /// Server retry hint.
    pub retry: Option<u64>,
    /// Raw non-blank lines, useful in diagnostics.
    pub raw: Vec<String>,
}

/// Boxed SSE event stream.
pub type SseEventStream = Pin<Box<dyn Stream<Item = Result<SseFrame, AiError>> + Send + 'static>>;

/// Streaming HTTP response metadata and decoded events.
pub struct SseResponse {
    /// HTTP status.
    pub status: u16,
    /// Response headers.
    pub headers: HttpHeaders,
    /// Decoded SSE frames.
    pub events: SseEventStream,
}

impl std::fmt::Debug for SseResponse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SseResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

/// Injectable HTTP execution contract.
#[async_trait]
pub trait HttpTransport: Send + Sync + std::fmt::Debug {
    /// Executes and buffers a normal response.
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, AiError>;
    /// Executes a request and incrementally decodes its body as SSE.
    async fn execute_sse(&self, request: HttpRequest) -> Result<SseResponse, AiError>;
    /// Executes a request and incrementally decodes AWS event-stream framing.
    async fn execute_aws_event_stream(
        &self,
        _request: HttpRequest,
    ) -> Result<SseResponse, AiError> {
        Err(AiError::Http(
            "transport does not support AWS event-stream responses".into(),
        ))
    }
}

/// Production reqwest transport.
#[derive(Clone, Debug)]
pub struct ReqwestTransport {
    client: reqwest::Client,
    max_error_body: usize,
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self {
            client: reqwest::Client::new(),
            max_error_body: 64 * 1024,
        }
    }
}

impl ReqwestTransport {
    /// Wraps a configured reqwest client.
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client,
            ..Self::default()
        }
    }

    /// Changes the maximum provider error body retained in memory.
    #[must_use]
    pub fn with_max_error_body(mut self, bytes: usize) -> Self {
        self.max_error_body = bytes;
        self
    }

    fn build(&self, request: &HttpRequest) -> Result<reqwest::RequestBuilder, AiError> {
        let mut builder = self
            .client
            .request(request.method.clone(), request.url.clone());
        for (name, value) in &request.headers {
            let name = reqwest::header::HeaderName::from_str(name)
                .map_err(|error| AiError::Validation(format!("invalid header {name}: {error}")))?;
            let value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|error| AiError::Validation(format!("invalid header value: {error}")))?;
            builder = builder.header(name, value);
        }
        if !request.body.is_empty() {
            builder = builder.body(request.body.clone());
        }
        if let Some(timeout) = request.timeout {
            builder = builder.timeout(timeout);
        }
        Ok(builder)
    }

    async fn send(&self, request: &HttpRequest) -> Result<reqwest::Response, AiError> {
        let builder = self.build(request)?;
        match &request.cancellation {
            Some(cancellation) => {
                tokio::select! {
                    () = cancellation.cancelled() => Err(AiError::Aborted),
                    response = builder.send() => {
                        response.map_err(|error| map_reqwest_error(&error))
                    },
                }
            }
            None => builder
                .send()
                .await
                .map_err(|error| map_reqwest_error(&error)),
        }
    }

    async fn response_error(&self, response: reqwest::Response) -> AiError {
        let status = response.status().as_u16();
        let reason = response
            .status()
            .canonical_reason()
            .unwrap_or("provider request failed")
            .to_owned();
        let body = response
            .bytes()
            .await
            .map(|body| {
                let end = body.len().min(self.max_error_body);
                String::from_utf8_lossy(&body[..end]).into_owned()
            })
            .unwrap_or_default();
        let message = provider_error_message(&body).unwrap_or_else(|| {
            if body.trim().is_empty() {
                reason
            } else {
                body.clone()
            }
        });
        AiError::ProviderResponse {
            status,
            message,
            body: (!body.is_empty()).then_some(body),
        }
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, AiError> {
        let response = self.send(&request).await?;
        if !response.status().is_success() {
            return Err(self.response_error(response).await);
        }
        let status = response.status().as_u16();
        let headers = response_headers(response.headers());
        let body = match &request.cancellation {
            Some(cancellation) => {
                tokio::select! {
                    () = cancellation.cancelled() => return Err(AiError::Aborted),
                    body = response.bytes() => {
                        body.map_err(|error| map_reqwest_error(&error))?
                    },
                }
            }
            None => response
                .bytes()
                .await
                .map_err(|error| map_reqwest_error(&error))?,
        };
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    async fn execute_sse(&self, request: HttpRequest) -> Result<SseResponse, AiError> {
        let response = self.send(&request).await?;
        if !response.status().is_success() {
            return Err(self.response_error(response).await);
        }
        let status = response.status().as_u16();
        let headers = response_headers(response.headers());
        let cancellation = request.cancellation;
        let mut bytes = response.bytes_stream();
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut decoder = SseDecoder::default();
            loop {
                let next = match &cancellation {
                    Some(cancellation) => {
                        tokio::select! {
                            () = cancellation.cancelled() => {
                                let _ = sender.send(Err(AiError::Aborted));
                                return;
                            }
                            next = bytes.next() => next,
                        }
                    }
                    None => bytes.next().await,
                };
                let Some(next) = next else {
                    break;
                };
                match next {
                    Ok(chunk) => match decoder.push(&chunk) {
                        Ok(frames) => {
                            for frame in frames {
                                if sender.send(Ok(frame)).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    },
                    Err(error) => {
                        let _ = sender.send(Err(map_reqwest_error(&error)));
                        return;
                    }
                }
            }
            match decoder.finish() {
                Ok(frames) => {
                    for frame in frames {
                        if sender.send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                }
            }
        });
        Ok(SseResponse {
            status,
            headers,
            events: Box::pin(UnboundedReceiverStream::new(receiver)),
        })
    }

    async fn execute_aws_event_stream(&self, request: HttpRequest) -> Result<SseResponse, AiError> {
        let response = self.send(&request).await?;
        if !response.status().is_success() {
            return Err(self.response_error(response).await);
        }
        let status = response.status().as_u16();
        let headers = response_headers(response.headers());
        let cancellation = request.cancellation;
        let mut bytes = response.bytes_stream();
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut decoder = AwsEventStreamDecoder::default();
            loop {
                let next = match &cancellation {
                    Some(cancellation) => {
                        tokio::select! {
                            () = cancellation.cancelled() => {
                                let _ = sender.send(Err(AiError::Aborted));
                                return;
                            }
                            next = bytes.next() => next,
                        }
                    }
                    None => bytes.next().await,
                };
                let Some(next) = next else {
                    break;
                };
                match next {
                    Ok(chunk) => match decoder.push(&chunk) {
                        Ok(frames) => {
                            for frame in frames {
                                if sender.send(Ok(frame)).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    },
                    Err(error) => {
                        let _ = sender.send(Err(map_reqwest_error(&error)));
                        return;
                    }
                }
            }
            if let Err(error) = decoder.finish() {
                let _ = sender.send(Err(error));
            }
        });
        Ok(SseResponse {
            status,
            headers,
            events: Box::pin(UnboundedReceiverStream::new(receiver)),
        })
    }
}

fn response_headers(headers: &reqwest::header::HeaderMap) -> HttpHeaders {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_owned(),
                value.to_str().unwrap_or_default().to_owned(),
            )
        })
        .collect()
}

fn map_reqwest_error(error: &reqwest::Error) -> AiError {
    if error.is_timeout() {
        AiError::Http(format!("request timed out: {error}"))
    } else {
        AiError::Http(error.to_string())
    }
}

fn provider_error_message(body: &str) -> Option<String> {
    let body = serde_json::from_str::<serde_json::Value>(body).ok()?;
    body.pointer("/error/message")
        .or_else(|| body.get("message"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// Incremental SSE decoder supporting CRLF, multiline data, comments, ids,
/// retry hints, and a trailing event without a final blank line.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    id: Option<String>,
    retry: Option<u64>,
    raw: Vec<String>,
}

impl SseDecoder {
    /// Decodes complete lines from a byte chunk.
    ///
    /// # Errors
    ///
    /// Returns a stream error when a completed SSE line is not valid UTF-8.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, AiError> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=newline).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8(line)
                .map_err(|error| AiError::Stream(format!("SSE was not UTF-8: {error}")))?;
            if let Some(frame) = self.consume_line(&line) {
                frames.push(frame);
            }
        }
        Ok(frames)
    }

    /// Flushes a trailing line/event at EOF.
    ///
    /// # Errors
    ///
    /// Returns a stream error when the trailing SSE line is not valid UTF-8.
    pub fn finish(&mut self) -> Result<Vec<SseFrame>, AiError> {
        let mut frames = Vec::new();
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8(line)
                .map_err(|error| AiError::Stream(format!("SSE was not UTF-8: {error}")))?;
            if let Some(frame) = self.consume_line(&line) {
                frames.push(frame);
            }
        }
        if let Some(frame) = self.flush() {
            frames.push(frame);
        }
        Ok(frames)
    }

    fn consume_line(&mut self, line: &str) -> Option<SseFrame> {
        if line.is_empty() {
            return self.flush();
        }
        self.raw.push(line.to_owned());
        if line.starts_with(':') {
            return None;
        }
        let (field, mut value) = line
            .split_once(':')
            .map_or((line, ""), |(field, value)| (field, value));
        if let Some(stripped) = value.strip_prefix(' ') {
            value = stripped;
        }
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            "id" if !value.contains('\0') => self.id = Some(value.to_owned()),
            "retry" => self.retry = value.parse().ok(),
            _ => {}
        }
        None
    }

    fn flush(&mut self) -> Option<SseFrame> {
        if self.data.is_empty() && self.event.is_none() && self.raw.is_empty() {
            return None;
        }
        Some(SseFrame {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
            id: self.id.clone(),
            retry: self.retry.take(),
            raw: std::mem::take(&mut self.raw),
        })
    }
}

/// Incremental decoder for the AWS binary event-stream framing protocol.
///
/// Bedrock's `ConverseStream` response uses this framing instead of SSE. Each
/// decoded event is projected into [`SseFrame`]: `event` contains the Smithy
/// `:event-type` header and `data` contains the UTF-8 JSON payload.
#[derive(Debug, Default)]
pub struct AwsEventStreamDecoder {
    buffer: BytesMut,
}

impl AwsEventStreamDecoder {
    /// Feed a response chunk and return every complete AWS event.
    ///
    /// # Errors
    ///
    /// Returns a stream error for invalid lengths, CRC mismatches, malformed
    /// headers, or non-UTF-8 event payloads.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseFrame>, AiError> {
        const PRELUDE_LEN: usize = 12;
        const TRAILER_LEN: usize = 4;
        const MAX_MESSAGE_LEN: usize = 16 * 1024 * 1024;

        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        loop {
            if self.buffer.len() < PRELUDE_LEN {
                break;
            }
            let total_len = usize::try_from(u32::from_be_bytes(
                self.buffer[0..4]
                    .try_into()
                    .map_err(|_| AiError::Stream("invalid AWS event prelude".into()))?,
            ))
            .map_err(|_| AiError::Stream("AWS event length does not fit usize".into()))?;
            let headers_len = usize::try_from(u32::from_be_bytes(
                self.buffer[4..8]
                    .try_into()
                    .map_err(|_| AiError::Stream("invalid AWS event prelude".into()))?,
            ))
            .map_err(|_| AiError::Stream("AWS header length does not fit usize".into()))?;
            if !(PRELUDE_LEN + TRAILER_LEN..=MAX_MESSAGE_LEN).contains(&total_len)
                || headers_len > total_len - PRELUDE_LEN - TRAILER_LEN
            {
                return Err(AiError::Stream("invalid AWS event-stream lengths".into()));
            }
            if self.buffer.len() < total_len {
                break;
            }
            let message = self.buffer.split_to(total_len).freeze();
            let expected_prelude_crc = u32::from_be_bytes(
                message[8..12]
                    .try_into()
                    .map_err(|_| AiError::Stream("invalid AWS prelude CRC".into()))?,
            );
            if crc32(&message[..8]) != expected_prelude_crc {
                return Err(AiError::Stream("AWS event prelude CRC mismatch".into()));
            }
            let expected_message_crc = u32::from_be_bytes(
                message[total_len - TRAILER_LEN..]
                    .try_into()
                    .map_err(|_| AiError::Stream("invalid AWS message CRC".into()))?,
            );
            if crc32(&message[..total_len - TRAILER_LEN]) != expected_message_crc {
                return Err(AiError::Stream("AWS event message CRC mismatch".into()));
            }
            let headers = decode_aws_headers(&message[PRELUDE_LEN..PRELUDE_LEN + headers_len])?;
            let payload =
                std::str::from_utf8(&message[PRELUDE_LEN + headers_len..total_len - TRAILER_LEN])
                    .map_err(|error| {
                        AiError::Stream(format!("AWS event payload was not UTF-8: {error}"))
                    })?
                    .to_owned();
            frames.push(SseFrame {
                event: headers.get(":event-type").cloned().or_else(|| {
                    headers
                        .get(":exception-type")
                        .map(|kind| format!("exception:{kind}"))
                }),
                id: None,
                retry: None,
                data: payload,
                raw: Vec::new(),
            });
        }
        Ok(frames)
    }

    /// Verify that no truncated event remains after EOF.
    ///
    /// # Errors
    ///
    /// Returns a stream error when EOF arrives with an incomplete AWS event.
    pub fn finish(self) -> Result<(), AiError> {
        if self.buffer.is_empty() {
            Ok(())
        } else {
            Err(AiError::Stream(
                "Bedrock event stream ended with a truncated frame".into(),
            ))
        }
    }
}

fn decode_aws_headers(bytes: &[u8]) -> Result<IndexMap<String, String>, AiError> {
    let mut headers = IndexMap::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let name_len = usize::from(
            *bytes
                .get(offset)
                .ok_or_else(|| AiError::Stream("truncated AWS event header".into()))?,
        );
        offset += 1;
        let name_end = offset.saturating_add(name_len);
        let name = std::str::from_utf8(
            bytes
                .get(offset..name_end)
                .ok_or_else(|| AiError::Stream("truncated AWS event header name".into()))?,
        )
        .map_err(|error| AiError::Stream(format!("invalid AWS header name: {error}")))?
        .to_owned();
        offset = name_end;
        let kind = *bytes
            .get(offset)
            .ok_or_else(|| AiError::Stream("truncated AWS event header type".into()))?;
        offset += 1;
        let value = match kind {
            // bool true / false
            0 => "true".into(),
            1 => "false".into(),
            // byte
            2 => {
                let value = *bytes
                    .get(offset)
                    .ok_or_else(|| AiError::Stream("truncated AWS byte header".into()))?;
                offset += 1;
                value.to_string()
            }
            // short
            3 => {
                let end = offset + 2;
                let value = i16::from_be_bytes(
                    bytes
                        .get(offset..end)
                        .ok_or_else(|| AiError::Stream("truncated AWS short header".into()))?
                        .try_into()
                        .map_err(|_| AiError::Stream("invalid AWS short header".into()))?,
                );
                offset = end;
                value.to_string()
            }
            // integer
            4 => {
                let end = offset + 4;
                let value = i32::from_be_bytes(
                    bytes
                        .get(offset..end)
                        .ok_or_else(|| AiError::Stream("truncated AWS integer header".into()))?
                        .try_into()
                        .map_err(|_| AiError::Stream("invalid AWS integer header".into()))?,
                );
                offset = end;
                value.to_string()
            }
            // long / timestamp
            5 | 8 => {
                let end = offset + 8;
                let value = i64::from_be_bytes(
                    bytes
                        .get(offset..end)
                        .ok_or_else(|| AiError::Stream("truncated AWS long header".into()))?
                        .try_into()
                        .map_err(|_| AiError::Stream("invalid AWS long header".into()))?,
                );
                offset = end;
                value.to_string()
            }
            // bytes / string
            6 | 7 => {
                let length_end = offset + 2;
                let length = usize::from(u16::from_be_bytes(
                    bytes
                        .get(offset..length_end)
                        .ok_or_else(|| AiError::Stream("truncated AWS header length".into()))?
                        .try_into()
                        .map_err(|_| AiError::Stream("invalid AWS header length".into()))?,
                ));
                offset = length_end;
                let end = offset.saturating_add(length);
                let value = bytes
                    .get(offset..end)
                    .ok_or_else(|| AiError::Stream("truncated AWS header value".into()))?;
                offset = end;
                if kind == 7 {
                    std::str::from_utf8(value)
                        .map_err(|error| {
                            AiError::Stream(format!("invalid AWS string header: {error}"))
                        })?
                        .to_owned()
                } else {
                    use base64::{Engine as _, engine::general_purpose::STANDARD};
                    STANDARD.encode(value)
                }
            }
            // UUID
            9 => {
                const HEX: &[u8; 16] = b"0123456789abcdef";

                let end = offset + 16;
                let value = bytes
                    .get(offset..end)
                    .ok_or_else(|| AiError::Stream("truncated AWS UUID header".into()))?;
                offset = end;
                let mut encoded = String::with_capacity(36);
                for (index, byte) in value.iter().enumerate() {
                    if matches!(index, 4 | 6 | 8 | 10) {
                        encoded.push('-');
                    }
                    encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                    encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
                }
                encoded
            }
            _ => {
                return Err(AiError::Stream(format!(
                    "unsupported AWS event header type {kind}"
                )));
            }
        };
        headers.insert(name, value);
    }
    Ok(headers)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Shared transport handle.
pub type DynHttpTransport = Arc<dyn HttpTransport>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_split_multiline_crlf_frames() {
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .push(b"event: message\r\ndata: {\"a\":")
                .expect("first chunk")
                .is_empty()
        );
        let frames = decoder
            .push(b"1}\r\ndata: second\r\nid: 7\r\n\r\n")
            .expect("second chunk");
        assert_eq!(
            frames,
            vec![SseFrame {
                event: Some("message".into()),
                data: "{\"a\":1}\nsecond".into(),
                id: Some("7".into()),
                retry: None,
                raw: vec![
                    "event: message".into(),
                    "data: {\"a\":1}".into(),
                    "data: second".into(),
                    "id: 7".into(),
                ],
            }]
        );
    }

    #[test]
    fn flushes_trailing_event_and_ignores_comments() {
        let mut decoder = SseDecoder::default();
        let frames = decoder.push(b": heartbeat\ndata: final").expect("decode");
        assert!(frames.is_empty());
        let frames = decoder.finish().expect("finish");
        assert_eq!(frames[0].data, "final");
    }

    #[test]
    fn extracts_nested_provider_error() {
        assert_eq!(
            provider_error_message(r#"{"error":{"message":"bad request"}}"#).as_deref(),
            Some("bad request")
        );
    }

    #[test]
    fn decodes_split_aws_event_stream_frames_and_checks_crc() {
        let name = b":event-type";
        let value = b"metadata";
        let mut headers = Vec::new();
        headers.push(u8::try_from(name.len()).expect("test header name fits in u8"));
        headers.extend_from_slice(name);
        headers.push(7);
        headers.extend_from_slice(
            &u16::try_from(value.len())
                .expect("test header value fits in u16")
                .to_be_bytes(),
        );
        headers.extend_from_slice(value);
        let payload = br#"{"usage":{"inputTokens":2}}"#;
        let total_len = 12 + headers.len() + payload.len() + 4;
        let mut message = Vec::new();
        message.extend_from_slice(
            &u32::try_from(total_len)
                .expect("test frame length fits in u32")
                .to_be_bytes(),
        );
        message.extend_from_slice(
            &u32::try_from(headers.len())
                .expect("test header length fits in u32")
                .to_be_bytes(),
        );
        let prelude_crc = crc32(&message[..8]);
        message.extend_from_slice(&prelude_crc.to_be_bytes());
        message.extend_from_slice(&headers);
        message.extend_from_slice(payload);
        let message_crc = crc32(&message);
        message.extend_from_slice(&message_crc.to_be_bytes());

        let mut decoder = AwsEventStreamDecoder::default();
        assert!(decoder.push(&message[..7]).expect("partial").is_empty());
        let frames = decoder.push(&message[7..]).expect("complete");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("metadata"));
        assert_eq!(frames[0].data, String::from_utf8_lossy(payload));
        decoder.finish().expect("clean end");

        let mut corrupt = message;
        corrupt[12] ^= 1;
        assert!(AwsEventStreamDecoder::default().push(&corrupt).is_err());
    }
}
