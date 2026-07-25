//! Strict LF-delimited JSONL framing.

use std::io;
use std::marker::PhantomData;

use bytes::BytesMut;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio_util::codec::{Decoder, Encoder};

/// Default maximum encoded JSON record size (16 MiB).
pub const DEFAULT_MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

/// JSONL encoding or decoding failure.
#[derive(Debug, thiserror::Error)]
pub enum JsonlError {
    /// Underlying asynchronous I/O failure.
    #[error("JSONL I/O failed: {0}")]
    Io(#[from] io::Error),
    /// A complete LF-delimited record was not valid JSON for the requested type.
    #[error("invalid JSONL record: {0}")]
    Json(#[from] serde_json::Error),
    /// A record exceeded the configured bound.
    #[error("JSONL record length {length} exceeds maximum {maximum}")]
    FrameTooLarge {
        /// Observed record length.
        length: usize,
        /// Configured maximum.
        maximum: usize,
    },
}

impl JsonlError {
    /// Whether this error belongs to one consumed record and decoding may continue.
    pub const fn is_recoverable_record_error(&self) -> bool {
        matches!(self, Self::Json(_) | Self::FrameTooLarge { .. })
    }
}

/// A strict JSONL codec with different incoming and outgoing types.
///
/// Only byte `LF` delimits records. A single `CR` immediately before the
/// delimiter (or before EOF for a final unterminated record) is stripped.
/// Unicode U+2028 and U+2029 are ordinary UTF-8 payload bytes.
#[derive(Debug, Clone)]
pub struct JsonlCodec<Incoming, Outgoing> {
    max_frame_len: usize,
    marker: PhantomData<fn() -> (Incoming, Outgoing)>,
}

impl<Incoming, Outgoing> Default for JsonlCodec<Incoming, Outgoing> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Incoming, Outgoing> JsonlCodec<Incoming, Outgoing> {
    /// Construct a codec with [`DEFAULT_MAX_FRAME_LEN`].
    pub const fn new() -> Self {
        Self {
            max_frame_len: DEFAULT_MAX_FRAME_LEN,
            marker: PhantomData,
        }
    }

    /// Construct a codec with an explicit maximum record length.
    pub const fn with_max_frame_len(max_frame_len: usize) -> Self {
        Self {
            max_frame_len,
            marker: PhantomData,
        }
    }

    /// Return the configured record-size limit.
    pub const fn max_frame_len(&self) -> usize {
        self.max_frame_len
    }

    fn decode_record(&self, mut record: BytesMut) -> Result<Incoming, JsonlError>
    where
        Incoming: DeserializeOwned,
    {
        if record.last() == Some(&b'\r') {
            record.truncate(record.len() - 1);
        }
        if record.len() > self.max_frame_len {
            return Err(JsonlError::FrameTooLarge {
                length: record.len(),
                maximum: self.max_frame_len,
            });
        }
        serde_json::from_slice(&record).map_err(JsonlError::from)
    }
}

impl<Incoming, Outgoing> Decoder for JsonlCodec<Incoming, Outgoing>
where
    Incoming: DeserializeOwned,
{
    type Item = Incoming;
    type Error = JsonlError;

    fn decode(&mut self, source: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if let Some(newline) = source.iter().position(|byte| *byte == b'\n') {
            let mut record = source.split_to(newline + 1);
            record.truncate(newline);
            return self.decode_record(record).map(Some);
        }

        if source.len() > self.max_frame_len {
            let length = source.len();
            source.clear();
            return Err(JsonlError::FrameTooLarge {
                length,
                maximum: self.max_frame_len,
            });
        }

        Ok(None)
    }

    fn decode_eof(&mut self, source: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if let Some(item) = self.decode(source)? {
            return Ok(Some(item));
        }
        if source.is_empty() {
            return Ok(None);
        }
        let record = source.split_to(source.len());
        self.decode_record(record).map(Some)
    }
}

impl<Incoming, Outgoing> Encoder<Outgoing> for JsonlCodec<Incoming, Outgoing>
where
    Outgoing: Serialize,
{
    type Error = JsonlError;

    fn encode(&mut self, item: Outgoing, destination: &mut BytesMut) -> Result<(), Self::Error> {
        let encoded = serde_json::to_vec(&item)?;
        if encoded.len() > self.max_frame_len {
            return Err(JsonlError::FrameTooLarge {
                length: encoded.len(),
                maximum: self.max_frame_len,
            });
        }
        destination.reserve(encoded.len() + 1);
        destination.extend_from_slice(&encoded);
        destination.extend_from_slice(b"\n");
        Ok(())
    }
}

/// Decode all strict JSONL records, including a final record without LF.
///
/// # Errors
///
/// Returns [`JsonlError::Json`] for an invalid record or
/// [`JsonlError::FrameTooLarge`] when a record exceeds the default limit.
pub fn decode_jsonl<T>(input: &[u8]) -> Result<Vec<T>, JsonlError>
where
    T: DeserializeOwned,
{
    let mut codec = JsonlCodec::<T, ()>::new();
    let mut source = BytesMut::from(input);
    let mut records = Vec::new();

    while let Some(record) = codec.decode(&mut source)? {
        records.push(record);
    }
    if let Some(record) = codec.decode_eof(&mut source)? {
        records.push(record);
    }
    Ok(records)
}

/// Encode values as strict LF-terminated JSONL records.
///
/// # Errors
///
/// Returns [`JsonlError::Json`] if a value cannot be serialized or
/// [`JsonlError::FrameTooLarge`] when its encoded record exceeds the default
/// limit.
pub fn encode_jsonl<T, I>(values: I) -> Result<Vec<u8>, JsonlError>
where
    T: Serialize,
    I: IntoIterator<Item = T>,
{
    let mut codec = JsonlCodec::<(), T>::new();
    let mut output = BytesMut::new();
    for value in values {
        codec.encode(value, &mut output)?;
    }
    Ok(output.to_vec())
}

/// Encode one strict LF-terminated JSONL record.
///
/// # Errors
///
/// Returns [`JsonlError::Json`] if `value` cannot be serialized or
/// [`JsonlError::FrameTooLarge`] when its encoded record exceeds the default
/// limit.
pub fn encode_json_line<T>(value: T) -> Result<Vec<u8>, JsonlError>
where
    T: Serialize,
{
    encode_jsonl(std::iter::once(value))
}
