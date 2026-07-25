//! Typed RPC protocol and strict JSONL runtime plumbing.
//!
//! The crate deliberately stops at [`RpcDispatch`]. A product SDK can implement
//! that trait once and bind the same runtime to stdio, sockets, or an in-memory
//! transport without duplicating command semantics.

pub mod client;
pub mod codec;
pub mod protocol;
pub mod server;
pub mod transport;
pub mod types;

pub use client::{ClientDriver, ClientError, RpcClient};
pub use codec::{
    DEFAULT_MAX_FRAME_LEN, JsonlCodec, JsonlError, decode_jsonl, encode_json_line, encode_jsonl,
};
pub use protocol::*;
pub use server::{
    DispatchContext, DispatchError, ExtensionUi, ExtensionUiError, RpcDispatch, RpcServer,
    ServerError,
};
pub use transport::{
    ChannelTransport, JsonlTransport, RpcTransport, TransportError, channel_transport_pair,
};
pub use types::*;
