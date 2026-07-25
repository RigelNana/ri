//! Pi RPC wire compatibility over the native typed protocol.

pub use ri_rpc::{
    AssistantMessageEvent as PiAssistantMessageEvent, ClientFrame as PiClientFrame,
    Command as PiRpcCommand, CommandName as PiRpcCommandName, Event as PiRpcEvent,
    ExtensionUiAction as PiExtensionUiAction, ExtensionUiRequest as PiExtensionUiRequest,
    ExtensionUiResponse as PiExtensionUiResponse, ExtensionUiResult as PiExtensionUiResult,
    JsonlCodec as PiRpcCodec, JsonlError as PiRpcCodecError, Request as PiRpcRequest,
    RequestId as PiRpcRequestId, Response as PiRpcResponse,
    ResponsePayload as PiRpcResponsePayload, ServerFrame as PiServerFrame,
};

/// Decode caller-provided Pi stdin records with strict LF framing.
///
/// # Errors
///
/// Returns an error for malformed, oversized, or type-invalid JSONL records.
pub fn decode_pi_client_jsonl(input: &[u8]) -> Result<Vec<PiClientFrame>, PiRpcCodecError> {
    ri_rpc::decode_jsonl(input)
}

/// Decode caller-provided Pi stdout records with strict LF framing.
///
/// # Errors
///
/// Returns an error for malformed, oversized, or type-invalid JSONL records.
pub fn decode_pi_server_jsonl(input: &[u8]) -> Result<Vec<PiServerFrame>, PiRpcCodecError> {
    ri_rpc::decode_jsonl(input)
}

/// Encode Pi-compatible stdin records with LF terminators.
///
/// # Errors
///
/// Returns an error if a frame cannot be serialized or exceeds the codec's
/// record-size limit.
pub fn encode_pi_client_jsonl(
    frames: impl IntoIterator<Item = PiClientFrame>,
) -> Result<Vec<u8>, PiRpcCodecError> {
    ri_rpc::encode_jsonl(frames)
}

/// Encode Pi-compatible stdout records with LF terminators.
///
/// # Errors
///
/// Returns an error if a frame cannot be serialized or exceeds the codec's
/// record-size limit.
pub fn encode_pi_server_jsonl(
    frames: impl IntoIterator<Item = PiServerFrame>,
) -> Result<Vec<u8>, PiRpcCodecError> {
    ri_rpc::encode_jsonl(frames)
}
