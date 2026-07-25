//! Narrow JSON boundary between this sandbox and the native extension host.

use crate::policy::CapabilityKind;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use thiserror::Error;

/// Generation-tagged descriptor publication sent to the native host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptorPublication {
    /// Stable extension ID.
    pub extension_id: String,
    /// Host-assigned generation that owns these registrations.
    pub generation: u64,
    /// Validated [`crate::ExtensionDescriptor`] serialized as JSON.
    pub descriptor: Value,
}

/// Generation-tagged descriptor retirement sent during unload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorRetirement {
    /// Stable extension ID.
    pub extension_id: String,
    /// Host-assigned generation to retire.
    pub generation: u64,
}

/// A capability operation forwarded to the native host.
///
/// The payload and response are JSON so this crate does not depend on
/// `ri-ext` internals. An adapter in the native layer can deserialize the
/// operation into its public request types and serialize its response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeCall {
    /// Calling extension ID.
    pub extension_id: String,
    /// Calling extension generation.
    pub generation: u64,
    /// Capability category.
    pub capability: CapabilityKind,
    /// Stable operation name within that category.
    pub operation: String,
    /// Effective resource scope.
    pub scope: Value,
    /// Operation-specific request payload.
    pub payload: Value,
}

/// Stable bridge failure classes mapped to WIT `capability-error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BridgeErrorCode {
    /// Policy or user consent denied the operation.
    Denied,
    /// The resource scope is malformed.
    InvalidScope,
    /// The operation payload is malformed.
    InvalidRequest,
    /// The native service is not available.
    Unavailable,
    /// A native service limit was exceeded.
    LimitExceeded,
    /// The native service failed.
    Failed,
}

impl fmt::Display for BridgeErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Denied => "denied",
            Self::InvalidScope => "invalid-scope",
            Self::InvalidRequest => "invalid-request",
            Self::Unavailable => "unavailable",
            Self::LimitExceeded => "limit-exceeded",
            Self::Failed => "failed",
        })
    }
}

/// Error returned by a native host bridge.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct BridgeError {
    /// Stable error class.
    pub code: BridgeErrorCode,
    /// Safe detail returned to the extension.
    pub message: String,
}

impl BridgeError {
    /// Constructs a bridge error.
    pub fn new(code: BridgeErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Adapter implemented by the native extension layer.
#[async_trait]
pub trait RiExtBridge: Send + Sync + fmt::Debug {
    /// Atomically publishes a validated generation's registrations.
    async fn publish_descriptor(
        &self,
        publication: DescriptorPublication,
    ) -> std::result::Result<(), BridgeError>;

    /// Retires registrations only if their generation still matches.
    async fn retire_descriptor(
        &self,
        retirement: DescriptorRetirement,
    ) -> std::result::Result<(), BridgeError>;

    /// Executes one explicitly granted capability operation.
    async fn call(&self, call: BridgeCall) -> std::result::Result<Value, BridgeError>;
}

/// Default bridge. It publishes no metadata and exposes no native services.
///
/// Using this bridge can instantiate components with no capability requests,
/// but every capability operation is denied even if a permissive policy was
/// accidentally supplied.
#[derive(Debug, Default)]
pub struct NoAmbientBridge;

#[async_trait]
impl RiExtBridge for NoAmbientBridge {
    async fn publish_descriptor(
        &self,
        _publication: DescriptorPublication,
    ) -> std::result::Result<(), BridgeError> {
        Ok(())
    }

    async fn retire_descriptor(
        &self,
        _retirement: DescriptorRetirement,
    ) -> std::result::Result<(), BridgeError> {
        Ok(())
    }

    async fn call(&self, call: BridgeCall) -> std::result::Result<Value, BridgeError> {
        Err(BridgeError::new(
            BridgeErrorCode::Unavailable,
            format!(
                "no native bridge is configured for {}.{}",
                call.capability, call.operation
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn default_bridge_exposes_no_ambient_service() {
        let error = NoAmbientBridge
            .call(BridgeCall {
                extension_id: "dev.ri.test".to_owned(),
                generation: 1,
                capability: CapabilityKind::Filesystem,
                operation: "read".to_owned(),
                scope: json!({"roots": ["C:/workspace"]}),
                payload: json!({"path": "C:/workspace/file.txt"}),
            })
            .await
            .expect_err("default bridge must not perform host I/O");
        assert_eq!(error.code, BridgeErrorCode::Unavailable);
    }
}
