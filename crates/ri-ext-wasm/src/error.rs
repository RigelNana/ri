//! Typed failures returned by the extension host.

use crate::policy::CapabilityKind;
use thiserror::Error;

/// Result type used by the extension host.
pub type Result<T, E = HostError> = std::result::Result<T, E>;

/// Failures that can occur while loading or invoking an extension component.
#[derive(Debug, Error)]
pub enum HostError {
    /// The host configuration is internally inconsistent or unsafe.
    #[error("invalid host limits: {0}")]
    InvalidLimits(String),

    /// The extension manifest did not pass structural validation.
    #[error("invalid extension manifest: {0}")]
    InvalidManifest(String),

    /// The component returned an invalid or inconsistent descriptor.
    #[error("invalid extension descriptor: {0}")]
    InvalidDescriptor(String),

    /// The manifest or descriptor targets an ABI this host does not support.
    #[error("unsupported extension ABI `{found}`; expected `{expected}`")]
    UnsupportedAbi {
        /// ABI version supplied by the extension.
        found: String,
        /// ABI version supported by this host.
        expected: &'static str,
    },

    /// The component exceeds the configured byte-size limit.
    #[error("component is {actual} bytes; the configured limit is {limit} bytes")]
    ComponentTooLarge {
        /// Actual encoded component size.
        actual: usize,
        /// Configured maximum component size.
        limit: usize,
    },

    /// Policy rejected a requested capability.
    #[error("capability `{kind}` denied: {reason}")]
    CapabilityDenied {
        /// Rejected capability category.
        kind: CapabilityKind,
        /// Human-readable denial reason.
        reason: String,
    },

    /// Wasmtime engine or linker setup failed.
    #[error("failed to configure the component host: {0}")]
    Configuration(String),

    /// The supplied bytes are not a valid component for this engine.
    #[error("failed to compile extension component: {0}")]
    Compilation(String),

    /// Required ABI imports could not be linked.
    #[error("failed to link extension component: {0}")]
    Linking(String),

    /// The component could not be instantiated.
    #[error("failed to instantiate extension component: {0}")]
    Instantiation(String),

    /// The guest returned a typed ABI error.
    #[error("extension returned `{code}`: {message}")]
    Guest {
        /// Stable guest error category.
        code: &'static str,
        /// Guest-provided detail.
        message: String,
    },

    /// Guest execution trapped for a reason other than a configured limit.
    #[error("extension trapped: {0}")]
    GuestTrap(String),

    /// Guest execution exhausted its deterministic fuel budget.
    #[error("extension exhausted its fuel budget")]
    FuelExhausted,

    /// Guest execution reached its epoch deadline.
    #[error("extension invocation exceeded {timeout_ms} ms")]
    Timeout {
        /// Configured call timeout.
        timeout_ms: u64,
    },

    /// A Wasm allocation exceeded the store resource limiter.
    #[error("extension exceeded its configured memory or table limit: {0}")]
    ResourceLimit(String),

    /// The integration bridge rejected or failed a host operation.
    #[error("extension bridge failed: {0}")]
    Bridge(String),

    /// An extension ID is not currently loaded.
    #[error("extension `{0}` is not loaded")]
    NotLoaded(String),

    /// A handle points at an extension generation that has been replaced.
    #[error(
        "stale handle for `{id}` generation {handle_generation}; current generation is {current_generation:?}"
    )]
    StaleHandle {
        /// Extension identifier embedded in the handle.
        id: String,
        /// Generation embedded in the handle.
        handle_generation: u64,
        /// Current generation, or `None` after unload.
        current_generation: Option<u64>,
    },

    /// A lifecycle call is not legal in the extension's current state.
    #[error("invalid lifecycle transition for `{id}`: {from} -> {operation}")]
    InvalidLifecycle {
        /// Extension identifier.
        id: String,
        /// Current lifecycle phase.
        from: &'static str,
        /// Attempted operation.
        operation: &'static str,
    },
}
