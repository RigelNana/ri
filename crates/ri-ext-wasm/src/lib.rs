//! Sandboxed WebAssembly Component Model host for ri extensions.
//!
//! The ABI is defined by `wit/ri-extension.wit` and versioned as
//! `ri:extension@1.0.0`. The host intentionally does not add WASI to its
//! linker. Extensions can only reach host services through the six explicit
//! capability resources in that world.

mod bridge;
mod capability;
mod descriptor;
mod error;
mod host;
mod limits;
mod model;
mod policy;

#[allow(missing_docs)]
pub mod bindings;

pub use bridge::{
    BridgeCall, BridgeError, BridgeErrorCode, DescriptorPublication, DescriptorRetirement,
    NoAmbientBridge, RiExtBridge,
};
pub use capability::{
    FilesystemResource, NetworkResource, ProcessResource, ProviderResource, SessionResource,
    UiResource,
};
pub use descriptor::{
    ABI_VERSION, CommandPlacement, CommandRegistration, ExtensionDescriptor, ExtensionManifest,
    MANIFEST_SCHEMA_VERSION, ToolRegistration, ViewLocation, ViewRegistration,
};
pub use error::{HostError, Result};
pub use host::{ExtensionHandle, ExtensionStatus, WasmExtensionHost, WasmExtensionHostBuilder};
pub use limits::HostLimits;
pub use model::{
    ActionBinding, ActionEvent, ActionKind, ActionResult, ActivationContext, ActivationResult,
    CommandInvocation, CommandResult, DeactivateReason, EventKind, ExtensionEvent, Invocation,
    InvocationResult, LifecyclePhase, ToolInvocation, ToolResult, View, ViewNode, ViewNodeKind,
    ViewProperty, ViewRequest,
};
pub use policy::{
    CapabilityKind, CapabilityPolicy, CapabilityRequest, GrantedCapabilities, ScopeRule,
    scope_contains, validate_scope,
};
