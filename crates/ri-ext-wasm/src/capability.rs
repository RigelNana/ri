//! Implementations of the six imported capability resources.

use crate::bindings::ri::extension::{filesystem, network, process, provider, session, types, ui};
use crate::bridge::{BridgeCall, BridgeError, BridgeErrorCode, RiExtBridge};
use crate::limits::HostLimits;
use crate::policy::{CapabilityKind, GrantedCapabilities, validate_scope};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use url::{Origin, Url};
use wasmtime::StoreLimits;
use wasmtime::component::{Resource, ResourceTable};

/// Host representation of a filesystem capability handle.
#[derive(Debug)]
pub struct FilesystemResource {
    scope: Value,
}

/// Host representation of a network capability handle.
#[derive(Debug)]
pub struct NetworkResource {
    scope: Value,
}

/// Host representation of a process capability handle.
#[derive(Debug)]
pub struct ProcessResource {
    scope: Value,
}

/// Host representation of a UI capability handle.
#[derive(Debug)]
pub struct UiResource {
    scope: Value,
}

/// Host representation of a session capability handle.
#[derive(Debug)]
pub struct SessionResource {
    scope: Value,
}

/// Host representation of a provider capability handle.
#[derive(Debug)]
pub struct ProviderResource {
    scope: Value,
}

/// Per-store state used by generated Component Model bindings.
#[derive(Debug)]
pub(crate) struct HostState {
    pub(crate) table: ResourceTable,
    pub(crate) store_limits: StoreLimits,
    extension_id: String,
    generation: u64,
    grants: GrantedCapabilities,
    bridge: Arc<dyn RiExtBridge>,
    max_resources: usize,
    live_resources: usize,
}

impl HostState {
    pub(crate) fn new(
        extension_id: String,
        generation: u64,
        grants: GrantedCapabilities,
        bridge: Arc<dyn RiExtBridge>,
        limits: &HostLimits,
    ) -> Self {
        Self {
            table: ResourceTable::new(),
            store_limits: limits.build_store_limits(),
            extension_id,
            generation,
            grants,
            bridge,
            max_resources: limits.max_capability_resources,
            live_resources: 0,
        }
    }

    fn authorize_resource(
        &self,
        kind: CapabilityKind,
        scope: &Value,
    ) -> std::result::Result<(), types::CapabilityError> {
        validate_scope(kind, scope)
            .map_err(|error| types::CapabilityError::InvalidScope(error.to_string()))?;
        if !self.grants.permits(kind, scope) {
            return Err(types::CapabilityError::Denied(format!(
                "requested {kind} resource scope is not granted"
            )));
        }
        if self.live_resources >= self.max_resources {
            return Err(types::CapabilityError::LimitExceeded(format!(
                "at most {} capability resources may be live",
                self.max_resources
            )));
        }
        Ok(())
    }

    async fn call_bridge(
        &mut self,
        capability: CapabilityKind,
        operation: &str,
        scope: Value,
        payload: Value,
    ) -> std::result::Result<Value, types::CapabilityError> {
        self.bridge
            .call(BridgeCall {
                extension_id: self.extension_id.clone(),
                generation: self.generation,
                capability,
                operation: operation.to_owned(),
                scope,
                payload,
            })
            .await
            .map_err(bridge_error)
    }

    fn inserted_resource(&mut self) {
        self.live_resources = self.live_resources.saturating_add(1);
    }

    fn dropped_resource(&mut self) {
        self.live_resources = self.live_resources.saturating_sub(1);
    }
}

fn bridge_error(error: BridgeError) -> types::CapabilityError {
    match error.code {
        BridgeErrorCode::Denied => types::CapabilityError::Denied(error.message),
        BridgeErrorCode::InvalidScope => types::CapabilityError::InvalidScope(error.message),
        BridgeErrorCode::InvalidRequest => types::CapabilityError::InvalidRequest(error.message),
        BridgeErrorCode::Unavailable => types::CapabilityError::Unavailable(error.message),
        BridgeErrorCode::LimitExceeded => types::CapabilityError::LimitExceeded(error.message),
        BridgeErrorCode::Failed => types::CapabilityError::Failed(error.message),
    }
}

fn invalid_request(message: impl Into<String>) -> types::CapabilityError {
    types::CapabilityError::InvalidRequest(message.into())
}

fn decode<T: DeserializeOwned>(
    value: Value,
    operation: &str,
) -> std::result::Result<T, types::CapabilityError> {
    serde_json::from_value(value).map_err(|error| {
        types::CapabilityError::Failed(format!(
            "bridge returned invalid `{operation}` response: {error}"
        ))
    })
}

fn filesystem_scope(scope: &filesystem::FilesystemScope) -> Value {
    json!({
        "roots": scope.roots,
        "access": match scope.access {
            filesystem::FilesystemAccess::Read => "read",
            filesystem::FilesystemAccess::ReadWrite => "read-write",
        },
        "max-read-bytes": scope.max_read_bytes,
        "max-write-bytes": scope.max_write_bytes,
    })
}

fn network_scope(scope: &network::NetworkScope) -> Value {
    json!({
        "origins": scope.origins,
        "methods": scope.methods,
        "max-response-bytes": scope.max_response_bytes,
    })
}

fn process_scope(scope: &process::ProcessScope) -> Value {
    json!({
        "programs": scope.programs,
        "max-runtime-ms": scope.max_runtime_ms,
        "max-output-bytes": scope.max_output_bytes,
        "allow-environment": scope.allow_environment,
    })
}

fn ui_scope(scope: ui::UiScope) -> Value {
    let surfaces: Vec<_> = scope
        .surfaces
        .into_iter()
        .map(|surface| match surface {
            ui::UiSurface::Notification => "notification",
            ui::UiSurface::Prompt => "prompt",
            ui::UiSurface::Clipboard => "clipboard",
            ui::UiSurface::ExternalUri => "external-uri",
        })
        .collect();
    json!({"surfaces": surfaces})
}

fn session_scope(scope: &session::SessionScope) -> Value {
    json!({
        "namespaces": scope.namespaces,
        "writable": scope.writable,
    })
}

fn provider_scope(scope: &provider::ProviderScope) -> Value {
    json!({
        "providers": scope.providers,
        "models": scope.models,
        "allow-streaming": scope.allow_streaming,
    })
}

impl types::Host for HostState {}

impl filesystem::Host for HostState {}

impl filesystem::HostFilesystem for HostState {
    async fn new(
        &mut self,
        scope: filesystem::FilesystemScope,
    ) -> wasmtime::Result<std::result::Result<Resource<FilesystemResource>, types::CapabilityError>>
    {
        let scope = filesystem_scope(&scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Filesystem, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(FilesystemResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn read(
        &mut self,
        resource: Resource<FilesystemResource>,
        path: String,
    ) -> wasmtime::Result<std::result::Result<Vec<u8>, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_filesystem_path(&scope, &path, false) {
            return Ok(Err(error));
        }
        let max = scope_u64(&scope, "max-read-bytes");
        let value = match self
            .call_bridge(
                CapabilityKind::Filesystem,
                "read",
                scope,
                json!({"path": path}),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let bytes: Vec<u8> = match decode(value, "filesystem.read") {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max {
            return Ok(Err(types::CapabilityError::LimitExceeded(format!(
                "filesystem read returned more than {max} bytes"
            ))));
        }
        Ok(Ok(bytes))
    }

    async fn write(
        &mut self,
        resource: Resource<FilesystemResource>,
        path: String,
        contents: Vec<u8>,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_filesystem_path(&scope, &path, true) {
            return Ok(Err(error));
        }
        let max = scope_u64(&scope, "max-write-bytes");
        if u64::try_from(contents.len()).unwrap_or(u64::MAX) > max {
            return Ok(Err(types::CapabilityError::LimitExceeded(format!(
                "filesystem write exceeds {max} bytes"
            ))));
        }
        match self
            .call_bridge(
                CapabilityKind::Filesystem,
                "write",
                scope,
                json!({"path": path, "contents": contents}),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn list_directory(
        &mut self,
        resource: Resource<FilesystemResource>,
        path: String,
    ) -> wasmtime::Result<
        std::result::Result<Vec<filesystem::DirectoryEntry>, types::CapabilityError>,
    > {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_filesystem_path(&scope, &path, false) {
            return Ok(Err(error));
        }
        let value = match self
            .call_bridge(
                CapabilityKind::Filesystem,
                "list-directory",
                scope,
                json!({"path": path}),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let entries: Vec<DirectoryEntryResponse> = match decode(value, "filesystem.list-directory")
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        Ok(Ok(entries.into_iter().map(Into::into).collect()))
    }

    async fn remove(
        &mut self,
        resource: Resource<FilesystemResource>,
        path: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_filesystem_path(&scope, &path, true) {
            return Ok(Err(error));
        }
        match self
            .call_bridge(
                CapabilityKind::Filesystem,
                "remove",
                scope,
                json!({"path": path}),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn drop(&mut self, resource: Resource<FilesystemResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

impl network::Host for HostState {}

impl network::HostNetwork for HostState {
    async fn new(
        &mut self,
        scope: network::NetworkScope,
    ) -> wasmtime::Result<std::result::Result<Resource<NetworkResource>, types::CapabilityError>>
    {
        let scope = network_scope(&scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Network, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(NetworkResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn request(
        &mut self,
        resource: Resource<NetworkResource>,
        method: String,
        url: String,
        headers: Vec<network::Header>,
        body: Option<Vec<u8>>,
    ) -> wasmtime::Result<std::result::Result<network::NetworkResponse, types::CapabilityError>>
    {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_network_request(&scope, &method, &url) {
            return Ok(Err(error));
        }
        let payload = json!({
            "method": method,
            "url": url,
            "headers": headers
                .into_iter()
                .map(|header| json!({"name": header.name, "value": header.value}))
                .collect::<Vec<_>>(),
            "body": body,
        });
        let value = match self
            .call_bridge(CapabilityKind::Network, "request", scope.clone(), payload)
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let response: NetworkResponse = match decode(value, "network.request") {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let max = scope_u64(&scope, "max-response-bytes");
        if u64::try_from(response.body.len()).unwrap_or(u64::MAX) > max {
            return Ok(Err(types::CapabilityError::LimitExceeded(format!(
                "network response exceeds {max} bytes"
            ))));
        }
        Ok(Ok(network::NetworkResponse {
            status: response.status,
            headers: response
                .headers
                .into_iter()
                .map(|header| network::Header {
                    name: header.name,
                    value: header.value,
                })
                .collect(),
            body: response.body,
        }))
    }

    async fn drop(&mut self, resource: Resource<NetworkResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

impl process::Host for HostState {}

impl process::HostProcess for HostState {
    async fn new(
        &mut self,
        scope: process::ProcessScope,
    ) -> wasmtime::Result<std::result::Result<Resource<ProcessResource>, types::CapabilityError>>
    {
        let scope = process_scope(&scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Process, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(ProcessResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn run(
        &mut self,
        resource: Resource<ProcessResource>,
        request: process::ProcessRequest,
    ) -> wasmtime::Result<std::result::Result<process::ProcessOutput, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_process_request(&scope, &request) {
            return Ok(Err(error));
        }
        let max_runtime_ms = scope_u64(&scope, "max-runtime-ms");
        let max_output_bytes = scope_u64(&scope, "max-output-bytes");
        let payload = json!({
            "program": request.program,
            "arguments": request.arguments,
            "working-directory": request.working_directory,
            "environment": request.environment,
            "stdin": request.stdin,
        });
        let value = match tokio::time::timeout(
            Duration::from_millis(max_runtime_ms),
            self.call_bridge(CapabilityKind::Process, "run", scope, payload),
        )
        .await
        {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => return Ok(Err(error)),
            Err(_) => {
                return Ok(Err(types::CapabilityError::LimitExceeded(format!(
                    "process exceeded its {max_runtime_ms} ms runtime limit"
                ))));
            }
        };
        let output: ProcessOutput = match decode(value, "process.run") {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let output_bytes = output.stdout.len().saturating_add(output.stderr.len());
        if u64::try_from(output_bytes).unwrap_or(u64::MAX) > max_output_bytes {
            return Ok(Err(types::CapabilityError::LimitExceeded(format!(
                "process output exceeds {max_output_bytes} bytes"
            ))));
        }
        Ok(Ok(process::ProcessOutput {
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            timed_out: output.timed_out,
        }))
    }

    async fn drop(&mut self, resource: Resource<ProcessResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

impl ui::Host for HostState {}

impl ui::HostUi for HostState {
    async fn new(
        &mut self,
        scope: ui::UiScope,
    ) -> wasmtime::Result<std::result::Result<Resource<UiResource>, types::CapabilityError>> {
        let scope = ui_scope(scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Ui, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(UiResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn notify(
        &mut self,
        resource: Resource<UiResource>,
        level: ui::NotificationLevel,
        message: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = require_surface(&scope, "notification") {
            return Ok(Err(error));
        }
        let level = match level {
            ui::NotificationLevel::Info => "info",
            ui::NotificationLevel::Warning => "warning",
            ui::NotificationLevel::Error => "error",
        };
        match self
            .call_bridge(
                CapabilityKind::Ui,
                "notify",
                scope,
                json!({"level": level, "message": message}),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn prompt(
        &mut self,
        resource: Resource<UiResource>,
        message: String,
        options: Vec<ui::PromptOption>,
    ) -> wasmtime::Result<std::result::Result<Option<String>, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = require_surface(&scope, "prompt") {
            return Ok(Err(error));
        }
        let value = match self
            .call_bridge(
                CapabilityKind::Ui,
                "prompt",
                scope,
                json!({
                    "message": message,
                    "options": options
                        .into_iter()
                        .map(|option| json!({"id": option.id, "label": option.label}))
                        .collect::<Vec<_>>()
                }),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        Ok(decode(value, "ui.prompt"))
    }

    async fn set_clipboard(
        &mut self,
        resource: Resource<UiResource>,
        text: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = require_surface(&scope, "clipboard") {
            return Ok(Err(error));
        }
        match self
            .call_bridge(
                CapabilityKind::Ui,
                "set-clipboard",
                scope,
                json!({"text": text}),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn open_uri(
        &mut self,
        resource: Resource<UiResource>,
        uri: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = require_surface(&scope, "external-uri") {
            return Ok(Err(error));
        }
        if http_origin(&uri).is_none() {
            return Ok(Err(invalid_request(
                "UI open-uri only accepts HTTP(S) URIs",
            )));
        }
        match self
            .call_bridge(CapabilityKind::Ui, "open-uri", scope, json!({"uri": uri}))
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn drop(&mut self, resource: Resource<UiResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

impl session::Host for HostState {}

impl session::HostSession for HostState {
    async fn new(
        &mut self,
        scope: session::SessionScope,
    ) -> wasmtime::Result<std::result::Result<Resource<SessionResource>, types::CapabilityError>>
    {
        let scope = session_scope(&scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Session, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(SessionResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn get(
        &mut self,
        resource: Resource<SessionResource>,
        namespace: String,
        key: String,
    ) -> wasmtime::Result<std::result::Result<Option<String>, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_namespace(&scope, &namespace, false) {
            return Ok(Err(error));
        }
        let value = match self
            .call_bridge(
                CapabilityKind::Session,
                "get",
                scope,
                json!({"namespace": namespace, "key": key}),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        Ok(decode(value, "session.get"))
    }

    async fn put(
        &mut self,
        resource: Resource<SessionResource>,
        namespace: String,
        key: String,
        value_json: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_namespace(&scope, &namespace, true) {
            return Ok(Err(error));
        }
        if serde_json::from_str::<Value>(&value_json).is_err() {
            return Ok(Err(invalid_request(
                "session value-json must contain valid JSON",
            )));
        }
        match self
            .call_bridge(
                CapabilityKind::Session,
                "put",
                scope,
                json!({
                    "namespace": namespace,
                    "key": key,
                    "value-json": value_json
                }),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn delete(
        &mut self,
        resource: Resource<SessionResource>,
        namespace: String,
        key: String,
    ) -> wasmtime::Result<std::result::Result<(), types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Err(error) = check_namespace(&scope, &namespace, true) {
            return Ok(Err(error));
        }
        match self
            .call_bridge(
                CapabilityKind::Session,
                "delete",
                scope,
                json!({"namespace": namespace, "key": key}),
            )
            .await
        {
            Ok(_) => Ok(Ok(())),
            Err(error) => Ok(Err(error)),
        }
    }

    async fn drop(&mut self, resource: Resource<SessionResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

impl provider::Host for HostState {}

impl provider::HostProvider for HostState {
    async fn new(
        &mut self,
        scope: provider::ProviderScope,
    ) -> wasmtime::Result<std::result::Result<Resource<ProviderResource>, types::CapabilityError>>
    {
        let scope = provider_scope(&scope);
        if let Err(error) = self.authorize_resource(CapabilityKind::Provider, &scope) {
            return Ok(Err(error));
        }
        let resource = self.table.push(ProviderResource { scope })?;
        self.inserted_resource();
        Ok(Ok(resource))
    }

    async fn complete(
        &mut self,
        resource: Resource<ProviderResource>,
        request_json: String,
    ) -> wasmtime::Result<std::result::Result<String, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        let request: Value = match serde_json::from_str(&request_json) {
            Ok(value) => value,
            Err(error) => {
                return Ok(Err(invalid_request(format!(
                    "provider request-json is invalid: {error}"
                ))));
            }
        };
        if let Err(error) = check_provider_request(&scope, &request) {
            return Ok(Err(error));
        }
        let value = match self
            .call_bridge(
                CapabilityKind::Provider,
                "complete",
                scope,
                json!({"request": request}),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        Ok(decode(value, "provider.complete"))
    }

    async fn list_models(
        &mut self,
        resource: Resource<ProviderResource>,
        provider_id: Option<String>,
    ) -> wasmtime::Result<std::result::Result<String, types::CapabilityError>> {
        let scope = self.table.get(&resource)?.scope.clone();
        if let Some(provider_id) = provider_id.as_deref()
            && !scope_array_contains(&scope, "providers", provider_id, false)
        {
            return Ok(Err(types::CapabilityError::Denied(format!(
                "provider `{provider_id}` is outside the resource scope"
            ))));
        }
        let value = match self
            .call_bridge(
                CapabilityKind::Provider,
                "list-models",
                scope,
                json!({"provider-id": provider_id}),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        Ok(decode(value, "provider.list-models"))
    }

    async fn drop(&mut self, resource: Resource<ProviderResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(resource)?;
        self.dropped_resource();
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct DirectoryEntryResponse {
    path: String,
    kind: EntryKindResponse,
    size: u64,
}

impl From<DirectoryEntryResponse> for filesystem::DirectoryEntry {
    fn from(value: DirectoryEntryResponse) -> Self {
        Self {
            path: value.path,
            kind: match value.kind {
                EntryKindResponse::File => filesystem::EntryKind::File,
                EntryKindResponse::Directory => filesystem::EntryKind::Directory,
                EntryKindResponse::Symlink => filesystem::EntryKind::Symlink,
                EntryKindResponse::Other => filesystem::EntryKind::Other,
            },
            size: value.size,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum EntryKindResponse {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug, Deserialize)]
struct HeaderResponse {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
struct NetworkResponse {
    status: u16,
    headers: Vec<HeaderResponse>,
    body: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct ProcessOutput {
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    timed_out: bool,
}

fn check_filesystem_path(
    scope: &Value,
    path: &str,
    write: bool,
) -> std::result::Result<(), types::CapabilityError> {
    if path.contains('\0') || path.split(['/', '\\']).any(|part| part == "..") {
        return Err(invalid_request("filesystem path is not lexically safe"));
    }
    if write && scope.get("access").and_then(Value::as_str) != Some("read-write") {
        return Err(types::CapabilityError::Denied(
            "filesystem resource is read-only".to_owned(),
        ));
    }
    let roots = scope
        .get("roots")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    if !roots.into_iter().any(|root| path_within(root, path)) {
        return Err(types::CapabilityError::Denied(format!(
            "path `{path}` is outside the filesystem resource roots"
        )));
    }
    Ok(())
}

fn path_within(root: &str, path: &str) -> bool {
    if root == "*" {
        return true;
    }
    let root = normalize_path(root);
    let path = normalize_path(path);
    path == root
        || path
            .strip_prefix(&root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn normalize_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_end_matches('/').to_owned();
    if cfg!(windows) {
        normalized.to_ascii_lowercase()
    } else {
        normalized
    }
}

fn check_network_request(
    scope: &Value,
    method: &str,
    url: &str,
) -> std::result::Result<(), types::CapabilityError> {
    let method = method.to_ascii_uppercase();
    if !scope_array_contains(scope, "methods", &method, true) {
        return Err(types::CapabilityError::Denied(format!(
            "HTTP method `{method}` is outside the network resource scope"
        )));
    }
    let origin = http_origin(url)
        .ok_or_else(|| invalid_request("network URL must be an absolute HTTP(S) URL"))?;
    if !scope_array_contains(scope, "origins", &origin, true) {
        return Err(types::CapabilityError::Denied(format!(
            "origin `{origin}` is outside the network resource scope"
        )));
    }
    Ok(())
}

fn http_origin(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return None;
    }
    match parsed.origin() {
        Origin::Tuple(..) => Some(parsed.origin().ascii_serialization()),
        Origin::Opaque(_) => None,
    }
}

fn check_process_request(
    scope: &Value,
    request: &process::ProcessRequest,
) -> std::result::Result<(), types::CapabilityError> {
    if !scope_array_contains(scope, "programs", &request.program, cfg!(windows)) {
        return Err(types::CapabilityError::Denied(format!(
            "program `{}` is outside the process resource scope",
            request.program
        )));
    }
    if !request.environment.is_empty()
        && scope.get("allow-environment").and_then(Value::as_bool) != Some(true)
    {
        return Err(types::CapabilityError::Denied(
            "process environment overrides are not granted".to_owned(),
        ));
    }
    Ok(())
}

fn require_surface(
    scope: &Value,
    surface: &str,
) -> std::result::Result<(), types::CapabilityError> {
    if scope_array_contains(scope, "surfaces", surface, false) {
        Ok(())
    } else {
        Err(types::CapabilityError::Denied(format!(
            "UI surface `{surface}` is outside the resource scope"
        )))
    }
}

fn check_namespace(
    scope: &Value,
    namespace: &str,
    write: bool,
) -> std::result::Result<(), types::CapabilityError> {
    if !scope_array_contains(scope, "namespaces", namespace, false) {
        return Err(types::CapabilityError::Denied(format!(
            "session namespace `{namespace}` is outside the resource scope"
        )));
    }
    if write && scope.get("writable").and_then(Value::as_bool) != Some(true) {
        return Err(types::CapabilityError::Denied(
            "session resource is read-only".to_owned(),
        ));
    }
    Ok(())
}

fn check_provider_request(
    scope: &Value,
    request: &Value,
) -> std::result::Result<(), types::CapabilityError> {
    let object = request
        .as_object()
        .ok_or_else(|| invalid_request("provider request must be a JSON object"))?;
    if let Some(provider) = object.get("provider").and_then(Value::as_str)
        && !scope_array_contains(scope, "providers", provider, false)
    {
        return Err(types::CapabilityError::Denied(format!(
            "provider `{provider}` is outside the resource scope"
        )));
    }
    if let Some(model) = object.get("model").and_then(Value::as_str) {
        let models = scope.get("models").and_then(Value::as_array);
        if models.is_some_and(|models| !models.is_empty())
            && !scope_array_contains(scope, "models", model, false)
        {
            return Err(types::CapabilityError::Denied(format!(
                "model `{model}` is outside the resource scope"
            )));
        }
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true)
        && scope.get("allow-streaming").and_then(Value::as_bool) != Some(true)
    {
        return Err(types::CapabilityError::Denied(
            "streaming provider calls are not granted".to_owned(),
        ));
    }
    Ok(())
}

fn scope_array_contains(scope: &Value, key: &str, needle: &str, case_insensitive: bool) -> bool {
    scope
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .any(|value| {
            value == "*"
                || if case_insensitive {
                    value.eq_ignore_ascii_case(needle)
                } else {
                    value == needle
                }
        })
}

fn scope_u64(scope: &Value, key: &str) -> u64 {
    scope.get(key).and_then(Value::as_u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoAmbientBridge;

    #[tokio::test]
    async fn resource_constructor_denies_ungranted_scope() {
        let limits = HostLimits::default();
        let mut state = HostState::new(
            "dev.ri.test".to_owned(),
            1,
            GrantedCapabilities::default(),
            Arc::new(NoAmbientBridge),
            &limits,
        );

        let result = <HostState as ui::HostUi>::new(
            &mut state,
            ui::UiScope {
                surfaces: vec![ui::UiSurface::Notification],
            },
        )
        .await
        .expect("host constructor must not trap");

        assert!(matches!(result, Err(types::CapabilityError::Denied(_))));
    }

    #[test]
    fn http_origin_uses_url_origin_canonicalization() {
        assert_eq!(
            http_origin("HTTPS://Example.COM:443/path?query#fragment"),
            Some("https://example.com".to_owned())
        );
        assert_eq!(
            http_origin("http://[::1]:8080/path"),
            Some("http://[::1]:8080".to_owned())
        );
    }

    #[test]
    fn http_origin_rejects_non_http_relative_and_credential_urls() {
        for invalid in [
            "/relative",
            "file:///tmp/data",
            "https://user@example.com/path",
            "not a url",
        ] {
            assert_eq!(http_origin(invalid), None, "accepted {invalid:?}");
        }
    }
}
