//! Wasmtime engine configuration, lifecycle, and generation registry.

use crate::bindings;
use crate::bindings::ri::extension::types;
use crate::bridge::{DescriptorPublication, DescriptorRetirement, NoAmbientBridge, RiExtBridge};
use crate::capability::HostState;
use crate::descriptor::{
    CommandPlacement, ExtensionDescriptor, ExtensionManifest, ToolRegistration, ViewLocation,
    ViewRegistration,
};
use crate::error::{HostError, Result};
use crate::limits::HostLimits;
use crate::model::{
    ActionBinding, ActionEvent, ActionKind, ActionResult, ActivationContext, ActivationResult,
    CommandInvocation, CommandResult, DeactivateReason, EventKind, ExtensionEvent, Invocation,
    InvocationResult, LifecyclePhase, ToolInvocation, ToolResult, View, ViewNode, ViewNodeKind,
    ViewProperty, ViewRequest,
};
use crate::policy::{CapabilityKind, CapabilityPolicy, CapabilityRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;
use tokio::sync::Mutex;
use wasmtime::component::{Component, HasSelf, Linker};
use wasmtime::{Config, Engine, Store};

/// Opaque identity for one loaded extension generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExtensionHandle {
    id: String,
    generation: u64,
}

impl ExtensionHandle {
    /// Stable extension ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Monotonic generation assigned by this host.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Read-only status for a loaded generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionStatus {
    /// Extension identity.
    pub handle: ExtensionHandle,
    /// Current lifecycle phase.
    pub phase: LifecyclePhase,
    /// Validated descriptor.
    pub descriptor: ExtensionDescriptor,
}

/// Builder for a deny-by-default extension host.
#[derive(Debug)]
pub struct WasmExtensionHostBuilder {
    policy: CapabilityPolicy,
    bridge: Arc<dyn RiExtBridge>,
    limits: HostLimits,
}

impl Default for WasmExtensionHostBuilder {
    fn default() -> Self {
        Self {
            policy: CapabilityPolicy::deny_all(),
            bridge: Arc::new(NoAmbientBridge),
            limits: HostLimits::default(),
        }
    }
}

impl WasmExtensionHostBuilder {
    /// Creates a builder with no ambient capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the capability policy used at load time.
    #[must_use]
    pub fn policy(mut self, policy: CapabilityPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets the narrow native-host bridge.
    #[must_use]
    pub fn bridge(mut self, bridge: Arc<dyn RiExtBridge>) -> Self {
        self.bridge = bridge;
        self
    }

    /// Sets finite component, store, fuel, and wall-clock limits.
    #[must_use]
    pub fn limits(mut self, limits: HostLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Configures the Wasmtime Component Model engine.
    ///
    /// # Errors
    ///
    /// Returns an error when limits are invalid or Wasmtime rejects the engine
    /// configuration.
    pub fn build(self) -> Result<WasmExtensionHost> {
        self.limits.validate()?;
        let mut config = Config::new();
        config.wasm_component_model(true);
        // ResourceLimiter does not receive shared-memory growth callbacks.
        config.wasm_threads(false);
        config.max_wasm_stack(self.limits.max_wasm_stack_bytes);
        config.consume_fuel(true);
        config.epoch_interruption(true);
        let engine =
            Engine::new(&config).map_err(|error| HostError::Configuration(error.to_string()))?;
        Ok(WasmExtensionHost {
            inner: Arc::new(HostInner {
                engine,
                policy: self.policy,
                bridge: self.bridge,
                limits: self.limits,
                next_generation: AtomicU64::new(1),
                ticker_started: AtomicBool::new(false),
                entries: RwLock::new(HashMap::new()),
            }),
        })
    }
}

/// Sandboxed host for versioned ri extension components.
///
/// The linker contains only imports generated from `ri:extension@1.0.0`.
/// WASI is deliberately not linked, so filesystem, sockets, environment,
/// clocks, and process APIs are absent unless represented by an explicit
/// scoped resource and serviced through [`RiExtBridge`].
#[derive(Clone)]
pub struct WasmExtensionHost {
    inner: Arc<HostInner>,
}

impl fmt::Debug for WasmExtensionHost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WasmExtensionHost")
            .field("limits", &self.inner.limits)
            .field("policy", &self.inner.policy)
            .field("bridge", &self.inner.bridge)
            .finish_non_exhaustive()
    }
}

impl WasmExtensionHost {
    /// Creates a configurable deny-by-default host builder.
    pub fn builder() -> WasmExtensionHostBuilder {
        WasmExtensionHostBuilder::new()
    }

    /// Returns the effective finite limits.
    pub fn limits(&self) -> &HostLimits {
        &self.inner.limits
    }

    /// Parses a strict JSON manifest and delegates to [`Self::load`].
    ///
    /// # Errors
    ///
    /// Returns any manifest parsing, policy, compilation, linking,
    /// instantiation, descriptor, bridge, or limit error from loading.
    pub async fn load_json(
        &self,
        manifest_json: &str,
        component_bytes: &[u8],
    ) -> Result<ExtensionHandle> {
        let manifest = ExtensionManifest::from_json(manifest_json)?;
        self.load(manifest, component_bytes).await
    }

    /// Compiles, links, instantiates, describes, and registers a component.
    ///
    /// Capability policy is evaluated before untrusted bytes are compiled.
    /// Loading the same extension ID replaces its registry entry and makes all
    /// earlier handles stale.
    ///
    /// # Errors
    ///
    /// Returns a typed error when validation, authorization, compilation,
    /// linking, instantiation, guest description, or bridge publication fails.
    pub async fn load(
        &self,
        manifest: ExtensionManifest,
        component_bytes: &[u8],
    ) -> Result<ExtensionHandle> {
        self.ensure_epoch_ticker();
        manifest.validate()?;
        if component_bytes.len() > self.inner.limits.max_component_bytes {
            return Err(HostError::ComponentTooLarge {
                actual: component_bytes.len(),
                limit: self.inner.limits.max_component_bytes,
            });
        }
        let grants = self.inner.policy.authorize(&manifest.capabilities)?;
        let generation = self
            .inner
            .next_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
                generation.checked_add(1)
            })
            .map_err(|_| {
                HostError::Configuration("extension generation counter exhausted".to_owned())
            })?;

        let component = Component::from_binary(&self.inner.engine, component_bytes)
            .map_err(|error| HostError::Compilation(error.to_string()))?;
        let mut linker = Linker::new(&self.inner.engine);
        bindings::Extension::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|error| HostError::Linking(error.to_string()))?;

        let state = HostState::new(
            manifest.id.clone(),
            generation,
            grants,
            Arc::clone(&self.inner.bridge),
            &self.inner.limits,
        );
        let mut store = Store::new(&self.inner.engine, state);
        store.limiter(|state| &mut state.store_limits);
        prepare_store(&mut store, &self.inner.limits)?;

        let timeout = Duration::from_millis(self.inner.limits.call_timeout_ms);
        let bindings = tokio::time::timeout(
            timeout,
            bindings::Extension::instantiate_async(&mut store, &component, &linker),
        )
        .await
        .map_err(|_| HostError::Timeout {
            timeout_ms: self.inner.limits.call_timeout_ms,
        })?
        .map_err(|error| classify_instantiation(&error))?;

        prepare_store(&mut store, &self.inner.limits)?;
        let raw_descriptor = tokio::time::timeout(
            timeout,
            bindings.ri_extension_guest().call_descriptor(&mut store),
        )
        .await
        .map_err(|_| HostError::Timeout {
            timeout_ms: self.inner.limits.call_timeout_ms,
        })?
        .map_err(|error| classify_execution(&error, &self.inner.limits))?
        .map_err(guest_error)?;
        let raw_descriptor_size = serde_json::to_vec(&raw_descriptor)
            .map_err(|error| HostError::InvalidDescriptor(error.to_string()))?
            .len();
        if raw_descriptor_size > self.inner.limits.max_descriptor_bytes {
            return Err(HostError::InvalidDescriptor(format!(
                "serialized ABI descriptor is {raw_descriptor_size} bytes; limit is {}",
                self.inner.limits.max_descriptor_bytes
            )));
        }
        let descriptor = descriptor_from_wit(raw_descriptor)?;
        descriptor.validate_against(&manifest)?;

        let descriptor_json = serde_json::to_value(&descriptor)
            .map_err(|error| HostError::InvalidDescriptor(error.to_string()))?;
        let descriptor_size = serde_json::to_vec(&descriptor_json)
            .map_err(|error| HostError::InvalidDescriptor(error.to_string()))?
            .len();
        if descriptor_size > self.inner.limits.max_descriptor_bytes {
            return Err(HostError::InvalidDescriptor(format!(
                "serialized descriptor is {descriptor_size} bytes; limit is {}",
                self.inner.limits.max_descriptor_bytes
            )));
        }
        tokio::time::timeout(
            timeout,
            self.inner.bridge.publish_descriptor(DescriptorPublication {
                extension_id: manifest.id.clone(),
                generation,
                descriptor: descriptor_json,
            }),
        )
        .await
        .map_err(|_| HostError::Timeout {
            timeout_ms: self.inner.limits.call_timeout_ms,
        })?
        .map_err(|error| HostError::Bridge(error.to_string()))?;

        let handle = ExtensionHandle {
            id: manifest.id,
            generation,
        };
        let entry = Arc::new(RegistryEntry {
            current: AtomicBool::new(true),
            generation,
            descriptor: descriptor.clone(),
            instance: Mutex::new(LoadedInstance {
                store,
                bindings,
                phase: LifecyclePhase::Loaded,
                poisoned: false,
            }),
        });
        let mut entries = write_lock(&self.inner.entries);
        if let Some(previous) = entries.insert(handle.id.clone(), entry) {
            previous.current.store(false, Ordering::SeqCst);
        }
        Ok(handle)
    }

    /// Returns a validated descriptor for a current handle.
    ///
    /// # Errors
    ///
    /// Returns a stale-handle error when the generation is no longer current.
    pub fn descriptor(&self, handle: &ExtensionHandle) -> Result<ExtensionDescriptor> {
        Ok(self.resolve(handle)?.descriptor.clone())
    }

    /// Returns lifecycle status for a current handle.
    ///
    /// # Errors
    ///
    /// Returns a stale-handle error when the generation is no longer current.
    pub async fn status(&self, handle: &ExtensionHandle) -> Result<ExtensionStatus> {
        let entry = self.resolve(handle)?;
        let instance = entry.instance.lock().await;
        ensure_entry_current(&entry, handle)?;
        Ok(ExtensionStatus {
            handle: handle.clone(),
            phase: instance.phase,
            descriptor: entry.descriptor.clone(),
        })
    }

    /// Invokes one lifecycle, event, registration, view, or action export.
    ///
    /// # Errors
    ///
    /// Returns a typed stale-generation, lifecycle, guest, timeout, fuel,
    /// resource-limit, or result-validation error.
    pub async fn invoke(
        &self,
        handle: &ExtensionHandle,
        invocation: Invocation,
    ) -> Result<InvocationResult> {
        self.ensure_epoch_ticker();
        let entry = self.resolve(handle)?;
        let mut instance = entry.instance.lock().await;
        ensure_entry_current(&entry, handle)?;
        if instance.poisoned {
            return Err(HostError::InvalidLifecycle {
                id: handle.id.clone(),
                from: instance.phase.as_str(),
                operation: "invoke-poisoned-instance",
            });
        }
        let loaded: &mut LoadedInstance = &mut instance;

        let timeout = Duration::from_millis(self.inner.limits.call_timeout_ms);
        macro_rules! call_guest {
            ($future:expr) => {{
                prepare_store(&mut loaded.store, &self.inner.limits)?;
                match tokio::time::timeout(timeout, $future).await {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => {
                        return Err(classify_execution(&error, &self.inner.limits));
                    }
                    Err(_) => {
                        loaded.poisoned = true;
                        return Err(HostError::Timeout {
                            timeout_ms: self.inner.limits.call_timeout_ms,
                        });
                    }
                }
            }};
        }

        match invocation {
            Invocation::Activate(context) => {
                require_phase(handle, loaded.phase, LifecyclePhase::Loaded, "activate")?;
                let context = activation_to_wit(context)?;
                let result = call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_activate(&mut loaded.store, &context)
                )
                .map_err(guest_error)?;
                let result = activation_from_wit(result)?;
                loaded.phase = LifecyclePhase::Active;
                Ok(InvocationResult::Activated(result))
            }
            Invocation::Event(event) => {
                require_active(handle, loaded.phase, "on-event")?;
                let event = event_to_wit(event)?;
                call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_on_event(&mut loaded.store, &event)
                )
                .map_err(guest_error)?;
                Ok(InvocationResult::EventDelivered)
            }
            Invocation::Tool(invocation) => {
                require_active(handle, loaded.phase, "invoke-tool")?;
                ensure_registered(
                    &entry.descriptor.tools,
                    &invocation.id,
                    |tool| tool.id.as_str(),
                    "tool",
                )?;
                let invocation = tool_to_wit(invocation)?;
                let result = call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_invoke_tool(&mut loaded.store, &invocation)
                )
                .map_err(guest_error)?;
                Ok(InvocationResult::Tool(tool_from_wit(&result)?))
            }
            Invocation::Command(invocation) => {
                require_active(handle, loaded.phase, "invoke-command")?;
                ensure_registered(
                    &entry.descriptor.commands,
                    &invocation.id,
                    |command| command.id.as_str(),
                    "command",
                )?;
                let invocation = command_to_wit(invocation)?;
                let result = call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_invoke_command(&mut loaded.store, &invocation)
                )
                .map_err(guest_error)?;
                let result = command_from_wit(result)?;
                for view_id in &result.refresh_views {
                    ensure_registered(
                        &entry.descriptor.views,
                        view_id,
                        |view| view.id.as_str(),
                        "view",
                    )?;
                }
                Ok(InvocationResult::Command(result))
            }
            Invocation::RenderView(request) => {
                require_active(handle, loaded.phase, "render-view")?;
                ensure_registered(
                    &entry.descriptor.views,
                    &request.id,
                    |view| view.id.as_str(),
                    "view",
                )?;
                let request = view_request_to_wit(request)?;
                let view = call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_render_view(&mut loaded.store, &request)
                )
                .map_err(guest_error)?;
                let view = view_from_wit(view)?;
                view.validate()?;
                Ok(InvocationResult::View(view))
            }
            Invocation::Action(event) => {
                require_active(handle, loaded.phase, "handle-action")?;
                ensure_registered(
                    &entry.descriptor.views,
                    &event.view_id,
                    |view| view.id.as_str(),
                    "view",
                )?;
                let event = action_to_wit(event)?;
                let result = call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_handle_action(&mut loaded.store, &event)
                )
                .map_err(guest_error)?;
                let result = action_from_wit(result)?;
                if let Some(view) = &result.replacement_view {
                    view.validate()?;
                }
                Ok(InvocationResult::Action(result))
            }
            Invocation::Deactivate(reason) => {
                if loaded.phase == LifecyclePhase::Deactivated {
                    return Err(HostError::InvalidLifecycle {
                        id: handle.id.clone(),
                        from: loaded.phase.as_str(),
                        operation: "deactivate",
                    });
                }
                let reason = deactivate_to_wit(reason);
                call_guest!(
                    loaded
                        .bindings
                        .ri_extension_guest()
                        .call_deactivate(&mut loaded.store, reason)
                )
                .map_err(guest_error)?;
                loaded.phase = LifecyclePhase::Deactivated;
                Ok(InvocationResult::Deactivated)
            }
        }
    }

    /// Deactivates when needed and removes a current generation.
    ///
    /// # Errors
    ///
    /// Returns a stale-generation, lifecycle, guest, timeout, fuel, or
    /// resource-limit error.
    pub async fn unload(&self, handle: &ExtensionHandle, reason: DeactivateReason) -> Result<()> {
        let entry = self.resolve(handle)?;
        let phase = {
            let instance = entry.instance.lock().await;
            ensure_entry_current(&entry, handle)?;
            instance.phase
        };
        if phase != LifecyclePhase::Deactivated {
            self.invoke(handle, Invocation::Deactivate(reason)).await?;
        }

        let timeout = Duration::from_millis(self.inner.limits.call_timeout_ms);
        tokio::time::timeout(
            timeout,
            self.inner.bridge.retire_descriptor(DescriptorRetirement {
                extension_id: handle.id.clone(),
                generation: handle.generation,
            }),
        )
        .await
        .map_err(|_| HostError::Timeout {
            timeout_ms: self.inner.limits.call_timeout_ms,
        })?
        .map_err(|error| HostError::Bridge(error.to_string()))?;

        let mut entries = write_lock(&self.inner.entries);
        let current = entries.get(&handle.id).map(|entry| entry.generation);
        ensure_generation(&handle.id, handle.generation, current)?;
        let removed = entries
            .remove(&handle.id)
            .ok_or_else(|| HostError::NotLoaded(handle.id.clone()))?;
        removed.current.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn resolve(&self, handle: &ExtensionHandle) -> Result<Arc<RegistryEntry>> {
        let entries = read_lock(&self.inner.entries);
        let current = entries.get(&handle.id);
        ensure_generation(
            &handle.id,
            handle.generation,
            current.map(|entry| entry.generation),
        )?;
        Ok(Arc::clone(
            current.expect("generation check guarantees a registry entry"),
        ))
    }

    fn ensure_epoch_ticker(&self) {
        if self
            .inner
            .ticker_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let weak = Arc::downgrade(&self.inner);
        let tick = Duration::from_millis(self.inner.limits.epoch_tick_ms);
        tokio::spawn(epoch_ticker(weak, tick));
    }
}

struct HostInner {
    engine: Engine,
    policy: CapabilityPolicy,
    bridge: Arc<dyn RiExtBridge>,
    limits: HostLimits,
    next_generation: AtomicU64,
    ticker_started: AtomicBool,
    entries: RwLock<HashMap<String, Arc<RegistryEntry>>>,
}

struct RegistryEntry {
    current: AtomicBool,
    generation: u64,
    descriptor: ExtensionDescriptor,
    instance: Mutex<LoadedInstance>,
}

struct LoadedInstance {
    store: Store<HostState>,
    bindings: bindings::Extension,
    phase: LifecyclePhase,
    poisoned: bool,
}

async fn epoch_ticker(inner: Weak<HostInner>, tick: Duration) {
    let mut interval = tokio::time::interval(tick);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval.tick().await;
    loop {
        interval.tick().await;
        let Some(inner) = inner.upgrade() else {
            break;
        };
        inner.engine.increment_epoch();
    }
}

fn prepare_store(store: &mut Store<HostState>, limits: &HostLimits) -> Result<()> {
    store
        .set_fuel(limits.fuel_per_call)
        .map_err(|error| HostError::Configuration(error.to_string()))?;
    store.set_epoch_deadline(limits.deadline_ticks());
    store.epoch_deadline_trap();
    Ok(())
}

fn classify_execution(error: &wasmtime::Error, limits: &HostLimits) -> HostError {
    if let Some(trap) = error.downcast_ref::<wasmtime::Trap>() {
        match trap {
            wasmtime::Trap::OutOfFuel => return HostError::FuelExhausted,
            wasmtime::Trap::Interrupt => {
                return HostError::Timeout {
                    timeout_ms: limits.call_timeout_ms,
                };
            }
            wasmtime::Trap::AllocationTooLarge | wasmtime::Trap::StackOverflow => {
                return HostError::ResourceLimit(error.to_string());
            }
            _ => {}
        }
    }
    let message = error.to_string();
    let lowercase = message.to_ascii_lowercase();
    if lowercase.contains("fuel") {
        HostError::FuelExhausted
    } else if lowercase.contains("epoch")
        || lowercase.contains("interrupt")
        || lowercase.contains("deadline")
    {
        HostError::Timeout {
            timeout_ms: limits.call_timeout_ms,
        }
    } else if (lowercase.contains("memory") || lowercase.contains("table"))
        && (lowercase.contains("limit")
            || lowercase.contains("grow")
            || lowercase.contains("allocation"))
    {
        HostError::ResourceLimit(message)
    } else {
        HostError::GuestTrap(message)
    }
}

fn classify_instantiation(error: &wasmtime::Error) -> HostError {
    let message = error.to_string();
    let lowercase = message.to_ascii_lowercase();
    if (lowercase.contains("memory") || lowercase.contains("table"))
        && (lowercase.contains("limit")
            || lowercase.contains("minimum")
            || lowercase.contains("grow")
            || lowercase.contains("allocation"))
    {
        HostError::ResourceLimit(message)
    } else {
        HostError::Instantiation(message)
    }
}

fn ensure_entry_current(entry: &RegistryEntry, handle: &ExtensionHandle) -> Result<()> {
    if entry.current.load(Ordering::SeqCst) && entry.generation == handle.generation {
        Ok(())
    } else {
        Err(HostError::StaleHandle {
            id: handle.id.clone(),
            handle_generation: handle.generation,
            current_generation: None,
        })
    }
}

fn ensure_generation(
    id: &str,
    handle_generation: u64,
    current_generation: Option<u64>,
) -> Result<()> {
    match current_generation {
        Some(current) if current == handle_generation => Ok(()),
        None if handle_generation == 0 => Err(HostError::NotLoaded(id.to_owned())),
        current => Err(HostError::StaleHandle {
            id: id.to_owned(),
            handle_generation,
            current_generation: current,
        }),
    }
}

fn require_phase(
    handle: &ExtensionHandle,
    actual: LifecyclePhase,
    expected: LifecyclePhase,
    operation: &'static str,
) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(HostError::InvalidLifecycle {
            id: handle.id.clone(),
            from: actual.as_str(),
            operation,
        })
    }
}

fn require_active(
    handle: &ExtensionHandle,
    phase: LifecyclePhase,
    operation: &'static str,
) -> Result<()> {
    require_phase(handle, phase, LifecyclePhase::Active, operation)
}

fn ensure_registered<T>(
    values: &[T],
    id: &str,
    get_id: impl Fn(&T) -> &str,
    kind: &'static str,
) -> Result<()> {
    if values.iter().any(|value| get_id(value) == id) {
        Ok(())
    } else {
        Err(HostError::Guest {
            code: "not-found",
            message: format!("{kind} `{id}` is not registered"),
        })
    }
}

fn descriptor_from_wit(descriptor: types::ExtensionDescriptor) -> Result<ExtensionDescriptor> {
    Ok(ExtensionDescriptor {
        id: descriptor.id,
        name: descriptor.name,
        version: descriptor.version,
        abi_version: descriptor.abi_version,
        description: descriptor.description,
        capabilities: descriptor
            .capabilities
            .into_iter()
            .map(|request| capability_from_wit(&request))
            .collect::<Result<_>>()?,
        tools: descriptor
            .tools
            .into_iter()
            .map(|tool| ToolRegistration {
                id: tool.id,
                title: tool.title,
                description: tool.description,
                input_schema_json: tool.input_schema_json,
                output_schema_json: tool.output_schema_json,
            })
            .collect(),
        commands: descriptor
            .commands
            .into_iter()
            .map(|command| crate::descriptor::CommandRegistration {
                id: command.id,
                title: command.title,
                description: command.description,
                placement: match command.placement {
                    types::CommandPlacement::Palette => CommandPlacement::Palette,
                    types::CommandPlacement::ContextMenu => CommandPlacement::ContextMenu,
                    types::CommandPlacement::SlashMenu => CommandPlacement::SlashMenu,
                    types::CommandPlacement::Hidden => CommandPlacement::Hidden,
                },
                argument_schema_json: command.argument_schema_json,
            })
            .collect(),
        views: descriptor
            .views
            .into_iter()
            .map(|view| ViewRegistration {
                id: view.id,
                title: view.title,
                location: match view.location {
                    types::ViewLocation::Sidebar => ViewLocation::Sidebar,
                    types::ViewLocation::Panel => ViewLocation::Panel,
                    types::ViewLocation::Editor => ViewLocation::Editor,
                    types::ViewLocation::Modal => ViewLocation::Modal,
                },
            })
            .collect(),
    })
}

fn capability_from_wit(request: &types::CapabilityRequest) -> Result<CapabilityRequest> {
    let kind = match request.kind {
        types::CapabilityKind::Filesystem => CapabilityKind::Filesystem,
        types::CapabilityKind::Network => CapabilityKind::Network,
        types::CapabilityKind::Process => CapabilityKind::Process,
        types::CapabilityKind::Ui => CapabilityKind::Ui,
        types::CapabilityKind::Session => CapabilityKind::Session,
        types::CapabilityKind::Provider => CapabilityKind::Provider,
    };
    let scope = serde_json::from_str(&request.scope_json).map_err(|error| {
        HostError::InvalidDescriptor(format!("{kind} capability scope-json is invalid: {error}"))
    })?;
    CapabilityRequest::new(kind, scope, request.required)
        .map_err(|error| HostError::InvalidDescriptor(error.to_string()))
}

fn guest_error(error: types::ExtensionError) -> HostError {
    match error {
        types::ExtensionError::InvalidArgument(message) => HostError::Guest {
            code: "invalid-argument",
            message,
        },
        types::ExtensionError::InvalidState(message) => HostError::Guest {
            code: "invalid-state",
            message,
        },
        types::ExtensionError::PermissionDenied(message) => HostError::Guest {
            code: "permission-denied",
            message,
        },
        types::ExtensionError::NotFound(message) => HostError::Guest {
            code: "not-found",
            message,
        },
        types::ExtensionError::Unavailable(message) => HostError::Guest {
            code: "unavailable",
            message,
        },
        types::ExtensionError::Failed(message) => HostError::Guest {
            code: "failed",
            message,
        },
    }
}

fn activation_to_wit(context: ActivationContext) -> Result<types::ActivationContext> {
    Ok(types::ActivationContext {
        session_id: context.session_id,
        workspace_uri: context.workspace_uri,
        configuration_json: encode_json(&context.configuration, "activation configuration")?,
    })
}

fn activation_from_wit(result: types::ActivationResult) -> Result<ActivationResult> {
    Ok(ActivationResult {
        subscriptions: result
            .subscriptions
            .into_iter()
            .map(event_kind_from_wit)
            .collect(),
        state: result
            .state_json
            .map(|json| decode_json(&json, "activation state"))
            .transpose()?,
    })
}

fn event_to_wit(event: ExtensionEvent) -> Result<types::ExtensionEvent> {
    Ok(types::ExtensionEvent {
        kind: event_kind_to_wit(event.kind),
        topic: event.topic,
        payload_json: encode_json(&event.payload, "event payload")?,
        sequence: event.sequence,
    })
}

fn tool_to_wit(invocation: ToolInvocation) -> Result<types::ToolInvocation> {
    Ok(types::ToolInvocation {
        id: invocation.id,
        input_json: encode_json(&invocation.input, "tool input")?,
        context_json: encode_json(&invocation.context, "tool context")?,
    })
}

fn tool_from_wit(result: &types::ToolResult) -> Result<ToolResult> {
    Ok(ToolResult {
        content: decode_json(&result.content_json, "tool content")?,
        is_error: result.is_error,
    })
}

fn command_to_wit(invocation: CommandInvocation) -> Result<types::CommandInvocation> {
    Ok(types::CommandInvocation {
        id: invocation.id,
        arguments_json: encode_json(&invocation.arguments, "command arguments")?,
        context_json: encode_json(&invocation.context, "command context")?,
    })
}

fn command_from_wit(result: types::CommandResult) -> Result<CommandResult> {
    Ok(CommandResult {
        output: decode_json(&result.output_json, "command output")?,
        refresh_views: result.refresh_views,
    })
}

fn view_request_to_wit(request: ViewRequest) -> Result<types::ViewRequest> {
    Ok(types::ViewRequest {
        id: request.id,
        context_json: encode_json(&request.context, "view context")?,
    })
}

fn action_to_wit(event: ActionEvent) -> Result<types::ActionEvent> {
    Ok(types::ActionEvent {
        view_id: event.view_id,
        action_id: event.action_id,
        kind: action_kind_to_wit(event.kind),
        payload_json: encode_json(&event.payload, "action payload")?,
    })
}

fn action_from_wit(result: types::ActionResult) -> Result<ActionResult> {
    Ok(ActionResult {
        output: decode_json(&result.output_json, "action output")?,
        replacement_view: result.replacement_view.map(view_from_wit).transpose()?,
    })
}

fn view_from_wit(view: types::View) -> Result<View> {
    Ok(View {
        root: view.root,
        nodes: view
            .nodes
            .into_iter()
            .map(|node| {
                Ok(ViewNode {
                    id: node.id,
                    kind: match node.kind {
                        types::ViewNodeKind::Container => ViewNodeKind::Container,
                        types::ViewNodeKind::Text => ViewNodeKind::Text,
                        types::ViewNodeKind::Markdown => ViewNodeKind::Markdown,
                        types::ViewNodeKind::Button => ViewNodeKind::Button,
                        types::ViewNodeKind::Input => ViewNodeKind::Input,
                        types::ViewNodeKind::Select => ViewNodeKind::Select,
                        types::ViewNodeKind::Image => ViewNodeKind::Image,
                        types::ViewNodeKind::Spacer => ViewNodeKind::Spacer,
                    },
                    text: node.text,
                    properties: node
                        .properties
                        .into_iter()
                        .map(|property| {
                            Ok(ViewProperty {
                                name: property.name,
                                value: decode_json(&property.value_json, "view property")?,
                            })
                        })
                        .collect::<Result<_>>()?,
                    actions: node
                        .actions
                        .into_iter()
                        .map(|action| {
                            Ok(ActionBinding {
                                action_id: action.action_id,
                                kind: action_kind_from_wit(action.kind),
                                payload: decode_json(
                                    &action.payload_json,
                                    "action binding payload",
                                )?,
                            })
                        })
                        .collect::<Result<_>>()?,
                    children: node.children,
                })
            })
            .collect::<Result<_>>()?,
    })
}

const fn event_kind_to_wit(kind: EventKind) -> types::EventKind {
    match kind {
        EventKind::Activated => types::EventKind::Activated,
        EventKind::Deactivating => types::EventKind::Deactivating,
        EventKind::ConfigurationChanged => types::EventKind::ConfigurationChanged,
        EventKind::SessionOpened => types::EventKind::SessionOpened,
        EventKind::SessionClosed => types::EventKind::SessionClosed,
        EventKind::ProviderChanged => types::EventKind::ProviderChanged,
        EventKind::ToolCompleted => types::EventKind::ToolCompleted,
        EventKind::CommandInvoked => types::EventKind::CommandInvoked,
        EventKind::Custom => types::EventKind::Custom,
    }
}

const fn event_kind_from_wit(kind: types::EventKind) -> EventKind {
    match kind {
        types::EventKind::Activated => EventKind::Activated,
        types::EventKind::Deactivating => EventKind::Deactivating,
        types::EventKind::ConfigurationChanged => EventKind::ConfigurationChanged,
        types::EventKind::SessionOpened => EventKind::SessionOpened,
        types::EventKind::SessionClosed => EventKind::SessionClosed,
        types::EventKind::ProviderChanged => EventKind::ProviderChanged,
        types::EventKind::ToolCompleted => EventKind::ToolCompleted,
        types::EventKind::CommandInvoked => EventKind::CommandInvoked,
        types::EventKind::Custom => EventKind::Custom,
    }
}

const fn action_kind_to_wit(kind: ActionKind) -> types::ActionKind {
    match kind {
        ActionKind::Activate => types::ActionKind::Activate,
        ActionKind::Change => types::ActionKind::Change,
        ActionKind::Submit => types::ActionKind::Submit,
        ActionKind::Dismiss => types::ActionKind::Dismiss,
        ActionKind::Custom => types::ActionKind::Custom,
    }
}

const fn action_kind_from_wit(kind: types::ActionKind) -> ActionKind {
    match kind {
        types::ActionKind::Activate => ActionKind::Activate,
        types::ActionKind::Change => ActionKind::Change,
        types::ActionKind::Submit => ActionKind::Submit,
        types::ActionKind::Dismiss => ActionKind::Dismiss,
        types::ActionKind::Custom => ActionKind::Custom,
    }
}

const fn deactivate_to_wit(reason: DeactivateReason) -> types::DeactivateReason {
    match reason {
        DeactivateReason::Disabled => types::DeactivateReason::Disabled,
        DeactivateReason::Reload => types::DeactivateReason::Reload,
        DeactivateReason::Shutdown => types::DeactivateReason::Shutdown,
        DeactivateReason::Failure => types::DeactivateReason::Failure,
    }
}

fn encode_json(value: &Value, label: &str) -> Result<String> {
    serde_json::to_string(value).map_err(|error| HostError::Guest {
        code: "invalid-argument",
        message: format!("{label} could not be encoded: {error}"),
    })
}

fn decode_json(json: &str, label: &str) -> Result<Value> {
    serde_json::from_str(json).map_err(|error| HostError::Guest {
        code: "invalid-argument",
        message: format!("guest returned invalid {label} JSON: {error}"),
    })
}

fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_generation_is_reported_with_current_generation() {
        let error =
            ensure_generation("dev.ri.test", 2, Some(3)).expect_err("old handle must be stale");
        assert!(matches!(
            error,
            HostError::StaleHandle {
                handle_generation: 2,
                current_generation: Some(3),
                ..
            }
        ));
    }

    #[test]
    fn host_engine_has_component_model_limits_configured() {
        let host = WasmExtensionHost::builder()
            .build()
            .expect("default host configuration must build");
        assert!(host.limits().fuel_per_call > 0);
        assert!(host.limits().max_memory_bytes >= 64 * 1024);
        assert!(host.limits().deadline_ticks() > 0);
    }

    #[tokio::test]
    async fn component_engine_parses_and_instantiates_empty_component() {
        const EMPTY_COMPONENT: &[u8] = b"\0asm\x0d\0\x01\0";
        let host = WasmExtensionHost::builder().build().expect("host builds");
        let component = Component::from_binary(&host.inner.engine, EMPTY_COMPONENT)
            .expect("component binary parses");
        let linker = Linker::<()>::new(&host.inner.engine);
        let mut store = Store::new(&host.inner.engine, ());
        linker
            .instantiate_async(&mut store, &component)
            .await
            .expect("empty component instantiates");
    }

    #[test]
    fn wit_source_declares_versioned_world_and_capabilities() {
        let wit = include_str!("../../../wit/ri-extension.wit");
        assert!(wit.contains("package ri:extension@1.0.0;"));
        assert!(wit.contains("world extension"));
        for capability in [
            "filesystem",
            "network",
            "process",
            "ui",
            "session",
            "provider",
        ] {
            assert!(wit.contains(&format!("resource {capability}")));
        }
    }

    #[test]
    fn bridge_error_codes_are_stable() {
        assert_eq!(crate::BridgeErrorCode::Denied.to_string(), "denied");
    }
}
