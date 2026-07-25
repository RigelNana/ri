//! Deterministic execution and store resource limits.

use crate::error::{HostError, Result};
use serde::{Deserialize, Serialize};
use wasmtime::{StoreLimits, StoreLimitsBuilder};

const WASM_PAGE_BYTES: usize = 64 * 1024;

/// Limits applied to every loaded component and every guest invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HostLimits {
    /// Maximum encoded component size accepted by `load`.
    pub max_component_bytes: usize,
    /// Maximum bytes in each guest linear memory.
    pub max_memory_bytes: usize,
    /// Approximate maximum native stack used by guest WebAssembly.
    pub max_wasm_stack_bytes: usize,
    /// Maximum elements in each guest table.
    pub max_table_elements: usize,
    /// Maximum core instances held by one store.
    pub max_instances: usize,
    /// Maximum linear memories held by one store.
    pub max_memories: usize,
    /// Maximum tables held by one store.
    pub max_tables: usize,
    /// Maximum simultaneously live imported capability resources.
    pub max_capability_resources: usize,
    /// Fuel reset before each guest export call.
    pub fuel_per_call: u64,
    /// Wall-clock deadline for each guest export call.
    pub call_timeout_ms: u64,
    /// Frequency at which the host advances Wasmtime's epoch.
    pub epoch_tick_ms: u64,
    /// Maximum serialized guest descriptor size.
    pub max_descriptor_bytes: usize,
}

impl Default for HostLimits {
    fn default() -> Self {
        Self {
            max_component_bytes: 32 * 1024 * 1024,
            max_memory_bytes: 64 * 1024 * 1024,
            max_wasm_stack_bytes: 1024 * 1024,
            max_table_elements: 100_000,
            max_instances: 32,
            max_memories: 8,
            max_tables: 16,
            max_capability_resources: 64,
            fuel_per_call: 10_000_000,
            call_timeout_ms: 5_000,
            epoch_tick_ms: 10,
            max_descriptor_bytes: 1024 * 1024,
        }
    }
}

impl HostLimits {
    /// Rejects zero, nonsensical, or platform-incompatible limits.
    ///
    /// # Errors
    ///
    /// Returns [`HostError::InvalidLimits`] when any limit is zero,
    /// internally inconsistent, or below Wasm's minimum page size.
    pub fn validate(&self) -> Result<()> {
        if self.max_component_bytes == 0 {
            return invalid("max_component_bytes must be non-zero");
        }
        if self.max_memory_bytes < WASM_PAGE_BYTES {
            return invalid("max_memory_bytes must allow at least one 64 KiB Wasm page");
        }
        if self.max_wasm_stack_bytes < WASM_PAGE_BYTES {
            return invalid("max_wasm_stack_bytes must be at least 64 KiB");
        }
        if self.max_table_elements == 0 {
            return invalid("max_table_elements must be non-zero");
        }
        if self.max_instances == 0 || self.max_memories == 0 || self.max_tables == 0 {
            return invalid("instance, memory, and table counts must be non-zero");
        }
        if self.max_capability_resources == 0 {
            return invalid("max_capability_resources must be non-zero");
        }
        if self.fuel_per_call == 0 {
            return invalid("fuel_per_call must be non-zero");
        }
        if self.call_timeout_ms == 0 || self.epoch_tick_ms == 0 {
            return invalid("call_timeout_ms and epoch_tick_ms must be non-zero");
        }
        if self.epoch_tick_ms > self.call_timeout_ms {
            return invalid("epoch_tick_ms must not exceed call_timeout_ms");
        }
        if self.max_descriptor_bytes == 0 || self.max_descriptor_bytes > self.max_component_bytes {
            return invalid(
                "max_descriptor_bytes must be non-zero and no larger than max_component_bytes",
            );
        }
        Ok(())
    }

    /// Number of epoch ticks assigned to one invocation.
    pub fn deadline_ticks(&self) -> u64 {
        self.call_timeout_ms.div_ceil(self.epoch_tick_ms)
    }

    pub(crate) fn build_store_limits(&self) -> StoreLimits {
        StoreLimitsBuilder::new()
            .memory_size(self.max_memory_bytes)
            .table_elements(self.max_table_elements)
            .instances(self.max_instances)
            .memories(self.max_memories)
            .tables(self.max_tables)
            .trap_on_grow_failure(true)
            .build()
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(HostError::InvalidLimits(message.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmtime::ResourceLimiter;

    #[test]
    fn default_limits_are_finite_and_consistent() {
        let limits = HostLimits::default();
        limits.validate().expect("defaults must be valid");
        assert!(limits.max_wasm_stack_bytes >= WASM_PAGE_BYTES);
        assert_eq!(
            limits.deadline_ticks(),
            limits.call_timeout_ms / limits.epoch_tick_ms
        );
        let mut store_limits = limits.build_store_limits();
        assert!(
            store_limits
                .memory_growing(0, limits.max_memory_bytes, None)
                .expect("growth through the configured ceiling is allowed")
        );
        assert!(
            store_limits
                .memory_growing(0, limits.max_memory_bytes + WASM_PAGE_BYTES, None,)
                .is_err(),
            "trap_on_grow_failure must reject memory beyond the ceiling"
        );
        assert_eq!(
            ResourceLimiter::instances(&store_limits),
            limits.max_instances
        );
        assert_eq!(
            ResourceLimiter::memories(&store_limits),
            limits.max_memories
        );
        assert_eq!(ResourceLimiter::tables(&store_limits), limits.max_tables);
    }

    #[test]
    fn zero_fuel_is_rejected() {
        let limits = HostLimits {
            fuel_per_call: 0,
            ..HostLimits::default()
        };
        assert!(matches!(
            limits.validate(),
            Err(HostError::InvalidLimits(_))
        ));
    }

    #[test]
    fn sub_page_memory_limit_is_rejected() {
        let limits = HostLimits {
            max_memory_bytes: WASM_PAGE_BYTES - 1,
            ..HostLimits::default()
        };
        assert!(matches!(
            limits.validate(),
            Err(HostError::InvalidLimits(_))
        ));
    }
}
