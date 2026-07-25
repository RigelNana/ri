//! Stable value types whose native and Pi wire representations are identical.
//!
//! Boundary DTOs with different required fields or serialization rules stay in
//! their owning crates and convert explicitly. This crate only owns values that
//! would otherwise be exact copies.

use serde::{Deserialize, Serialize};

/// Provider-neutral reasoning effort, ordered from disabled to maximum.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /// Disable reasoning.
    #[default]
    Off,
    /// Minimum supported effort.
    Minimal,
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
    /// Extended high effort.
    Xhigh,
    /// Provider maximum effort.
    Max,
}

impl ThinkingLevel {
    /// All levels in clamp order.
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    /// Stable Pi spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Delivery policy for steering and follow-up queues.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    /// Deliver every queued message at the next safe point.
    All,
    /// Deliver only the oldest queued message at each safe point.
    #[default]
    OneAtATime,
}

impl QueueMode {
    /// Stable Pi spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::OneAtATime => "one-at-a-time",
        }
    }
}

/// Reason a compaction operation started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompactionReason {
    /// Explicit user or API request.
    Manual,
    /// Configured context threshold was reached.
    Threshold,
    /// Provider context overflow recovery.
    Overflow,
}

impl CompactionReason {
    /// Stable Pi spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Threshold => "threshold",
            Self::Overflow => "overflow",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompactionReason, QueueMode, ThinkingLevel};

    #[test]
    fn shared_values_keep_pi_wire_spellings() {
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::Xhigh).expect("thinking level"),
            "\"xhigh\""
        );
        assert_eq!(
            serde_json::to_string(&QueueMode::OneAtATime).expect("queue mode"),
            "\"one-at-a-time\""
        );
        assert_eq!(
            serde_json::to_string(&CompactionReason::Threshold).expect("compaction reason"),
            "\"threshold\""
        );
    }
}
