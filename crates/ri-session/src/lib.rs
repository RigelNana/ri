//! Typed, append-only conversation sessions.
//!
//! A session is a tree of immutable entries.  Appending normally creates a
//! child of the active leaf; moving the leaf writes a `leaf` entry so branch
//! selection survives process restarts.  This crate provides in-memory and
//! durable JSONL repositories plus backend-neutral traversal and context
//! projection.  Applications may implement [`SessionStore`] and
//! [`Repository`] to add another persistence backend.

pub mod backend;
mod error;
pub mod legacy;
mod model;
mod repository;
mod state;

pub use error::{Error, Result};
pub use model::{
    ActiveToolsChangeEntry, BranchSummaryEntry, CURRENT_SESSION_VERSION, CompactionEntry,
    CreateOptions, CustomEntry, CustomMessageEntry, EntryBase, ForkOptions, ForkPosition,
    LabelEntry, LeafEntry, ListOptions, MessageEntry, ModelChangeEntry, ModelSelection,
    SequencedEntry, SessionContext, SessionEntry, SessionHeader, SessionInfoEntry, SessionMetadata,
    SessionStats, ThinkingLevelChangeEntry, Usage, UsageCost,
};
pub use repository::{
    FileRepository, InMemoryRepository, InMemorySessionRepository, JsonlRepository,
    JsonlSessionRepository, MalformedEntryPolicy, MemoryRepository, Repository, Session,
    SessionStore,
};
pub use repository::{Repository as SessionRepository, SessionStore as Storage};
pub use state::{SessionSnapshot, SessionTreeNode};

#[cfg(test)]
mod tests;
